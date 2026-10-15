"""Golden tests against a running service (local uvicorn or the Docker container).

mode 'imported': the cohort was loaded with import_cohort.py; every meal of the report's prediction inspector (176) must come
                 back with the same numbers (demo values are rounded to 0.1 mg/dL / 1 mg/dL·min).
mode 'onboard' : starts from raw data through the API, like a new user: POST /history with days 1-7, POST /signals with the
                 rest, then /predict for every labelled day-8+ meal; reports accuracy and the difference to the batch evaluation.
usage: python -I golden_test.py <url> imported <demo_data.json> <out.json>
       python -I golden_test.py <url> onboard  <CGMacros dir> <stage1.pkl> <final_preds.pkl> <out.json> [max people]
"""
import sys, os, json, time, pickle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, httpx

URL, MODE = sys.argv[1].rstrip("/"), sys.argv[2]
H = {"x-api-key": os.environ["PMRF_API_KEY"]} if os.environ.get("PMRF_API_KEY") else {}
cli = httpx.Client(base_url=URL, timeout=300, headers=H)


def post(path, body):
    r = cli.post(path, json=body)
    if r.status_code != 200:
        raise RuntimeError(f"{path}: {r.status_code} {r.text[:500]}")
    return r.json()


def ts(t):
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S")


if MODE == "imported":
    DEMO, OUT = sys.argv[3], sys.argv[4]
    demo = json.load(open(DEMO))
    rows, lat = [], []
    for mid, ref in demo["meals"].items():
        pid = ref["person_id"]
        meal_photo = ref["meal"].get("photo_key")
        # the demo stores thumbnails, not keys: recover the key from the stage-1 table via the meal id
        rows.append((mid, pid, ref))
    st1 = pickle.load(open(os.path.join(ROOT_DIR, "results", "final", "stage1.pkl"), "rb"))
    Mo = st1["outer"]["M"]
    DATA = os.path.join(ROOT_DIR, "cgmacros", "CGMacros")
    diffs = []
    for mid, pid, ref in rows:
        r = Mo.loc[int(mid.split("_")[1])]
        t = time.perf_counter()
        a = post(f"/v1/people/{pid}/predict", dict(photo_key=os.path.relpath(r.photo, DATA), meal_time=ts(r.time), meal_type=r.meal_type, log_meal=False))
        lat.append((time.perf_counter() - t) * 1000)
        p, q = a["prediction"], ref["prediction"]
        diffs.append(dict(meal=mid, peak_rise=abs(p["peak_rise_mgdl"] - q["peak_rise_mgdl"]), peak=abs(p["peak_glucose_mgdl"] - q["peak_glucose_mgdl"]),
                          g30_120=max(abs(p["glucose_at_min"][k] - q["glucose_at_min"][k]) for k in q["glucose_at_min"]),
                          ttp=abs(p["time_to_peak_min"] - q["time_to_peak_min"]), iauc=abs(p["iauc_0_120_mgdl_min"] - q["iauc_0_120_mgdl_min"]),
                          curve=max(abs(x - y) for x, y in zip(p["curve"]["glucose_mgdl"], q["curve"]["glucose_mgdl"])),
                          band=max(abs(x - y) for k in ("lower80", "upper80") for x, y in zip(p["curve"][k], q["curve"][k])),
                          p180=abs(p["prob_peak_above_mgdl"]["180"] - q["prob_peak_above_mgdl"]["180"]),
                          carbs=abs(a["meal"]["estimated_nutrition"]["carbs_g"] - ref["meal"]["estimated_nutrition"]["carbs_g"]),
                          what_if=max(abs(x["peak_rise_mgdl"] - y["peak_rise_mgdl"]) for x, y in zip(a["what_if"], ref["what_if"])),
                          contrib=max(abs(x["mgdl"] - y["mgdl"]) for x, y in zip(a["explanation"]["contributions"], ref["explanation"]["contributions"])),
                          flags_equal=a["confidence"] == ref["confidence"], tags_equal=a["meal"]["photo_tags"] == ref["meal"]["photo_tags"]))
    D = pd.DataFrame(diffs)
    num = D.drop(columns=["meal", "flags_equal", "tags_equal"])
    tol = dict(peak_rise=0.1, peak=0.1, g30_120=0.1, ttp=0, iauc=1, curve=0.1, band=0.1, p180=0.001, carbs=0.1, what_if=0.1, contrib=0.1)
    fails = D[(num > pd.Series(tol)).any(axis=1) | ~D.flags_equal | ~D.tags_equal]
    res = dict(mode=MODE, url=URL, meals=len(D), max_abs_diff=num.max().round(4).to_dict(), tolerance=tol, failed_meals=fails.meal.tolist(),
               passed=len(fails) == 0, latency_ms=dict(median=round(float(np.median(lat)), 1), p90=round(float(np.percentile(lat, 90)), 1)))
else:
    DATA, S1, FP, OUT = sys.argv[3:7]
    MAXP = int(sys.argv[7]) if len(sys.argv) > 7 else None
    from pmrf.data import load_cgmacros
    st1 = pickle.load(open(S1, "rb")); fp = pickle.load(open(FP, "rb"))
    TE = st1["TE"]
    batch = {i: {t: fp["blend"][t][k] for t in ("dpeak", "G60", "iauc", "ttp")} for k, i in enumerate(fp["te_index"])}
    persons = load_cgmacros(DATA)
    rel = lambda p: os.path.relpath(p, DATA) if isinstance(p, str) else None
    pids = sorted(TE.pid.unique())[:MAXP]
    rows, build = [], []
    for pid in pids:
        p = persons[pid]
        day = lambda t: p.day(t)
        end7 = p.t0 + pd.Timedelta(days=7) + pd.Timedelta(hours=3)     # 3 h after day 7 so the last history meal has its response
        c, h, m = p.cgm.dropna(), p.hr.dropna(), p.meals
        hm = m[day(m.time) <= 7]
        meal = lambda r, nut=True: dict(timestamp=ts(r.time), meal_type=r.meal_type, photo_key=rel(r.photo),
                                        **({} if not nut else dict(carbs_g=_f(r.carbs), protein_g=_f(r.protein), fat_g=_f(r.fat), fiber_g=_f(r.fiber))))
        _f = lambda v: None if not np.isfinite(v) else float(v)
        body = dict(start_time=ts(p.t0), cgm=[dict(timestamp=ts(t), glucose_mgdl=float(v)) for t, v in c[c.index < end7].items()],
                    hr=[dict(timestamp=ts(t), hr_bpm=float(v)) for t, v in h[h.index < end7].items()],
                    meals=[meal(r) for r in hm.itertuples()])
        t = time.perf_counter()
        s = post(f"/v1/people/{pid}/history", body)
        build.append(dict(pid=pid, ms=(time.perf_counter() - t) * 1000, usable=s["usable_meals"], warnings=s.get("warnings")))
        lm = m[day(m.time) > 7]
        post(f"/v1/people/{pid}/signals", dict(cgm=[dict(timestamp=ts(t), glucose_mgdl=float(v)) for t, v in c[c.index >= end7].items()],
                                               hr=[dict(timestamp=ts(t), hr_bpm=float(v)) for t, v in h[h.index >= end7].items()],
                                               meals=[meal(r) for r in lm.itertuples()]))
        for i, r in TE[TE.pid == pid].iterrows():
            a = post(f"/v1/people/{pid}/predict", dict(photo_key=rel(r.photo), meal_time=ts(r.time), meal_type=r.meal_type, log_meal=False))
            pr = a["prediction"]
            rows.append(dict(meal=int(i), pid=pid, seen=bool(r.seen), obs_dpeak=r.dpeak, obs_G60=r.G60, obs_iauc=r.iauc, obs_ttp=r.ttp,
                             api_dpeak=pr["peak_rise_mgdl"], api_G60=pr["glucose_at_min"]["60"], api_iauc=pr["iauc_0_120_mgdl_min"],
                             api_ttp=pr["time_to_peak_min"], **{f"batch_{k}": v for k, v in batch[i].items()}))
        print(pid, f"history {build[-1]['ms']:.0f} ms, {len(rows)} meals", flush=True)
    D = pd.DataFrame(rows)
    un = ~D.seen
    mae = lambda a, b, m: float(np.mean(np.abs(D.loc[m, a] - D.loc[m, b])))
    res = dict(mode=MODE, url=URL, people=len(pids), meals=len(D), unseen_meals=int(un.sum()),
               unseen_mae=dict(api={t: round(mae(f"api_{t}", f"obs_{t}", un), 2) for t in ("dpeak", "G60", "iauc", "ttp")},
                               batch={t: round(mae(f"batch_{t}", f"obs_{t}", un), 2) for t in ("dpeak", "G60", "iauc", "ttp")}),
               api_vs_batch_abs_diff=dict(median={t: round(float(np.median(np.abs(D[f"api_{t}"] - D[f"batch_{t}"]))), 2) for t in ("dpeak", "G60", "iauc")},
                                          p95={t: round(float(np.percentile(np.abs(D[f"api_{t}"] - D[f"batch_{t}"]), 95)), 2) for t in ("dpeak", "G60", "iauc")},
                                          max={t: round(float(np.max(np.abs(D[f"api_{t}"] - D[f"batch_{t}"]))), 2) for t in ("dpeak", "G60", "iauc")}),
               history_build_ms=dict(median=round(float(np.median([b["ms"] for b in build])))),
               history_warnings={b["pid"]: b["warnings"] for b in build if b["warnings"]})
    D.to_csv(OUT.replace(".json", "_meals.csv"), index=False)
json.dump(res, open(OUT, "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
