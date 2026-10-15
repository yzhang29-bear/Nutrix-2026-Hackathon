"""Bundle the trained components and export the demo dataset by running the single-meal Predictor (the future API path)
on selected day-8+ meals. Also checks that the API path reproduces the batch evaluation.
usage: python -I export_demo.py <CGMacros dir> <stage1 dir> <blend dir> <out dir>
"""
import sys, os, json, time, pickle, io, base64
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, torch
from PIL import Image, ImageOps
from pmrf.data import load_cgmacros
from pmrf.features import CURVE_T, PRE_T, TARGETS
from pmrf.predict import Predictor, build_response, group_contributions, confidence, _r
from pmrf.guide import exceed_prob, person_guide, person_guide_measured

DATA, S1, BL, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
summ = json.load(open(os.path.join(BL, "summary.json")))
fp = pickle.load(open(os.path.join(BL, "final_preds.pkl"), "rb"))
outer = st["outer"]; Mo = outer["M"]; E = outer["E"]; CURV = outer["CURV"]; PRE = outer["PRE"]
TE = st["TE"]
persons = load_cgmacros(DATA)

# ------------------------------------------------------------------ bundle (everything the API needs)
libmask = (Mo.in_hist & Mo.label & (Mo.emb_idx >= 0)).to_numpy()
state = {}
for pid in persons:
    H = Mo[libmask & (Mo.pid == pid).to_numpy()].copy()     # libmask needs a photo, so extra photo-less cohorts never appear
    H["emb"] = list(E[H.index]); H["curve"] = list(CURV[H.index]); H["pre"] = list(PRE[H.index])
    trow = TE[TE.pid == pid]
    src = trow.iloc[0] if len(trow) else (H.iloc[0] if len(H) else None)
    prof = {c: src[c] for c in Mo.columns if c.startswith("p_")} if src is not None else {}
    if not len(trow) and len(H):    # no test meal: fall back to history means (profile of test rows already uses the full history)
        prof.update(p_n=len(H), p_dpeak=H.dpeak.mean(), p_iauc=H.iauc.mean(), p_ttp=H.ttp.mean())
    state[pid] = dict(hist=H, profile=prof)
bundle = dict(encoder=("MobileCLIP-S2", "datacompdr"), macro_est=outer["mest"], pca=outer["pca"], gbm=st["gbm_final"],
              transformer_states=[torch.load(os.path.join(BL, f"transformer_seed{s}.pt"), map_location="cpu") for s in (0, 1, 2)], img_dim=E.shape[1],
              blend_weights=summ["blend_weights"], curve_weight=summ["curve_weight"], band_scale=summ["band_scale"],
              tf_sources=summ.get("tf_sources", {}), num_cols=summ.get("num_cols", "v1"),
              lib_emb=E[libmask], lib_y=Mo.loc[libmask, ["dpeak", "iauc", "d60"]].to_numpy(float), lib_pid=Mo.pid.to_numpy()[libmask],
              person_state=state, pop_avg_rise=float(st["TR"].dpeak.mean()))
pickle.dump(bundle, open(os.path.join(OUT, "bundle.pkl"), "wb"))
pred = Predictor(bundle)
EFFECTS = pickle.load(open(os.path.join(os.path.dirname(OUT.rstrip("/")), "guide", "shake_effects.pkl"), "rb"))

def _js(o):
    return o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))


def thumb(path, w=220):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im = ImageOps.fit(im, (w, int(w * 4 / 3)))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=70, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

# ------------------------------------------------------------------ choose demo meals: up to 4 per person, unseen first, varied meal types
pick = []
for pid, g in TE.groupby("pid"):
    g = g.sort_values(["seen", "time"])
    chosen = []
    for mt in ["lunch", "dinner", "breakfast", "snack"]:
        c = g[(g.meal_type == mt) & ~g.index.isin(chosen)]
        if len(c): chosen.append(c.index[0])
    for i in g.index:
        if len(chosen) >= 4: break
        if i not in chosen: chosen.append(i)
    pick += chosen[:4]
pos = {i: k for k, i in enumerate(fp["te_index"])}
images, meals_out, diffs, lats = {}, {}, [], []
for n, i in enumerate(pick):
    r = Mo.loc[i]; p = persons[r.pid]; H = state[r.pid]["hist"]
    lk = r.get("meal_kcal_logged", np.nan) if "meal_kcal_logged" in bundle["gbm"].features else None   # demo build: logged meal kcal
    out, X, pre, e, tags, lat, sidx = pred.predict_meal(p, r.photo, r.time, r.meal_type, logged_kcal=lk)
    k = pos[i]
    diffs.append({t: abs(float(out[t][0]) - float(fp["blend"][t][k])) for t in ["G60", "dpeak", "iauc"]})
    lats.append(lat)
    wi = pred.what_if(X, pre[None], H, e[None])
    sim = []
    for j in np.atleast_1d(sidx)[:3]:
        h = H.iloc[int(j)]; hid = f"h{r.pid}_{int(H.index[int(j)])}"
        if hid not in images: images[hid] = thumb(h.photo)
        sim.append(dict(photo_id=hid, time=str(h.time), meal_type=h.meal_type, logged_carbs_g=_r(h.carbs), observed_peak_rise_mgdl=_r(h.dpeak),
                        similarity=_r(float(H.emb.iloc[int(j)] @ e), 2)))
    hist_types = set(H.meal_type); hmax = H.carbs.max() if len(H) else 0
    conf = confidence(X.iloc[0].to_dict(), len(H), hist_types, hmax, float(X.photo_sim_pop.iloc[0]))
    contrib = group_contributions(out["contrib"].iloc[0], bundle["gbm"].features)
    mid = f"m{r.pid}_{i}"
    images[mid] = thumb(r.photo)
    obs = dict(curve_glucose_mgdl=[_r(r.G0 + v) for v in CURV[i]], pre_meal_minutes=list(PRE_T.astype(int)), pre_meal_glucose_mgdl=[_r(r.G0 + v) for v in PRE[i]],
               glucose_at_min={str(m): _r(r[f"G{m}"]) for m in (30, 60, 90, 120)}, peak_glucose_mgdl=_r(r.Gmax), peak_rise_mgdl=_r(r.dpeak),
               time_to_peak_min=_r(r.ttp, 0), iauc_0_120_mgdl_min=_r(r.iauc, 0),
               logged_nutrition=dict(carbs_g=_r(r.carbs), protein_g=_r(r.protein), fat_g=_r(r.fat), fiber_g=_r(r.fiber), kcal=_r(r.kcal, 0)),
               seen_recipe_in_history=bool(TE.loc[i, "seen"]), study_day=int(r.day))
    pr = {t: float(out[t][0]) for t in TARGETS}; pr["curve"] = out["curve"][0]
    probs = exceed_prob(float(X.G0.iloc[0]), out, 0)
    resp = build_response(r.pid, X.iloc[0].to_dict(), pr, (out["band"][0][0], out["band"][1][0]), contrib, float(out["base"][0]),
                          state[r.pid]["profile"], bundle["pop_avg_rise"], sim, wi, conf, tags, latency=lat, observed=obs, probs=probs)
    resp["meal"]["photo_id"] = mid; resp["meal_id"] = mid
    meals_out[mid] = resp
    if n % 20 == 0: print(f"{n}/{len(pick)} meals, last total {lat['total']:.0f} ms", flush=True)

# ------------------------------------------------------------------ person summaries (7-day CGM trace with meals, diet guide, metabolic profile)
RISK = os.path.join(os.path.dirname(OUT.rstrip("/")), "risk", "cgmacros_profiles.json")
risk_profiles = json.load(open(RISK)) if os.path.exists(RISK) else {}

# relative 3-year T2D risk: rank among participants without diabetes (the absolute value is not calibrated on Dexcom data)
_risk = {k: v["t2d_risk"]["probability"] for k, v in risk_profiles.items() if v.get("t2d_risk", {}).get("available")}
_rv = np.array(list(_risk.values())) if _risk else np.array([])

def norm_profile(d, pid):
    if not d:
        return None
    hb = d.get("hba1c_estimate", {})
    st = hb.get("status")
    tier = None
    if pid in _risk and len(_rv) >= 6:
        q = float(np.mean(_rv <= _risk[pid]))
        tier = "lower than most" if q <= 1 / 3 else ("about average" if q <= 2 / 3 else "higher than most")
    return dict(est_hba1c=hb.get("model"), est_hba1c_lo=(hb.get("interval95") or [None, None])[0], est_hba1c_hi=(hb.get("interval95") or [None, None])[1],
                gmi=hb.get("gmi"), status=dict(label=st, prob=(hb.get("status_probabilities") or {}).get(st)),
                gri_zone=d.get("literature_risk", {}).get("gri", {}).get("zone"), t2d_risk_tier=tier,
                note="Estimated from the first week of CGM. Research estimates, not a diagnosis.")

people = []
for pid, p in persons.items():
    if pid not in TE.pid.values or pid.startswith("B"): continue
    c = p.cgm[p.day(p.cgm.index) <= 7]
    c15 = c.resample("15min").mean()
    h = state[pid]["hist"]
    pm = p.meals[p.day(p.meals.time) <= 7]
    gd = person_guide_measured(state[pid]["hist"], EFFECTS, pid) or person_guide(pred, pid)
    for sp_ in gd.get("biggest_spikes", []):
        hid = f"g{pid}_{abs(hash(sp_['photo_path'])) % 10 ** 8}"
        images.setdefault(hid, thumb(sp_.pop("photo_path")))
        sp_["photo_id"] = hid
    people.append(dict(person_id=pid, study_group=p.meta["group"], history_meals=int(len(h)), guide=gd,
                       metabolic_profile=norm_profile(risk_profiles.get(pid), pid),
                       profile={k: _r(v, 2) for k, v in state[pid]["profile"].items()},
                       trace=dict(start=str(c15.index[0]), step_min=15, glucose_mgdl=[_r(v, 0) for v in c15.to_numpy()]),
                       history_meal_markers=[dict(time=str(t), meal_type=mt, carbs_g=_r(cb)) for t, mt, cb in zip(pm.time, pm.meal_type, pm.carbs)],
                       demo_meals=[m for m in meals_out if m.startswith(f"m{pid}_")]))
lat_df = pd.DataFrame(lats)
meta = dict(api_version="0.1", n_people=len(people), n_meals=len(meals_out), population_avg_peak_rise_mgdl=round(bundle["pop_avg_rise"], 1),
            latency_ms_median={k: round(float(v), 1) for k, v in lat_df.median().items()},
            latency_ms_p90={k: round(float(v), 1) for k, v in lat_df.quantile(0.9).items()},
            api_vs_batch_max_abs_diff=pd.DataFrame(diffs).max().round(4).to_dict(), curve_minutes=list(CURVE_T.astype(int)))
json.dump(dict(meta=meta, people=people, meals=meals_out, images=images), open(os.path.join(OUT, "demo_data.json"), "w"), separators=(",", ":"), default=_js)
example = meals_out[next(iter(meals_out))].copy(); example.pop("observed", None)
json.dump(example, open(os.path.join(OUT, "api_example_response.json"), "w"), indent=2, default=_js)
print(json.dumps(meta, indent=1, default=_js))
print("demo_data.json MB:", round(os.path.getsize(os.path.join(OUT, "demo_data.json")) / 1e6, 2))
