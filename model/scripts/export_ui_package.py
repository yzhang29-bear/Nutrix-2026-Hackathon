"""Export the showcase meals (results/best_subjects/showcase_meals.csv) as a self-contained package for UI/UX design:
for every meal the model input (photo + request + recent CGM/HR), the real API response (in-process FastAPI service, imported
cohort, so the numbers equal the report), and the logged outcome; for every person the week-1 input and the /profile response.
usage: python -I export_ui_package.py <CGMacros dir> <model store dir> <imported data dir> <best_subjects dir> <out dir>
"""
import sys, os, json, shutil, tempfile, pickle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd
from PIL import Image, ImageOps

DATA, STORE, IMPORTED, BEST, OUT = [os.path.abspath(a) for a in sys.argv[1:6]]
work = tempfile.mkdtemp(prefix="pmrf_ui_")
shutil.copytree(IMPORTED, os.path.join(work, "data"))          # the service may write caches; never touch the original
os.environ.update(PMRF_MODEL_DIR=STORE, PMRF_DATA_DIR=os.path.join(work, "data"), PMRF_PHOTO_DIR=DATA, PMRF_THREADS="4")
os.environ.pop("PMRF_API_KEY", None)
from fastapi.testclient import TestClient
from service.app import app
from pmrf.data import load_cgmacros

show = pd.read_csv(os.path.join(BEST, "showcase_meals.csv"), dtype=dict(pid=str), parse_dates=["time"])
st1 = pickle.load(open(os.path.join(ROOT_DIR, "results/final/stage1.pkl"), "rb"))
TE = st1["TE"]; CURV, PRE = st1["outer"]["CURV"], st1["outer"]["PRE"]
persons = load_cgmacros(DATA)
for d in ("photos", "meals", "people"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
ts = lambda t: pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S")
r1 = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 1)


def photo_file(key, w=640):
    """copy a CGMacros photo (key = path relative to the dataset) into photos/, resized; return the package-relative path"""
    if not key:
        return None
    pid, name = key.split("/")[0][-3:], os.path.basename(key)
    rel = f"photos/{pid}_{name.split('-')[0]}.jpg"
    dst = os.path.join(OUT, rel)
    if not os.path.exists(dst):
        im = ImageOps.exif_transpose(Image.open(os.path.join(DATA, key))).convert("RGB")
        im.thumbnail((w, w))
        im.save(dst, "JPEG", quality=82, optimize=True)
    return rel


def swap_keys(o):
    """replace every photo_key (dataset path) in an API response with the package photo file"""
    if isinstance(o, dict):
        return {k: (photo_file(v) if k == "photo_key" else swap_keys(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [swap_keys(v) for v in o]
    return o


def series(s, step="5min"):
    s = s.dropna().resample(step).mean().dropna()
    return [dict(timestamp=ts(t), value=r1(v)) for t, v in s.items()]


index = dict(about="Showcase meals of the post-meal glucose model (CGMacros, unseen days 8+). See README.md.",
             model_version=None, people=[], meals=[])
with TestClient(app) as c:
    index["model_version"] = c.get("/v1/version").json()["model_version"]
    for pid, g in show.groupby("pid", sort=False):
        p = persons[pid]
        end7 = p.t0 + pd.Timedelta(days=7)
        prof = c.get(f"/v1/people/{pid}/profile").json()
        hm = p.meals[p.day(p.meals.time) <= 7]
        rel = lambda x: os.path.relpath(x, DATA) if isinstance(x, str) else None
        week1 = dict(note="What the app uploads after the first week (POST /history). CGM and heart rate are shown every 5 min.",
                     start_time=ts(p.t0), end_time=ts(end7),
                     cgm_mgdl=series(p.cgm[p.cgm.index < end7]), heart_rate_bpm=series(p.hr[p.hr.index < end7]),
                     meals=[dict(timestamp=ts(r.time), meal_type=r.meal_type, photo_key=photo_file(rel(r.photo)), carbs_g=r1(r.carbs),
                                 protein_g=r1(r.protein), fat_g=r1(r.fat), fiber_g=r1(r.fiber), kcal=r1(r.kcal)) for r in hm.itertuples()])
        os.makedirs(os.path.join(OUT, "people", pid), exist_ok=True)
        json.dump(week1, open(os.path.join(OUT, "people", pid, "input_week1.json"), "w"), indent=1)
        json.dump(swap_keys(prof), open(os.path.join(OUT, "people", pid, "output_profile.json"), "w"), indent=1)
        index["people"].append(dict(person_id=pid, study_group=p.meta["group"], week1_meals=len(hm),
                                    input=f"people/{pid}/input_week1.json", output=f"people/{pid}/output_profile.json"))
        for row in g.itertuples():
            i = int(row.meal_index); r = TE.loc[i]
            key = rel(r.photo)
            req = dict(photo_key=key, meal_time=ts(r.time), meal_type=r.meal_type, log_meal=False)
            resp = c.post(f"/v1/people/{pid}/predict", json=req)
            assert resp.status_code == 200, resp.text
            a = resp.json()
            assert abs(a["prediction"]["peak_rise_mgdl"] - row.pred_dpeak) < 0.6, (pid, i, a["prediction"]["peak_rise_mgdl"], row.pred_dpeak)
            mid = f"{pid}_{r.time:%m%d_%H%M}_{r.meal_type}"
            before = (p.cgm.index >= r.time - pd.Timedelta("2h")) & (p.cgm.index <= r.time)
            inp = dict(note="What the app sends for this meal (POST /predict) plus the recent signals the service already has.",
                       request={**req, "photo_key": photo_file(key)},
                       recent_cgm_mgdl=series(p.cgm[before]), recent_heart_rate_bpm=series(p.hr[before]))
            obs = dict(note="Demo only: what really happened (Dexcom CGM and the participant's own food log). The live API never returns this.",
                       minutes=list(range(0, 121, 5)), glucose_mgdl=[r1(r.G0 + v) for v in CURV[i]],
                       pre_meal_minutes=list(range(-120, 1, 5)), pre_meal_glucose_mgdl=[r1(r.G0 + v) for v in PRE[i]],
                       peak_glucose_mgdl=r1(r.Gmax), peak_rise_mgdl=r1(r.dpeak), time_to_peak_min=r1(r.ttp), iauc_0_120_mgdl_min=r1(r.iauc),
                       glucose_at_min={str(m): r1(r[f"G{m}"]) for m in (30, 60, 90, 120)},
                       logged_nutrition=dict(carbs_g=r1(r.carbs), protein_g=r1(r.protein), fat_g=r1(r.fat), fiber_g=r1(r.fiber), kcal=r1(r.kcal)),
                       curve_error_mgdl=round(float(row.curve_mae), 1))
            json.dump(dict(meal_id=mid, person_id=pid, input=inp, output=swap_keys(a), observed=obs),
                      open(os.path.join(OUT, "meals", f"{mid}.json"), "w"), indent=1)
            index["meals"].append(dict(meal_id=mid, person_id=pid, meal_type=r.meal_type, time=ts(r.time), photo=photo_file(key),
                                       file=f"meals/{mid}.json", predicted_peak_rise_mgdl=a["prediction"]["peak_rise_mgdl"],
                                       logged_peak_rise_mgdl=r1(r.dpeak), curve_error_mgdl=round(float(row.curve_mae), 1)))
            print(mid, "ok", flush=True)
json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
shutil.copy(os.path.join(BEST, "showcase_best_meals.png"), os.path.join(OUT, "showcase_best_meals.png"))
shutil.rmtree(work)
print("package written to", OUT)
