"""Library-level check: the model store (stable formats) reproduces the demo predictions made with the training bundle.
usage: python -I check_store.py <CGMacros dir> <store version dir> <demo_data.json> <bundle.pkl>
"""
import sys, os, json, pickle, time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, torch
from pmrf.data import load_cgmacros
from pmrf.store import load_store
from pmrf.predict import Predictor

DATA, STORE, DEMO, BUNDLE = sys.argv[1:5]
torch.set_num_threads(8)
t = time.time(); b = load_store(STORE); print(f"store loaded in {time.time() - t:.1f}s")
old = pickle.load(open(BUNDLE, "rb"))
pred = Predictor(b)
persons = load_cgmacros(DATA)
demo = json.load(open(DEMO))
lib = b["macro_est"]
diffs = []
for mid, ref in demo["meals"].items():
    pid = ref["person_id"]; idx = int(mid.split("_")[1])
    st = dict(old["person_state"][pid]); own = lib.Lpid == pid
    st["own_lib"] = (lib.LE[own], lib.LY[own]); st["in_library"] = True
    p = persons[pid]
    row = p.meals.iloc[0]
    mt = pd.Timestamp(ref["meal"]["time"])
    m = p.meals[pd.to_datetime(p.meals.time) == mt].iloc[0]
    lk = None
    if "meal_kcal_logged" in b["gbm"].features:                      # demo build: the logged meal calories are an input
        lk = float(m.calories_logged) if pd.notna(m.get("calories_logged", np.nan)) else float(m.kcal)
    resp, _ = pred.respond(p, m.photo, mt, ref["meal"]["meal_type"], state=st, photo_ref=lambda h: h.photo, logged_kcal=lk)
    a, r = resp["prediction"], ref["prediction"]
    d = dict(peak=abs(a["peak_rise_mgdl"] - r["peak_rise_mgdl"]), g60=abs(a["glucose_at_min"]["60"] - r["glucose_at_min"]["60"]),
             iauc=abs(a["iauc_0_120_mgdl_min"] - r["iauc_0_120_mgdl_min"]),
             curve=max(abs(x - y) for x, y in zip(a["curve"]["glucose_mgdl"], r["curve"]["glucose_mgdl"])),
             carbs=abs(resp["meal"]["estimated_nutrition"]["carbs_g"] - ref["meal"]["estimated_nutrition"]["carbs_g"]),
             whatif=max(abs(x["peak_rise_mgdl"] - y["peak_rise_mgdl"]) for x, y in zip(resp["what_if"], ref["what_if"])),
             p180=abs(a["prob_peak_above_mgdl"]["180"] - r["prob_peak_above_mgdl"]["180"]),
             contrib=max(abs(x["mgdl"] - y["mgdl"]) for x, y in zip(resp["explanation"]["contributions"], ref["explanation"]["contributions"])),
             tags=resp["meal"]["photo_tags"] != ref["meal"]["photo_tags"], conf=resp["confidence"] != ref["confidence"])
    diffs.append(d)
D = pd.DataFrame(diffs)
print(f"{len(D)} meals; max abs differences vs the demo (values are rounded to 0.1 in the demo):")
print(D.max().to_string())
