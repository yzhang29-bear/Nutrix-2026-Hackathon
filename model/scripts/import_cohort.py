"""Import the development cohort (CGMacros) into a person store with the person states used in training, so the service
reproduces the report's predictions exactly (golden test, mode 'imported'). Photo keys are paths relative to the CGMacros
folder, which is then used as the photo directory.
usage: python -I import_cohort.py <CGMacros dir> <bundle.pkl> <data dir>
"""
import sys, os, pickle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd
from pmrf.data import load_cgmacros
from pmrf.state import PersonStore

DATA, BUNDLE, OUT = sys.argv[1:4]
b = pickle.load(open(BUNDLE, "rb"))
persons = load_cgmacros(DATA)
store = PersonStore(OUT)
me = b["macro_est"]
rel = lambda p: os.path.relpath(p, DATA) if isinstance(p, str) else None
for pid, st in b["person_state"].items():
    p = persons[pid]
    store.write_meta(pid, t0=str(p.t0), hist_days=7, person={}, model_version="imported-cohort", source="cgmacros",
                     cgm_fill_gap_min=0)          # CGMacros CGM is already on a 1-min grid; keep its gaps as in training
    c = p.cgm.dropna(); h = p.hr.dropna()
    store.append_signal(pid, "cgm", pd.DataFrame(dict(timestamp=c.index, glucose_mgdl=c.to_numpy())), replace=True)
    store.append_signal(pid, "hr", pd.DataFrame(dict(timestamp=h.index, hr_bpm=h.to_numpy())), replace=True)
    m = p.meals
    store.append_signal(pid, "meals", pd.DataFrame(dict(time=m.time, meal_type=m.meal_type, carbs=m.carbs, protein=m.protein, fat=m.fat,
                                                        fiber=m.fiber, kcal=m.kcal, photo_key=[rel(x) for x in m.photo],
                                                        source=np.where(p.day(m.time) <= 7, "history", "logged"))), replace=True)
    store.append_signal(pid, "meds", pd.DataFrame(columns=["time", "med_class", "dose"]), replace=True)
    H = st["hist"].copy()
    H["photo_key"] = [rel(x) for x in H.photo]
    H["photo"] = H.photo_key
    own = me.Lpid == pid
    store.save_state(pid, dict(hist=H, profile=st["profile"], own_lib=(me.LE[own], me.LY[own]), in_library=True,
                               summary=dict(history_meals=int(st["profile"].get("p_n", 0) or 0), usable_meals=len(H), imported=True)))
print("imported", len(b["person_state"]), "people into", OUT)
