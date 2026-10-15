"""End-to-end API flow for a brand-new user against a running service (with PMRF_API_KEY set and a writable data dir):
photo upload -> /history -> /signals -> /predict (logged) -> /profile, plus error cases. Uses CGMacros participant 001's data.
usage: PMRF_API_KEY=... python -I tests/test_api_flow.py <url> <CGMacros dir>"""
import os, sys, json, time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, httpx
from pmrf.data import load_cgmacros

URL, DATA = sys.argv[1].rstrip("/"), sys.argv[2]
KEY = os.environ["PMRF_API_KEY"]
c = httpx.Client(base_url=URL, timeout=300, headers={"x-api-key": KEY})
pid = "flowtest-user"
p = load_cgmacros(DATA)["001"]
ts = lambda t: pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S")
f = lambda v: None if not np.isfinite(v) else float(v)
checks = []

def check(name, cond, info=""):
    checks.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name, info)

check("auth required", httpx.post(f"{URL}/v1/people/{pid}/signals", json={}).status_code == 401)
check("unknown person -> 404", c.post(f"/v1/people/nobody/predict", json=dict(photo_key="x.jpg", meal_time="2020-01-01T12:00:00")).status_code == 404)
r = c.post(f"/v1/people/{pid}/photos", files={"file": ("x.txt", b"not an image", "text/plain")})
check("non-image upload -> 422", r.status_code == 422)
# upload week-1 photos through the API
keys = {}
m7 = p.meals[p.day(p.meals.time) <= 7]
t = time.perf_counter()
for ph in m7.photo.dropna():
    with open(ph, "rb") as fh:
        r = c.post(f"/v1/people/{pid}/photos", files={"file": (os.path.basename(ph), fh.read(), "image/jpeg")})
    keys[ph] = r.json()["photo_key"]
check("photo uploads", len(keys) == m7.photo.notna().sum(), f"{len(keys)} photos in {time.perf_counter() - t:.1f}s")
end7 = p.t0 + pd.Timedelta(days=7, hours=3)
cg, hr = p.cgm.dropna(), p.hr.dropna()
body = dict(start_time=ts(p.t0), person=dict(age=45, bmi=27.5, sex="female"),
            cgm=[dict(timestamp=ts(a), glucose_mgdl=float(v)) for a, v in cg[cg.index < end7].items()],
            hr=[dict(timestamp=ts(a), hr_bpm=float(v)) for a, v in hr[hr.index < end7].items()],
            meals=[dict(timestamp=ts(r.time), meal_type=r.meal_type, photo_key=keys.get(r.photo), carbs_g=f(r.carbs), protein_g=f(r.protein),
                        fat_g=f(r.fat), fiber_g=f(r.fiber)) for r in m7.itertuples()])
check("predict before history -> 404", c.post(f"/v1/people/{pid}/predict", json=dict(photo_key=list(keys.values())[0], meal_time=ts(p.t0))).status_code == 404)
r = c.post(f"/v1/people/{pid}/history", json=body)
h = r.json()
check("history builds state", r.status_code == 200 and h["usable_meals"] >= 10, json.dumps({k: h.get(k) for k in ("usable_meals", "history_meals", "build_ms", "warnings")}))
later = p.meals[(pd.to_datetime(p.meals.time) > end7 + pd.Timedelta("1h")) & p.meals.photo.notna()]
meal = later.iloc[0]
with open(meal.photo, "rb") as fh:
    k8 = c.post(f"/v1/people/{pid}/photos", files={"file": ("m.jpg", fh.read(), "image/jpeg")}).json()["photo_key"]
r = c.post(f"/v1/people/{pid}/predict", json=dict(photo_key=k8, meal_time=ts(meal.time), meal_type=meal.meal_type))
check("predict without recent CGM -> 409", r.status_code == 409, r.text[:120])
r = c.post(f"/v1/people/{pid}/signals", json=dict(cgm=[dict(timestamp=ts(a), glucose_mgdl=float(v)) for a, v in cg[(cg.index >= end7) & (cg.index <= meal.time)].items()],
                                                  hr=[dict(timestamp=ts(a), hr_bpm=float(v)) for a, v in hr[(hr.index >= end7) & (hr.index <= meal.time)].items()]))
check("signals stored", r.status_code == 200, json.dumps(r.json().get("stored")))
r = c.post(f"/v1/people/{pid}/predict", json=dict(photo_key=k8, meal_time=ts(meal.time)))       # meal type inferred from the time
a = r.json()
check("predict ok", r.status_code == 200 and len(a["prediction"]["curve"]["glucose_mgdl"]) == 25,
      f"peak rise {a['prediction']['peak_rise_mgdl']} mg/dL (observed rise of this meal is in the dataset), type {a['meal']['meal_type']}, {a['latency_ms']['total']:.0f} ms core")
check("similar meals point to uploaded photos", all(s["photo_key"] in keys.values() for s in a["meal"]["similar_past_meals"]))
r = c.get(f"/v1/people/{pid}/profile")
pr = r.json()
check("profile ok", r.status_code == 200 and pr["diet_guide"]["available"] and pr["metabolic_profile"]["est_hba1c"] is not None,
      json.dumps(dict(method=pr["diet_guide"].get("method"), budget=pr["diet_guide"].get("carb_budget_g_per_meal"), hba1c=pr["metabolic_profile"]["est_hba1c"],
                      tier=pr["metabolic_profile"]["t2d_risk_tier"])))
r = c.post(f"/v1/people/{pid}/predict", json=dict(photo_key="../../etc/passwd", meal_time=ts(meal.time)))
check("path traversal key rejected", r.status_code in (404, 422))
r = c.post(f"/v1/people/{pid}/history", json=dict(cgm=[dict(timestamp="2020-01-01T00:00:00", glucose_mgdl=5000)], meals=[]))
check("out-of-range glucose -> 422", r.status_code == 422)
n_fail = sum(1 for _, ok in checks if not ok)
print(f"{len(checks) - n_fail}/{len(checks)} checks passed")
sys.exit(1 if n_fail else 0)
