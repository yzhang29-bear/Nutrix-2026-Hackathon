"""Predict the glucose response of one CGMacros participant to unseen hospital meal photos, against a running service.

Onboards the participant as a new user (week-1 photos -> /history), sends CGM/HR up to a chosen meal time (/signals),
then calls /predict for every photo in the photo folder at that same time, so only the meal differs.
No observed curve exists for these photos: check plausibility, nutrition estimates, tags and confidence, not error.

usage: PMRF_API_KEY=... python -I tests/integration/demo_hospital_meals.py <url> <CGMacros dir> <photo dir> <out.json>
       [--person 001] [--day 8] [--time 12:30]
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
import httpx, numpy as np, pandas as pd
from pmrf.data import load_cgmacros

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("data"); ap.add_argument("photos"); ap.add_argument("out")
ap.add_argument("--person", default="001"); ap.add_argument("--day", type=int, default=8)
ap.add_argument("--time", default="12:30")
a = ap.parse_args()

headers = {"x-api-key": os.environ["PMRF_API_KEY"]} if os.environ.get("PMRF_API_KEY") else {}
c = httpx.Client(base_url=a.url.rstrip("/"), timeout=300, headers=headers)
ts = lambda t: pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S")
num = lambda v: None if v is None or not np.isfinite(v) else float(v)


def call(method, path, **kw):
    r = c.request(method, path, **kw)
    if r.status_code != 200:
        sys.exit(f"{method} {path}: {r.status_code} {r.text[:300]}")
    return r.json()


def upload(pid, path):
    with open(path, "rb") as fh:
        return call("POST", f"/v1/people/{pid}/photos", files={"file": (Path(path).name, fh.read(), "image/jpeg")})["photo_key"]


p = load_cgmacros(a.data)[a.person]
pid = f"demo-hospital-{a.person}"
end7 = p.t0 + pd.Timedelta(days=7, hours=3)
cg, hr = p.cgm.dropna(), p.hr.dropna()

# 1. Week 1 -> person state
m7 = p.meals[p.day(p.meals.time) <= 7]
keys = {ph: upload(pid, ph) for ph in m7.photo.dropna()}
h = call("POST", f"/v1/people/{pid}/history", json=dict(
    start_time=ts(p.t0),
    cgm=[dict(timestamp=ts(t), glucose_mgdl=float(v)) for t, v in cg[cg.index < end7].items()],
    hr=[dict(timestamp=ts(t), hr_bpm=float(v)) for t, v in hr[hr.index < end7].items()],
    meals=[dict(timestamp=ts(r.time), meal_type=r.meal_type, photo_key=keys.get(r.photo), carbs_g=num(r.carbs),
                protein_g=num(r.protein), fat_g=num(r.fat), fiber_g=num(r.fiber)) for r in m7.itertuples()]))
print(f"history: {h['usable_meals']} usable meals, {len(keys)} photos, warnings {h.get('warnings')}")

# 2. Signals up to the meal time; it needs a CGM reading in the 30 min before
meal_time = (p.t0.normalize() + pd.Timedelta(days=a.day - 1)) + pd.Timedelta(a.time + ":00")
pre = cg[(cg.index >= meal_time - pd.Timedelta(minutes=30)) & (cg.index <= meal_time)]
if pre.empty:
    sys.exit(f"no CGM in the 30 min before {meal_time}; pick another --day/--time")
call("POST", f"/v1/people/{pid}/signals", json=dict(
    cgm=[dict(timestamp=ts(t), glucose_mgdl=float(v)) for t, v in cg[(cg.index >= end7) & (cg.index <= meal_time)].items()],
    hr=[dict(timestamp=ts(t), hr_bpm=float(v)) for t, v in hr[(hr.index >= end7) & (hr.index <= meal_time)].items()]))

# 3. One prediction per photo, same person and time; log_meal off so photos don't become each other's previous meal
rows, full = [], {}
for photo in sorted(Path(a.photos).glob("*.jpg")):
    r = call("POST", f"/v1/people/{pid}/predict",
             json=dict(photo_key=upload(pid, photo), meal_time=ts(meal_time), log_meal=False))
    full[photo.name] = r
    n, pr = r["meal"]["estimated_nutrition"], r["prediction"]
    rows.append(dict(photo=photo.stem, top_tag=r["meal"]["photo_tags"][0]["label"],
                     tag_p=r["meal"]["photo_tags"][0]["p"], carbs_g=n["carbs_g"], kcal=n["kcal"],
                     pre_meal=r["pre_meal_state"]["glucose_mgdl"], peak=pr["peak_glucose_mgdl"],
                     rise=pr["peak_rise_mgdl"], ttp_min=pr["time_to_peak_min"],
                     p_gt180=pr["prob_peak_above_mgdl"]["180"], confidence=r["confidence"]["level"],
                     flags="; ".join(r["confidence"]["flags"]), ms=round(r["latency_ms"]["total"])))

json.dump(dict(person=a.person, meal_time=ts(meal_time), history=h, predictions=full), open(a.out, "w"), indent=1)
pd.set_option("display.width", 250, "display.max_colwidth", 60)
print(f"person {a.person}, meal time {meal_time}\n" + pd.DataFrame(rows).to_string(index=False))
