"""Onboard CGMacros participants into the deployed service as patients `cgmacros-<id>`.

For each participant: upload the week-1 meal photos, POST /history (week 1, with age, sex and BMI from bio.csv), then
POST /signals with all later CGM, heart rate, activity and logged meals, so a prediction can be made at any later time in the record (a prediction only sees data
up to its meal time; the earlier meals of that day feed its previous-meal features, as in training).
Re-running replaces the patient's history.

Test meals: the participant's own day-8 meals with a photo and enough CGM around them (the training rule: a reading at
the meal time and >= 90% of minutes -30..+120) are uploaded too and listed in tests/integration/day8_meals.csv
(rows of other patients are kept), with the logged calories to send as `logged_kcal` (the v7 model expects them).
/predict on one of them returns the observed curve next to the prediction.
Optionally predicts one photo for every patient at the same moment of their study (day/time) to compare responses.

usage: PMRF_API_KEY=... python -I ops/data/onboard_patients.py <url> <CGMacros dir> <id> [<id> ...]
       [--compare-photo tests/integration/hospital_meals/korea_snuh_evening_meal.jpg --day 8 --time 12:30]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
import httpx
import numpy as np
import pandas as pd
from pmrf.data import load_cgmacros
from pmrf.features import meal_table

TEST_DAY = 8
MANIFEST = ROOT / "tests" / "integration" / "day8_meals.csv"

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("data"); ap.add_argument("ids", nargs="+")
ap.add_argument("--compare-photo"); ap.add_argument("--day", type=int, default=8); ap.add_argument("--time", default="12:30")
a = ap.parse_args()

c = httpx.Client(base_url=a.url.rstrip("/"), timeout=300, headers={"x-api-key": os.environ["PMRF_API_KEY"]})
ts = lambda t: pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S")
num = lambda v: None if v is None or not np.isfinite(v) else float(v)


def call(method, path, **kw):
    r = c.request(method, path, **kw)
    if r.status_code != 200:
        raise RuntimeError(f"{method} {path}: {r.status_code} {r.text[:300]}")
    return r.json()


def activity(p, mask):
    """per-minute METs and active kcal where either is known, inside the API's ranges"""
    if p.act is None:
        return []
    k = p.act_kcal if p.act_kcal is not None else pd.Series(np.nan, index=p.act.index)
    a = pd.DataFrame(dict(mets=p.act.where(p.act.between(0.5, 25)), kcal=k.reindex(p.act.index).where(lambda v: v.between(0, 60))))
    a = a[mask(a.index) & (a.mets.notna() | a.kcal.notna())]
    return [dict(timestamp=ts(t), mets=num(r.mets), active_kcal=num(r.kcal)) for t, r in a.iterrows()]


def person_info(p):
    i = p.info or {}
    sex = i.get("sex_female")
    return dict(age=num(i.get("age")), bmi=num(i.get("bmi")),
                sex=None if sex is None or not np.isfinite(sex) else ("female" if sex else "male"))


def upload(pid, path):
    with open(path, "rb") as fh:
        return call("POST", f"/v1/people/{pid}/photos", files={"file": (Path(path).name, fh.read(), "image/jpeg")})["photo_key"]


people = load_cgmacros(a.data)
rows, tests = [], []
for sid in a.ids:
    p, pid = people[sid], f"cgmacros-{sid}"
    end7 = p.t0 + pd.Timedelta(days=7, hours=3)
    cg, hr = p.cgm.dropna(), p.hr.dropna()
    m7 = p.meals[p.day(p.meals.time) <= 7]
    keys = {ph: upload(pid, ph) for ph in m7.photo.dropna()}
    h = call("POST", f"/v1/people/{pid}/history", json=dict(
        start_time=ts(p.t0), person=person_info(p),
        cgm=[dict(timestamp=ts(t), glucose_mgdl=float(v)) for t, v in cg[cg.index < end7].items()],
        hr=[dict(timestamp=ts(t), hr_bpm=float(v)) for t, v in hr[hr.index < end7].items()],
        activity=activity(p, lambda i: i < end7),
        meals=[dict(timestamp=ts(r.time), meal_type=r.meal_type, photo_key=keys.get(r.photo), carbs_g=num(r.carbs),
                    protein_g=num(r.protein), fat_g=num(r.fat), fiber_g=num(r.fiber), kcal=num(r.calories_logged))
               for r in m7.itertuples()]))
    later = p.meals[p.day(p.meals.time) > 7]
    s = call("POST", f"/v1/people/{pid}/signals", json=dict(
        cgm=[dict(timestamp=ts(t), glucose_mgdl=float(v)) for t, v in cg[cg.index >= end7].items()],
        hr=[dict(timestamp=ts(t), hr_bpm=float(v)) for t, v in hr[hr.index >= end7].items()],
        activity=activity(p, lambda i: i >= end7),
        meals=[dict(timestamp=ts(r.time), meal_type=r.meal_type, carbs_g=num(r.carbs), protein_g=num(r.protein), fat_g=num(r.fat),
                    fiber_g=num(r.fiber), kcal=num(r.calories_logged)) for r in later.itertuples()]))
    M = meal_table({sid: p})[0]
    M = M[(M.day == TEST_DAY) & M.valid & M.photo.map(lambda f: isinstance(f, str) and os.path.exists(f))]
    for i, r in enumerate(M.sort_values("time").itertuples(), 1):
        tests.append(dict(patient=pid, group=p.meta.get("group"), n=i, meal_time=ts(r.time), meal_type=r.meal_type,
                          photo_key=upload(pid, r.photo), logged_carbs_g=num(r.carbs), logged_protein_g=num(r.protein),
                          logged_fat_g=num(r.fat), logged_kcal=num(r.meal_kcal_logged),
                          observed_peak_rise_mgdl=round(float(r.dpeak), 1)))
    row = dict(patient=pid, group=p.meta.get("group"), usable_meals=h["usable_meals"], photos=len(keys), day8_test_meals=len(M),
               cgm_until=ts(cg.index.max()), warnings="; ".join(h.get("warnings") or []))
    if a.compare_photo:
        mt = p.t0.normalize() + pd.Timedelta(days=a.day - 1) + pd.Timedelta(a.time + ":00")
        r = call("POST", f"/v1/people/{pid}/predict",
                 json=dict(photo_key=upload(pid, a.compare_photo), meal_time=ts(mt), log_meal=False))
        pr = r["prediction"]
        row.update(meal_time=ts(mt), pre_meal=r["pre_meal_state"]["glucose_mgdl"], peak=pr["peak_glucose_mgdl"],
                   rise=pr["peak_rise_mgdl"], p_gt140=pr["prob_peak_above_mgdl"]["140"], confidence=r["confidence"]["level"])
    rows.append(row)
    print(f"{pid}: {h['usable_meals']} usable meals, {len(keys)} photos, signals {s.get('stored')}", flush=True)

T = pd.DataFrame(tests)
if MANIFEST.exists():                   # keep the rows of patients not onboarded in this run
    old = pd.read_csv(MANIFEST)
    T = pd.concat([old[~old.patient.isin([f"cgmacros-{i}" for i in a.ids])], T], ignore_index=True)
if len(T):
    T.sort_values(["patient", "n"]).to_csv(MANIFEST, index=False, lineterminator="\n")
print(f"{len(tests)} day-{TEST_DAY} test meals -> {MANIFEST.relative_to(ROOT)}")

pd.set_option("display.width", 250, "display.max_colwidth", 50)
print(pd.DataFrame(rows).to_string(index=False))
