"""Read-only view of a person's stored data, one calendar day at a time, for the app's history screen.

Days are calendar dates in the person's wall-clock time (timestamps are stored timezone-naive, as in training). That is
also how server/store.py chunks signals in DynamoDB, so one day view reads one item per signal kind. The model's first
week (days 1-7 counted from t0, pmrf.data.Person.day) usually starts mid-day, so it is reported separately as
`history_window`, and each date says whether it overlaps it.

Each meal carries its observed 0-120 min response (server/observed.py, the training-target definitions) when the CGM
around it is complete enough. Values are the stored readings, not the model's 1-min grid; heart rate is averaged per
5 minutes to keep a day's payload small.
"""
import numpy as np
import pandas as pd

from pmrf.state import CGM_INTERP_LIMIT_MIN, SIGNAL_COLUMNS, _naive, assemble_person
from server.observed import _r, observed_outcome

DAY = pd.Timedelta(days=1)
BIN = "5min"
TIR_LOW, TIR_HIGH = 70, 180


def _date(date):
    """'YYYY-MM-DD' -> midnight Timestamp; ValueError (422) otherwise"""
    try:
        d = pd.Timestamp(date)
    except ValueError:
        d = None
    if d is None or len(date) != 10 or d != d.normalize():
        raise ValueError(f"invalid date {date!r}: use YYYY-MM-DD")
    return d


def _iso(t):
    return pd.Timestamp(t).isoformat()


def _window(meta):
    t0 = pd.Timestamp(meta["t0"])
    days = int(meta.get("hist_days", 7))
    return t0, t0 + days * DAY, days


def _glucose_stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(cgm_readings=0, mean_glucose_mgdl=None, min_glucose_mgdl=None, max_glucose_mgdl=None, time_in_range_pct=None)
    return dict(cgm_readings=int(len(v)), mean_glucose_mgdl=_r(v.mean()), min_glucose_mgdl=_r(v.min()), max_glucose_mgdl=_r(v.max()),
                time_in_range_pct=_r(100 * np.mean((v >= TIR_LOW) & (v <= TIR_HIGH))))


def _bins(df, key, col):
    return df.set_index(key)[col].astype(float).resample(BIN).mean().dropna() if len(df) else pd.Series(dtype=float)


def _overlaps(d, start, end):
    return bool(d < end and d + DAY > start)


def days_overview(store, pid):
    """every date with stored data: glucose summary and counts, oldest first"""
    meta = store.meta(pid)
    start, end, days = _window(meta)
    cgm = store.read_signal(pid, "cgm")
    g = pd.Series(cgm.glucose_mgdl.to_numpy(float), index=_naive(cgm.timestamp)).dropna() if len(cgm) else pd.Series(dtype=float)
    stats = {d.strftime("%Y-%m-%d"): _glucose_stats(v) for d, v in g.groupby(g.index.normalize())} if len(g) else {}
    counts = {k: store.signal_dates(pid, k) for k in ("hr", "meals", "meds")}
    dates = sorted((set(stats) | {d for c in counts.values() for d in c}) - {"none"})
    rows = [dict(date=d, in_history_window=_overlaps(pd.Timestamp(d), start, end), **stats.get(d, _glucose_stats([])),
                 hr_readings=counts["hr"].get(d, 0), meals=counts["meals"].get(d, 0), meds=counts["meds"].get(d, 0)) for d in dates]
    return dict(person_id=pid, history_window=dict(start=_iso(start), end=_iso(end), days=days), days=rows)


def day_view(store, pid, date, photo_url=lambda key: None):
    """one date: CGM readings, 5-min heart rate, meals (with photo URL and observed response) and medication"""
    d = _date(date)
    meta = store.meta(pid)
    start, end, _ = _window(meta)
    s = {k: store.signal_day(pid, k, date) for k in SIGNAL_COLUMNS}
    if not any(len(v) for v in s.values()):
        raise FileNotFoundError(f"no data for {pid} on {date}")

    cgm = s["cgm"].assign(timestamp=_naive(s["cgm"].timestamp)).dropna(subset=["glucose_mgdl"]).sort_values("timestamp")
    hr = s["hr"].assign(timestamp=_naive(s["hr"].timestamp)).dropna(subset=["hr_bpm"])
    cgm5 = _bins(cgm, "timestamp", "glucose_mgdl")
    hr5 = _bins(hr, "timestamp", "hr_bpm")
    meals = s["meals"].assign(time=_naive(s["meals"].time)).sort_values("time")
    meds = s["meds"].assign(time=_naive(s["meds"].time)).sort_values("time")

    obs = {}
    if len(meals):
        # a meal's response window reaches into the neighbouring days (-120 min before, +180 min after)
        near = [store.signal_day(pid, "cgm", (d + k * DAY).strftime("%Y-%m-%d")) for k in (-1, 1)]
        around = pd.concat([x for x in near + [s["cgm"]] if len(x)], ignore_index=True)
        if len(around):
            empty = {k: pd.DataFrame(columns=SIGNAL_COLUMNS[k]) for k in ("hr", "meals", "meds")}
            person = assemble_person(pid, meta.get("t0"), around, empty["hr"], empty["meals"], empty["meds"],
                                     fill_gap_min=meta.get("cgm_fill_gap_min", CGM_INTERP_LIMIT_MIN))
            obs = {i: observed_outcome(person, m.time) for i, m in meals.iterrows()}

    meal_rows = [dict(time=_iso(m.time), meal_type=m.meal_type, carbs_g=_r(m.carbs), protein_g=_r(m.protein), fat_g=_r(m.fat),
                      fiber_g=_r(m.fiber), kcal=_r(m.kcal, 0), source=m.source if isinstance(m.source, str) else None,
                      photo_key=m.photo_key if isinstance(m.photo_key, str) else None,
                      photo_url=photo_url(m.photo_key) if isinstance(m.photo_key, str) else None,
                      in_history_window=bool(start <= m.time < end), observed=obs.get(i))
                 for i, m in meals.iterrows()]
    return dict(person_id=pid, date=date, in_history_window=_overlaps(d, start, end), summary=_glucose_stats(cgm.glucose_mgdl),
                cgm=[dict(t=_iso(t), mgdl=_r(v)) for t, v in cgm5.items()],
                hr=[dict(t=_iso(t), bpm=_r(v)) for t, v in hr5.items()],
                meals=meal_rows,
                meds=[dict(time=_iso(m.time), med_class=m.med_class, dose=_r(m.dose, 2)) for m in meds.itertuples()])
