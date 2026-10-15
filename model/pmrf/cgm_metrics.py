"""Standard CGM summary metrics for 5-min or 15-min sensor data.

Definitions follow the international consensus on CGM reporting:
  Battelino T, et al. Clinical targets for continuous glucose monitoring data interpretation: recommendations from the
  International Consensus on Time in Range. Diabetes Care 2019;42(8):1593-1603. doi:10.2337/dci19-0028
  Battelino T, et al. Continuous glucose monitoring and metrics for clinical trials: an international consensus statement.
  Lancet Diabetes Endocrinol 2023;11(1):42-57.
GMI (Bergenstal RM, et al. Diabetes Care 2018;41(11):2275-2280): GMI(%) = 3.31 + 0.02392 * mean glucose (mg/dL).
GRI (Klonoff DC, et al. A Glycemia Risk Index (GRI) of hypoglycemia and hyperglycemia for continuous glucose monitoring
  validated by clinician ratings. J Diabetes Sci Technol 2023;17(5):1226-1242, online 2022. doi:10.1177/19322968221085273):
  GRI = 3.0*VLow + 2.4*Low + 1.6*VHigh + 0.8*High (percent time <54, 54-69, >250, 181-250 mg/dL), capped at 100.

Every reading on the regular grid gets equal weight, so the percentages are percent of *recorded* time. Input is a pandas
Series of mg/dL indexed by timestamps; 1-min interpolated streams (CGMacros) are averaged into 5-min bins first.
"""
import numpy as np
import pandas as pd

# percent of readings: below 54, below 70, 70-140 and 70-180 inclusive, above 140 / 180 / 250 (mg/dL)
TIME_RANGES = {
    "tbr54": lambda v: v < 54, "tbr70": lambda v: v < 70, "tir70_140": lambda v: (v >= 70) & (v <= 140),
    "tir70_180": lambda v: (v >= 70) & (v <= 180), "tar140": lambda v: v > 140, "tar180": lambda v: v > 180,
    "tar250": lambda v: v > 250,
}
# Klonoff 2022: GRI zones are the quintiles of the clinician-ranked GRI grid: A 0-20, B 21-40, C 41-60, D 61-80, E 81-100
GRI_ZONES = [("A", 20), ("B", 40), ("C", 60), ("D", 80), ("E", 100)]


def infer_interval(index):
    """Native sampling interval in minutes (5 or 15 for the supported sensors; 1-min interpolated input -> 5)."""
    dt = np.diff(index.values).astype("timedelta64[s]").astype(float) / 60.0
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 5
    med = float(np.median(dt))
    if med < 10:          # 5-min sensors, or 1-min interpolated copies of them
        return 5
    if med < 22:
        return 15
    return int(round(med))


def regularize(glucose, interval_min=None):
    """Series(DatetimeIndex, mg/dL) -> series on a regular grid (mean per bin, NaN where nothing was recorded)."""
    g = pd.Series(glucose, dtype=float).dropna()
    g = g[~g.index.isna()].sort_index()
    g = g[~g.index.duplicated()]
    if len(g) == 0:
        return g
    step = interval_min or infer_interval(g.index)
    r = g.resample(f"{step}min", origin="start").mean()
    r.attrs["interval_min"] = step
    return r


def first_days(g, days):
    """Keep [t0, t0 + days) where t0 is the first reading (days=None keeps everything)."""
    if days is None or len(g) == 0:
        return g
    t0 = g.dropna().index[0]
    out = g[(g.index >= t0) & (g.index < t0 + pd.Timedelta(days=days))]
    out.attrs.update(g.attrs)
    return out


def gri(v):
    """Glycemia risk index and its components from readings v (mg/dL)."""
    v = np.asarray(v, float)
    vlow = 100 * np.mean(v < 54)
    low = 100 * np.mean((v >= 54) & (v < 70))
    high = 100 * np.mean((v > 180) & (v <= 250))
    vhigh = 100 * np.mean(v > 250)
    hypo = vlow + 0.8 * low
    hyper = vhigh + 0.5 * high
    return dict(gri=float(min(100.0, 3.0 * hypo + 1.6 * hyper)), gri_hypo=float(hypo), gri_hyper=float(hyper),
                gri_vlow=float(vlow), gri_low=float(low), gri_high=float(high), gri_vhigh=float(vhigh))


def gri_zone(value):
    for z, ub in GRI_ZONES:
        if value <= ub:
            return z
    return "E"


def _fill_short_gaps(v, maxlen):
    """Linearly fill interior NaN runs of at most maxlen samples; longer runs stay NaN (no partial filling)."""
    v = v.copy()
    if maxlen <= 0:
        return v
    nan = ~np.isfinite(v)
    i, n = 0, len(v)
    while i < n:
        if nan[i]:
            j = i
            while j < n and nan[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= maxlen:
                v[i:j] = np.interp(np.arange(i, j), [i - 1, j], [v[i - 1], v[j]])
            i = j
        else:
            i += 1
    return v


def _segments(g, max_gap_min):
    """Contiguous runs of the regular series; gaps up to max_gap_min are linearly filled, longer gaps split runs."""
    step = g.attrs.get("interval_min") or infer_interval(g.index)
    v = _fill_short_gaps(g.to_numpy(float), int(max_gap_min // step))
    ok = np.isfinite(v)
    segs, start = [], None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        if (not flag) and start is not None:
            segs.append(v[start:i]); start = None
    if start is not None:
        segs.append(v[start:])
    return segs, step


def _excursions(x, thr):
    """Alternating peaks/nadirs whose consecutive differences are >= thr (hysteresis turning points)."""
    if len(x) < 3 or not np.isfinite(thr) or thr <= 0:
        return []
    pts = []
    hi = lo = x[0]; hi_i = lo_i = 0; direction = 0
    for i in range(1, len(x)):
        v = x[i]
        if direction == 0:
            if v > hi: hi, hi_i = v, i
            if v < lo: lo, lo_i = v, i
            if hi - lo >= thr:
                if hi_i > lo_i:
                    pts.append(lo); direction = 1; lo = v
                else:
                    pts.append(hi); direction = -1; hi = v
        elif direction == 1:            # rising, tracking the peak
            if v > hi:
                hi = v
            elif hi - v >= thr:
                pts.append(hi); lo = v; direction = -1
        else:                           # falling, tracking the nadir
            if v < lo:
                lo = v
            elif v - lo >= thr:
                pts.append(lo); hi = v; direction = 1
    if direction == 1 and pts and hi - pts[-1] >= thr:
        pts.append(hi)
    if direction == -1 and pts and pts[-1] - lo >= thr:
        pts.append(lo)
    return pts


def mage(g, sd=None, max_gap_min=30):
    """Mean amplitude of glycemic excursions: mean |peak - nadir| over all up- and down-excursions larger than one SD
    of the whole record (Service 1970 criterion; both directions averaged, i.e. 'MAGE avg'). Runs are split at gaps
    longer than max_gap_min. Returns (mage, number of excursions)."""
    segs, _ = _segments(g, max_gap_min)
    allv = g.dropna().to_numpy(float)
    sd = float(np.std(allv, ddof=1)) if sd is None and len(allv) > 1 else sd
    amps = []
    for s in segs:
        p = _excursions(s, sd)
        amps.extend(np.abs(np.diff(p)).tolist())
    return (float(np.mean(amps)) if amps else np.nan), len(amps)


def dfa_alpha(g, scales_min=(60, 90, 120, 180, 240, 360), max_gap_min=60):
    """Detrended fluctuation analysis exponent (DFA-1). Gaps <= max_gap_min are interpolated, longer gaps are skipped
    by concatenating runs. Box sizes are fixed in minutes so 5- and 15-min data are comparable; boxes that leave fewer
    than 4 windows are dropped. Exploratory (cf. Colas A, et al., PLoS One 2019, which used DFA on the same Colas data)."""
    segs, step = _segments(g, max_gap_min)
    x = np.concatenate(segs) if segs else np.array([])
    if len(x) < 32:
        return np.nan
    y = np.cumsum(x - x.mean())
    ns, fs = [], []
    for smin in scales_min:
        n = int(round(smin / step))
        m = len(y) // n
        if n < 4 or m < 4:
            continue
        seg = y[: m * n].reshape(m, n).T                  # n x m
        t = np.arange(n, dtype=float)
        A = np.vstack([t, np.ones(n)]).T
        coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
        res = seg - A @ coef
        ns.append(n); fs.append(np.sqrt(np.mean(res ** 2)))
    if len(ns) < 3:
        return np.nan
    return float(np.polyfit(np.log(ns), np.log(fs), 1)[0])


def cgm_metrics(glucose, interval_min=None, days=None, regular=False):
    """All metrics for one record. glucose: Series(DatetimeIndex, mg/dL). days: keep only the first `days` days.
    regular=True means the series is already on its native regular grid (output of regularize)."""
    g = glucose if regular else regularize(glucose, interval_min)
    g = first_days(g, days)
    step = g.attrs.get("interval_min") or interval_min or infer_interval(g.index)
    v = g.dropna().to_numpy(float)
    out = dict(n_readings=int(len(v)), interval_min=int(step))
    if len(v) < 12:
        return out
    t = g.dropna().index
    span_days = ((t[-1] - t[0]).total_seconds() / 60 + step) / 1440
    out.update(days_span=float(span_days), days_of_data=float(len(v) * step / 1440),
               coverage=float(len(v) / max(1, len(g))))
    mean, sd = float(np.mean(v)), float(np.std(v, ddof=1))
    out.update(mean=mean, sd=sd, cv=100 * sd / mean, gmi=3.31 + 0.02392 * mean, median=float(np.median(v)),
               min=float(v.min()), max=float(v.max()))
    for q in (5, 10, 25, 75, 90, 95):
        out[f"p{q:02d}"] = float(np.percentile(v, q))
    out["iqr"] = out["p75"] - out["p25"]
    for k, fn in TIME_RANGES.items():
        out[k] = float(100 * np.mean(fn(v)))
    out.update(gri(v))
    out["gri_zone"] = gri_zone(out["gri"])
    out["mage"], out["n_excursions"] = mage(g, sd)
    out["dfa_alpha"] = dfa_alpha(g)
    hours = t.hour
    night = v[(hours >= 0) & (hours < 6)]
    day = v[hours >= 6]
    out["mean_nocturnal"] = float(np.mean(night)) if len(night) >= 12 else np.nan
    out["mean_daytime"] = float(np.mean(day)) if len(day) >= 12 else np.nan
    return out
