"""Observed outcome of a meal, from the CGM stored after it, for predicted-vs-observed comparisons.

Same definitions as the model's training targets (pmrf.features.meal_table): CGM on the person's 1-min grid, the
meal counts only if there is a reading at the meal time and at least 90% of minutes -30..+120 have one; short gaps
are interpolated; G0 is glucose at the meal time, iAUC is the area above G0 over 0-120 min.
"""
import numpy as np
import pandas as pd

from pmrf.features import CURVE_T, _window

PRE_MIN, POST_MIN, CGM_AFTER_MIN = 120, 120, 180     # CGM up to +180 min fills gaps at the end, as in training


def _r(x, nd=1):
    """pmrf.predict._r, copied so this module does not import torch"""
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def observed_outcome(person, meal_time, prediction=None):
    """dict like the demo's `observed` (plus CGM coverage and how much of it lies in the 80% band), or None without enough CGM"""
    g = _window(person.cgm, meal_time, np.arange(-PRE_MIN, CGM_AFTER_MIN + 1))
    o = lambda k: k + PRE_MIN
    if not (np.isfinite(g[o(0)]) and np.mean(np.isfinite(g[o(-30):o(POST_MIN) + 1])) >= 0.9):
        return None
    gi = pd.Series(g).interpolate(limit_direction="both").to_numpy()
    g0 = gi[o(0)]
    post = gi[o(0):o(POST_MIN) + 1]
    k = int(np.argmax(post))
    curve = [_r(v) for v in post[CURVE_T]]
    out = dict(curve=dict(minutes=[int(m) for m in CURVE_T], glucose_mgdl=curve),
               glucose_at_min={str(m): _r(post[m]) for m in (30, 60, 90, 120)},
               pre_meal_glucose_mgdl=_r(g0), peak_glucose_mgdl=_r(post[k]), peak_rise_mgdl=_r(post[k] - g0),
               time_to_peak_min=k, iauc_0_120_mgdl_min=_r(float(np.trapz(np.clip(post - g0, 0, None), dx=1)), 0),
               cgm_coverage=_r(float(np.mean(np.isfinite(g[o(0):o(POST_MIN) + 1]))), 2))
    if prediction is not None:
        c = prediction["curve"]
        lo, hi = np.array(c["lower80"], float), np.array(c["upper80"], float)
        obs = np.array(curve, float)
        out["share_of_curve_inside_band80"] = _r(float(np.mean((obs >= lo) & (obs <= hi))), 2)
    return out
