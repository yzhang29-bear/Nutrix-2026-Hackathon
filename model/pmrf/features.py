"""Meal-level targets and features. Every feature uses only information available before the meal starts."""
import numpy as np
import pandas as pd

CURVE_T = np.arange(0, 121, 5)            # predicted trajectory grid, minutes after meal start (25 points)
PRE_T = np.arange(-120, 1, 5)             # pre-meal CGM sequence for sequence models (25 points)
HR_PRE_T = np.arange(-60, 1, 5)           # pre-meal heart-rate sequence (13 points)
DELTA_T = [15, 30, 45, 60, 75, 90, 105, 120]
TARGETS = ["G30", "G60", "G90", "G120", "Gmax", "dpeak", "ttp", "iauc"]
MED_CLASSES = ["rapid_insulin", "basal_insulin", "metformin", "sulfonylurea", "glp1", "sglt2", "dpp4", "other"]
IOB_TAU = 90.0                            # min, exponential decay of rapid-acting insulin activity

CONTEXT_FEATURES = ["G0", "g0_rel24", "slope30", "slope60", "sd60", "gmean3h", "hour", "tod_sin", "tod_cos",
                    "prev_gap", "prev_carbs", "n_meals_6h", "hr_pre", "hr_max60"]
MED_FEATURES = [f"med_{c}_since" for c in MED_CLASSES] + [f"med_{c}_dose24" for c in MED_CLASSES] + ["iob", "meal_bolus", "bolus_lead"]
ACTIVITY_FEATURES = ["act_mets30", "act_mets120", "act_active120", "act_kcal120"]   # activity before the meal only
DEMO_FEATURES = ["age", "sex_female", "bmi"]
LOGGED_KCAL_FEATURES = ["meal_kcal_logged"]   # demo only: the logged meal calories are not available for a day-8 meal
PROFILE_FEATURES = ["p_n", "p_dpeak", "p_dpeak_med", "p_iauc", "p_ttp", "p_dpeak_per10g", "p_carb_slope",
                    "p_g_mean", "p_g_cv", "p_g_tir", "p_g_night", "p_hr_rest"]


def _window(series, t, offs):
    idx = pd.Timestamp(t).floor("min") + pd.to_timedelta(offs, unit="min")
    return series.reindex(idx).to_numpy(dtype=float)


def resting_hr(p, days=7):
    dd = p.day(p.hr.index)
    h = p.hr[(dd >= 1) & (dd <= days)].dropna()        # days 1..days of the history window
    return float(np.percentile(h, 10)) if len(h) > 500 else np.nan


def med_features(meds, t):
    out = {k: np.nan for k in MED_FEATURES}
    if meds is None or len(meds) == 0:
        return out
    dt = (pd.Timestamp(t) - pd.to_datetime(meds.time)).dt.total_seconds().to_numpy() / 60   # minutes before the meal
    for c in MED_CLASSES:
        sel = (meds.med_class.to_numpy() == c)
        past = sel & (dt >= 0)
        if past.any():
            out[f"med_{c}_since"] = min(dt[past].min(), 1440.0)
            out[f"med_{c}_dose24"] = float(meds.dose.to_numpy()[past & (dt <= 1440)].sum())
    rapid = meds.med_class.to_numpy() == "rapid_insulin"
    if rapid.any():
        d = meds.dose.to_numpy(float)
        prior = rapid & (dt > 15) & (dt <= 480)
        out["iob"] = float(np.sum(d[prior] * np.exp(-dt[prior] / IOB_TAU)))
        around = rapid & (dt >= -15) & (dt <= 15)        # meal bolus: within 15 min either side of the meal
        out["meal_bolus"] = float(d[around].sum())
        out["bolus_lead"] = float(dt[around].max()) if around.any() else np.nan   # >0 = given before eating
    return out


def meal_table(persons, hist_days=7):
    """One row per logged meal: identifiers, macros, pre-meal context, targets and the observed curve."""
    rows, curves, pre_seq, hr_seq = [], [], [], []
    for pid, p in persons.items():
        rest = resting_hr(p, hist_days)
        m = p.meals.sort_values("time").reset_index(drop=True)
        days = p.day(m.time)
        times = pd.to_datetime(m.time)
        prev_gap = times.diff().dt.total_seconds().to_numpy() / 60
        next_gap = -times.diff(-1).dt.total_seconds().to_numpy() / 60
        for i, r in m.iterrows():
            t = pd.Timestamp(r.time)
            g = _window(p.cgm, t, np.arange(-1440, 181))
            o = lambda k: k + 1440
            win = g[o(-30):o(120) + 1]
            valid = bool(np.isfinite(g[o(0)]) and np.mean(np.isfinite(win)) >= 0.9)
            gi = pd.Series(g[o(-120):]).interpolate(limit_direction="both").to_numpy()   # -120..+180, edges held
            q = lambda k: k + 120
            row = dict(pid=pid, time=t, day=int(days[i]), meal_type=r.meal_type, carbs=r.carbs, protein=r.protein, fat=r.fat,
                       fiber=r.fiber, kcal=r.kcal, photo=r.photo, valid=valid, next_gap=next_gap[i], group=p.meta.get("group"))
            if np.isfinite(gi).any():
                g0 = gi[q(0)]
                post = gi[q(0):q(120) + 1]
                k = int(np.nanargmax(post))
                row.update(G0=g0, G30=gi[q(30)], G60=gi[q(60)], G90=gi[q(90)], G120=gi[q(120)], Gmax=post[k], dpeak=post[k] - g0,
                           ttp=float(k), iauc=float(np.trapz(np.clip(post - g0, 0, None), dx=1)))
                for dtm in DELTA_T:
                    row[f"d{dtm}"] = gi[q(dtm)] - g0
                prior = g[:o(0)]
                row["g0_rel24"] = g0 - np.nanmedian(prior) if np.isfinite(prior).sum() > 600 else np.nan
                row["slope30"] = (g0 - gi[q(-30)]) / 30
                row["slope60"] = (g0 - gi[q(-60)]) / 60
                row["sd60"] = float(np.nanstd(g[o(-60):o(0) + 1])) if np.isfinite(g[o(-60):o(0) + 1]).sum() > 20 else np.nan
                row["gmean3h"] = float(np.nanmean(g[o(-180):o(0) + 1])) if np.isfinite(g[o(-180):o(0) + 1]).sum() > 60 else np.nan
                curves.append(gi[q(0):q(120) + 1:5] - g0)
                pre_seq.append(gi[q(-120):q(0) + 1:5] - g0)
            else:
                curves.append(np.full(len(CURVE_T), np.nan)); pre_seq.append(np.full(len(PRE_T), np.nan))
            h = _window(p.hr, t, np.arange(-60, 121))
            row["hr_pre"] = np.nanmean(h[30:61]) - rest if np.isfinite(h[30:61]).any() else np.nan
            row["hr_max60"] = np.nanmax(h[0:61]) - rest if np.isfinite(h[0:61]).any() else np.nan
            row["hr_post"] = np.nanmean(h[60:]) - rest if np.isfinite(h[60:]).any() else np.nan     # analysis only (not known at prediction time)
            hs = h[0:61:5] - rest
            hr_seq.append(hs)
            for c in ACTIVITY_FEATURES:                        # missing activity = NaN (the columns always exist)
                row[c] = np.nan
            if getattr(p, "act", None) is not None:
                a = _window(p.act, t, np.arange(-120, 0))
                row["act_mets30"] = np.nanmean(a[-30:]) if np.isfinite(a[-30:]).any() else np.nan
                row["act_mets120"] = np.nanmean(a) if np.isfinite(a).any() else np.nan
                row["act_active120"] = float(np.sum(a >= 3.0)) if np.isfinite(a).any() else np.nan     # minutes at >= 3 METs
            if getattr(p, "act_kcal", None) is not None:
                k = _window(p.act_kcal, t, np.arange(-120, 0))
                row["act_kcal120"] = float(np.nansum(k)) if np.isfinite(k).any() else np.nan
            for c in DEMO_FEATURES:
                row[c] = (getattr(p, "info", None) or {}).get(c, np.nan)
            cl = r.get("calories_logged", np.nan) if hasattr(r, "get") else np.nan
            row["meal_kcal_logged"] = float(cl) if pd.notna(cl) else float(r.kcal) if pd.notna(r.kcal) else np.nan
            row["hour"] = t.hour + t.minute / 60
            row["tod_sin"], row["tod_cos"] = np.sin(2 * np.pi * row["hour"] / 24), np.cos(2 * np.pi * row["hour"] / 24)
            row["prev_gap"] = min(prev_gap[i], 720.0) if np.isfinite(prev_gap[i]) else 720.0
            row["prev_carbs"] = float(m.carbs.iloc[i - 1]) if i > 0 and prev_gap[i] < 360 else 0.0
            row["n_meals_6h"] = int(np.sum((times < t) & (times >= t - pd.Timedelta("6h"))))
            row.update(med_features(p.meds, t))
            rows.append(row)
    M = pd.DataFrame(rows)
    M["label"] = M.valid & (M.next_gap.isna() | (M.next_gap >= 120))     # targets not contaminated by a following meal
    return M, np.array(curves), np.array(pre_seq), np.array(hr_seq)


def cgm_profile(p, hist_days=7):
    """CGM and heart-rate descriptors of days 1..hist_days"""
    dd = p.day(p.cgm.index)
    c = p.cgm[(dd >= 1) & (dd <= hist_days)].dropna()
    night = c[(c.index.hour >= 0) & (c.index.hour < 6)]
    return dict(p_g_mean=c.mean() if len(c) else np.nan, p_g_cv=c.std() / c.mean() if len(c) else np.nan,
                p_g_tir=float(np.mean((c >= 70) & (c <= 180))) if len(c) else np.nan,
                p_g_night=night.median() if len(night) else np.nan, p_hr_rest=resting_hr(p, hist_days))


def meal_profile(h):
    """response descriptors from labelled history meals h"""
    f = dict(p_n=len(h))
    if len(h):
        big = h[h.carbs >= 15]
        hc = h[h.carbs.notna() & h.dpeak.notna()]          # meals recorded without nutrients (Shanghai) are left out of the fit
        slope = np.polyfit(hc.carbs, hc.dpeak, 1)[0] if len(hc) >= 5 and hc.carbs.std() > 5 else np.nan
        f.update(p_dpeak=h.dpeak.mean(), p_dpeak_med=h.dpeak.median(), p_iauc=h.iauc.mean(), p_ttp=h.ttp.mean(),
                 p_dpeak_per10g=float(np.median(big.dpeak / (big.carbs / 10))) if len(big) else np.nan,
                 p_carb_slope=slope * len(h) / (len(h) + 5) if np.isfinite(slope) else np.nan)
    return f


def person_profile(persons, M, hist_mask, hist_days=7):
    """Person descriptors from history meals only. Rows that are themselves history meals get leave-one-out values."""
    H = M[hist_mask & M.label]
    cgm_stats = {pid: cgm_profile(p, hist_days) for pid, p in persons.items()}
    out = []
    for idx, r in M.iterrows():
        f = meal_profile(H[(H.pid == r.pid) & (H.index != idx)])
        f.update(cgm_stats[r.pid])
        out.append(f)
    return pd.DataFrame(out, index=M.index).reindex(columns=PROFILE_FEATURES)
