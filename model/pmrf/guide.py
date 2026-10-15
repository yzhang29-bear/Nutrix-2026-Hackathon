"""Personal diet guide built on the trained response model (Predictor):
- carb budget per meal: the largest carbohydrate amount whose predicted rise above the pre-meal level stays at or below a target
  (30 mg/dL by default), averaged over the person's own week of meals (their real contexts). A rise target, unlike an absolute
  peak target, does not depend on a sensor's overall calibration offset;
- the person's biggest spikes in days 1-7 and, for each, which single change the model expects to help most;
- probability that a meal's peak exceeds 140 or 180 mg/dL, from the calibrated 80% band.
All counterfactuals are model-based estimates, not measured outcomes.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

CARB_GRID = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120], float)
SWAPS = [("Half the carbs", dict(carbs_mult=0.5)), ("Add 20 g protein", dict(protein_add=20)), ("Add 10 g fiber", dict(fiber_add=10)),
         ("Swap 30 g carbs for 30 g protein", dict(carbs_add=-30, protein_add=30))]
Z80 = 1.2816


def _set(X, carbs=None, protein=None, fat=None, fiber=None):
    Z = X.copy()
    for c, v in [("carbs", carbs), ("protein", protein), ("fat", fat), ("fiber", fiber)]:
        if v is not None:
            Z[f"est_{c}"] = np.clip(np.asarray(v, float), 0, None)
    Z["est_kcal"] = 4 * Z.est_carbs + 4 * Z.est_protein + 9 * Z.est_fat
    return Z


def exceed_prob(g0, out, i, thresholds=(140, 180)):
    """P(peak glucose > threshold) from the blended peak and the band width at the predicted time of peak"""
    k = int(np.clip(round(float(out["ttp"][i]) / 5), 0, out["curve"].shape[1] - 1))
    sd = max((out["band"][1][i, k] - out["band"][0][i, k]) / (2 * Z80), 3.0)
    mu = g0 + float(out["dpeak"][i])
    return {str(t): float(1 - norm.cdf((t - mu) / sd)) for t in thresholds}


def _predict(pred, X, pre, emb, H):
    return pred._predict_rows(X.reset_index(drop=True), pre, H, emb, contributions=False)


def person_guide(pred, pid, target_rise=30.0, H=None):
    """pred: pmrf.predict.Predictor; uses the person's history state (days 1-7): H, or the bundle's state for pid."""
    H = H if H is not None else pred.b["person_state"][pid]["hist"]
    if len(H) < 3:
        return dict(available=False, reason="fewer than 3 usable meals in the 7-day history")
    # 1) carb budget: replace carbs in each of the person's meals with each grid value (logged protein/fat/fiber kept)
    rows, pres, embs, meta = [], [], [], []
    for j in range(len(H)):
        h = H.iloc[[j]]
        base = _set(h, protein=h.protein, fat=h.fat, fiber=h.fiber.fillna(0))
        for c in CARB_GRID:
            rows.append(_set(base, carbs=[c])); pres.append(h.pre.iloc[0]); embs.append(h.emb.iloc[0]); meta.append((j, c))
    X = pd.concat(rows, ignore_index=True)
    out = _predict(pred, X, np.stack(pres), np.stack(embs), H)
    g0 = X.G0.to_numpy()
    peak = g0 + out["dpeak"]
    df = pd.DataFrame(meta, columns=["meal", "carbs"]).assign(peak=peak, rise=out["dpeak"])
    curve = df.groupby("carbs")[["peak", "rise"]].mean()
    ok = curve.index[curve.rise <= target_rise]
    if len(ok) == 0:
        budget = 0.0
    elif ok.max() >= CARB_GRID.max():
        budget = float(CARB_GRID.max())
    else:
        hi = curve.index[curve.index > ok.max()].min()          # interpolate the crossing between grid points
        a, b = curve.rise.loc[ok.max()], curve.rise.loc[hi]
        budget = float(ok.max() + (target_rise - a) / max(b - a, 1e-6) * (hi - ok.max()))
    mid = curve.loc[(curve.index >= 15) & (curve.index <= 90)]
    slope10 = float(np.polyfit(mid.index, mid.rise, 1)[0] * 10)
    # 2) biggest spikes in the week and the most helpful single change for each
    spikes = []
    for j in np.argsort(-H.dpeak.to_numpy())[:3]:
        h = H.iloc[[j]]
        base = _set(h, carbs=h.carbs, protein=h.protein, fat=h.fat, fiber=h.fiber.fillna(0))
        variants = [base]
        for _, mod in SWAPS:
            variants.append(_set(base, carbs=np.clip(base.est_carbs * mod.get("carbs_mult", 1) + mod.get("carbs_add", 0), 0, None),
                                 protein=base.est_protein + mod.get("protein_add", 0), fiber=base.est_fiber + mod.get("fiber_add", 0)))
        Xv = pd.concat(variants, ignore_index=True)
        ov = _predict(pred, Xv, np.repeat(h.pre.iloc[0][None], len(Xv), 0), np.repeat(h.emb.iloc[0][None], len(Xv), 0), H)
        effects = [dict(change=name, predicted_rise_mgdl=round(float(ov["dpeak"][k + 1]), 1),
                        change_vs_as_eaten_mgdl=round(float(ov["dpeak"][k + 1] - ov["dpeak"][0]), 1)) for k, (name, _) in enumerate(SWAPS)]
        best = min(effects, key=lambda e: e["change_vs_as_eaten_mgdl"])
        spikes.append(dict(photo_path=h.photo.iloc[0], time=str(h.time.iloc[0]), meal_type=h.meal_type.iloc[0],
                           logged_carbs_g=round(float(h.carbs.iloc[0]), 1), observed_rise_mgdl=round(float(h.dpeak.iloc[0]), 1),
                           model_rise_as_eaten_mgdl=round(float(ov["dpeak"][0]), 1), best_change=best["change"], options=effects))
    # 3) time-of-day pattern from the person's own observed responses (needs at least 3 meals in each window)
    rate = H.dpeak / (H.carbs.clip(lower=10) / 10)
    tod = {}
    for name, sel in [("morning", H.hour < 11), ("evening", H.hour >= 17)]:
        if sel.sum() >= 3:
            tod[name] = round(float(rate[sel].median()), 1)
    return dict(available=True, target_rise_mgdl=target_rise, carb_budget_g_per_meal=round(budget, 0),
                predicted_rise_per_10g_carbs_mgdl=round(slope10, 1),
                carb_curve=[dict(carbs_g=float(c), mean_predicted_rise_mgdl=round(float(r.rise), 1), mean_predicted_peak_mgdl=round(float(r.peak), 1))
                            for c, r in curve.iterrows()],
                biggest_spikes=spikes, rise_per_10g_by_time_of_day_mgdl=tod,
                note="Model-based estimates from this person's first week; not medical advice.")


# ---------------------------------------------------------------------------------------------------------------------
# Measured personal nutrient effects (preferred when available). In CGMacros every participant drank standardized shakes in
# week 1 that differ in one nutrient; the person's own contrasts, shrunk toward the group, predicted their day-8+ shakes better
# than group effects (MAE 24.8 vs 29.1 mg/dL, scripts/nutrient_effects.py). Model counterfactuals from the main predictor are
# attenuated (photo features stay fixed), so they are only the fallback.
SHAKE_CHANGES = {"low carb": ("carbs", -42.0), "+protein": ("protein", 44.0), "+fat": ("fat", 31.5)}


def personal_effects(effects, pid, kshrink=1.0):
    """per-gram effects on the rise (mg/dL per g) for carbs, protein and fat, plus the measured contrasts"""
    pop, r7 = effects["pop"], effects["r7"]
    out, measured = {}, {}
    for name, (nut, grams) in SHAKE_CHANGES.items():
        if name not in pop:
            continue
        d, n = np.nan, 0
        if pid in r7.index and name in r7["mean"].columns:
            d = r7["mean"].loc[pid, name] - r7["mean"].loc[pid, "reference"]
            n = np.nanmin([r7["size"].loc[pid, name], r7["size"].loc[pid, "reference"]])
        e = pop[name] if not np.isfinite(d) else (n * d + kshrink * pop[name]) / (n + kshrink)
        out[nut] = e / grams
        measured[name] = dict(you_mgdl=None if not np.isfinite(d) else round(float(d), 1), adjusted_mgdl=round(float(e), 1),
                              group_mgdl=round(float(pop[name]), 1))
    return out, measured


def person_guide_measured(H, effects, pid, target_rise=50.0):
    """H: the person's labelled history meals (days 1-7)."""
    if pid not in effects["r7"].index or not np.isfinite(effects["r7"]["mean"].loc[pid].get("reference", np.nan)):
        return None
    per_g, measured = personal_effects(effects, pid)
    ref_rise = float(effects["r7"]["mean"].loc[pid, "reference"])
    slope = max(per_g.get("carbs", np.nan), 1e-3)     # mg/dL per extra gram of carbs (measured as a 42 g cut, so the ratio is positive)
    budget = float(np.clip(66 + (target_rise - ref_rise) / slope, 0, 120))
    spikes = []
    for j in np.argsort(-H.dpeak.to_numpy())[:3]:
        h = H.iloc[j]
        opts = [("Half the carbs", -slope * 0.5 * h.carbs), ("Add 20 g protein", per_g.get("protein", 0) * 20),
                ("Swap 30 g carbs for 30 g protein", -slope * min(30, h.carbs) + per_g.get("protein", 0) * min(30, h.carbs))]
        effects_ = [dict(change=n, predicted_rise_mgdl=round(float(max(h.dpeak + d, 0)), 1), change_vs_as_eaten_mgdl=round(float(d), 1)) for n, d in opts]
        best = min(effects_, key=lambda e: e["change_vs_as_eaten_mgdl"])
        spikes.append(dict(photo_path=h.photo, time=str(h.time), meal_type=h.meal_type, logged_carbs_g=round(float(h.carbs), 1),
                           observed_rise_mgdl=round(float(h.dpeak), 1), best_change=best["change"], options=effects_))
    rate = H.dpeak / (H.carbs.clip(lower=10) / 10)
    tod = {name: round(float(rate[sel].median()), 1) for name, sel in [("morning", H.hour < 11), ("evening", H.hour >= 17)] if sel.sum() >= 3}
    return dict(available=True, method="measured in your first-week standardized meals", target_rise_mgdl=target_rise,
                carb_budget_g_per_meal=round(budget, 0), predicted_rise_per_10g_carbs_mgdl=round(slope * 10, 1),
                measured_effects=measured, biggest_spikes=spikes, rise_per_10g_by_time_of_day_mgdl=tod,
                note="Based on your own week-1 standardized meals (liquid shakes, which usually raise glucose faster than solid food), adjusted toward the group average; estimates, not medical advice.")


def person_guide_group(H, effects, target_rise=50.0):
    """For people without their own standardized meals (new users): the group-average nutrient effects measured in the
    CGMacros standardized meals (better than no adjustment on day-8+ shakes: MAE 29.1 vs 35.8 mg/dL), applied to the person's
    own biggest week-1 spikes. A personal carb budget needs the person's own reference meal, so it is not given.
    (Model counterfactuals were checked as a fallback and under-react to carbs about five-fold, so they are not used.)"""
    if len(H) == 0:
        return dict(available=False, reason="no usable meals in the first week")
    per_g = {nut: effects["pop"][name] / grams for name, (nut, grams) in SHAKE_CHANGES.items() if name in effects["pop"]}
    slope = max(per_g.get("carbs", np.nan), 1e-3)
    spikes = []
    for j in np.argsort(-H.dpeak.to_numpy())[:3]:
        h = H.iloc[j]
        opts = [("Half the carbs", -slope * 0.5 * h.carbs), ("Add 20 g protein", per_g.get("protein", 0) * 20),
                ("Swap 30 g carbs for 30 g protein", -slope * min(30, h.carbs) + per_g.get("protein", 0) * min(30, h.carbs))]
        eff = [dict(change=n, predicted_rise_mgdl=round(float(max(h.dpeak + d, 0)), 1), change_vs_as_eaten_mgdl=round(float(d), 1)) for n, d in opts]
        spikes.append(dict(photo_path=h.photo, time=str(h.time), meal_type=h.meal_type, logged_carbs_g=round(float(h.carbs), 1),
                           observed_rise_mgdl=round(float(h.dpeak), 1), best_change=min(eff, key=lambda e: e["change_vs_as_eaten_mgdl"])["change"],
                           options=eff))
    rate = H.dpeak / (H.carbs.clip(lower=10) / 10)
    tod = {name: round(float(rate[sel].median()), 1) for name, sel in [("morning", H.hour < 11), ("evening", H.hour >= 17)] if sel.sum() >= 3}
    return dict(available=True, method="group-average effects (no standardized meals of your own)", personalized=False, target_rise_mgdl=target_rise,
                carb_budget_g_per_meal=None, carb_budget_note="needs one standardized test meal of your own in week 1",
                predicted_rise_per_10g_carbs_mgdl=round(slope * 10, 1),
                measured_effects={name: dict(you_mgdl=None, adjusted_mgdl=round(float(v), 1), group_mgdl=round(float(v), 1)) for name, v in effects["pop"].items()
                                  if name in SHAKE_CHANGES},
                biggest_spikes=spikes, rise_per_10g_by_time_of_day_mgdl=tod,
                note="Your biggest first-week spikes with the average effect of each change measured in standardized meals (liquid shakes, which usually raise glucose faster than solid food); estimates, not medical advice.")
