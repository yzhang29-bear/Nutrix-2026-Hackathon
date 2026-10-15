"""Metabolic profile & risk from CGM: estimated HbA1c, glycemic status, GRI zone, literature-based complication note and
(for people without diabetes) a future type 2 diabetes risk.

Fitted parameters live in results/risk/models.pkl (written by scripts/run_risk.py); `metabolic_profile` only loads them.
Validation numbers quoted in the output come from that file (they were computed by run_risk.py, see summary.json).

What each part is and is not:
  * Estimated HbA1c: ridge regression from CGM summary metrics (days 1-7) fitted on the outpatient cohorts (CGMacros,
    BIG IDEAs, Colas; HbA1c 4.6-8.5 %). The pooled model that also used the Shanghai T1DM/T2DM cohorts was evaluated but
    not shipped: in those cohorts every CGM-based estimate under-read lab HbA1c by 2-3 %-points, and including them biased
    the outpatient estimates. The 95 % interval is the empirical spread of cross-validated residuals in the Dexcom cohorts
    (CGMacros + BIG IDEAs). Not a lab test; not validated for insulin-treated or hospitalised patients.
  * Glycemic status: the HbA1c class (<5.7 / 5.7-6.4 / >=6.5 %) of the estimate; probabilities propagate the residual spread.
    HbA1c class is not a diagnosis.
  * GRI zone and the TIR -> complication relation are LITERATURE-DERIVED, POPULATION-LEVEL relations (no fitting on our
    data); they are not an individualized clinical risk.
  * Future T2D risk: logistic model fitted on the Colas cohort (208 non-diabetic hypertensive Spanish adults, 17 incident
    T2D over a median ~2.7 y); returned only for people without diabetes and only if run_risk.py marked it as shippable.
"""
import os
import pickle

import numpy as np
import pandas as pd

from pmrf.cgm_metrics import cgm_metrics, regularize

DEFAULT_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "risk", "models.pkl")

# CGM metrics used by the HbA1c model: level, spread and time-in-range summaries that do not depend on the sensor's
# sampling interval or on record length (min/max/coverage/days are left out on purpose: they identify cohorts).
A1C_FEATURES = ["mean", "sd", "cv", "median", "p10", "p25", "p75", "p90", "tir70_140", "tir70_180", "tar180", "tar250",
                "tbr70", "mean_nocturnal", "mage"]
STATUS_CLASSES = ["normal", "prediabetes", "diabetes"]          # HbA1c < 5.7, 5.7-6.4, >= 6.5 %
STATUS_CUTS = (5.7, 6.5)


def status_of(a1c):
    a = np.asarray(a1c, float)
    return np.where(a < STATUS_CUTS[0], 0, np.where(a < STATUS_CUTS[1], 1, 2))


def gmi(mean_glucose):
    """Bergenstal RM, et al. Glucose Management Indicator (GMI): a new term for estimating A1C from continuous glucose
    monitoring. Diabetes Care 2018;41(11):2275-2280. GMI(%) = 3.31 + 0.02392 x mean glucose (mg/dL)."""
    return 3.31 + 0.02392 * np.asarray(mean_glucose, float)


# ------------------------------------------------------------------ literature-derived relations (no fitting)
# Beck RW, Bergenstal RM, Riddlesworth TD, Kollman C, Li Z, Brown AS, Close KL. Validation of time in range as an outcome
# measure for diabetes clinical trials. Diabetes Care 2019;42(3):400-405. doi:10.2337/dc18-1444.
# DCCT data (type 1 diabetes; TIR 70-180 mg/dL computed from quarterly 7-point fingerstick profiles, not CGM):
# each 10 percentage points LOWER TIR -> hazard of retinopathy progression x1.64 and of microalbuminuria development x1.40.
BECK_2019 = dict(
    citation="Beck RW, et al. Validation of time in range as an outcome measure for diabetes clinical trials. "
             "Diabetes Care 2019;42(3):400-405. doi:10.2337/dc18-1444",
    hr_per_10pct_lower_tir={"retinopathy_progression": 1.64, "microalbuminuria_development": 1.40},
    population="type 1 diabetes (DCCT); TIR from 7-point fingerstick profiles, not CGM",
)
# Reference TIR: the >70 % target of the international consensus (Battelino T, et al. Clinical targets for continuous
# glucose monitoring data interpretation: recommendations from the International Consensus on Time in Range.
# Diabetes Care 2019;42(8):1593-1603. doi:10.2337/dci19-0028).
TIR_REFERENCE = 70.0
# Klonoff DC, et al. A Glycemia Risk Index (GRI) of hypoglycemia and hyperglycemia for continuous glucose monitoring
# validated by clinician ratings.
# J Diabetes Sci Technol 2023;17(5):1226-1242 (online 2022). doi:10.1177/19322968221085273.
# Zones are the quintiles of the clinician-ranked GRI distribution (developed on CGM tracings of adults with diabetes).
GRI_ZONE_TEXT = {
    "A": "GRI 0-20 (zone A, lowest-risk quintile of the reference tracings)",
    "B": "GRI 21-40 (zone B)", "C": "GRI 41-60 (zone C)", "D": "GRI 61-80 (zone D)",
    "E": "GRI 81-100 (zone E, highest-risk quintile)",
}
GRI_CITATION = ("Klonoff DC, et al. A Glycemia Risk Index (GRI) of hypoglycemia and hyperglycemia for continuous "
                "glucose monitoring validated by clinician ratings. J Diabetes Sci Technol 2023;17(5):1226-1242. "
                "doi:10.1177/19322968221085273")
LITERATURE_DISCLAIMER = ("Literature-derived, population-level relation applied to this CGM record; not fitted on our data "
                         "and not an individualized clinical risk estimate.")


def tir_relative_hazard(tir, reference=TIR_REFERENCE):
    """Relative hazard vs a person at the reference TIR, extrapolating Beck 2019's per-10-point hazard ratios
    (log-linear in TIR, as in their Cox models)."""
    d = (reference - float(tir)) / 10.0
    return {k: float(hr ** d) for k, hr in BECK_2019["hr_per_10pct_lower_tir"].items()}


def literature_risk(metrics, applicable=True):
    tir = metrics.get("tir70_180")
    out = dict(source="literature", disclaimer=LITERATURE_DISCLAIMER,
               gri=dict(value=metrics.get("gri"), zone=metrics.get("gri_zone"),
                        zone_text=GRI_ZONE_TEXT.get(metrics.get("gri_zone")), hypo_component=metrics.get("gri_hypo"),
                        hyper_component=metrics.get("gri_hyper"), citation=GRI_CITATION,
                        note="Zones were derived from CGM tracings of adults with diabetes."),
               tir_complications=dict(tir70_180=tir, reference_tir=TIR_REFERENCE, citation=BECK_2019["citation"],
                                      population=BECK_2019["population"], applicable=bool(applicable)))
    if tir is not None and np.isfinite(tir):
        rh = tir_relative_hazard(tir)
        out["tir_complications"]["relative_hazard_vs_reference"] = rh if applicable else None
        out["tir_complications"]["note"] = (
            f"Beck et al. 2019 (DCCT, type 1 diabetes): each 10 points lower TIR multiplies the hazard of retinopathy "
            f"progression by 1.64 and of microalbuminuria by 1.40. "
            + (f"At TIR {tir:.0f}% vs the {TIR_REFERENCE:.0f}% target: x{rh['retinopathy_progression']:.2f} and "
               f"x{rh['microalbuminuria_development']:.2f}." if applicable else
               "Not applied: the relation was derived in diabetes and does not transfer to people without diabetes."))
    return out


# ------------------------------------------------------------------ model application helpers
def load_models(path=DEFAULT_MODELS):
    with open(path, "rb") as f:
        return pickle.load(f)


def _row(metrics, features, extra=None):
    d = {k: metrics.get(k, np.nan) for k in features}
    if extra:
        d.update({k: (np.nan if v is None else v) for k, v in extra.items() if k in features})
    return pd.DataFrame([d], columns=features).astype(float)


def _predict(model, X, proba=False):
    """Pass a DataFrame only to models fitted on one (avoids sklearn feature-name warnings either way)."""
    X = X if hasattr(model, "feature_names_in_") else X.to_numpy()
    return model.predict_proba(X)[:, 1] if proba else model.predict(X)


def estimate_hba1c(metrics, bundle):
    """Model estimate, GMI, empirical interval and class probabilities from the cross-validated residual distribution."""
    a = bundle["a1c"]
    pred = float(_predict(a["model"], _row(metrics, a["features"]))[0])
    res = np.asarray(a["residuals"], float)             # lab - prediction, out-of-fold
    lo, hi = np.quantile(res, [0.025, 0.975])
    sims = pred + res
    probs = {c: float(np.mean(status_of(sims) == i)) for i, c in enumerate(STATUS_CLASSES)}
    g = float(gmi(metrics["mean"]))
    return dict(model=pred, interval95=[pred + float(lo), pred + float(hi)], gmi=g,
                status=STATUS_CLASSES[int(status_of(pred))], status_gmi=STATUS_CLASSES[int(status_of(g))],
                status_probabilities=probs, model_name=a["name"], validation=a["validation"],
                note="CGM-based estimate, not a laboratory HbA1c; the HbA1c class is not a diagnosis.")


def t2d_risk(metrics, bundle, age=None, bmi=None, fasting_glucose=None, hba1c=None):
    t = bundle.get("t2d")
    if not t or not t.get("ship"):
        return dict(available=False, reason=(t or {}).get("reason", "no validated model"))
    clin = dict(age=age, bmi=bmi, fasting_glucose=fasting_glucose, hba1c=hba1c)
    use = None
    for name in t["order"]:                               # most informative model whose inputs are all present
        m = t["models"][name]
        if all((k in metrics and np.isfinite(metrics[k])) or (clin.get(k) is not None) for k in m["features"]):
            use = name
            break
    if use is None:
        return dict(available=False, reason="required inputs missing")
    m = t["models"][use]
    p = float(_predict(m["model"], _row(metrics, m["features"], clin), proba=True)[0])
    return dict(available=True, probability=p, model=use, features=m["features"], validation=m["validation"],
                horizon=t["horizon"], population=t["population"], caveat=t["caveat"],
                calibration_warning=t.get("calibration_warning"))


def metabolic_profile(glucose_series, age=None, bmi=None, sex=None, fasting_glucose=None, hba1c_lab=None,
                      has_diabetes=None, days=7, models=None, interval_min=None):
    """CGM-based metabolic profile.

    glucose_series: pandas Series of mg/dL indexed by timestamp (5-min or 15-min sensor data; 1-min interpolated is fine).
    days: use only the first `days` days of the record (None = all). has_diabetes: known diagnosis if available; when None
    the estimated HbA1c class decides whether the future-T2D risk is shown. hba1c_lab / fasting_glucose / age / bmi are
    optional inputs to the T2D risk model (the CGM-only model is used when they are missing).
    """
    bundle = models if isinstance(models, dict) else load_models(models or DEFAULT_MODELS)
    g = regularize(glucose_series, interval_min)
    m = cgm_metrics(g, days=days, regular=True)
    out = dict(cgm_metrics=m, inputs=dict(age=age, bmi=bmi, sex=sex, fasting_glucose=fasting_glucose, hba1c_lab=hba1c_lab,
                                          has_diabetes=has_diabetes, days=days))
    if "mean" not in m:
        out["error"] = "too few glucose readings"
        return out
    est = estimate_hba1c(m, bundle)
    out["hba1c_estimate"] = est
    diabetic = has_diabetes if has_diabetes is not None else (est["status"] == "diabetes")
    out["literature_risk"] = literature_risk(m, applicable=bool(diabetic))
    if diabetic:
        out["t2d_risk"] = dict(available=False, reason="not shown for people with (known or estimated) diabetes")
    else:
        out["t2d_risk"] = t2d_risk(m, bundle, age=age, bmi=bmi, fasting_glucose=fasting_glucose, hba1c=hba1c_lab)
    out["coverage_warning"] = (None if m.get("days_of_data", 0) >= 0.7 * min(days or 14, 14) else
                               f"only {m.get('days_of_data', 0):.1f} days of CGM data in the window")
    return out


def app_summary(d, tier_reference=None):
    """metabolic_profile output -> the compact `metabolic_profile` object of the API. The future-T2D risk is reported as a
    tier relative to tier_reference (risk values of the development cohort's participants without diabetes), because the
    absolute value is not calibrated for Dexcom data."""
    if not d or "hba1c_estimate" not in d:
        return None
    hb = d["hba1c_estimate"]
    st = hb.get("status")
    tier = None
    t2 = d.get("t2d_risk", {})
    ref = np.asarray(tier_reference or [], float)
    if t2.get("available") and len(ref) >= 6:
        q = float(np.mean(ref <= t2["probability"]))
        tier = "lower than most" if q <= 1 / 3 else ("about average" if q <= 2 / 3 else "higher than most")
    iv = hb.get("interval95") or [None, None]
    return dict(est_hba1c=hb.get("model"), est_hba1c_lo=iv[0], est_hba1c_hi=iv[1], gmi=hb.get("gmi"),
                status=dict(label=st, prob=(hb.get("status_probabilities") or {}).get(st)),
                gri_zone=d.get("literature_risk", {}).get("gri", {}).get("zone"), t2d_risk_tier=tier,
                note="Estimated from the first week of CGM. Research estimates, not a diagnosis.")
