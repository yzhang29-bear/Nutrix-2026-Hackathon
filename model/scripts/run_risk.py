"""Metabolic profile & risk module: runs analyses B-E end to end, fits and saves the shipped models (F), writes
results/risk/summary.json, tables (*.csv), figures (fig/*.png) and the CGMacros day-1-7 profiles for the app.

usage (from the project root):  OMP_NUM_THREADS=8 .venv/bin/python -I scripts/run_risk.py [out_dir] [n_boot]

B  HbA1c / glycemic status from CGM, pooled cohorts (CGMacros, BIG IDEAs, Shanghai T1DM+T2DM, Colas)
C  incident T2D in the Colas cohort (17 events)
D  existing micro/macrovascular complications in Shanghai T2DM (cross-sectional association)
E  literature-based mapping (GRI zones, Beck 2019 TIR hazard ratios) - no fitting
"""
import sys, os, glob, json, pickle, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr
from scipy.optimize import minimize
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

from pmrf.risk_data import load_all
from pmrf.cgm_metrics import cgm_metrics
from pmrf import risk as R
from eda_common import save_fig

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "risk")
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)
T0 = time.time()
RNG = np.random.default_rng(20261010)
COHORTS = ["CGMacros", "BIG IDEAs", "Shanghai T1DM", "Shanghai T2DM", "Colas"]
LOCO_GROUP = {"CGMacros": "CGMacros", "BIG IDEAs": "BIG IDEAs", "Shanghai T1DM": "Shanghai", "Shanghai T2DM": "Shanghai",
              "Colas": "Colas"}
OUTPATIENT = ["CGMacros", "BIG IDEAs", "Colas"]
N_REP_B, N_REP_C = 10, 20
SUMMARY = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M"), n_boot=NB, ci="95% percentile bootstrap over persons "
                         "(stratified by cohort for B, by outcome for C/D); each bootstrap replicate averages the metric over all CV repeats"))


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def js(x):
    """JSON-safe conversion (NaN -> None, numpy -> python)."""
    if isinstance(x, dict):
        return {str(k): js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [js(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 5)
    if isinstance(x, np.ndarray):
        return js(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# =============================================================== data + features
log("loading cohorts")
RECS = load_all(ROOT)
META_KEYS = ["cohort", "pid", "hba1c", "sex", "age", "bmi", "fasting_glucose", "duration", "micro_any", "macro_any",
             "t2dm", "followup_days", "patient", "n_sessions", "agents", "micro", "macro", "clinical_row_label"]


def features(days, recs=RECS, thin=1):
    rows = []
    for r in recs:
        g = r["glucose"]
        if thin > 1:                                      # sampling sensitivity: keep every thin-th reading
            step = g.attrs.get("interval_min", 5)
            g = g.iloc[::thin].copy(); g.attrs["interval_min"] = step * thin
        m = cgm_metrics(g, days=days, regular=True)
        rows.append({**{k: r.get(k, np.nan) for k in META_KEYS}, **m})
    return pd.DataFrame(rows)


F7, FALL = features(7), features(None)
F7.to_csv(os.path.join(OUT, "features_7d.csv"), index=False)
FALL.to_csv(os.path.join(OUT, "features_all.csv"), index=False)
inv = FALL.groupby("cohort").agg(n_persons=("pid", "size"), n_hba1c=("hba1c", "count"), hba1c_min=("hba1c", "min"),
                                 hba1c_median=("hba1c", "median"), hba1c_max=("hba1c", "max"),
                                 days_median=("days_of_data", "median"), days_min=("days_of_data", "min"),
                                 days_max=("days_of_data", "max"), interval_min=("interval_min", "first"),
                                 mean_glucose_median=("mean", "median")).reindex(COHORTS)
inv.to_csv(os.path.join(OUT, "data_inventory.csv"))
SUMMARY["data_inventory"] = inv.reset_index().to_dict(orient="records")
log("features done", len(F7))


# =============================================================== generic stats helpers
def boot_idx(strata, nb=NB, rng=RNG):
    """Stratified bootstrap index sets (persons resampled within strata)."""
    strata = np.asarray(strata)
    groups = [np.where(strata == s)[0] for s in pd.unique(strata)]
    out = np.empty((nb, len(strata)), dtype=int)
    for b in range(nb):
        out[b] = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
    return out


def ci(stat_fn, idx_sets, *arrays):
    """Point value on the full sample + percentile CI over bootstrap index sets. arrays are (n,) or (R, n)."""
    pt = stat_fn(*arrays)
    vals = np.array([stat_fn(*[a[..., ix] for a in arrays]) for ix in idx_sets])
    vals = vals[np.isfinite(vals)]
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) > 10 else (np.nan, np.nan))
    return dict(value=float(pt), lo=float(lo), hi=float(hi))


def auc_rows(y, S):
    """AUC of score rows S (R, n) against binary y (n,), via the rank-sum formula (ties averaged)."""
    S = np.atleast_2d(S)
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.full(len(S), np.nan)
    rk = rankdata(S, axis=1)
    return (rk[:, y].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def mean_auc(y, S):
    return float(np.mean(auc_rows(y, S)))


def harrell_c(time_, event, score):
    t, e, s = map(np.asarray, (time_, event, score))
    ok = np.isfinite(t) & np.isfinite(s)
    t, e, s = t[ok], e[ok].astype(bool), s[ok]
    comp = (t[:, None] < t[None, :]) & e[:, None]          # i had the event before j's time
    conc = (s[:, None] > s[None, :]) & comp
    tie = (s[:, None] == s[None, :]) & comp
    return float((conc.sum() + 0.5 * tie.sum()) / comp.sum()) if comp.sum() else np.nan


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calib_stats(y, p):
    """Calibration-in-the-large (intercept with logit(p) offset), calibration slope, E/O ratio, Brier score."""
    y = np.asarray(y, float); lp = logit(p)

    def nll(theta, slope_free):
        a, b = (theta[0], theta[1]) if slope_free else (theta[0], 1.0)
        z = a + b * lp
        return np.sum(np.logaddexp(0, z) - y * z)
    citl = minimize(nll, [0.0], args=(False,), method="BFGS").x[0]
    ab = minimize(nll, [0.0, 1.0], args=(True,), method="BFGS").x
    return dict(citl=float(citl), slope=float(ab[1]), e_over_o=float(np.sum(p) / max(1, np.sum(y))),
                brier=float(np.mean((p - y) ** 2)), mean_pred=float(np.mean(p)), obs_rate=float(np.mean(y)))


# =============================================================== B: HbA1c from CGM
log("B: HbA1c from CGM")
FEAT = R.A1C_FEATURES


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 21)))


def recal():
    return make_pipeline(SimpleImputer(strategy="median"), LinearRegression())


def lgbm():
    return make_pipeline(SimpleImputer(strategy="median"),
                         LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10, subsample=0.8,
                                       subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4, verbose=-1,
                                       random_state=0))


B_MODELS = {"Recalibrated mean (linear)": (recal, ["mean"]), "Ridge (CGM metrics)": (ridge, FEAT),
            "LightGBM (CGM metrics)": (lgbm, FEAT)}


def cv_reg(D, make, feats, n_rep=N_REP_B, train_mask=None, seed=0):
    """Repeated 5-fold CV stratified by cohort -> OOF predictions (n_rep, n). train_mask restricts training rows."""
    X, y, strat = D[feats].to_numpy(float), D.hba1c.to_numpy(float), D.cohort.to_numpy()
    tm = np.ones(len(D), bool) if train_mask is None else np.asarray(train_mask)
    P = np.full((n_rep, len(D)), np.nan)
    for r in range(n_rep):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, strat):
            tr = tr[tm[tr]]
            m = make().fit(X[tr], y[tr])
            P[r, te] = m.predict(X[te])
    return P


def loco_reg(D, make, feats):
    X, y, grp = D[feats].to_numpy(float), D.hba1c.to_numpy(float), D.cohort.map(LOCO_GROUP).to_numpy()
    P = np.full(len(D), np.nan)
    for g in pd.unique(grp):
        te = grp == g
        P[te] = make().fit(X[~te], y[~te]).predict(X[te])
    return P


def rowcorr(y, P):
    yc = y - y.mean()
    Pc = P - P.mean(1, keepdims=True)
    den = np.sqrt((Pc ** 2).sum(1) * (yc ** 2).sum())
    return np.where(den > 0, (Pc @ yc) / np.where(den > 0, den, 1), np.nan)


def reg_metrics(y, P, idx_sets):
    """y (n,), P (R, n) -> dict of metrics with CIs; class accuracy uses the HbA1c cuts 5.7 / 6.5 %."""
    P = np.atleast_2d(P)
    yc = R.status_of(y)
    f = dict(
        mae=lambda y, P: float(np.mean(np.abs(P - y))),
        bias=lambda y, P: float(np.mean(P - y)),
        r=lambda y, P: float(np.mean(rowcorr(y, P))),
        within_0_5=lambda y, P: float(np.mean(np.abs(P - y) <= 0.5)),
        acc3=lambda y, P: float(np.mean(R.status_of(P) == R.status_of(y))),
    )
    out = {k: ci(fn, idx_sets, y, P) for k, fn in f.items()}
    out["n"] = int(len(y))
    out["class_counts"] = {c: int(np.sum(yc == i)) for i, c in enumerate(R.STATUS_CLASSES)}
    return out


def evaluate_b(D, preds, label):
    """preds: {model: (R, n) array}. Pooled (bootstrap stratified by cohort) and per-cohort metrics."""
    rows, res = [], {}
    y = D.hba1c.to_numpy(float)
    coh = D.cohort.to_numpy()
    for name, P in preds.items():
        P = np.atleast_2d(P)
        ok = np.isfinite(P).all(0)
        res[name] = {}
        scopes = [("pooled", ok)] + [(c, ok & (coh == c)) for c in COHORTS]
        if "pooled training" in label:
            scopes.insert(1, ("outpatient (CGMacros+BIG IDEAs+Colas)", ok & np.isin(coh, OUTPATIENT)))
        for scope, mk in scopes:
            if mk.sum() < 5:
                continue
            m = reg_metrics(y[mk], P[:, mk], boot_idx(coh[mk]))
            res[name][scope] = m
            rows.append(dict(eval=label, model=name, scope=scope, n=m["n"],
                             **{f"{k}{s}": m[k][v] for k in ["mae", "bias", "r", "within_0_5", "acc3"]
                                for s, v in [("", "value"), ("_lo", "lo"), ("_hi", "hi")]}))
    return res, rows


DB = {w: F[F.hba1c.notna()].reset_index(drop=True) for w, F in [("7d", F7), ("all", FALL)]}
B_ROWS, B_RES, OOF, B_DIFF = [], {}, {}, {}
OUTPT_NAME = "Ridge (CGM metrics), outpatient-trained"
RECAL_O_NAME = "Recalibrated mean (linear), outpatient-trained"


def mae_diff(y, P1, P2):
    return float(np.mean(np.abs(np.atleast_2d(P1) - y)) - np.mean(np.abs(np.atleast_2d(P2) - y)))


def loco_outpatient(D, make, feats):
    """Leave-one-cohort-out within the outpatient cohorts (train on the other two); Shanghai is predicted by the model
    trained on all three outpatient cohorts."""
    X, y, coh = D[feats].to_numpy(float), D.hba1c.to_numpy(float), D.cohort.to_numpy()
    out = np.isin(coh, OUTPATIENT)
    P = np.full(len(D), np.nan)
    for g in OUTPATIENT:
        te = coh == g
        P[te] = make().fit(X[out & ~te], y[out & ~te]).predict(X[te])
    P[~out] = make().fit(X[out], y[out]).predict(X[~out])
    return P


for w, D in DB.items():
    gmi_all = R.gmi(D["mean"].to_numpy())[None, :]
    # (1) models trained on all five cohorts pooled
    preds = {"GMI formula": gmi_all}
    for name, (mk, feats) in B_MODELS.items():
        preds[name] = cv_reg(D, mk, feats)
        log(f"  B {w} {name} done")
    res, rows = evaluate_b(D, preds, f"cv {N_REP_B}x5-fold, pooled training, window {w}")
    B_RES[f"cv_pooled_training_{w}"] = res; B_ROWS += rows
    lpreds = {"GMI formula": gmi_all}
    for name, (mk, feats) in B_MODELS.items():
        lpreds[name] = loco_reg(D, mk, feats)[None, :]
    res, rows = evaluate_b(D, lpreds, f"leave-one-cohort-out, pooled training, window {w}")
    B_RES[f"loco_pooled_training_{w}"] = res; B_ROWS += rows
    # (2) models trained on the outpatient cohorts only (the population the app serves)
    Do = D[D.cohort.isin(OUTPATIENT)].reset_index(drop=True)
    gmi_o = R.gmi(Do["mean"].to_numpy())[None, :]
    opreds = {"GMI formula": gmi_o, RECAL_O_NAME: cv_reg(Do, recal, ["mean"]), OUTPT_NAME: cv_reg(Do, ridge, FEAT)}
    res, rows = evaluate_b(Do, opreds, f"cv {N_REP_B}x5-fold, outpatient training, window {w}")
    B_RES[f"cv_outpatient_training_{w}"] = res; B_ROWS += rows
    olpreds = {"GMI formula": gmi_all, RECAL_O_NAME: loco_outpatient(D, recal, ["mean"])[None, :],
               OUTPT_NAME: loco_outpatient(D, ridge, FEAT)[None, :]}
    res, rows = evaluate_b(D, olpreds, f"leave-one-cohort-out, outpatient training, window {w}")
    B_RES[f"loco_outpatient_training_{w}"] = res; B_ROWS += rows
    # paired MAE differences (negative = first model better)
    yo, co = Do.hba1c.to_numpy(float), Do.cohort.to_numpy()
    diffs = {}
    for scope, mk in [("outpatient pooled", np.ones(len(Do), bool))] + [(c, co == c) for c in OUTPATIENT]:
        ix = boot_idx(co[mk])
        diffs[scope] = {
            f"{OUTPT_NAME} minus GMI": ci(mae_diff, ix, yo[mk], opreds[OUTPT_NAME][:, mk], gmi_o[:, mk]),
            f"{OUTPT_NAME} minus {RECAL_O_NAME}": ci(mae_diff, ix, yo[mk], opreds[OUTPT_NAME][:, mk], opreds[RECAL_O_NAME][:, mk]),
        }
    y, co_all = D.hba1c.to_numpy(float), D.cohort.to_numpy()
    for c in COHORTS:
        mk = co_all == c
        diffs.setdefault(c, {})[f"Ridge (CGM metrics) pooled-trained minus GMI"] = ci(
            mae_diff, boot_idx(co_all[mk]), y[mk], preds["Ridge (CGM metrics)"][:, mk], gmi_all[:, mk])
    B_DIFF[w] = diffs
    OOF[w] = dict(D=D, Do=Do, cv=preds, loco=lpreds, cv_o=opreds, loco_o=olpreds)
pd.DataFrame(B_ROWS).to_csv(os.path.join(OUT, "b_hba1c_metrics.csv"), index=False)
D7, D7o = OOF["7d"]["D"], OOF["7d"]["Do"]
oof_tab = D7[["cohort", "pid", "hba1c", "mean", "days_of_data"]].copy()
for name, P in OOF["7d"]["cv"].items():
    oof_tab[f"cv_pooled_{name}"] = np.nanmean(np.atleast_2d(P), 0)
key = D7o.cohort + "|" + D7o.pid
for name, P in OOF["7d"]["cv_o"].items():
    if name != "GMI formula":
        oof_tab[f"cv_{name}"] = (D7.cohort + "|" + D7.pid).map(dict(zip(key, np.nanmean(P, 0))))
for name, P in OOF["7d"]["loco"].items():
    if name != "GMI formula":
        oof_tab[f"loco_pooled_{name}"] = P[0]
for name, P in OOF["7d"]["loco_o"].items():
    if name != "GMI formula":
        oof_tab[f"loco_{name}"] = P[0]
oof_tab.to_csv(os.path.join(OUT, "b_oof_predictions_7d.csv"), index=False)
SUMMARY["B_hba1c"] = dict(
    design="one record per person; features from the first 7 days of CGM (or all available if shorter; Colas has 1-2 days) "
           f"and from all days; {N_REP_B}x5-fold CV stratified by cohort; leave-one-cohort-out with Shanghai T1DM+T2DM as one "
           "cohort. Two training sets: all five cohorts pooled, and the outpatient cohorts only (CGMacros, BIG IDEAs, Colas).",
    features=FEAT, results=B_RES, mae_differences=B_DIFF)
log("B main done")

# ---- days of CGM used (records >= 7 days), outpatient (CGMacros+BIG IDEAs) and Shanghai separately
DAY_ROWS = []
for dw in [1, 2, 3, 5, 7, None]:
    Fd = F7 if dw == 7 else (FALL if dw is None else features(dw))
    D = Fd[Fd.hba1c.notna()].reset_index(drop=True)
    Do = D[D.cohort.isin(OUTPATIENT)].reset_index(drop=True)
    P_o = cv_reg(Do, ridge, FEAT, n_rep=5)
    P_p = cv_reg(D, ridge, FEAT, n_rep=5)
    for grp, DD_, mk, P in [("CGMacros+BIG IDEAs", Do, Do.cohort.isin(["CGMacros", "BIG IDEAs"]).to_numpy(), P_o),
                            ("Shanghai T1DM+T2DM", D, D.cohort.str.startswith("Shanghai").to_numpy(), P_p)]:
        for name, PP in [("GMI formula", R.gmi(DD_["mean"].to_numpy())[None, :]),
                         (OUTPT_NAME if grp.startswith("CGM") else "Ridge (CGM metrics), pooled-trained", P)]:
            m = reg_metrics(DD_.hba1c.to_numpy()[mk], PP[:, mk], boot_idx(DD_.cohort.to_numpy()[mk], nb=min(NB, 1000)))
            DAY_ROWS.append(dict(group=grp, days="all" if dw is None else dw, model=name, n=m["n"], mae=m["mae"]["value"],
                                 mae_lo=m["mae"]["lo"], mae_hi=m["mae"]["hi"], bias=m["bias"]["value"], r=m["r"]["value"],
                                 acc3=m["acc3"]["value"], median_days_used=float(DD_.days_of_data[mk].median())))
pd.DataFrame(DAY_ROWS).to_csv(os.path.join(OUT, "b_days_of_cgm.csv"), index=False)
SUMMARY["B_hba1c"]["days_of_cgm"] = dict(scope="first k days of each record; ridge CV 5x5-fold (outpatient-trained for "
                                         "CGMacros+BIG IDEAs, pooled-trained for Shanghai)", rows=DAY_ROWS)
log("B days done")

# ---- sampling-interval sensitivity: 5-min cohorts thinned to 15 min
five = [r for r in RECS if r["cohort"] in ("CGMacros", "BIG IDEAs")]
F5, F15 = features(7, five), features(7, five, thin=3)
samp = {}
for k in ["mean", "sd", "cv", "tir70_180", "tar180", "tbr70", "mage", "p90"]:
    d = (F15[k] - F5[k]).to_numpy()
    samp[k] = dict(mean_abs_diff=float(np.nanmean(np.abs(d))), mean_diff=float(np.nanmean(d)), value_median_5min=float(np.nanmedian(F5[k])))
m_full = ridge().fit(D7o[FEAT], D7o.hba1c)
dp = m_full.predict(F15[FEAT]) - m_full.predict(F5[FEAT])
samp["outpatient_ridge_prediction_change"] = dict(mean_abs_diff=float(np.mean(np.abs(dp))), max_abs_diff=float(np.max(np.abs(dp))))
SUMMARY["B_hba1c"]["sampling_sensitivity"] = dict(n=len(five), what="CGMacros+BIG IDEAs day 1-7 metrics at 5 min vs the "
                                                 "same data thinned to 15 min", metrics=samp)

# ---- sensor check inside CGMacros: the same 45 people wore a Libre (15-min) and a Dexcom G6 (5-min)
sens = []
for f in sorted(glob.glob(os.path.join(ROOT, "cgmacros", "CGMacros", "CGMacros-*", "CGMacros-*.csv"))):
    d = pd.read_csv(f)
    d.columns = [c.strip() for c in d.columns]
    d["Timestamp"] = pd.to_datetime(d["Timestamp"])
    d = d.drop_duplicates("Timestamp").set_index("Timestamp").sort_index()
    both = d[["Libre GL", "Dexcom GL"]].dropna()
    pid = os.path.basename(f)[9:12]
    lab = float(D7.loc[(D7.cohort == "CGMacros") & (D7.pid == pid), "hba1c"].iloc[0])
    sens.append(dict(pid=pid, lab=lab, gmi_libre=cgm_metrics(d["Libre GL"], interval_min=15, days=7)["gmi"],
                     gmi_dexcom=cgm_metrics(d["Dexcom GL"], interval_min=5, days=7)["gmi"],
                     dexcom_minus_libre=float((both["Dexcom GL"] - both["Libre GL"]).mean())))
sens = pd.DataFrame(sens)
sens.to_csv(os.path.join(OUT, "b_sensor_check_cgmacros.csv"), index=False)
ixs = boot_idx(np.zeros(len(sens)))
SUMMARY["B_hba1c"]["sensor_check_cgmacros"] = dict(
    n=len(sens), what="GMI from days 1-7 of each sensor vs lab HbA1c, same participants",
    gmi_libre_bias=ci(lambda y, p: float(np.mean(p - y)), ixs, sens.lab.to_numpy(), sens.gmi_libre.to_numpy()),
    gmi_dexcom_bias=ci(lambda y, p: float(np.mean(p - y)), ixs, sens.lab.to_numpy(), sens.gmi_dexcom.to_numpy()),
    gmi_libre_mae=ci(lambda y, p: float(np.mean(np.abs(p - y))), ixs, sens.lab.to_numpy(), sens.gmi_libre.to_numpy()),
    gmi_dexcom_mae=ci(lambda y, p: float(np.mean(np.abs(p - y))), ixs, sens.lab.to_numpy(), sens.gmi_dexcom.to_numpy()),
    dexcom_minus_libre_mgdl_median=float(sens.dexcom_minus_libre.median()))
log("B sensor check done")

# ---- status probabilities + interval coverage (second-level CV over the OOF residuals)
SHIP_A1C = OUTPT_NAME


def prob_eval(D, P, res_pool_mask=None, k=5, seed=7):
    """For each person, residual distribution = OOF residuals of persons in OTHER folds (optionally restricted to a
    cohort mask) -> 95% interval coverage and class probabilities (Brier, log loss, accuracy)."""
    y = D.hba1c.to_numpy(float); coh = D.cohort.to_numpy()
    pred = P[0]
    res = y - pred
    pool = np.ones(len(y), bool) if res_pool_mask is None else res_pool_mask
    cover = np.zeros(len(y), bool); prob = np.zeros((len(y), 3)); width = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(y, coh):
        rr = res[tr[pool[tr]]]
        lo, hi = np.quantile(rr, [0.025, 0.975])
        cover[te] = (y[te] >= pred[te] + lo) & (y[te] <= pred[te] + hi)
        width[te] = hi - lo
        sims = R.status_of(pred[te][:, None] + rr[None, :])
        prob[te] = np.stack([(sims == c).mean(1) for c in range(3)], 1)
    yc = R.status_of(y)
    onehot = np.eye(3)[yc]
    out = {}
    for scope, mk in [("all evaluated", np.ones(len(y), bool))] + [(c, coh == c) for c in COHORTS]:
        if mk.sum() == 0:
            continue
        out[scope] = dict(n=int(mk.sum()), coverage95=float(cover[mk].mean()), interval_width=float(np.median(width[mk])),
                          brier3=float(np.mean(np.sum((prob[mk] - onehot[mk]) ** 2, 1))),
                          brier3_reference_prevalence=float(np.mean(np.sum((onehot[mk].mean(0)[None, :] - onehot[mk]) ** 2, 1))),
                          logloss3=float(-np.mean(np.log(np.clip(prob[mk][np.arange(mk.sum()), yc[mk]], 1e-3, 1)))),
                          acc_argmax=float(np.mean(prob[mk].argmax(1) == yc[mk])),
                          mean_prob=dict(zip(R.STATUS_CLASSES, [float(v) for v in prob[mk].mean(0)])),
                          observed=dict(zip(R.STATUS_CLASSES, [float(v) for v in onehot[mk].mean(0)])))
    return out


Pship = OOF["7d"]["cv_o"][SHIP_A1C]
DEXCOM = ["CGMacros", "BIG IDEAs"]
SUMMARY["B_hba1c"]["interval_and_probabilities"] = dict(
    model=SHIP_A1C,
    residual_pool_all_outpatient=prob_eval(D7o, Pship),
    residual_pool_dexcom_cohorts=prob_eval(D7o, Pship, D7o.cohort.isin(DEXCOM).to_numpy()),
    shipped_pool="dexcom_cohorts (CGMacros + BIG IDEAs): same sensor type and wider HbA1c range than Colas, i.e. closest to "
                 "the app's 5-min CGM users; the all-outpatient pool is dominated by Colas (narrow HbA1c range) and "
                 "under-covers CGMacros",
    note="repeat-1 OOF predictions of the outpatient-trained ridge (7-day window, n=261); each person's 95% interval and "
         "class probabilities use only the OOF residuals of persons in the other 5 folds. brier3_reference_prevalence = "
         "Brier score of always predicting the observed class frequencies of that scope (lower is better).")
log("B probabilities done")


# =============================================================== C / D: binary outcomes with nested L2 logistic regression
def logit_model(seed):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                              scoring="neg_log_loss", penalty="l2", max_iter=5000))


def cv_bin(D, feats, ycol, n_rep=N_REP_C, seed=100):
    X, y = D[feats].to_numpy(float), D[ycol].to_numpy(int)
    P = np.full((n_rep, len(D)), np.nan)
    Cs = []
    for r in range(n_rep):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, y):
            m = logit_model(seed + 1000 + r).fit(X[tr], y[tr])
            P[r, te] = m.predict_proba(X[te])[:, 1]
            Cs.append(float(m[-1].C_[0]))
    return P, Cs


def roc_curve_rows(y, P, grid):
    """Mean ROC over CV repeats on a common FPR grid."""
    y = np.asarray(y).astype(bool)
    tprs = []
    for p in np.atleast_2d(P):
        o = np.argsort(-p, kind="mergesort")
        ys = y[o]
        tpr = np.concatenate([[0], np.cumsum(ys) / ys.sum()])
        fpr = np.concatenate([[0], np.cumsum(~ys) / (~ys).sum()])
        tprs.append(np.interp(grid, fpr, tpr))
    return np.mean(tprs, 0)


def binary_task(D, ycol, models, tag, time_col=None, strata=None):
    y = D[ycol].to_numpy(int)
    idx = boot_idx(y if strata is None else strata)
    res, rows, curves, P_all = {}, [], {}, {}
    grid = np.linspace(0, 1, 101)
    for name, feats in models.items():
        P, Cs = cv_bin(D, feats, ycol)
        P_all[name] = P
        a = ci(mean_auc, idx, y, P)
        per_rep = auc_rows(y, P)
        pm = P.mean(0)
        cal = calib_stats(y, pm)
        cal_ci = {k: ci(lambda yy, pp, k=k: calib_stats(yy, pp)[k], idx[: min(NB, 500)], y, pm) for k in ["citl", "slope"]}
        entry = dict(features=feats, n=int(len(y)), events=int(y.sum()), auc=a,
                     auc_repeat_range=[float(per_rep.min()), float(per_rep.max())],
                     chosen_C_median=float(np.median(Cs)), calibration={**cal, "citl_ci": cal_ci["citl"], "slope_ci": cal_ci["slope"]})
        if time_col is not None:
            t = D[time_col].to_numpy(float)
            ok = np.isfinite(t)
            entry["c_index"] = ci(lambda tt, ee, ss: harrell_c(tt, ee, ss), boot_idx(y[ok])[: min(NB, 1000)], t[ok], y[ok], pm[ok])
            entry["c_index"]["n"] = int(ok.sum())
        res[name] = entry
        bands = np.array([roc_curve_rows(y[ix], P[:, ix], grid) for ix in idx[:500]])
        curves[name] = dict(grid=grid, mean=roc_curve_rows(y, P, grid), lo=np.percentile(bands, 2.5, 0), hi=np.percentile(bands, 97.5, 0))
        rows.append(dict(task=tag, model=name, n=len(y), events=int(y.sum()), n_features=len(feats), auc=a["value"], auc_lo=a["lo"],
                         auc_hi=a["hi"], brier=cal["brier"], citl=cal["citl"], slope=cal["slope"], slope_lo=cal_ci["slope"]["lo"],
                         slope_hi=cal_ci["slope"]["hi"], e_over_o=cal["e_over_o"],
                         c_index=entry.get("c_index", {}).get("value"), features=" + ".join(feats)))
        log(f"  {tag} {name}: AUC {a['value']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]")
    key = {k: next((n for n in models if n.startswith(k)), None) for k in ["CGM-only", "Clinical", "Combined"]}
    diffs = {}
    for a_, b_ in [("Combined", "Clinical"), ("CGM-only", "Clinical"), ("Combined", "CGM-only")]:
        if key[a_] and key[b_]:
            diffs[f"{key[a_]} minus {key[b_]}"] = ci(lambda yy, p1, p2: mean_auc(yy, p1) - mean_auc(yy, p2), idx, y,
                                                     P_all[key[a_]], P_all[key[b_]])
    return res, rows, curves, P_all, diffs


def univariate(D, ycol, cols, strata=None):
    y = D[ycol].to_numpy(int)
    idx = boot_idx(y if strata is None else strata)
    rows = []
    for c in cols:
        x = D[c].to_numpy(float)
        ok = np.isfinite(x)
        a = ci(mean_auc, idx[:, :] if ok.all() else boot_idx(y[ok]), y[ok], x[ok][None, :]) if ok.sum() > 10 else None
        if a is None:
            continue
        rows.append(dict(variable=c, n=int(ok.sum()), events=int(y[ok].sum()), auc=a["value"], auc_lo=a["lo"], auc_hi=a["hi"],
                         direction="higher value -> outcome" if a["value"] >= 0.5 else "lower value -> outcome",
                         median_events=float(np.median(x[ok][y[ok] == 1])), median_nonevents=float(np.median(x[ok][y[ok] == 0]))))
    return pd.DataFrame(rows)


CGM_UNI = ["mean", "median", "sd", "cv", "p10", "p25", "p75", "p90", "p95", "max", "tir70_140", "tir70_180", "tar140", "tar180",
           "tbr70", "mage", "mean_nocturnal", "mean_daytime", "gri", "dfa_alpha"]

# ---------------------------------------------------------------- C: Colas incident T2D
log("C: Colas incident T2D")
DC = FALL[FALL.cohort == "Colas"].reset_index(drop=True).copy()
DC["t2dm"] = DC.t2dm.astype(int)
C_MODELS = {
    "CGM-only (mean, CV, %>140, MAGE)": ["mean", "cv", "tar140", "mage"],
    "Clinical (age, BMI, fasting glucose, HbA1c)": ["age", "bmi", "fasting_glucose", "hba1c"],
    "Combined (clinical + CGM, 8 predictors)": ["age", "bmi", "fasting_glucose", "hba1c", "mean", "cv", "tar140", "mage"],
    "CGM all metrics (exploratory, 15 predictors)": FEAT,
}
c_res, c_rows, c_curves, c_P, c_diff = binary_task(DC, "t2dm", C_MODELS, "C incident T2D (Colas)", time_col="followup_days")
c_uni = univariate(DC, "t2dm", CGM_UNI + ["age", "bmi", "fasting_glucose", "hba1c"])
c_uni.to_csv(os.path.join(OUT, "c_univariate_auc.csv"), index=False)
# sensitivity: drop people already in the diabetes range at baseline
base_dm = (DC.hba1c >= 6.5) | (DC.fasting_glucose >= 126)
DCs = DC[~base_dm].reset_index(drop=True)
cs_res, cs_rows, _, _, _ = binary_task(DCs, "t2dm", {k: v for k, v in C_MODELS.items() if "exploratory" not in k},
                                       "C sensitivity: baseline HbA1c<6.5 and FPG<126", time_col="followup_days")
# calibration table by predicted-risk quartile
cal_rows = []
yC = DC.t2dm.to_numpy()
for name, P in c_P.items():
    pm = P.mean(0)
    q = pd.qcut(pm, 4, labels=False, duplicates="drop")
    for g in sorted(np.unique(q)):
        mk = q == g
        k, n = int(yC[mk].sum()), int(mk.sum())
        z = 1.96; ph = k / n
        den = 1 + z * z / n; cen = (ph + z * z / (2 * n)) / den; half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        cal_rows.append(dict(model=name, quartile=int(g) + 1, n=n, events=k, mean_predicted=float(pm[mk].mean()),
                             observed=ph, observed_lo=cen - half, observed_hi=cen + half))
pd.DataFrame(cal_rows).to_csv(os.path.join(OUT, "c_calibration_quartiles.csv"), index=False)
pd.DataFrame(c_rows + cs_rows).to_csv(os.path.join(OUT, "c_t2d_auc.csv"), index=False)
fu = DC.followup_days
SUMMARY["C_incident_t2d"] = dict(
    cohort="Colas et al. 2019 (PLoS One): 208 non-diabetic hypertensive adults, CGM 1-2 days at baseline",
    n=int(len(DC)), events=int(DC.t2dm.sum()), followup_days=dict(median=float(fu.median()), q1=float(fu.quantile(.25)),
                                                                  q3=float(fu.quantile(.75)), missing=int(fu.isna().sum())),
    cgm_days=dict(median=float(DC.days_of_data.median()), min=float(DC.days_of_data.min())),
    design=f"{N_REP_C}x5-fold stratified CV; L2 logistic regression with C chosen by inner 5-fold CV (log loss) inside every "
           "training fold; missing predictors median-imputed inside the pipeline. AUC = mean over repeats of the pooled OOF AUC. "
           "C-index (Harrell) uses follow-up time on the repeat-averaged OOF risk. 17 events: the 4-predictor models are "
           "near the ~4-5 events-per-variable floor; the 8- and 15-predictor models rely on the L2 penalty.",
    models=c_res, auc_differences=c_diff,
    sensitivity_excluding_baseline_diabetes_range=dict(n=int(len(DCs)), events=int(DCs.t2dm.sum()), excluded=int(base_dm.sum()),
                                                      models=cs_res),
    univariate_auc=c_uni.to_dict(orient="records"),
    multiplicity_note=f"{len(c_uni)} univariate AUCs on 17 events; CIs are unadjusted and overlapping.")
log("C done")

# ---------------------------------------------------------------- D: Shanghai T2DM complications (cross-sectional)
log("D: Shanghai T2DM complications")
DD = FALL[FALL.cohort == "Shanghai T2DM"].reset_index(drop=True).copy()
D_MODELS = {
    "CGM-only (TIR, mean, CV, %<70)": ["tir70_180", "mean", "cv", "tbr70"],
    "Clinical (age, duration, HbA1c, BMI)": ["age", "duration", "hba1c", "bmi"],
    "Combined (clinical + CGM, 8 predictors)": ["age", "duration", "hba1c", "bmi", "tir70_180", "mean", "cv", "tbr70"],
}
D_RES, d_rows, d_curves, d_uni_all = {}, [], {}, []
for yc, lab in [("micro_any", "any microvascular complication"), ("macro_any", "any macrovascular complication")]:
    r_, rows_, cur_, P_, diff_ = binary_task(DD, yc, D_MODELS, f"D {lab} (Shanghai T2DM)")
    u = univariate(DD, yc, CGM_UNI + ["age", "duration", "hba1c", "bmi"])
    u.insert(0, "outcome", lab)
    d_uni_all.append(u)
    D_RES[yc] = dict(outcome=lab, n=int(len(DD)), events=int(DD[yc].sum()), models=r_, auc_differences=diff_,
                     univariate_auc=u.to_dict(orient="records"))
    d_rows += rows_; d_curves[yc] = cur_
pd.DataFrame(d_rows).to_csv(os.path.join(OUT, "d_complications_auc.csv"), index=False)
pd.concat(d_uni_all).to_csv(os.path.join(OUT, "d_univariate_auc.csv"), index=False)
SUMMARY["D_complications"] = dict(
    cohort="Shanghai T2DM (Zhao et al. 2023), one CGM session per patient (earliest-dated session with an HbA1c)",
    interpretation="cross-sectional association with complications that already exist; not prediction of future complications",
    hba1c_missing=int(DD.hba1c.isna().sum()), design=SUMMARY["C_incident_t2d"]["design"].split(" C-index")[0], results=D_RES)
log("D done")

# =============================================================== E: literature mapping (no fitting)
E_rows = []
for c in COHORTS:
    d = F7[F7.cohort == c]
    for z in "ABCDE":
        E_rows.append(dict(cohort=c, zone=z, n=int((d.gri_zone == z).sum()), pct=float(100 * (d.gri_zone == z).mean())))
pd.DataFrame(E_rows).to_csv(os.path.join(OUT, "e_gri_zones_7d.csv"), index=False)
lit = []
for _, r in F7.iterrows():
    dm = r.cohort.startswith("Shanghai")
    rh = R.tir_relative_hazard(r.tir70_180)
    lit.append(dict(cohort=r.cohort, pid=r.pid, tir70_180=r.tir70_180, gri=r.gri, gri_zone=r.gri_zone,
                    rh_retinopathy=rh["retinopathy_progression"] if dm else np.nan,
                    rh_microalbuminuria=rh["microalbuminuria_development"] if dm else np.nan))
pd.DataFrame(lit).to_csv(os.path.join(OUT, "e_literature_mapping_7d.csv"), index=False)
SUMMARY["E_literature"] = dict(
    label="LITERATURE-DERIVED, POPULATION-LEVEL; no fitting on our data; not an individualized clinical risk",
    beck_2019=R.BECK_2019, tir_reference=R.TIR_REFERENCE, gri_citation=R.GRI_CITATION,
    gri_zone_counts_7d=E_rows,
    low_glucose_by_cohort_7d={c: dict(n=int((F7.cohort == c).sum()), median_tbr70=float(F7.tbr70[F7.cohort == c].median()),
                                      pct_persons_tbr70_over_4=float(100 * (F7.tbr70[F7.cohort == c] > 4).mean()),
                                      pct_persons_tbr54_over_1=float(100 * (F7.tbr54[F7.cohort == c] > 1).mean()))
                              for c in COHORTS},
    applied_to="relative hazards computed only for the Shanghai cohorts (people with diabetes); GRI zones for everyone")

# =============================================================== F: fit + save shipped models
log("F: fitting final models")
a_ship = SUMMARY["B_hba1c"]["results"]["cv_outpatient_training_7d"][SHIP_A1C]
res_ship = D7o.hba1c.to_numpy()[None, :] - Pship                     # (10 repeats, 261) OOF residuals
a1c_final = ridge().fit(D7o[FEAT], D7o.hba1c)
VAL_A1C = {s: dict(n=a_ship[s]["n"], mae=a_ship[s]["mae"], bias=a_ship[s]["bias"], r=a_ship[s]["r"], acc3=a_ship[s]["acc3"])
           for s in ["pooled", "CGMacros", "BIG IDEAs", "Colas"] if s in a_ship}
VAL_A1C["interval_coverage95"] = {k: v["coverage95"] for k, v in
                                  SUMMARY["B_hba1c"]["interval_and_probabilities"]["residual_pool_dexcom_cohorts"].items()}
dex_mask = D7o.cohort.isin(DEXCOM).to_numpy()
VAL_A1C["not_valid_for"] = ("people with HbA1c far outside 4.6-8.5 % (training range) and insulin-treated / hospitalised "
                            "patients: in the Shanghai cohorts every CGM-based estimate under-read lab HbA1c by 2-3 %-points")
a1c_bundle = dict(name=f"{SHIP_A1C}, 7-day window, trained on {len(D7o)} persons (CGMacros, BIG IDEAs, Colas)", features=FEAT,
                  model=a1c_final, residuals=res_ship[:, dex_mask].ravel(),
                  residual_scope=f"out-of-fold residuals ({N_REP_B} CV repeats) of the same model on the {int(dex_mask.sum())} "
                                 "CGMacros + BIG IDEAs persons (Dexcom G6, outpatients)",
                  validation=VAL_A1C, alpha=float(a1c_final[-1].alpha_),
                  coefficients=dict(zip(FEAT, a1c_final[-1].coef_.tolist())), intercept=float(a1c_final[-1].intercept_))

# T2D model ship rule (fixed before looking at the numbers): CV AUC >= 0.70 and the CI lower bound > 0.50
T2D_ORDER = ["Combined (clinical + CGM, 8 predictors)", "CGM-only (mean, CV, %>140, MAGE)"]
t2d_models, ship_any = {}, False
for name in T2D_ORDER:
    e = c_res[name]
    ok = e["auc"]["value"] >= 0.70 and e["auc"]["lo"] > 0.50
    feats = C_MODELS[name]
    fitted = logit_model(4242).fit(DC[feats].to_numpy(float), DC.t2dm.to_numpy(int))
    t2d_models[name] = dict(features=feats, model=fitted, ship=bool(ok),
                            validation=dict(n=e["n"], events=e["events"], auc=e["auc"], calibration_slope=e["calibration"]["slope"],
                                            c_index=e.get("c_index")))
    ship_any |= ok
t2d_bundle = dict(ship=bool(ship_any), order=[n for n in T2D_ORDER if t2d_models[n]["ship"]], models=t2d_models,
                  horizon=f"incident T2D during the Colas follow-up (median {fu.median() / 365.25:.1f} years; "
                          f"IQR {fu.quantile(.25) / 365.25:.1f}-{fu.quantile(.75) / 365.25:.1f})",
                  population="Colas cohort: 208 non-diabetic hypertensive adults (Spain), 17 incident T2D; CGM 1-2 days",
                  caveat="small development cohort (17 events), different population and CGM duration from the app; "
                         "use as a relative indicator, not a clinical prediction",
                  ship_rule="cross-validated AUC >= 0.70 with 95% CI lower bound > 0.50",
                  reason=None if ship_any else "no model met the ship rule (CV AUC >= 0.70 and CI lower bound > 0.50)")
for n in T2D_ORDER:
    t2d_models[n]["model_C"] = float(t2d_models[n]["model"][-1].C_[0])
# transportability check (no outcomes): the CGM-only Colas model applied to CGMacros (Dexcom) days 1-7, non-diabetic by lab
cgm_name = "CGM-only (mean, CV, %>140, MAGE)"
cg7 = F7[(F7.cohort == "CGMacros") & (F7.hba1c < 6.5)].reset_index(drop=True)
risk_cg = t2d_models[cgm_name]["model"].predict_proba(cg7[C_MODELS[cgm_name]].to_numpy(float))[:, 1]
gmi_bias = {c: float(np.mean(R.gmi(F7["mean"][F7.cohort == c]) - F7.hba1c[F7.cohort == c])) for c in ["CGMacros", "BIG IDEAs", "Colas"]}
offset = (np.mean([gmi_bias["CGMacros"], gmi_bias["BIG IDEAs"]]) - gmi_bias["Colas"]) / 0.02392    # mg/dL
shifted = []
for r in RECS:
    if r["cohort"] == "CGMacros" and r["hba1c"] < 6.5:
        g = r["glucose"] - offset
        g.attrs.update(r["glucose"].attrs)
        shifted.append(cgm_metrics(g, days=7, regular=True))
shifted = pd.DataFrame(shifted)
risk_sh = t2d_models[cgm_name]["model"].predict_proba(shifted[C_MODELS[cgm_name]].to_numpy(float))[:, 1]
norm, pre = (cg7.hba1c < 5.7).to_numpy(), (cg7.hba1c >= 5.7).to_numpy()
colas_pred = c_P[cgm_name].mean(0)
transport = dict(
    n_normal=int(norm.sum()), n_prediabetes=int(pre.sum()),
    median_risk_lab_normal=float(np.median(risk_cg[norm])), median_risk_lab_prediabetes=float(np.median(risk_cg[pre])),
    auc_prediabetes_vs_normal=float(mean_auc(pre.astype(int), risk_cg[None, :])),
    colas_observed_incidence=float(DC.t2dm.mean()), colas_median_oof_risk=float(np.median(colas_pred)),
    gmi_bias_by_cohort=gmi_bias, implied_sensor_offset_mgdl=float(offset),
    after_offset_median_risk_lab_normal=float(np.median(risk_sh[norm])),
    after_offset_median_risk_lab_prediabetes=float(np.median(risk_sh[pre])),
    note="descriptive only (CGMacros has no follow-up); the offset is the GMI-minus-lab bias of the Dexcom cohorts minus that "
         "of Colas, converted to mg/dL; it is a rough sensitivity analysis, not applied in the shipped function")
t2d_bundle["transport_check"] = transport
t2d_bundle["calibration_warning"] = (
    f"Applied to Dexcom days 1-7 in CGMacros, the CGM-only model gave median risks of {transport['median_risk_lab_normal']:.2f} "
    f"(lab HbA1c < 5.7 %) and {transport['median_risk_lab_prediabetes']:.2f} (5.7-6.4 %), against an observed incidence of "
    f"{transport['colas_observed_incidence']:.2f} in the Colas cohort; Dexcom readings run about {offset:.0f} mg/dL higher "
    "than the Colas sensor relative to lab HbA1c, so absolute risks on Dexcom data are likely overestimated. "
    "Use the value to rank, not as a calibrated probability.")
SUMMARY["C_incident_t2d"]["transport_check_cgmacros"] = transport
BUNDLE = dict(version=1, created=SUMMARY["meta"]["created"], a1c=a1c_bundle, t2d=t2d_bundle,
              literature=dict(beck=R.BECK_2019, gri=R.GRI_CITATION))
with open(os.path.join(OUT, "models.pkl"), "wb") as f:
    pickle.dump(BUNDLE, f)
SUMMARY["F_shipped"] = dict(
    hba1c=dict(model=a1c_bundle["name"], alpha=a1c_bundle["alpha"], residual_scope=a1c_bundle["residual_scope"],
               interval95_offsets=[float(np.quantile(a1c_bundle["residuals"], .025)), float(np.quantile(a1c_bundle["residuals"], .975))],
               validation=VAL_A1C),
    t2d=dict(ship=t2d_bundle["ship"], order=t2d_bundle["order"], rule=t2d_bundle["ship_rule"],
             per_model={n: dict(ship=t2d_models[n]["ship"], auc=t2d_models[n]["validation"]["auc"], C=t2d_models[n]["model_C"])
                        for n in T2D_ORDER}))
log("models saved")

# =============================================================== G: CGMacros day 1-7 profiles (leave-one-out HbA1c model)
log("G: CGMacros profiles")
profiles = {}
cg = [r for r in RECS if r["cohort"] == "CGMacros"]
for r in cg:
    keep = ~((D7o.cohort == "CGMacros") & (D7o.pid == r["pid"])).to_numpy()
    b = dict(BUNDLE)
    a = dict(a1c_bundle)
    a["model"] = ridge().fit(D7o.loc[keep, FEAT], D7o.hba1c[keep])
    a["residuals"] = res_ship[:, keep & dex_mask].ravel()
    a["name"] = a1c_bundle["name"] + " (refit without this participant)"
    b["a1c"] = a
    known_dm = bool(r["hba1c"] >= 6.5)
    prof = R.metabolic_profile(r["glucose"], age=r["age"], bmi=r["bmi"], sex=r["sex"], has_diabetes=known_dm, days=7, models=b)
    prof["analysis_only"] = dict(lab_hba1c=r["hba1c"], lab_fasting_glucose=r["fasting_glucose"],
                                 diabetes_flag_source="lab HbA1c >= 6.5 % in bio.csv used as the 'known diagnosis' input",
                                 hba1c_model="leave-one-out refit; this participant's lab HbA1c was not used")
    profiles[r["pid"]] = prof
with open(os.path.join(OUT, "cgmacros_profiles.json"), "w") as f:
    json.dump(js(profiles), f, indent=1)
pe = pd.DataFrame([dict(pid=k, lab=v["analysis_only"]["lab_hba1c"], est=v["hba1c_estimate"]["model"], gmi=v["hba1c_estimate"]["gmi"],
                        lo=v["hba1c_estimate"]["interval95"][0], hi=v["hba1c_estimate"]["interval95"][1],
                        t2d=v["t2d_risk"].get("probability"), zone=v["cgm_metrics"]["gri_zone"]) for k, v in profiles.items()])
pe.to_csv(os.path.join(OUT, "cgmacros_profiles_summary.csv"), index=False)
SUMMARY["G_cgmacros_profiles"] = dict(
    n=len(pe), window="days 1-7", mae_model=float(np.mean(np.abs(pe.est - pe.lab))), mae_gmi=float(np.mean(np.abs(pe.gmi - pe.lab))),
    interval_coverage=float(np.mean((pe.lab >= pe.lo) & (pe.lab <= pe.hi))),
    n_with_t2d_risk=int(pe.t2d.notna().sum()), gri_zones=pe.zone.value_counts().to_dict(),
    t2d_risk_descriptive=dict(
        note="no outcomes in CGMacros: descriptive only (CGM-only Colas model applied to Dexcom days 1-7)",
        median_risk_lab_normal=float(pe.t2d[pe.lab < 5.7].median()), n_lab_normal=int((pe.lab < 5.7).sum()),
        median_risk_lab_prediabetes=float(pe.t2d[(pe.lab >= 5.7) & (pe.lab < 6.5)].median()),
        n_lab_prediabetes=int(((pe.lab >= 5.7) & (pe.lab < 6.5)).sum()),
        colas_observed_incidence=float(DC.t2dm.mean())),
    note="HbA1c model refit without the participant (leave-one-out) and residual pool without the participant's own residuals; "
         "T2D risk only for participants with lab HbA1c < 6.5 % (stand-in for a known diagnosis), CGM-only inputs")
log("profiles done")

def kn(d):
    return f"{d['value']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


B7p, B7o = B_RES["cv_pooled_training_7d"], B_RES["cv_outpatient_training_7d"]
SUMMARY["key_numbers"] = dict(
    B_mae_GMI_by_cohort={c: dict(n=B7p["GMI formula"][c]["n"], mae=kn(B7p["GMI formula"][c]["mae"]), bias=kn(B7p["GMI formula"][c]["bias"]))
                         for c in COHORTS},
    B_pooled_training_ridge_cv={c: dict(n=B7p["Ridge (CGM metrics)"][c]["n"], mae=kn(B7p["Ridge (CGM metrics)"][c]["mae"]))
                                for c in ["pooled"] + COHORTS},
    B_outpatient_ridge_cv={c: dict(n=B7o[OUTPT_NAME][c]["n"], mae=kn(B7o[OUTPT_NAME][c]["mae"]), r=kn(B7o[OUTPT_NAME][c]["r"]),
                                   acc3=kn(B7o[OUTPT_NAME][c]["acc3"])) for c in ["pooled", "CGMacros", "BIG IDEAs", "Colas"]},
    B_majority_class_accuracy_baseline={c: round(max(v["class_counts"].values()) / v["n"], 3)
                                        for c, v in B7o[OUTPT_NAME].items()},
    B_outpatient_gmi_same_persons={c: dict(mae=kn(B7o["GMI formula"][c]["mae"]), acc3=kn(B7o["GMI formula"][c]["acc3"]))
                                   for c in ["pooled", "CGMacros", "BIG IDEAs", "Colas"]},
    C_auc={k: dict(n=v["n"], events=v["events"], auc=kn(v["auc"]), c_index=kn(v["c_index"]) if "c_index" in v else None,
                   calibration_slope=round(v["calibration"]["slope"], 3)) for k, v in c_res.items()},
    D_auc={yc: {k: dict(n=v["n"], events=v["events"], auc=kn(v["auc"])) for k, v in D_RES[yc]["models"].items()} for yc in D_RES},
)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(js(SUMMARY), f, indent=1)
log("summary written")


# =============================================================== figures
log("figures")
COH_COLOR = {c: i for i, c in enumerate(COHORTS)}


def f_est_vs_lab(T):
    O = OOF["7d"]
    D, Do = O["D"], O["Do"]
    panels = [("GMI formula (no fitting)", D, O["cv"]["GMI formula"], (4, 18)),
              ("Ridge, pooled training (CV)", D, O["cv"]["Ridge (CGM metrics)"], (4, 18)),
              ("Ridge, outpatient training (CV)", Do, O["cv_o"][OUTPT_NAME], (4.3, 9))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw=dict(wspace=0.22))
    for ax, (title, DD_, P, lim) in zip(axes, panels):
        est = np.nanmean(np.atleast_2d(P), 0)
        ax.plot(lim, lim, color=T["ref"], lw=1.2, zorder=1)
        for c in [5.7, 6.5]:
            ax.axvline(c, color=T["grid"], lw=1, zorder=0); ax.axhline(c, color=T["grid"], lw=1, zorder=0)
        for c in COHORTS:
            mk = (DD_.cohort == c).to_numpy()
            if mk.any():
                ax.scatter(DD_.hba1c[mk], est[mk], s=26, color=T["s"][COH_COLOR[c]], edgecolor=T["panel"], linewidth=0.8,
                           label=f"{c} (n={mk.sum()})", zorder=3, alpha=0.9)
        mae = np.mean(np.abs(est - DD_.hba1c.to_numpy()))
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_title(title); ax.set_xlabel("lab HbA1c (%)")
        ax.text(0.03, 0.97, f"MAE {mae:.2f} %-points\n(repeat-averaged OOF)", transform=ax.transAxes, va="top", fontsize=9, color=T["ink2"])
    axes[0].set_ylabel("CGM-based estimate (%), days 1-7")
    axes[1].legend(loc="lower right", fontsize=8.5)
    axes[2].legend(loc="lower right", fontsize=8.5)
    return fig


save_fig(f_est_vs_lab, "b_estimated_vs_lab_hba1c", FIG)


def f_mae_cohort(T):
    sets = [("GMI formula", B_RES["cv_pooled_training_7d"]["GMI formula"]),
            ("Ridge, pooled training (CV)", B_RES["cv_pooled_training_7d"]["Ridge (CGM metrics)"]),
            ("Ridge, pooled training (leave-cohort-out)", B_RES["loco_pooled_training_7d"]["Ridge (CGM metrics)"]),
            ("Ridge, outpatient training (CV)", B_RES["cv_outpatient_training_7d"][OUTPT_NAME]),
            ("Ridge, outpatient training (leave-cohort-out)", B_RES["loco_outpatient_training_7d"][OUTPT_NAME])]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    off = np.linspace(-0.3, 0.3, len(sets))
    for j, (lab, res) in enumerate(sets):
        for i, c in enumerate(COHORTS):
            if c not in res:
                continue
            m = res[c]["mae"]
            ax.plot([m["lo"], m["hi"]], [i + off[j]] * 2, color=T["s"][j], lw=2, solid_capstyle="round")
            ax.plot(m["value"], i + off[j], "o", ms=6.5, color=T["s"][j], mec=T["panel"], mew=1, label=lab if i == 0 else None)
    ax.set_yticks(range(len(COHORTS)), [f"{c}\n(n={B_RES['cv_pooled_training_7d']['GMI formula'][c]['n']})" for c in COHORTS])
    ax.invert_yaxis(); ax.set_xlim(0, None); ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean absolute error vs lab HbA1c (%-points), 95% bootstrap CI")
    ax.set_title("HbA1c estimation error by cohort (days 1-7)")
    ax.legend(loc="lower right", fontsize=8.5)
    return fig


save_fig(f_mae_cohort, "b_mae_by_cohort", FIG)


def f_days(T):
    d = pd.DataFrame(DAY_ROWS)
    order = [1, 2, 3, 5, 7, "all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(wspace=0.18))
    for ax, grp in zip(axes, ["CGMacros+BIG IDEAs", "Shanghai T1DM+T2DM"]):
        g = d[d.group == grp]
        for j, (name, gg) in enumerate(g.groupby("model", sort=False)):
            gg = gg.set_index(gg.days.astype(str)).reindex([str(o) for o in order])
            x = np.arange(len(order))
            ax.fill_between(x, gg.mae_lo, gg.mae_hi, color=T["s"][j], alpha=T["band_alpha"], lw=0)
            ax.plot(x, gg.mae, "o-", color=T["s"][j], ms=5, label=name)
        ax.set_xticks(np.arange(len(order)), ["1", "2", "3", "5", "7", "all"])
        ax.set_xlabel("days of CGM used (first k days)"); ax.set_ylim(0, None)
        ax.set_title(f"{grp} (n={int(g.n.iloc[0])})")
        ax.legend(loc="lower left", fontsize=8.5)
    axes[0].set_ylabel("MAE vs lab HbA1c (%-points), 95% CI")
    return fig


save_fig(f_days, "b_mae_vs_days_of_cgm", FIG)


def f_roc(T, curves, titles, res_by):
    fig, axes = plt.subplots(1, len(curves), figsize=(5.3 * len(curves), 5.0), squeeze=False)
    for ax, cur, title, res in zip(axes[0], curves, titles, res_by):
        ax.plot([0, 1], [0, 1], color=T["ref"], lw=1)
        for j, (name, c) in enumerate([(k, v) for k, v in cur.items() if "exploratory" not in k]):
            a = res[name]["auc"]
            ax.fill_between(c["grid"], c["lo"], c["hi"], color=T["s"][j], alpha=T["band_alpha"], lw=0)
            ax.plot(c["grid"], c["mean"], color=T["s"][j], lw=2,
                    label=f"{name.split(' (')[0]}: AUC {a['value']:.2f} [{a['lo']:.2f}, {a['hi']:.2f}]")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.set_title(title, fontsize=10.5)
        ax.legend(loc="lower right", fontsize=8.3)
    return fig


save_fig(lambda T: f_roc(T, [c_curves], [f"Incident T2D, Colas (n={len(DC)}, {int(DC.t2dm.sum())} events)\n"
                                         f"{N_REP_C}x5-fold CV, mean ROC, 95% bootstrap band"], [c_res]), "c_roc_t2d", FIG)
save_fig(lambda T: f_roc(T, [d_curves["micro_any"], d_curves["macro_any"]],
                         [f"Existing microvascular complication\nShanghai T2DM (n={len(DD)}, {int(DD.micro_any.sum())} with)",
                          f"Existing macrovascular complication\nShanghai T2DM (n={len(DD)}, {int(DD.macro_any.sum())} with)"],
                         [D_RES["micro_any"]["models"], D_RES["macro_any"]["models"]]), "d_roc_complications", FIG)


def f_cal(T):
    cr = pd.DataFrame(cal_rows)
    names = [n for n in C_MODELS if "exploratory" not in n]
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.plot([0, 0.45], [0, 0.45], color=T["ref"], lw=1)
    for j, n in enumerate(names):
        g = cr[cr.model == n]
        dx = (j - 1) * 0.004
        ax.vlines(g.mean_predicted + dx, g.observed_lo, g.observed_hi, color=T["s"][j], lw=2)
        cal = c_res[n]["calibration"]
        ax.plot(g.mean_predicted + dx, g.observed, "o", ms=7, color=T["s"][j], mec=T["panel"], mew=1,
                label=f"{n.split(' (')[0]}: slope {cal['slope']:.2f}, Brier {cal['brier']:.3f}")
    ax.set_xlim(0, 0.45); ax.set_ylim(0, 0.45); ax.set_aspect("equal")
    ax.set_xlabel("mean predicted risk (OOF, quartiles of predicted risk)")
    ax.set_ylabel("observed incident T2D (Wilson 95% CI)")
    ax.set_title("Calibration, Colas (52 persons per quartile)", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.3)
    return fig


save_fig(f_cal, "c_calibration_t2d", FIG)


def f_uni(T, tabs, titles):
    fig, axes = plt.subplots(1, len(tabs), figsize=(5.6 * len(tabs), 7.2), squeeze=False, gridspec_kw=dict(wspace=0.45))
    for ax, tab, title in zip(axes[0], tabs, titles):
        t = tab.sort_values("auc").reset_index(drop=True)
        clin = t.variable.isin(["age", "bmi", "fasting_glucose", "hba1c", "duration"])
        y = np.arange(len(t))
        ax.axvline(0.5, color=T["ref"], lw=1)
        for i, r in t.iterrows():
            col = T["s"][1] if clin[i] else T["s"][0]
            ax.plot([r.auc_lo, r.auc_hi], [i, i], color=col, lw=2, solid_capstyle="round")
            ax.plot(r.auc, i, "o", ms=6, color=col, mec=T["panel"], mew=1)
        ax.set_yticks(y, t.variable)
        ax.set_xlim(0.1, 0.95); ax.grid(axis="y", visible=False)
        ax.set_xlabel("AUC of the raw variable (95% CI)\n<0.5 = lower values go with the outcome")
        ax.set_title(title, fontsize=10.5)
    axes[0][0].plot([], [], "o", color=T["s"][0], label="CGM metric"); axes[0][0].plot([], [], "o", color=T["s"][1], label="clinical")
    axes[0][0].legend(loc="lower right", fontsize=8.5)
    return fig


save_fig(lambda T: f_uni(T, [c_uni], ["Univariate AUC, incident T2D (Colas, 17 events)"]), "c_univariate_auc", FIG)
du = pd.concat(d_uni_all)
save_fig(lambda T: f_uni(T, [du[du.outcome.str.contains("micro")], du[du.outcome.str.contains("macro")]],
                         ["Microvascular (Shanghai T2DM)", "Macrovascular (Shanghai T2DM)"]), "d_univariate_auc", FIG)


def f_gri(T):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    xs = np.linspace(0, 62.5, 200)
    for z, v in [(20, "A|B"), (40, "B|C"), (60, "C|D"), (80, "D|E")]:
        ax.plot(xs, (z - 1.6 * xs) / 3.0, color=T["ref"], lw=1, ls="--")
    for z, (x, y) in {"A": (3, 1.5), "B": (17, 3), "C": (29, 4), "D": (41, 5), "E": (54, 6)}.items():
        ax.text(x, y, z, fontsize=13, color=T["ink2"], fontweight="bold")
    for c in COHORTS:
        d = F7[F7.cohort == c]
        ax.scatter(d.gri_hyper, d.gri_hypo, s=22, color=T["s"][COH_COLOR[c]], edgecolor=T["panel"], linewidth=0.7,
                   label=f"{c} (n={len(d)})", alpha=0.85, zorder=3)
    ax.set_xlim(0, max(62.5, float(F7.gri_hyper.max()) * 1.05)); ax.set_ylim(0, max(33.3, float(F7.gri_hypo.max()) * 1.05))
    ax.set_xlabel("hyperglycemia component (%>250 + 0.5 x %181-250)")
    ax.set_ylabel("hypoglycemia component (%<54 + 0.8 x %54-69)")
    ax.set_title("GRI grid, days 1-7 (zones per Klonoff et al. 2022; literature-derived)", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5)
    return fig


save_fig(f_gri, "e_gri_grid", FIG)


def f_tir(T):
    tir = np.linspace(20, 100, 161)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for j, (k, lab) in enumerate([("retinopathy_progression", "retinopathy progression (x1.64 per 10 points)"),
                                  ("microalbuminuria_development", "microalbuminuria development (x1.40 per 10 points)")]):
        ax.plot(tir, [R.tir_relative_hazard(t)[k] for t in tir], color=T["s"][j], lw=2, label=lab)
    ax.set_yscale("log"); ax.axhline(1, color=T["ref"], lw=1); ax.axvline(R.TIR_REFERENCE, color=T["ref"], lw=1, ls="--")
    ticks = [0.25, 0.5, 1, 2, 4, 8]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks]); ax.minorticks_off()
    ax.set_xlabel("time in range 70-180 mg/dL (%)"); ax.set_ylabel("relative hazard vs TIR 70% (log scale)")
    ax.set_title("Literature-derived: Beck et al. 2019 (DCCT, type 1 diabetes); population-level", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5)
    return fig


save_fig(f_tir, "e_tir_relative_hazard_literature", FIG)
log("figures done")
