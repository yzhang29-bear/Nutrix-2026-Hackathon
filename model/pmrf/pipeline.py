"""End-to-end Day-8 protocol: history = days 1-7 of every person (with nutrition labels), test = meals on day 8+
predicted from the photo, recent CGM/HR and the person's history only."""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats

from .features import (meal_table, person_profile, CONTEXT_FEATURES, MED_FEATURES, PROFILE_FEATURES, CURVE_T, TARGETS)
from .vision import MacroEstimator, MACROS, _knn
from .models.gbm import GBMResponse, ALL_TARGETS

N_PCA = 8
MEALTYPE_FEATURES = ["is_breakfast", "is_lunch", "is_dinner", "is_snack"]
PHOTO_FEATURES = ["est_carbs", "est_protein", "est_fat", "est_fiber", "est_kcal", "est_carbs_sd", "photo_sim_pop"] + [f"img{i}" for i in range(N_PCA)]
POP_RETRIEVAL = ["pop_nn_dpeak", "pop_nn_iauc"]
PERS_RETRIEVAL = ["own_nn_dpeak", "own_nn_iauc", "own_nn_d60", "own_nn_sim"]
LOGGED_MACROS = ["carbs", "protein", "fat", "fiber", "kcal"]


def attach_embeddings(M, npz_path, root):
    z = np.load(npz_path, allow_pickle=True)
    key = {p: i for i, p in enumerate(z["path"])}
    rel = M.photo.map(lambda p: os.path.relpath(p, root) if isinstance(p, str) else None)
    M = M.copy()
    M["emb_idx"] = rel.map(lambda r: key.get(r, -1) if r else -1).astype(int)
    return M, z["emb"]


def build(persons, root, emb_npz, hist_days=7):
    M, CURV, PRE, HRS = meal_table(persons, hist_days)
    M, EMB = attach_embeddings(M, emb_npz, root)
    M["in_hist"] = M.day <= hist_days
    for mt in ["breakfast", "lunch", "dinner", "snack"]:
        M[f"is_{mt}"] = (M.meal_type == mt).astype(float)
    return M, CURV, PRE, HRS, EMB


def photo_features(M, EMB, macro_est=None):
    """Macro estimates + image PCA + retrieval of responses to similar-looking meals.
    History rows get out-of-fold estimates; test rows get estimates from a model fitted on history photos only."""
    M = M.copy()
    has = M.emb_idx.to_numpy() >= 0
    E = np.zeros((len(M), EMB.shape[1]), np.float32); E[has] = EMB[M.emb_idx.to_numpy()[has]]
    H = M.in_hist.to_numpy() & has
    if macro_est is None:
        macro_est = MacroEstimator().fit(E[H], M.loc[H, MACROS].to_numpy(float), M.pid.to_numpy()[H])
    est = np.full((len(M), 4), np.nan); unc = np.full(len(M), np.nan); simp = np.full(len(M), np.nan)
    est[H] = macro_est.oof_pred; unc[H] = macro_est.oof_unc[:, 0]
    T = (~M.in_hist.to_numpy()) & has
    if T.any():
        e_, s_, smp_, _ = macro_est.predict(E[T], M.pid.to_numpy()[T])
        est[T], unc[T], simp[T] = e_, s_[:, 0], smp_
    # population similarity for history rows (to other people's photos)
    if H.any():
        _, _, smp_h, _ = _knn(E[H], macro_est.LE, macro_est.LY, 10, M.pid.to_numpy()[H][:, None] == macro_est.Lpid[None, :])
        simp[H] = smp_h
    for j, c in enumerate(MACROS):
        M[f"est_{c}"] = est[:, j]
    M["est_kcal"] = 4 * M.est_carbs + 4 * M.est_protein + 9 * M.est_fat
    M["est_carbs_sd"], M["photo_sim_pop"] = unc, simp
    pca = PCA(N_PCA, random_state=0).fit(E[H])
    P = pca.transform(E); P[~has] = np.nan
    for i in range(N_PCA):
        M[f"img{i}"] = P[:, i]
    # retrieval of observed responses: other people's history meals (population) and the person's own history (personal)
    lib = H & M.label.to_numpy()
    LE, LY, Lpid = E[lib], M.loc[lib, ["dpeak", "iauc", "d60"]].to_numpy(float), M.pid.to_numpy()[lib]
    Lidx = np.flatnonzero(lib)
    q = has
    pid = M.pid.to_numpy()[q]
    mp, _, _, _ = _knn(E[q], LE, LY, 10, pid[:, None] == Lpid[None, :])
    ex_own = (pid[:, None] != Lpid[None, :]) | (np.flatnonzero(q)[:, None] == Lidx[None, :])
    mo, _, so, _ = _knn(E[q], LE, LY, 3, ex_own)
    none = ex_own.all(1)
    mo[none] = np.nan; so = np.where(none, np.nan, so)
    for c, v in [("pop_nn_dpeak", mp[:, 0]), ("pop_nn_iauc", mp[:, 1]), ("own_nn_dpeak", mo[:, 0]), ("own_nn_iauc", mo[:, 1]),
                 ("own_nn_d60", mo[:, 2]), ("own_nn_sim", so)]:
        M[c] = np.nan; M.loc[q, c] = v
    return M, macro_est, pca, E


def to_challenge_outputs(df, P):
    """delta-scale predictions -> the 8 challenge outputs (absolute glucose where the challenge asks for it)"""
    g0 = df.G0.to_numpy()
    out = {f"G{k}": g0 + P[f"d{k}"]["pred"] for k in (30, 60, 90, 120)}
    out.update(Gmax=g0 + P["dpeak"]["pred"], dpeak=P["dpeak"]["pred"], ttp=np.clip(P["ttp"]["pred"], 0, 120), iauc=np.clip(P["iauc"]["pred"], 0, None))
    return out


def metrics(df, out, subsets):
    rows = []
    for name, mask in subsets.items():
        for t in TARGETS:
            o, p = df[t].to_numpy()[mask], out[t][mask]
            ok = np.isfinite(o) & np.isfinite(p)
            rows.append(dict(subset=name, target=t, n=int(ok.sum()), mae=float(np.mean(np.abs(o[ok] - p[ok]))),
                             r=float(stats.pearsonr(o[ok], p[ok])[0]) if t != "ttp" else float(stats.spearmanr(o[ok], p[ok]).statistic)))
    return pd.DataFrame(rows)


def feature_sets():
    base_ctx = CONTEXT_FEATURES + MED_FEATURES + MEALTYPE_FEATURES
    return dict(
        population=base_ctx + PHOTO_FEATURES + POP_RETRIEVAL,
        personalized=base_ctx + PHOTO_FEATURES + POP_RETRIEVAL + PROFILE_FEATURES + PERS_RETRIEVAL,
        oracle_macros=base_ctx + LOGGED_MACROS + [f"img{i}" for i in range(N_PCA)] + POP_RETRIEVAL + PROFILE_FEATURES + PERS_RETRIEVAL,
    )


FEATURE_GROUPS = {
    "Meal (from photo)": ["est_carbs", "est_protein", "est_fat", "est_fiber", "est_kcal", "est_carbs_sd", "photo_sim_pop"] + [f"img{i}" for i in range(N_PCA)] + ["carbs", "protein", "fat", "fiber", "kcal"],
    "Meal type & time": MEALTYPE_FEATURES + ["hour", "tod_sin", "tod_cos"],
    "Glucose before meal": ["G0", "g0_rel24", "gmean3h"],
    "Glucose trend before meal": ["slope30", "slope60", "sd60"],
    "Previous meals": ["prev_gap", "prev_carbs", "n_meals_6h"],
    "Heart rate": ["hr_pre", "hr_max60"],
    "Medication": MED_FEATURES,
    "Personal profile (days 1-7)": PROFILE_FEATURES,
    "Similar meals: your history": PERS_RETRIEVAL,
    "Similar meals: other people": POP_RETRIEVAL,
    "Activity before meal": ["act_mets30", "act_mets120", "act_active120", "act_kcal120"],
    "Age, sex, BMI": ["age", "sex_female", "bmi"],
    "Logged meal calories": ["meal_kcal_logged"],
}


def enforce_consistency(out, g0, curve):
    """Outcomes are predicted and blended separately, so the peak can fall below a predicted time point. Make them agree:
    the peak is the highest of the blended peak, the 30/60/90/120-min predictions and the curve maximum; the peak rise and the
    time to peak follow it."""
    g0 = np.asarray(g0, float)
    cand_v = np.column_stack([out["Gmax"], out["G30"], out["G60"], out["G90"], out["G120"], g0 + curve.max(1)])
    cand_t = np.column_stack([out["ttp"], np.full(len(g0), 30.0), np.full(len(g0), 60.0), np.full(len(g0), 90.0), np.full(len(g0), 120.0),
                              curve.argmax(1) * 5.0])
    k = np.argmax(cand_v, 1)
    res = dict(out)
    res["Gmax"] = cand_v[np.arange(len(g0)), k]
    res["dpeak"] = res["Gmax"] - g0
    res["ttp"] = np.where(k == 0, out["ttp"], cand_t[np.arange(len(g0)), k])
    return res
