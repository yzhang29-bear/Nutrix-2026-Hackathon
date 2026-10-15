"""Challenge #3 on CGMacros: learn each person from days 1-7, predict post-meal glucose for meals on day 8+.
usage: python -I eda_challenge.py <CGMacros dir> <eda dir>
Reads <eda dir>/cache/photo_emb.npz (scripts/photo_embed.py). Writes challenge_stats.json, challenge_meals.csv, fig/cx_*.png
"""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_common import save_fig, GROUPS
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT, E = sys.argv[1], sys.argv[2]
FIG = os.path.join(E, "fig")
S = {}
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ load
bio = pd.read_csv(os.path.join(ROOT, "bio.csv")); bio.columns = [c.strip() for c in bio.columns]
bio["sid"] = bio.subject.map(lambda x: f"{int(x):03d}")
grp = bio.set_index("sid")["A1c PDL (Lab)"].map(lambda a: GROUPS[0] if a < 5.65 else (GROUPS[1] if a <= 6.45 else GROUPS[2]))

series, meals = {}, []
for f in sorted(glob.glob(os.path.join(ROOT, "CGMacros-*", "CGMacros-*.csv"))):
    sid = os.path.basename(f)[9:12]
    d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]
    d["Timestamp"] = pd.to_datetime(d["Timestamp"])
    d = d.drop_duplicates("Timestamp").sort_values("Timestamp")
    idx = pd.date_range(d.Timestamp.iloc[0].floor("min"), d.Timestamp.iloc[-1].ceil("min"), freq="min")
    g = d.set_index("Timestamp")["Dexcom GL"].reindex(idx)      # piecewise-linear between native 5-min readings
    hr = d.set_index("Timestamp")["HR"].reindex(idx)
    t_first = d.Timestamp.iloc[0]
    day_all = ((idx - t_first).total_seconds() // 86400).astype(int) + 1
    series[sid] = dict(g=g, hr=hr, day=pd.Series(day_all, index=idx))
    m = d[d["Meal Type"].notna()].copy()
    m["sid"] = sid
    m["meal"] = m["Meal Type"].map(lambda x: "snack" if str(x).strip().lower().startswith("snack") else str(x).strip().lower())
    m["day"] = ((m.Timestamp - t_first).dt.total_seconds() // 86400).astype(int) + 1
    meals.append(m[["sid", "Timestamp", "day", "meal", "Calories", "Carbs", "Protein", "Fat", "Fiber", "Image path"]])
M = pd.concat(meals, ignore_index=True).sort_values(["sid", "Timestamp"]).reset_index(drop=True)
M["group"] = M.sid.map(grp)
M["fiber_c"] = M.Fiber.where((M.Fiber <= 50) & (M.Fiber <= M.Carbs.clip(lower=1)))
M["kcal_atw"] = 4 * M.Carbs + 4 * M.Protein + 9 * M.Fat
M["prev_gap"] = M.groupby("sid").Timestamp.diff().dt.total_seconds() / 60
M["next_gap"] = -M.groupby("sid").Timestamp.diff(-1).dt.total_seconds() / 60
M["prev_carbs"] = M.groupby("sid").Carbs.shift(1).where(M.prev_gap < 360, 0).fillna(0)
M["recipe"] = np.where(M.meal.isin(["breakfast", "lunch"]),
                       M.meal + ":" + M.Carbs.astype(str) + "/" + M.Protein.astype(str) + "/" + M.Fat.astype(str) + "/" + M.Fiber.astype(str),
                       "free:" + M.index.astype(str))

# ------------------------------------------------------------------ targets (Dexcom, challenge definitions) + pre-meal context
OFF = np.arange(-60, 181)
TARGETS = ["G30", "G60", "G90", "G120", "Gmax", "dpeak", "ttp", "iauc"]
rows, curves = [], []
hr_rest = {}
for sid, s in series.items():
    h = s["hr"][s["day"] <= 7].dropna()
    hr_rest[sid] = np.percentile(h, 10) if len(h) > 500 else np.nan
for i, r in M.iterrows():
    s = series[r.sid]
    t = r.Timestamp.floor("min") + pd.to_timedelta(OFF, unit="min")
    g = s["g"].reindex(t).to_numpy(); hr = s["hr"].reindex(t).to_numpy()
    o = lambda k: k + 60
    win = g[o(-30):o(120) + 1]
    out = dict(valid=bool(np.isfinite(g[o(0)]) and np.mean(np.isfinite(win)) >= 0.9))
    if out["valid"]:
        gi = pd.Series(g).interpolate(limit_direction="both").to_numpy()   # edges held at nearest reading
        g0 = gi[o(0)]; post = gi[o(0):o(120) + 1]
        k = int(np.nanargmax(post))
        out.update(G0=g0, G30=gi[o(30)], G60=gi[o(60)], G90=gi[o(90)], G120=gi[o(120)], Gmax=post[k], dpeak=post[k] - g0, ttp=float(k),
                   iauc=float(np.trapz(np.clip(post - g0, 0, None), dx=1)),
                   slope30=(g0 - gi[o(-30)]) / 30 if np.isfinite(gi[o(-30)]) else np.nan,
                   sd60=np.nanstd(gi[o(-60):o(0) + 1]))
        prior = s["g"][(s["g"].index < r.Timestamp) & (s["g"].index >= r.Timestamp - pd.Timedelta("24h"))]
        out["g0_rel"] = g0 - prior.median() if prior.notna().sum() > 600 else np.nan
        curves.append(gi[o(0):o(120) + 1:5] - g0)
    else:
        curves.append(np.full(25, np.nan))
    rest = hr_rest[r.sid]
    out.update(hr_pre=np.nanmean(hr[o(-30):o(0)]) - rest if np.isfinite(hr[o(-30):o(0)]).any() else np.nan,
               hr_max60=np.nanmax(hr[o(-60):o(0)]) - rest if np.isfinite(hr[o(-60):o(0)]).any() else np.nan,
               hr_post=np.nanmean(hr[o(0):o(120)]) - rest if np.isfinite(hr[o(0):o(120)]).any() else np.nan)
    rows.append(out)
M = pd.concat([M, pd.DataFrame(rows)], axis=1)
CURV = np.array(curves)
M["hour"] = M.Timestamp.dt.hour + M.Timestamp.dt.minute / 60
M["label"] = M.valid & (M.next_gap.isna() | (M.next_gap >= 120))
M["split"] = np.where(M.day <= 7, "train", "test")
S["n_meals"], S["n_valid"], S["n_label"] = len(M), int(M.valid.sum()), int(M.label.sum())
S["n_label_by_split"] = M[M.label].split.value_counts().to_dict()
S["peak_censored_share"] = float((M.loc[M.label, "ttp"] >= 120).mean())
S["target_summary"] = M.loc[M.label, TARGETS + ["G0"]].describe().round(1).to_dict()
S["target_corr"] = M.loc[M.label, ["dpeak", "iauc", "G60", "G120", "ttp"]].corr(method="spearman").round(2).to_dict()

# seen-before: this person ate the same standardized recipe during days 1-7
seen = M[M.split == "train"].groupby("sid").recipe.apply(set)
M["seen"] = [(r.split == "test") and (r.recipe in seen.get(r.sid, set())) and not r.recipe.startswith("free") for r in M.itertuples()]
L = M[M.label].copy()
S["test_seen_share"] = float(L[L.split == "test"].seen.mean())
S["test_by_meal"] = L[L.split == "test"].meal.value_counts().to_dict()

# ================================================================== 1. variance decomposition on designed meals
def anova2_rep(df, y, a="sid", b="recipe"):
    """balanced two-way random(person) x fixed(recipe) ANOVA with replicates -> variance shares"""
    A = df[a].unique(); B = df[b].unique(); n = df.groupby([a, b]).size().iloc[0]
    gm = df[y].mean()
    ma = df.groupby(a)[y].mean(); mb = df.groupby(b)[y].mean(); mab = df.groupby([a, b])[y].mean()
    nA, nB = len(A), len(B)
    ssa = nB * n * ((ma - gm) ** 2).sum(); ssb = nA * n * ((mb - gm) ** 2).sum()
    ssab = n * ((mab - mab.index.get_level_values(0).map(ma).to_numpy() - mab.index.get_level_values(1).map(mb).to_numpy() + gm) ** 2).sum()
    sse = ((df[y] - df.set_index([a, b]).index.map(mab).to_numpy()) ** 2).sum()
    msa, msb, msab, mse = ssa / (nA - 1), ssb / (nB - 1), ssab / ((nA - 1) * (nB - 1)), sse / (nA * nB * (n - 1))
    comp = dict(recipe=max((msb - msab) / (nA * n), 0) * (nB - 1) / nB, person=max((msa - msab) / (nB * n), 0),
                person_x_recipe=max((msab - mse) / n, 0), day_to_day=mse)
    tot = sum(comp.values())
    return {k: v / tot for k, v in comp.items()}, dict(persons=nA, recipes=nB, reps=int(n))

def anova2_norep(df, y, a="sid", b="recipe"):
    """additive two-way fit by alternating means (unbalanced, one serving per cell) -> shares"""
    x = df[[a, b, y]].copy(); pa = pd.Series(0.0, index=x[a].unique()); pb = pd.Series(0.0, index=x[b].unique()); gm = x[y].mean()
    for _ in range(50):
        pa = (x[y] - gm - x[b].map(pb)).groupby(x[a]).mean()
        pb = (x[y] - gm - x[a].map(pa)).groupby(x[b]).mean()
    res = x[y] - gm - x[a].map(pa) - x[b].map(pb)
    nA = x[a].nunique(); nB = x[b].nunique(); dfe = len(x) - nA - nB + 1
    se2 = (res ** 2).sum() / dfe
    va = max(pa.var() - se2 / x.groupby(a).size().mean(), 0); vb = max(pb.var() - se2 / x.groupby(b).size().mean(), 0)
    tot = va + vb + se2
    return dict(recipe=vb / tot, person=va / tot, residual=se2 / tot), dict(persons=nA, recipes=nB, n=len(x))

L["log_iauc"] = np.log(L.iauc + 100)
core = ["breakfast:24.0/22.0/10.5/0.0", "breakfast:66.0/22.0/10.5/0.0", "breakfast:66.0/66.0/10.5/0.0", "breakfast:66.0/22.0/42.0/0.0"]
sh = L[L.recipe.isin(core)].sort_values("Timestamp").groupby(["sid", "recipe"]).head(2)
ok = sh.groupby("sid").recipe.apply(lambda r: (r.value_counts() == 2).sum() == 4)
sh = sh[sh.sid.isin(ok[ok].index)]
VD = {}
for y in ["dpeak", "log_iauc"]:
    VD[f"shakes_{y}"], VD[f"shakes_{y}_n"] = anova2_rep(sh, y)
lu = L[L.meal == "lunch"]; lu = lu[lu.recipe.map(lu.recipe.value_counts()) >= 20]
for y in ["dpeak", "log_iauc"]:
    VD[f"lunch_{y}"], VD[f"lunch_{y}_n"] = anova2_norep(lu, y)
# how much of the person component is explained by diabetes group
pm = sh.groupby("sid").dpeak.mean(); eta = stats.f_oneway(*[pm[pm.index.map(grp) == g] for g in GROUPS])
ss_b = sum(len(pm[pm.index.map(grp) == g]) * (pm[pm.index.map(grp) == g].mean() - pm.mean()) ** 2 for g in GROUPS)
VD["person_share_explained_by_group_dpeak"] = float(ss_b / ((pm - pm.mean()) ** 2).sum())
pm2 = sh.groupby("sid").log_iauc.mean()
ss_b2 = sum(len(pm2[pm2.index.map(grp) == g]) * (pm2[pm2.index.map(grp) == g].mean() - pm2.mean()) ** 2 for g in GROUPS)
VD["person_share_explained_by_group_log_iauc"] = float(ss_b2 / ((pm2 - pm2.mean()) ** 2).sum())
S["variance"] = VD

# ================================================================== 2. pre-meal context, within person and after what was eaten
CTX = [("G0", "pre-meal glucose"), ("g0_rel", "pre-meal glucose vs own 24 h median"), ("slope30", "glucose trend, 30 min before"),
       ("sd60", "glucose variability, 60 min before"), ("hour", "time of day"), ("prev_gap", "minutes since previous meal"),
       ("prev_carbs", "carbs in previous meal (≤ 6 h)"), ("hr_pre", "heart rate before meal (vs resting)"),
       ("hr_max60", "peak heart rate, 60 min before (vs resting)"), ("day", "day of sensor wear"),
       ("hr_post", "heart rate after meal (vs resting) *")]
def resid_within(df, y):
    X = pd.get_dummies(df[["sid", "meal"]].astype(str), drop_first=False).astype(float)
    X[["Carbs", "Protein", "Fat", "kcal_atw"]] = df[["Carbs", "Protein", "Fat", "kcal_atw"]].to_numpy()
    beta, *_ = np.linalg.lstsq(X.to_numpy(), df[y].to_numpy(), rcond=None)
    return df[y].to_numpy() - X.to_numpy() @ beta
CX = []
for y in ["dpeak", "iauc"]:
    df = L.copy(); df["prev_gap"] = df.prev_gap.clip(upper=720).fillna(720)
    df["res"] = resid_within(df, y)
    for c, lab in CTX:
        x = df[c] - df.groupby("sid")[c].transform("mean")
        m_ = x.notna() & df.res.notna()
        if m_.sum() < 30: continue
        rho = stats.spearmanr(x[m_], df.res[m_]).statistic
        xv, rv, sv = x[m_].to_numpy(), df.res[m_].to_numpy(), df.sid[m_].to_numpy()
        pos = {s_: np.flatnonzero(sv == s_) for s_ in np.unique(sv)}; keys = list(pos); bs = []
        for _ in range(400):
            ii = np.concatenate([pos[k_] for k_ in rng.choice(keys, len(keys))])
            bs.append(stats.spearmanr(xv[ii], rv[ii]).statistic)
        CX.append(dict(target=y, feature=c, label=lab, rho=rho, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), n=int(m_.sum())))
CX = pd.DataFrame(CX)
S["context"] = CX.round(3).to_dict("records")

# ================================================================== 3. simulated challenge: train days 1-7 (all people), test day 8+
MEAL_F = ["Carbs", "Protein", "Fat", "fiber_c", "kcal_atw", "is_breakfast", "is_lunch", "is_dinner", "is_snack"]
CTX_F = ["G0", "g0_rel", "slope30", "sd60", "hour", "prev_gap", "prev_carbs", "hr_pre", "hr_max60", "day"]
PERS_F = ["p_dpeak", "p_iauc", "p_dpeak_per_carb", "p_mean_g", "p_cv_g", "p_hr_rest", "p_n"]
for mt in ["breakfast", "lunch", "dinner", "snack"]:
    L[f"is_{mt}"] = (L.meal == mt).astype(float)
L["prev_gap"] = L.prev_gap.clip(upper=720).fillna(720)
DT = ["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"]       # modelled on the delta scale, G0 added back
for k in (30, 60, 90, 120):
    L[f"d{k}"] = L[f"G{k}"] - L.G0

def person_feats(df, hist):
    """person descriptors from that person's history meals only (leave-one-out when the row is itself in history)"""
    out = []
    cg = {sid: series[sid]["g"][series[sid]["day"] <= 7] for sid in series}
    for r in df.itertuples():
        h = hist[(hist.sid == r.sid) & (hist.index != r.Index)]
        c = cg[r.sid].dropna()
        dpc = (h.dpeak / h.Carbs.clip(lower=10)).median() if len(h) else np.nan
        out.append(dict(p_dpeak=h.dpeak.mean() if len(h) else np.nan, p_iauc=h.iauc.mean() if len(h) else np.nan, p_dpeak_per_carb=dpc,
                        p_mean_g=c.mean() if len(c) else np.nan, p_cv_g=c.std() / c.mean() if len(c) else np.nan,
                        p_hr_rest=hr_rest[r.sid], p_n=len(h)))
    return pd.DataFrame(out, index=df.index)

TR, TE = L[L.split == "train"].copy(), L[L.split == "test"].copy()
TR = TR.join(person_feats(TR, TR)); TE = TE.join(person_feats(TE, TR))
S["split_sizes"] = dict(train=len(TR), test=len(TE), test_unseen=int((~TE.seen).sum()), persons=int(TE.sid.nunique()))

def hgb():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20, l2_regularization=1.0, random_state=0)

def fit_predict(Xtr, ytr, Xte):
    return hgb().fit(Xtr, ytr).predict(Xte)

def oof(df, feats, y):
    p = np.zeros(len(df))
    for a, b in GroupKFold(5).split(df, groups=df.sid):
        p[b] = fit_predict(df.iloc[a][feats], df.iloc[a][y], df.iloc[b][feats])
    return p

def shrink_offset(train_df, res, k):
    t = pd.DataFrame(dict(sid=train_df.sid.to_numpy(), res=res))
    g_ = t.groupby("sid").res.agg(["sum", "count"])
    return g_["sum"] / (g_["count"] + k)

# choose the shrinkage k with an internal temporal split (days 1-5 -> 6-7)
inner_tr, inner_va = TR[TR.day <= 5], TR[TR.day >= 6]
best_k = {}
for y in DT:
    oo = oof(inner_tr, MEAL_F + CTX_F, y)
    pv = fit_predict(inner_tr[MEAL_F + CTX_F], inner_tr[y], inner_va[MEAL_F + CTX_F])
    errs = {}
    for k in [0, 1, 2, 4, 8, 16, 1e9]:
        off = shrink_offset(inner_tr, inner_tr[y].to_numpy() - oo, k)
        errs[k] = np.mean(np.abs(inner_va[y] - (pv + inner_va.sid.map(off).fillna(0).to_numpy())))
    best_k[y] = min(errs, key=errs.get)
S["shrinkage_k"] = {k: (v if v < 1e8 else "inf") for k, v in best_k.items()}

PRED = {}
for y in DT:
    pop_mean = TR.groupby("meal")[y].mean()
    PRED[("M0", y)] = TE.meal.map(pop_mean).to_numpy()
    pm_ = TR.groupby("sid")[y].agg(["sum", "count"]); off_p = (pm_["sum"] + 3 * TR[y].mean()) / (pm_["count"] + 3)
    PRED[("M1", y)] = TE.sid.map(off_p).fillna(TR[y].mean()).to_numpy()
    PRED[("M2", y)] = fit_predict(TR[MEAL_F + CTX_F], TR[y], TE[MEAL_F + CTX_F])
    oo = oof(TR, MEAL_F + CTX_F, y)
    off = shrink_offset(TR, TR[y].to_numpy() - oo, best_k[y])
    PRED[("M3", y)] = PRED[("M2", y)] + TE.sid.map(off).fillna(0).to_numpy()
    PRED[("M4", y)] = fit_predict(TR[MEAL_F + CTX_F + PERS_F], TR[y], TE[MEAL_F + CTX_F + PERS_F])
    oo4 = oof(TR, MEAL_F + CTX_F + PERS_F, y)
    off4 = shrink_offset(TR, TR[y].to_numpy() - oo4, best_k[y])
    PRED[("M5", y)] = PRED[("M4", y)] + TE.sid.map(off4).fillna(0).to_numpy()
MODELS = {"M0": "Population mean by meal type", "M1": "Person's own average", "M2": "Population model (meal + pre-meal context)",
          "M3": "M2 + personal offset", "M4": "M2 + person descriptors", "M5": "M4 + personal offset"}

def to_abs(model):
    """convert delta-scale predictions to the challenge outputs"""
    p = {y: PRED[(model, y)] for y in DT}
    out = {f"G{k}": TE.G0.to_numpy() + p[f"d{k}"] for k in (30, 60, 90, 120)}
    out.update(Gmax=TE.G0.to_numpy() + p["dpeak"], dpeak=p["dpeak"], ttp=np.clip(p["ttp"], 0, 120), iauc=np.clip(p["iauc"], 0, None))
    return out

def boot_mae(err, sids, n=500):
    u = np.unique(sids); b = []
    for _ in range(n):
        pick = rng.choice(u, len(u)); idx = np.concatenate([np.flatnonzero(sids == s_) for s_ in pick]); b.append(err[idx].mean())
    return np.percentile(b, [2.5, 97.5])

RES = []
for mdl in MODELS:
    P = to_abs(mdl)
    for t in TARGETS:
        e = np.abs(P[t] - TE[t].to_numpy())
        for sub, mask in [("all", np.ones(len(TE), bool)), ("unseen", ~TE.seen.to_numpy())]:
            lo, hi = boot_mae(e[mask], TE.sid.to_numpy()[mask], 300)
            RES.append(dict(model=mdl, target=t, subset=sub, mae=e[mask].mean(), lo=lo, hi=hi,
                            r=stats.pearsonr(P[t][mask], TE[t].to_numpy()[mask])[0] if t not in ("ttp",) else stats.spearmanr(P[t][mask], TE[t].to_numpy()[mask]).statistic))
RES = pd.DataFrame(RES)
S["model_results"] = RES.round(3).to_dict("records")
best = RES[(RES.subset == "unseen") & (RES.target == "iauc")].sort_values("mae").model.iloc[0]
S["best_model_iauc_unseen"] = best
# person-level personalization: correlation of per-person mean predicted vs observed dpeak on test meals
for mdl in ["M2", "M3", "M4", "M5"]:
    P = to_abs(mdl); t_ = TE.assign(p=P["dpeak"]).groupby("sid")[["p", "dpeak"]].mean()
    S[f"person_level_r_dpeak_{mdl}"] = float(stats.pearsonr(t_.p, t_.dpeak)[0])

# learning curve: personal offset estimated from the first d days only (population model fixed)
LC = []
for y in ["dpeak", "iauc", "d60"]:
    oo = oof(TR, MEAL_F + CTX_F, y); base = PRED[("M2", y)]
    for dd in range(0, 8):
        h = TR.day.to_numpy() <= dd
        off = shrink_offset(TR[h], TR[y].to_numpy()[h] - oo[h], best_k[y]) if h.any() else pd.Series(dtype=float)
        p = base + TE.sid.map(off).fillna(0).to_numpy()
        e = np.abs(p - TE[y].to_numpy())
        LC.append(dict(target=y, days=dd, mae=e.mean(), mae_unseen=e[~TE.seen.to_numpy()].mean(),
                       meals_median=float(TR[h].groupby("sid").size().reindex(TE.sid.unique()).fillna(0).median())))
LC = pd.DataFrame(LC)
S["learning_curve"] = LC.round(2).to_dict("records")

# permutation importance (feature groups) for the best model on test, dpeak and iauc
GROUPS_F = {"meal composition": ["Carbs", "Protein", "Fat", "fiber_c", "kcal_atw"], "meal type": ["is_breakfast", "is_lunch", "is_dinner", "is_snack"],
            "pre-meal glucose level": ["G0", "g0_rel"], "pre-meal glucose trend": ["slope30", "sd60"], "time of day": ["hour"],
            "previous meal": ["prev_gap", "prev_carbs"], "heart rate": ["hr_pre", "hr_max60"], "day of wear": ["day"],
            "person descriptors": PERS_F}
IMP = []
for y in ["dpeak", "iauc"]:
    feats = MEAL_F + CTX_F + PERS_F
    mdl = hgb().fit(TR[feats], TR[y]); base_e = np.mean(np.abs(mdl.predict(TE[feats]) - TE[y]))
    for gname, cols in GROUPS_F.items():
        inc = []
        for _ in range(20):
            X = TE[feats].copy(); perm = rng.permutation(len(X))
            X[cols] = X[cols].to_numpy()[perm]
            inc.append(np.mean(np.abs(mdl.predict(X) - TE[y])) - base_e)
        IMP.append(dict(target=y, group=gname, inc=np.mean(inc), sd=np.std(inc), rel=np.mean(inc) / base_e))
IMP = pd.DataFrame(IMP)
S["importance"] = IMP.round(4).to_dict("records")

# ================================================================== 4. photo -> macros, and how macro error propagates
PE = np.load(os.path.join(E, "cache", "photo_emb.npz"), allow_pickle=True)
emb = pd.DataFrame(PE["emb"]); emb["sid"] = PE["sid"]; emb["ts"] = pd.to_datetime(PE["ts"])
emb = emb.drop_duplicates(["sid", "ts"])
Mp = M.merge(emb, left_on=["sid", "Timestamp"], right_on=["sid", "ts"], how="inner")
Mp = Mp[Mp.Carbs.notna()].reset_index(drop=True)
EC = list(range(PE["emb"].shape[1]))
S["photo_n"] = len(Mp)
PH = {}
reg = lambda: make_pipeline(StandardScaler(), PCA(128, random_state=0), RidgeCV(alphas=np.logspace(-1, 4, 12)))
for c in ["Carbs", "Protein", "Fat", "kcal_atw"]:
    p = np.zeros(len(Mp))
    for a, b in GroupKFold(5).split(Mp, groups=Mp.sid):
        p[b] = np.expm1(reg().fit(Mp.iloc[a][EC], np.log1p(Mp.iloc[a][c].clip(lower=0))).predict(Mp.iloc[b][EC]))
    Mp[f"ph_{c}"] = np.clip(p, 0, None)
    for sub, mask in [("standardized (breakfast/lunch)", Mp.meal.isin(["breakfast", "lunch"])), ("free-living (dinner/snack)", Mp.meal.isin(["dinner", "snack"]))]:
        t_, p_ = Mp.loc[mask, c], Mp.loc[mask, f"ph_{c}"]
        PH[f"{c}|{sub}"] = dict(mae=float(np.mean(np.abs(p_ - t_))), median_ape=float(np.median(np.abs(p_ - t_) / t_.clip(lower=5)) * 100),
                               r=float(stats.pearsonr(p_, t_)[0]), true_median=float(t_.median()))
S["photo_macro"] = PH

# end-to-end: best-model glucose predictions on test meals with photo-estimated macros, and with noisy macros
best_feats = MEAL_F + CTX_F + (PERS_F if best in ("M4", "M5") else [])
_fit_cache = {}
def predict_with(TEx, y):
    if y not in _fit_cache:
        mdl_ = hgb().fit(TR[best_feats], TR[y])
        off_ = shrink_offset(TR, TR[y].to_numpy() - oof(TR, best_feats, y), best_k[y]) if best in ("M3", "M5") else pd.Series(dtype=float)
        _fit_cache[y] = (mdl_, off_)
    mdl_, off_ = _fit_cache[y]
    return mdl_.predict(TEx[best_feats]) + TEx.sid.map(off_).fillna(0).to_numpy()
TEp = TE.merge(Mp[["sid", "Timestamp", "ph_Carbs", "ph_Protein", "ph_Fat", "ph_kcal_atw"]], on=["sid", "Timestamp"], how="left")
TEp.index = TE.index
SENS = []
for y in ["dpeak", "iauc"]:
    base_p = predict_with(TE, y)
    SENS.append(dict(target=y, cond="logged macros", err_pct=0, mae=float(np.mean(np.abs(base_p - TE[y])))))
    for pct in [10, 20, 30, 50, 75]:
        maes = []
        for rep in range(10):
            X = TE.copy(); sig = pct / 100 / 0.8   # mean |log-normal error| ≈ 0.8σ for small σ
            for c in ["Carbs", "Protein", "Fat", "fiber_c"]:
                X[c] = X[c] * np.exp(rng.normal(0, sig, len(X)))
            X["kcal_atw"] = 4 * X.Carbs + 4 * X.Protein + 9 * X.Fat
            maes.append(np.mean(np.abs(predict_with(X, y) - TE[y])))
        SENS.append(dict(target=y, cond="noisy macros", err_pct=pct, mae=float(np.mean(maes)), sd=float(np.std(maes))))
    X = TEp.copy(); has = X.ph_Carbs.notna()
    X.loc[has, "Carbs"], X.loc[has, "Protein"], X.loc[has, "Fat"] = X.ph_Carbs[has], X.ph_Protein[has], X.ph_Fat[has]
    X.loc[has, "fiber_c"] = np.nan; X["kcal_atw"] = 4 * X.Carbs + 4 * X.Protein + 9 * X.Fat
    SENS.append(dict(target=y, cond="photo-estimated macros (ResNet50 + ridge)", err_pct=None, mae=float(np.mean(np.abs(predict_with(X, y) - TE[y]))), n_photo=int(has.sum())))
    X = TE.copy(); X[["Carbs", "Protein", "Fat", "fiber_c", "kcal_atw"]] = np.nan
    SENS.append(dict(target=y, cond="no macros (meal type + context only)", err_pct=None, mae=float(np.mean(np.abs(predict_with(X, y) - TE[y])))))
S["macro_sensitivity"] = SENS

# ================================================================== 5. uncertainty: personal residual spread -> intervals; low-information flags
UNC = {}
Pb = to_abs(best)
for y in ["dpeak", "iauc", "G60"]:
    yy = {"G60": "d60"}.get(y, y)
    feats = MEAL_F + CTX_F + (PERS_F if best in ("M4", "M5") else [])
    oo_ = oof(TR, feats, yy); res = TR[yy].to_numpy() - oo_
    gsd = np.std(res); psd = pd.Series(res).groupby(TR.sid.to_numpy()).agg(lambda r: np.sqrt((np.sum(r ** 2) + 5 * gsd ** 2) / (len(r) + 5)))
    sd = TE.sid.map(psd).fillna(gsd).to_numpy()
    pred = Pb[y]; obs = TE[y].to_numpy()
    cov_p = np.mean(np.abs(obs - pred) <= 1.645 * sd); cov_g = np.mean(np.abs(obs - pred) <= 1.645 * gsd)
    UNC[y] = dict(coverage_personal_sd=float(cov_p), coverage_global_sd=float(cov_g), mean_width_personal=float(2 * 1.645 * sd.mean()),
                  width_global=float(2 * 1.645 * gsd), sd_ratio_max_min=float(psd.max() / psd.min()))
    if y == "dpeak":
        TE["pi_sd"] = sd
S["uncertainty"] = UNC
trmax = TR.groupby("sid").Carbs.max(); trtypes = TR.groupby("sid").meal.apply(set); trn = TR.groupby("sid").size()
TE["flag_carb_extrap"] = TE.Carbs > TE.sid.map(trmax).fillna(0)
TE["flag_new_type"] = [r.meal not in trtypes.get(r.sid, set()) for r in TE.itertuples()]
TE["flag_few_meals"] = TE.sid.map(trn).fillna(0) < 10
TE["abs_err_dpeak"] = np.abs(Pb["dpeak"] - TE.dpeak.to_numpy())
FL = []
for f, lab in [("flag_carb_extrap", "more carbs than any meal in days 1–7"), ("flag_new_type", "meal type not seen in days 1–7"),
               ("flag_few_meals", "fewer than 10 usable meals in days 1–7")]:
    a, b = TE[TE[f]].abs_err_dpeak, TE[~TE[f]].abs_err_dpeak
    FL.append(dict(flag=lab, n_flag=len(a), mae_flag=a.mean() if len(a) else np.nan, mae_rest=b.mean(),
                   p=stats.mannwhitneyu(a, b).pvalue if len(a) > 3 else np.nan))
S["flags"] = pd.DataFrame(FL).round(3).to_dict("records")

json.dump(S, open(os.path.join(E, "challenge_stats.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
L.drop(columns=["Image path"]).to_csv(os.path.join(E, "challenge_meals.csv"), index=False)
print("stats written; best model:", best)

# ================================================================== FIGURES
GC = lambda T: dict(zip(GROUPS, T["s"][:3]))

def f_variance(T):
    fig, ax = plt.subplots(figsize=(12.5, 3.9))
    bars = [("Shakes · glucose rise (Δpeak)", VD["shakes_dpeak"], VD["person_share_explained_by_group_dpeak"]),
            ("Shakes · iAUC (log)", VD["shakes_log_iauc"], VD["person_share_explained_by_group_log_iauc"]),
            ("Lunches · glucose rise (Δpeak)", VD["lunch_dpeak"], None), ("Lunches · iAUC (log)", VD["lunch_log_iauc"], None)]
    cols = dict(recipe=T["ref"], person=T["s"][0], person_x_recipe=T["s"][2], day_to_day=T["grid"], residual=T["grid"])
    labs = dict(recipe="which meal", person="which person", person_x_recipe="person × meal (personal ranking)",
                day_to_day="same person, same meal, different day", residual="unexplained (incl. person × meal and day-to-day)")
    seen_l = set()
    for i, (name, v, gshare) in enumerate(bars):
        left = 0
        for k in ["recipe", "person", "person_x_recipe", "day_to_day", "residual"]:
            if k not in v: continue
            w = v[k] * 100
            ax.barh(i, w, left=left, color=cols[k], height=0.62, label=labs[k] if k not in seen_l else None, edgecolor=T["panel"], lw=1.5)
            if k == "person" and gshare is not None:
                ax.barh(i, w * gshare, left=left, color="none", height=0.62, hatch="////", edgecolor=T["panel"], lw=0)
            if w >= 6: ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=9, color=T["panel"] if k in ("person", "person_x_recipe") else T["ink"])
            seen_l.add(k); left += w
    ax.set_yticks(range(len(bars)), [b[0] for b in bars]); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel("share of variance (%)"); ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0)
    ax.text(100, 4.05, "hatched = part of the person share explained by diabetes group", ha="right", fontsize=8.5, color=T["ink2"])
    return fig
save_fig(f_variance, "cx_variance", FIG)

def f_context(T):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True, gridspec_kw=dict(wspace=0.06))
    order_ = [c for c, _ in CTX]
    for ax, y, ttl in [(axes[0], "dpeak", "Glucose rise (Δpeak)"), (axes[1], "iauc", "iAUC 0–120 min")]:
        d = CX[CX.target == y].set_index("feature").reindex(order_)
        for i, (c, row) in enumerate(d.iterrows()):
            col = T["ref"] if c == "hr_post" else (T["s"][0] if row.lo > 0 or row.hi < 0 else T["ink2"])
            ax.plot([row.lo, row.hi], [i, i], color=col, lw=2)
            ax.plot(row.rho, i, "o", color=col, ms=7, mec=T["panel"], mew=1.5)
        ax.axvline(0, color=T["ink2"], lw=1); ax.set_title(ttl); ax.set_xlabel("within-person Spearman ρ with residual response (95% CI)")
        ax.grid(axis="y", visible=False); ax.set_xlim(-0.45, 0.45)
    axes[0].set_yticks(range(len(order_)), [l for _, l in CTX]); axes[0].invert_yaxis()
    fig.text(0.125, -0.02, "* not available at prediction time; shown to size the effect of post-meal activity. Blue = 95% CI excludes 0.", fontsize=8.5, color=T["ink2"])
    return fig
save_fig(f_context, "cx_context", FIG)

def f_models(T):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw=dict(wspace=0.32, width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    tg = ["G30", "G60", "G90", "G120", "Gmax", "dpeak", "iauc", "ttp"]
    d = RES[RES.subset == "unseen"].pivot(index="target", columns="model", values="mae").reindex(tg)
    rel = d.div(d["M0"], axis=0) * 100
    mcol = dict(M0=T["grid"], M1=T["s"][3], M2=T["ref"], M3=T["s"][1], M4=T["s"][2], M5=T["s"][0])
    w = 0.13
    for j, mdl in enumerate(MODELS):
        ax.bar(np.arange(len(tg)) + (j - 2.5) * w, rel[mdl], width=w, color=mcol[mdl], label=f"{mdl} {MODELS[mdl]}", lw=0)
    ax.axhline(100, color=T["ink2"], lw=0.8)
    ax.set_xticks(range(len(tg)), ["G30", "G60", "G90", "G120", "max", "Δpeak", "iAUC", "t-peak"]); ax.set_ylim(40, 115)
    ax.set_ylabel("MAE, % of population-mean baseline"); ax.set_title("Error on unseen day-8+ meals (lower is better)")
    ax.legend(fontsize=7.6, loc="lower left", ncol=2); ax.grid(axis="x", visible=False)
    ax = axes[1]
    for y, col, lab in [("dpeak", T["s"][0], "Δpeak (mg/dL)"), ("d60", T["s"][1], "Δ glucose at 60 min (mg/dL)")]:
        q = LC[LC.target == y]; ax.plot(q.days, q.mae_unseen, "o-", color=col, label=lab, ms=5, mec=T["panel"])
    ax.set_xlabel("days of personal history used for the offset"); ax.set_ylabel("MAE on unseen test meals"); ax.set_title("How much history personalization needs")
    ax.legend(fontsize=8.5); ax.set_xticks(range(8))
    ax = axes[2]
    q = LC[LC.target == "iauc"]; ax.plot(q.days, q.mae_unseen, "o-", color=T["s"][2], ms=5, mec=T["panel"], label="iAUC 0–120 min")
    ax.set_xlabel("days of personal history used for the offset"); ax.set_ylabel("MAE (mg/dL·min)"); ax.set_title("Same, for iAUC"); ax.set_xticks(range(8))
    ax.legend(fontsize=8.5)
    return fig
save_fig(f_models, "cx_models", FIG)

def f_day8(T):
    fig = plt.figure(figsize=(14.5, 7.4))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.28)
    Pp, Pb_ = to_abs("M2"), to_abs(best)
    TEr = TE.reset_index(drop=True)
    cand = TEr[~TEr.seen]
    picks = []
    for g in GROUPS:
        for mt in ["lunch", "dinner"]:
            c = cand[(cand.group == g) & (cand.meal == mt)]
            if len(c): picks.append(c.sample(1, random_state=3).index[0])
    tt = np.arange(0, 121, 5)
    for k, i in enumerate(picks[:6]):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        r = TEr.loc[i]; ci = TE.index[i]
        ax.plot(tt, r.G0 + CURV[ci], color=T["ink"], lw=2, label="observed (Dexcom)")
        for P_, col, lab in [(Pp, T["ref"], "population model"), (Pb_, T["s"][0], f"personalized ({best})")]:
            ys = [r.G0] + [P_[f"G{m}"][i] for m in (30, 60, 90, 120)]
            ax.plot([0, 30, 60, 90, 120], ys, "o--", color=col, ms=5, lw=1.5, label=lab, mec=T["panel"])
            ax.plot(P_["ttp"][i], P_["Gmax"][i], "*", color=col, ms=11, mec=T["panel"])
        ax.set_title(f"#{r.sid} ({r.group}) · day {r.day} {r.meal}\n{r.Carbs:g} g carb · {r.kcal_atw:.0f} kcal", fontsize=9.8)
        ax.set_xticks([0, 30, 60, 90, 120]); ax.set_xlabel("min after meal", fontsize=9)
        if k % 3 == 0: ax.set_ylabel("glucose (mg/dL)")
        if k == 0: ax.legend(fontsize=7.8, loc="upper right")
    ax = fig.add_subplot(gs[:, 3])
    for P_, col, lab in [(Pp, T["ref"], "population (M2)"), (Pb_, T["s"][0], f"personalized ({best})")]:
        ax.scatter(TE.dpeak, P_["dpeak"], s=10, color=col, alpha=0.6, lw=0,
                   label=f"{lab}: r = {stats.pearsonr(P_['dpeak'], TE.dpeak)[0]:.2f}")
    mx = np.percentile(TE.dpeak, 99); ax.plot([0, mx], [0, mx], color=T["ink2"], ls="--", lw=1)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_xlabel("observed Δpeak (mg/dL)"); ax.set_ylabel("predicted Δpeak (mg/dL)")
    ax.set_title("All day-8+ test meals"); ax.legend(fontsize=8, loc="upper left")
    return fig
save_fig(f_day8, "cx_day8", FIG)

def f_importance(T):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True, gridspec_kw=dict(wspace=0.06))
    order_g = IMP[IMP.target == "dpeak"].sort_values("rel", ascending=False).group.tolist()
    for ax, y, ttl in [(axes[0], "dpeak", "Glucose rise (Δpeak)"), (axes[1], "iauc", "iAUC 0–120 min")]:
        d = IMP[IMP.target == y].set_index("group").reindex(order_g)
        ax.barh(range(len(d)), d.rel * 100, xerr=d.sd / (d.inc / d.rel).replace(0, np.nan) * 100, color=T["s"][0], height=0.65,
                error_kw=dict(ecolor=T["ink2"], lw=1))
        ax.axvline(0, color=T["ink2"], lw=0.8); ax.set_title(ttl); ax.set_xlabel("MAE increase when shuffled (% of model MAE)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order_g)), order_g); axes[0].invert_yaxis()
    return fig
save_fig(f_importance, "cx_importance", FIG)

def f_photo(T):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw=dict(wspace=0.3))
    for ax, c, ttl in [(axes[0], "Carbs", "Carbs from photo"), (axes[1], "kcal_atw", "Calories from photo")]:
        for sub, mask, col in [("standardized", Mp.meal.isin(["breakfast", "lunch"]), T["ref"]), ("free-living", Mp.meal.isin(["dinner", "snack"]), T["s"][0])]:
            ax.scatter(Mp.loc[mask, c], Mp.loc[mask, f"ph_{c}"], s=9, alpha=0.5, color=col, lw=0,
                       label=f"{sub}: median error {PH[f'{c}|' + ('standardized (breakfast/lunch)' if sub == 'standardized' else 'free-living (dinner/snack)')]['median_ape']:.0f}%")
        mx = np.percentile(Mp[c], 99); ax.plot([0, mx], [0, mx], color=T["ink2"], ls="--", lw=1); ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.set_xlabel("logged"); ax.set_ylabel("estimated from photo"); ax.set_title(ttl); ax.legend(fontsize=8, loc="upper left")
    ax = axes[2]
    sd = pd.DataFrame(SENS)
    for y, col, lab in [("dpeak", T["s"][0], "Δpeak")]:
        q = sd[(sd.target == y) & (sd.cond.isin(["logged macros", "noisy macros"]))]
        ax.plot(q.err_pct, q.mae, "o-", color=col, label=f"{lab}: macros with random error", ms=5, mec=T["panel"])
        for cond, mk, cc in [("photo-estimated macros (ResNet50 + ridge)", "D", T["s"][1]), ("no macros (meal type + context only)", "s", T["s"][7])]:
            v = sd[(sd.target == y) & (sd.cond == cond)].mae.iloc[0]
            ax.axhline(v, color=cc, ls=":", lw=1.3); ax.text(76, v + 0.15, cond.split(" (")[0], fontsize=8.3, color=cc, ha="right", va="bottom")
    ax.set_xlabel("mean error in macros (%)"); ax.set_ylabel("Δpeak MAE on test meals (mg/dL)"); ax.set_title("What macro error costs"); ax.legend(fontsize=8, loc="upper left")
    return fig
save_fig(f_photo, "cx_photo", FIG)
print("figures written")
