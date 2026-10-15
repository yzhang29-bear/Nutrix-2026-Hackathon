"""CGMacros EDA: stats JSON + figures (light/dark).
usage: python -I eda_cgmacros.py <CGMacros dir> <out dir>
"""
import sys, os, glob, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_common import save_fig, GROUPS
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from PIL import Image, ImageOps

ROOT, OUT = sys.argv[1], sys.argv[2]
FIG = os.path.join(OUT, "fig")
S = {}

# ------------------------------------------------------------------ load
bio = pd.read_csv(os.path.join(ROOT, "bio.csv"))
bio.columns = [c.strip() for c in bio.columns]
bio = bio.rename(columns={"A1c PDL (Lab)": "A1c", "Fasting GLU - PDL (Lab)": "FPG", "Body weight": "Weight_lb",
                          "Height": "Height_in", "Self-identify": "Ethnicity", "LDL (Cal)": "LDL", "VLDL (Cal)": "VLDL",
                          "Cho/HDL Ratio": "Chol_HDL", "Non HDL": "NonHDL"})
bio["sid"] = bio.subject.map(lambda x: f"{int(x):03d}")
S["bio_sentinels"] = dict(LDL=int((bio.LDL >= 800).sum()), VLDL=int((bio.VLDL >= 400).sum()), Chol_HDL=int((bio.Chol_HDL >= 400).sum()))
for c, v in [("LDL", 800), ("VLDL", 400), ("Chol_HDL", 400)]:
    bio.loc[bio[c] >= v, c] = np.nan
bio["HOMA_IR"] = bio.FPG * bio.Insulin / 405
bio["group"] = pd.cut(bio.A1c, [0, 5.65, 6.45, 99], labels=GROUPS).astype(str)
grp = bio.set_index("sid")["group"]
S["group_n"] = bio.group.value_counts().reindex(GROUPS).to_dict()

data, raw_labels, schemas = {}, {}, {}
for f in sorted(glob.glob(os.path.join(ROOT, "CGMacros-*", "CGMacros-*.csv"))):
    sid = os.path.basename(f)[9:12]
    d = pd.read_csv(f)
    d.columns = [c.strip() for c in d.columns]
    d = d.drop(columns=[c for c in d.columns if c.startswith("Unnamed")])
    key = ",".join(d.columns); schemas.setdefault(key, []).append(sid)
    d["activity"] = d["METs"] if "METs" in d else (d["Intensity"] if "Intensity" in d else np.nan)
    d["Timestamp"] = pd.to_datetime(d["Timestamp"])
    for k, v in d["Meal Type"].dropna().value_counts().items():
        raw_labels[k] = raw_labels.get(k, 0) + int(v)
    d["meal"] = d["Meal Type"].map(lambda x: ("snack" if str(x).strip().lower().startswith("snack") else str(x).strip().lower()) if pd.notna(x) else np.nan)
    data[sid] = d
S["n_subjects"] = len(data)
S["missing_ids"] = sorted(set(f"{i:03d}" for i in range(1, 50)) - set(data))
S["meal_label_variants"] = raw_labels
S["n_rows"] = int(sum(len(d) for d in data.values()))
S["schemas"] = [dict(columns=k, n=len(v), sids=v) for k, v in sorted(schemas.items(), key=lambda kv: -len(kv[1]))]

# ------------------------------------------------------------------ per-subject quality + CGM metrics
def kink_interval(x):
    """native sampling interval of a linearly-interpolated 1-min series = spacing of slope changes"""
    v = x.to_numpy()
    d2 = np.abs(np.diff(v, 2))
    k = np.flatnonzero(d2 > 1e-6) + 1
    if len(k) < 10: return np.nan, np.nan
    iv = np.diff(k)
    mode = stats.mode(iv, keepdims=False).mode
    return int(mode), float(np.mean(iv % mode == 0))

rows = []
for sid, d in data.items():
    dt = d.Timestamp.diff().dt.total_seconds().div(60)
    lib, dex = d["Libre GL"], d["Dexcom GL"]
    li, lfrac = kink_interval(lib.dropna())
    di, dfrac = kink_interval(dex.dropna())
    v = lib.dropna().to_numpy()
    r = dict(sid=sid, group=grp.get(sid), rows=len(d), days=len(d) / 1440,
             span_days=(d.Timestamp.max() - d.Timestamp.min()).total_seconds() / 86400,
             gaps_gt1min=int((dt > 1).sum()), max_gap_h=float(dt.max() / 60), dup_ts=int((dt == 0).sum()),
             libre_cov=lib.notna().mean(), dexcom_cov=dex.notna().mean(), hr_cov=d.HR.notna().mean(),
             act_cov=d.activity.notna().mean(), act_col="METs" if "METs" in d else ("Intensity" if "Intensity" in d else "none"), libre_native=li, libre_native_frac=lfrac, dexcom_native=di, dexcom_native_frac=dfrac,
             mean_g=v.mean(), cv=v.std() / v.mean() * 100, tir=np.mean((v >= 70) & (v <= 180)) * 100,
             tbr=np.mean(v < 70) * 100, tar=np.mean(v > 180) * 100, gmi=3.31 + 0.02392 * v.mean(),
             mean_g_dex=dex.mean(), meals=int(d.meal.notna().sum()))
    w = dex.dropna().to_numpy()
    r.update(tir_dex=np.mean((w >= 70) & (w <= 180)) * 100, tbr_dex=np.mean(w < 70) * 100, tar_dex=np.mean(w > 180) * 100,
             cv_dex=w.std() / w.mean() * 100)
    rows.append(r)
Q = pd.DataFrame(rows).merge(bio[["sid", "A1c", "group"]], on=["sid", "group"], how="left")
Q.to_csv(os.path.join(OUT, "cgmacros_subjects.csv"), index=False)
S["days_median"], S["days_min"], S["days_max"] = Q.days.median(), Q.days.min(), Q.days.max()
S["libre_native_mode"], S["dexcom_native_mode"] = Q.libre_native.mode()[0], Q.dexcom_native.mode()[0]
S["libre_native_frac_median"], S["dexcom_native_frac_median"] = Q.libre_native_frac.median(), Q.dexcom_native_frac.median()
S["coverage_median"] = Q[["libre_cov", "dexcom_cov", "hr_cov", "act_cov"]].median().round(3).to_dict()
S["coverage_min"] = Q[["libre_cov", "dexcom_cov", "hr_cov", "act_cov"]].min().round(3).to_dict()
S["subjects_with_time_gaps"] = Q.loc[Q.gaps_gt1min > 0, ["sid", "gaps_gt1min", "max_gap_h"]].round(1).to_dict("records")
r_l, p_l = stats.pearsonr(Q.mean_g, Q.A1c)
r_d, p_d = stats.pearsonr(Q.mean_g_dex, Q.A1c)
S["meanG_vs_A1c"] = dict(r_libre=r_l, r_dexcom=r_d)
S["cgm_by_group"] = Q.groupby("group")[["mean_g", "cv", "tir", "tbr", "tar", "mean_g_dex", "cv_dex", "tir_dex", "tbr_dex", "tar_dex"]].median().round(1).reindex(GROUPS).to_dict("index")

# ------------------------------------------------------------------ Libre vs Dexcom agreement
pairs, lags = [], []
for sid, d in data.items():
    x = d[["Timestamp", "Libre GL", "Dexcom GL"]].dropna()
    x = x[x.Timestamp.dt.minute % 15 == 0]          # one pair per 15 min, avoids interpolation pseudo-replication
    pairs.append(x.assign(sid=sid))
    a = d.set_index("Timestamp")[["Libre GL", "Dexcom GL"]].dropna()
    if len(a) > 2000:
        la, da = np.diff(a["Libre GL"].to_numpy()), np.diff(a["Dexcom GL"].to_numpy())
        la, da = la - la.mean(), da - da.mean()
        cc = [np.corrcoef(la[max(0, k):len(la) + min(0, k)], da[max(0, -k):len(da) - max(0, k)])[0, 1] for k in range(-30, 31)]
        lags.append(np.arange(-30, 31)[int(np.argmax(cc))])
P = pd.concat(pairs)
diff = P["Dexcom GL"] - P["Libre GL"]
mean_ = (P["Dexcom GL"] + P["Libre GL"]) / 2
S["agreement"] = dict(n_pairs=len(P), bias=diff.mean(), sd=diff.std(), loa_lo=diff.mean() - 1.96 * diff.std(),
                      loa_hi=diff.mean() + 1.96 * diff.std(), mard_vs_libre=float(np.mean(np.abs(diff) / P["Libre GL"]) * 100),
                      r=float(np.corrcoef(P["Libre GL"], P["Dexcom GL"])[0, 1]),
                      lag_median_min=float(np.median(lags)), lag_iqr=[float(np.percentile(lags, 25)), float(np.percentile(lags, 75))],
                      subj_bias_range=[float(P.assign(dd=diff).groupby("sid").dd.mean().min()), float(P.assign(dd=diff).groupby("sid").dd.mean().max())])

# ------------------------------------------------------------------ meals table + quality checks
ML = []
for sid, d in data.items():
    m = d[d.meal.notna()].copy()
    m["sid"] = sid
    m["prev_gap_min"] = m.Timestamp.diff().dt.total_seconds() / 60
    m["next_gap_min"] = -m.Timestamp.diff(-1).dt.total_seconds() / 60
    ML.append(m)
M = pd.concat(ML, ignore_index=True)
M["group"] = M.sid.map(grp)
M["atwater"] = 4 * M.Carbs + 4 * M.Protein + 9 * M.Fat
M["atw_ratio"] = M.Calories / M.atwater.replace(0, np.nan)
M["hour"] = M.Timestamp.dt.hour + M.Timestamp.dt.minute / 60
S["n_meals"] = len(M)
S["meals_by_type"] = M.meal.value_counts().to_dict()
S["meals_per_day_median"] = float((Q.meals / Q.days).median())
S["photo_share"] = float(M["Image path"].notna().mean())
S["amount_consumed_nonnull"] = int(M["Amount Consumed"].notna().sum())
S["amount_consumed_dist"] = M["Amount Consumed"].round(-1).value_counts().head(8).to_dict()
S["fiber_error_subjects"] = M.loc[M.Fiber > 50].sid.value_counts().to_dict()
S["atwater_within_25pct"] = float(np.mean(np.abs(np.log(M.atw_ratio.dropna().clip(lower=1e-6))) < np.log(1.25)))
S["zero_kcal_meals"] = int((M.Calories == 0).sum())
S["fiber_gt_carbs"] = int((M.Fiber > M.Carbs).sum())
S["fiber_gt_50"] = M.loc[M.Fiber > 50, ["sid", "meal", "Calories", "Carbs", "Fiber"]].to_dict("records")
S["dup_meal_ts"] = int(M.duplicated(["sid", "Timestamp"]).sum())
S["meals_with_next_within_2h"] = float((M.next_gap_min < 120).mean())
bf = M[M.meal == "breakfast"]
S["breakfast_variants"] = bf.groupby(["Calories", "Carbs", "Protein", "Fat", "Fiber"]).size().sort_values(ascending=False).reset_index(name="n").to_dict("records")
lu = M[M.meal == "lunch"]
S["lunch_variants"] = lu.groupby(["Calories", "Carbs", "Protein", "Fat", "Fiber"]).size().sort_values(ascending=False).reset_index(name="n").head(12).to_dict("records")

# ------------------------------------------------------------------ postprandial response (Libre, 1-min)
WIN = np.arange(-15, 181)
curves, cm = [], []
for i, r in M.iterrows():
    s = data[r.sid].drop_duplicates("Timestamp").set_index("Timestamp")["Libre GL"]
    v = s.reindex(r.Timestamp + pd.to_timedelta(WIN, unit="min")).to_numpy()
    if np.isnan(v).mean() > 0.1:
        curves.append(np.full(len(WIN), np.nan)); continue
    v = pd.Series(v).interpolate(limit_direction="both").to_numpy()
    base = v[:16].mean()
    inc = v - base
    curves.append(inc)
C = np.array(curves)
post = C[:, 15:15 + 121]
M["iauc120"] = np.trapz(np.clip(post, 0, None), dx=1, axis=1)
M["peak_delta"] = np.nanmax(C[:, 15:], axis=1)
M["t_peak"] = np.nanargmax(np.nan_to_num(C[:, 15:], nan=-1e9), axis=1).astype(float)
M.loc[np.isnan(C).all(1), ["iauc120", "peak_delta", "t_peak"]] = np.nan
M["clean"] = (M.next_gap_min.isna() | (M.next_gap_min >= 120)) & (M.prev_gap_min.isna() | (M.prev_gap_min >= 120))
S["ppgr_valid"] = int(M.iauc120.notna().sum())
S["ppgr_clean"] = int((M.iauc120.notna() & M.clean).sum())

# shake variants: label + contrast vs the 448 kcal reference
def shake_label(r):
    return f"C{r.Carbs:g} P{r.Protein:g} F{r.Fat:g} Fb{r.Fiber:g}"
M["variant"] = np.where(M.meal == "breakfast", M.apply(shake_label, axis=1), None)
VAR_ORDER = ["C24 P22 F10.5 Fb0", "C66 P22 F10.5 Fb0", "C66 P66 F10.5 Fb0", "C66 P22 F42 Fb0", "C73 P22 F42 Fb7", "C73 P66 F42 Fb7"]
VAR_NAME = {"C24 P22 F10.5 Fb0": "Low carb", "C66 P22 F10.5 Fb0": "Reference (high carb)", "C66 P66 F10.5 Fb0": "+ Protein",
            "C66 P22 F42 Fb0": "+ Fat", "C73 P22 F42 Fb7": "+ Fat + Fiber", "C73 P66 F42 Fb7": "+ Protein + Fat + Fiber"}
S["variant_kcal_logged"] = M[M.meal == "breakfast"].groupby("variant").Calories.agg(lambda x: sorted(x.unique().tolist())).to_dict()
S["variant_kcal_atwater"] = M[M.meal == "breakfast"].groupby("variant").atwater.first().to_dict()

B = M[(M.meal == "breakfast") & M.iauc120.notna()]
per = B.groupby(["sid", "variant"]).iauc120.mean().unstack()
eff = []
for v in VAR_ORDER[1:] if False else VAR_ORDER:
    if v == "C66 P22 F10.5 Fb0": continue
    for g in GROUPS + ["All"]:
        sids = per.index if g == "All" else [s for s in per.index if grp.get(s) == g]
        dd = (per.loc[sids, v] - per.loc[sids, "C66 P22 F10.5 Fb0"]).dropna()
        ratio = (per.loc[sids, v] / per.loc[sids, "C66 P22 F10.5 Fb0"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(dd) < 3: continue
        ci = stats.t.interval(0.95, len(dd) - 1, loc=dd.mean(), scale=stats.sem(dd))
        eff.append(dict(variant=v, name=VAR_NAME[v], group=g, n=len(dd), mean_diff=dd.mean(), lo=ci[0], hi=ci[1],
                        p=stats.wilcoxon(dd).pvalue, median_ratio=ratio.median()))
E = pd.DataFrame(eff)
S["shake_effects"] = E.round(3).to_dict("records")
ref_by_group = per["C66 P22 F10.5 Fb0"].groupby(per.index.map(grp)).median().reindex(GROUPS)
S["ref_shake_iauc_median"] = ref_by_group.round(0).to_dict()

# repeatability of the same shake (ICC(1,1) over subjects with 2 repeats)
icc = []
for v in VAR_ORDER:
    x = B[B.variant == v].groupby("sid").iauc120.apply(list)
    x = x[x.map(len) >= 2].map(lambda a: a[:2])
    if len(x) < 5: continue
    A = np.array(x.tolist())
    k = 2; n = len(A)
    msb = k * np.var(A.mean(1), ddof=1)
    msw = np.sum((A - A.mean(1, keepdims=True)) ** 2) / (n * (k - 1))
    icc.append(dict(variant=v, n=n, icc=(msb - msw) / (msb + (k - 1) * msw)))
S["shake_icc"] = icc
S["shake_icc_median"] = float(np.median([x["icc"] for x in icc]))

# iAUC vs carbs (all clean meals)
cl = M[M.clean & M.iauc120.notna()]
S["iauc_vs_carbs_spearman"] = {g: float(stats.spearmanr(cl[cl.group == g].Carbs, cl[cl.group == g].iauc120).statistic) for g in GROUPS}
S["iauc_by_group_median"] = cl.groupby("group").iauc120.median().reindex(GROUPS).round(0).to_dict()
fl = cl[cl.meal.isin(["dinner", "snack"])]
S["free_iauc_vs_carbs_spearman"] = {g: float(stats.spearmanr(fl[fl.group == g].Carbs, fl[fl.group == g].iauc120).statistic) for g in GROUPS}
S["free_n"] = int(len(fl))

# ------------------------------------------------------------------ bio table
BIO_VARS = [("Age", "Age (y)"), ("BMI", "BMI (kg/m²)"), ("A1c", "HbA1c (%)"), ("FPG", "Fasting glucose (mg/dL)"),
            ("Insulin", "Fasting insulin (µU/mL)"), ("HOMA_IR", "HOMA-IR"), ("Triglycerides", "Triglycerides (mg/dL)"),
            ("HDL", "HDL (mg/dL)"), ("LDL", "LDL (mg/dL)")]
tab = []
for c, lab in BIO_VARS:
    row = dict(var=lab)
    for g in GROUPS:
        x = bio.loc[bio.group == g, c].dropna()
        row[g] = f"{x.median():.1f} [{x.quantile(.25):.1f}–{x.quantile(.75):.1f}]"
    row["p"] = stats.kruskal(*[bio.loc[bio.group == g, c].dropna() for g in GROUPS]).pvalue
    tab.append(row)
cgm_merge = bio.merge(Q[["sid", "mean_g", "cv", "tir", "tar", "mean_g_dex", "cv_dex", "tir_dex", "tar_dex"]], on="sid")
for c, lab in [("mean_g_dex", "Mean glucose, Dexcom (mg/dL)"), ("cv_dex", "Glucose CV, Dexcom (%)"), ("tir_dex", "Time 70–180 mg/dL, Dexcom (%)"), ("tar_dex", "Time >180 mg/dL, Dexcom (%)")]:
    row = dict(var=lab)
    for g in GROUPS:
        x = cgm_merge.loc[cgm_merge.group == g, c]
        row[g] = f"{x.median():.1f} [{x.quantile(.25):.1f}–{x.quantile(.75):.1f}]"
    row["p"] = stats.kruskal(*[cgm_merge.loc[cgm_merge.group == g, c] for g in GROUPS]).pvalue
    tab.append(row)
iref = per["C66 P22 F10.5 Fb0"].rename("ref_iauc")
cgm_merge = cgm_merge.merge(iref, left_on="sid", right_index=True, how="left")
row = dict(var="Reference-shake iAUC 0–120 min (mg/dL·min)")
for g in GROUPS:
    x = cgm_merge.loc[cgm_merge.group == g, "ref_iauc"].dropna()
    row[g] = f"{x.median():.0f} [{x.quantile(.25):.0f}–{x.quantile(.75):.0f}]"
row["p"] = stats.kruskal(*[cgm_merge.loc[cgm_merge.group == g, "ref_iauc"].dropna() for g in GROUPS]).pvalue
tab.append(row)
S["bio_table"] = tab
S["sex"] = bio.groupby("group").Gender.value_counts().unstack().reindex(GROUPS).fillna(0).astype(int).to_dict("index")
S["ethnicity"] = bio.Ethnicity.str.strip().value_counts().to_dict()

# ------------------------------------------------------------------ gut + microbes
gut = pd.read_csv(os.path.join(ROOT, "gut_health_test.csv"))
gut.columns = [c.strip().lstrip("﻿") for c in gut.columns]
gut["sid"] = gut.subject.map(lambda x: f"{int(x):03d}")
S["gut_rows"], S["gut_extra_subjects"] = len(gut), sorted(set(gut.sid) - set(data))
mic = pd.read_csv(os.path.join(ROOT, "microbes.csv"))
mic["sid"] = mic.subject.map(lambda x: f"{int(x):03d}")
Xall = mic.drop(columns=["subject", "sid"]).to_numpy().astype(bool)
bad = Xall.mean(1) > 0.5
S["microbes_invalid_rows"] = dict(sids=mic.sid[bad].tolist(), richness=Xall.sum(1)[bad].tolist())
mic = mic[~bad].reset_index(drop=True)
Xm = Xall[~bad]
prev = Xm.mean(0)
S["microbes"] = dict(n_taxa=int(Xm.shape[1]), n_subj=int(Xm.shape[0]), richness_median=float(np.median(Xm.sum(1))),
                     richness_range=[int(Xm.sum(1).min()), int(Xm.sum(1).max())],
                     taxa_absent_all=int((prev == 0).sum()), taxa_in_one=int((Xm.sum(0) == 1).sum()),
                     taxa_ge50=int((prev >= 0.5).sum()), taxa_ge90=int((prev >= 0.9).sum()), sparsity=float(1 - Xm.mean()))
D = squareform(pdist(Xm, "jaccard"))
n = len(D); Jc = np.eye(n) - 1 / n
Bm = -0.5 * Jc @ (D ** 2) @ Jc
w, V = np.linalg.eigh(Bm); o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
pcoa = V[:, :2] * np.sqrt(np.clip(w[:2], 0, None))
S["pcoa_var_expl"] = (w[:2] / w[w > 0].sum()).round(3).tolist()
gl = mic.sid.map(grp).to_numpy()
batch = np.where(mic.sid.astype(int) <= 18, "IDs 001–018", "IDs 019–049")
taxa = np.array([c.strip() for c in mic.columns if c not in ("subject", "sid")])
sp = np.array([" ".join(t.replace("[", "").replace("]", "").split()[:2]) for t in taxa])
uniq_sp = np.unique(sp)
Xs = np.stack([Xm[:, sp == u].any(1) for u in uniq_sp], 1)
Ds = squareform(pdist(Xs, "jaccard"))
Bs = -0.5 * Jc @ (Ds ** 2) @ Jc
ws, Vs = np.linalg.eigh(Bs); o2 = np.argsort(ws)[::-1]; ws, Vs = ws[o2], Vs[:, o2]
pcoa_s = Vs[:, :2] * np.sqrt(np.clip(ws[:2], 0, None))
S["microbe_batches"] = dict(n_batch1=int((batch == "IDs 001–018").sum()), n_batch2=int((batch == "IDs 019–049").sum()),
                            only_b1=int(((Xm[batch == "IDs 001–018"].sum(0) > 0) & (Xm[batch == "IDs 019–049"].sum(0) == 0)).sum()),
                            only_b2=int(((Xm[batch == "IDs 019–049"].sum(0) > 0) & (Xm[batch == "IDs 001–018"].sum(0) == 0)).sum()),
                            richness_b1=float(np.median(Xm[batch == "IDs 001–018"].sum(1))), richness_b2=float(np.median(Xm[batch == "IDs 019–049"].sum(1))),
                            n_species=int(len(uniq_sp)), pcoa_species_var=(ws[:2] / ws[ws > 0].sum()).round(3).tolist())

def pseudoF(Dm, labels):
    labs = np.unique(labels); N = len(labels); a = len(labs)
    sst = np.sum(Dm[np.triu_indices(N, 1)] ** 2) / N
    ssw = sum(np.sum(Dm[np.ix_(labels == l, labels == l)][np.triu_indices((labels == l).sum(), 1)] ** 2) / (labels == l).sum() for l in labs)
    return ((sst - ssw) / (a - 1)) / (ssw / (N - a))
rng = np.random.default_rng(0)
def permanova(Dm, lab):
    f0 = pseudoF(Dm, lab); pr = [pseudoF(Dm, rng.permutation(lab)) for _ in range(999)]
    return dict(F=float(f0), p=float((1 + np.sum(np.array(pr) >= f0)) / 1000))
S["permanova"] = permanova(D, gl)
S["permanova_batch"] = permanova(D, batch)
S["permanova_species_group"] = permanova(Ds, gl)
S["permanova_species_batch"] = permanova(Ds, batch)
rich = pd.Series(Xm.sum(1), index=mic.sid)
S["richness_kw_p"] = float(stats.kruskal(*[rich[[grp.get(s) == g for s in rich.index]] for g in GROUPS]).pvalue)
gut_m = gut.merge(bio[["sid", "A1c"]], on="sid")
gcols = [c for c in gut.columns if c not in ("subject", "sid")]
S["gut_vs_a1c"] = sorted([(c, float(stats.spearmanr(gut_m[c], gut_m.A1c, nan_policy="omit").statistic)) for c in gcols], key=lambda x: -abs(x[1]))[:5]

json.dump(S, open(os.path.join(OUT, "cgmacros_stats.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
M.drop(columns=["Image path"]).to_csv(os.path.join(OUT, "cgmacros_meals.csv"), index=False)
print("stats written")

# ================================================================== FIGURES
GC = lambda T: dict(zip(GROUPS, T["s"][:3]))

# --- F1 study timeline (swimmer plot)
order = Q.sort_values(["group", "A1c"], key=lambda s: s.map({g: i for i, g in enumerate(GROUPS)}) if s.name == "group" else s).sid.tolist()
def seg(mask, t0, tt):
    m = np.concatenate([[False], mask, [False]]).astype(np.int8); dd = np.diff(m)
    st, en = np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1
    return [((tt[a] - t0) / 86400, max((tt[b] - tt[a]) / 86400, 1 / 1440)) for a, b in zip(st, en)]
def f_timeline(T):
    fig, ax = plt.subplots(figsize=(11, 9.5))
    for y, sid in enumerate(order):
        d = data[sid]; tt = d.Timestamp.astype("int64").to_numpy() / 1e9; t0 = tt[0]
        ax.broken_barh(seg(d["Libre GL"].notna().to_numpy(), t0, tt), (y - 0.38, 0.32), color=T["s"][0], lw=0)
        ax.broken_barh(seg(d["Dexcom GL"].notna().to_numpy(), t0, tt), (y + 0.02, 0.32), color=T["s"][1], lw=0)
        mt = (d.Timestamp[d.meal.notna()].astype("int64").to_numpy() / 1e9 - t0) / 86400
        ax.plot(mt, np.full(len(mt), y), "|", color=T["ink"], ms=5, mew=0.9)
    ax.set_yticks(range(len(order)), [f"{s} · {grp[s][:4] if grp[s] != 'Prediabetes' else 'Pre'}" for s in order], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("days since first record"); ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color=T["s"][0], label="Libre Pro present"), Patch(color=T["s"][1], label="Dexcom G6 Pro present"),
                       Line2D([], [], marker="|", ls="", color=T["ink"], label="logged meal")], loc="lower right", ncol=3)
    ax.set_title("Per-participant recording timeline (sorted by group, then HbA1c)")
    return fig
save_fig(f_timeline, "cg_timeline", FIG)

# --- F2 sampling artefact + sensor agreement
def f_sensors(T):
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.42, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    d = data["001"]; w = d[(d.Timestamp >= "2020-05-03 08:30") & (d.Timestamp < "2020-05-03 11:30")]
    ax.plot(w.Timestamp, w["Libre GL"], color=T["s"][0], lw=1.6, label="Libre (1-min rows)")
    knots = w[w.Timestamp.dt.minute % 15 == 0]
    ax.plot(knots.Timestamp, knots["Libre GL"], "o", color=T["s"][0], ms=6, mec=T["panel"], mew=1.5, label="Libre native 15-min reading")
    ax.plot(w.Timestamp, w["Dexcom GL"], color=T["s"][1], lw=1.6, label="Dexcom (1-min rows)")
    k5 = w[w.Timestamp.dt.minute % 5 == 0]
    ax.plot(k5.Timestamp, k5["Dexcom GL"], "o", color=T["s"][1], ms=4.5, mec=T["panel"], mew=1.2, label="Dexcom 5-min grid")
    ax.axvline(pd.Timestamp("2020-05-03 09:07"), color=T["ref"], ls=":", lw=1); ax.text(pd.Timestamp("2020-05-03 09:08"), 117, "breakfast", fontsize=9, color=T["ink2"])
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); ax.set_ylabel("glucose (mg/dL)")
    ax.legend(ncol=2, loc="upper right", fontsize=8.5); ax.set_title("Both sensors are linearly interpolated to 1-min rows (participant 001)")
    ax = fig.add_subplot(gs[1, 0])
    ax.hexbin(P["Libre GL"], P["Dexcom GL"], gridsize=60, bins="log", cmap=LinearSegmentedColormap.from_list("q", [T["panel"]] + T["seq"]), mincnt=1, extent=(40, 400, 40, 400))
    ax.plot([40, 400], [40, 400], color=T["ref"], lw=1, ls="--"); ax.set_xlim(40, 400); ax.set_ylim(40, 400)
    ax.set_xlabel("Libre (mg/dL)"); ax.set_ylabel("Dexcom (mg/dL)"); ax.set_title(f"Paired readings, r = {S['agreement']['r']:.2f}")
    ax = fig.add_subplot(gs[1, 1])
    ax.hexbin(mean_, diff, gridsize=60, bins="log", cmap=LinearSegmentedColormap.from_list("q", [T["panel"]] + T["seq"]), mincnt=1, extent=(40, 400, -120, 120))
    a = S["agreement"]
    for yv, ls in [(a["bias"], "-"), (a["loa_lo"], "--"), (a["loa_hi"], "--")]:
        ax.axhline(yv, color=T["ink2"], lw=1, ls=ls)
    ax.text(395, a["bias"] + 4, f"bias {a['bias']:+.1f}", ha="right", fontsize=8.5, color=T["ink"])
    ax.text(395, a["loa_hi"] + 4, f"+1.96 SD {a['loa_hi']:+.0f}", ha="right", fontsize=8.5, color=T["ink2"])
    ax.text(395, a["loa_lo"] - 12, f"−1.96 SD {a['loa_lo']:+.0f}", ha="right", fontsize=8.5, color=T["ink2"])
    ax.set_ylim(-120, 120); ax.set_xlabel("mean of sensors (mg/dL)"); ax.set_ylabel("Dexcom − Libre (mg/dL)"); ax.set_title("Bland–Altman")
    ax = fig.add_subplot(gs[1, 2])
    sb = P.assign(dd=diff).groupby("sid").dd.mean().sort_values()
    ax.barh(range(len(sb)), sb.values, color=[GC(T)[grp[s]] for s in sb.index], height=0.75)
    ax.axvline(0, color=T["ink2"], lw=0.8); ax.set_yticks([]); ax.set_xlabel("mean Dexcom − Libre (mg/dL)")
    ax.set_title("Per-participant bias"); ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GC(T)[g], label=g) for g in GROUPS], fontsize=8, loc="lower right")
    return fig
save_fig(f_sensors, "cg_sensors", FIG)

# --- F3 glycaemic profile: mean glucose vs A1c, TIR stack, diurnal
def f_glyc(T):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1.35, 1.25], wspace=0.3))
    ax = axes[0]
    xs = np.linspace(4.5, 8.7, 10)
    ax.plot(xs, (xs - 3.31) / 0.02392, color=T["ref"], ls="--", lw=1.2, label="GMI formula")
    ax.scatter(Q.A1c, Q.mean_g_dex, s=40, color=T["s"][1], edgecolor=T["panel"], lw=1.4, label=f"Dexcom (r = {S['meanG_vs_A1c']['r_dexcom']:.2f})", zorder=3)
    ax.scatter(Q.A1c, Q.mean_g, s=40, color=T["s"][0], edgecolor=T["panel"], lw=1.4, label=f"Libre (r = {S['meanG_vs_A1c']['r_libre']:.2f})", zorder=3)
    ax.set_xlabel("lab HbA1c (%)"); ax.set_ylabel("participant mean glucose (mg/dL)")
    ax.set_title("Sensor mean vs HbA1c"); ax.legend(fontsize=8.5)
    ax = axes[1]
    q = Q.sort_values("A1c").reset_index(drop=True)
    lo, tir, hi = q.tbr_dex, q.tir_dex, q.tar_dex
    ax.bar(q.index, lo, color=T["s"][7], width=0.82, label="<70")
    ax.bar(q.index, tir, bottom=lo, color=T["s"][2], width=0.82, label="70–180")
    ax.bar(q.index, hi, bottom=lo + tir, color=T["s"][3], width=0.82, label=">180")
    for i, g in enumerate(q.group):
        ax.plot(i, -4, "s", color=GC(T)[g], ms=4.5, clip_on=False)
    ax.set_ylim(-7, 100); ax.set_xticks([]); ax.set_xlabel("participants, sorted by HbA1c →  (marker = group)")
    ax.set_ylabel("% of time (Dexcom)"); ax.set_title("Time in ranges"); ax.legend(ncol=3, loc="lower left", fontsize=8.5, bbox_to_anchor=(0, 1.0), borderaxespad=1.9)
    ax.grid(axis="x", visible=False)
    ax = axes[2]
    for g in GROUPS:
        prof = []
        for sid in Q[Q.group == g].sid:
            d = data[sid]; h = (d.Timestamp.dt.hour * 4 + d.Timestamp.dt.minute // 15)
            prof.append(d.groupby(h)["Dexcom GL"].median().reindex(range(96)).to_numpy())
        prof = np.array(prof); hx = np.arange(96) / 4
        ax.fill_between(hx, np.nanpercentile(prof, 25, 0), np.nanpercentile(prof, 75, 0), color=GC(T)[g], alpha=T["band_alpha"], lw=0)
        ax.plot(hx, np.nanmedian(prof, 0), color=GC(T)[g], label=g)
    ax.set_xticks([0, 6, 12, 18, 24], ["00", "06", "12", "18", "24"]); ax.set_xlabel("hour of day"); ax.set_ylabel("glucose (mg/dL)")
    ax.set_title("Diurnal Dexcom profile (median, IQR)"); ax.legend(fontsize=8.5, loc="upper left")
    return fig
save_fig(f_glyc, "cg_glycaemia", FIG)

# --- F4 meals: timing, macros, Atwater check
MT = ["breakfast", "lunch", "dinner", "snack"]
def f_meals(T):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    bins = np.arange(0, 24.5, 0.5)
    bottom = np.zeros(len(bins) - 1)
    for i, mt in enumerate(MT):
        h = np.histogram(M[M.meal == mt].hour, bins=bins)[0]
        ax.bar(bins[:-1] + 0.25, h, bottom=bottom, width=0.5, color=T["s"][i], label=f"{mt} ({(M.meal == mt).sum()})", lw=0)
        bottom += h
    ax.set_xticks([0, 6, 12, 18, 24]); ax.set_xlabel("hour of meal start"); ax.set_ylabel("meals"); ax.set_title(f"Meal timing ({len(M)} logged meals)")
    ax.legend(fontsize=8.5); ax.grid(axis="x", visible=False)
    ax = axes[1]
    macro = ["Carbs", "Protein", "Fat", "Fiber"]
    pos = 0; ticks = []
    for j, mc in enumerate(macro):
        for i, mt in enumerate(MT):
            x = M.loc[M.meal == mt, mc].clip(upper=200)
            bp = ax.boxplot(x, positions=[pos], widths=0.7, showfliers=False, patch_artist=True,
                            boxprops=dict(facecolor=T["s"][i], edgecolor=T["s"][i], alpha=0.85), medianprops=dict(color=T["panel"], lw=1.6),
                            whiskerprops=dict(color=T["ink2"]), capprops=dict(color=T["ink2"]))
            pos += 1
        ticks.append(pos - 2.5); pos += 1
    ax.set_xticks(ticks, [m + " (g)" for m in macro]); ax.set_ylabel("grams per meal"); ax.set_title("Macronutrients by meal type")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=T["s"][i], label=mt) for i, mt in enumerate(MT)], fontsize=8.5, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax = axes[2]
    lim = 1800
    ok = (np.abs(np.log(M.atw_ratio.clip(lower=1e-6))) < np.log(1.25)).to_numpy()
    ax.scatter(M.atwater[ok], M.Calories[ok], s=9, color=T["s"][0], alpha=0.45, lw=0, label=f"within ±25% ({S['atwater_within_25pct'] * 100:.0f}% of meals with macros)")
    ax.scatter(M.atwater[~ok], M.Calories[~ok], s=14, color=T["s"][7], alpha=0.8, lw=0, label="outside ±25%")
    ax.plot([0, lim], [0, lim], color=T["ref"], ls="--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel("4·carbs + 4·protein + 9·fat (kcal)"); ax.set_ylabel("logged calories (kcal)")
    ax.set_title("Atwater consistency check"); ax.legend(fontsize=8.5, loc="upper left")
    return fig
save_fig(f_meals, "cg_meals", FIG)

# --- F5 breakfast shake curves: one panel per variant, one line per group
def f_shakes(T):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.32, wspace=0.08))
    bidx = np.flatnonzero((M.meal == "breakfast").to_numpy() & M.iauc120.notna().to_numpy())
    for ax, v in zip(axes.flat, VAR_ORDER):
        for g in GROUPS:
            sel = [i for i in bidx if M.variant.iat[i] == v and M.group.iat[i] == g]
            if not sel: continue
            A = C[sel]
            ax.fill_between(WIN, np.nanpercentile(A, 25, 0), np.nanpercentile(A, 75, 0), color=GC(T)[g], alpha=T["band_alpha"], lw=0)
            ax.plot(WIN, np.nanmedian(A, 0), color=GC(T)[g], label=f"{g} (n={len(sel)})")
        ax.axvline(0, color=T["ref"], ls=":", lw=1); ax.axhline(0, color=T["ref"], lw=0.8)
        c_, p_, f_, fb_ = [float(t[2:] if t.startswith("Fb") else t[1:]) for t in v.split()]
        kc = 4 * c_ + 4 * p_ + 9 * f_
        ax.set_title(f"{VAR_NAME[v]}\ncarb {c_:g} · protein {p_:g} · fat {f_:g} · fiber {fb_:g} g  (≈{kc:.0f} kcal)", fontsize=10)
        ax.legend(fontsize=7.8, loc="upper right")
    for ax in axes[1]: ax.set_xlabel("minutes after shake")
    for ax in axes[:, 0]: ax.set_ylabel("Δ glucose (mg/dL)")
    fig.suptitle("Standardized breakfast shakes: postprandial Libre response (median, IQR)", x=0.125, ha="left", fontsize=12.5, fontweight="bold", color=T["ink"], y=0.995)
    return fig
save_fig(f_shakes, "cg_shakes", FIG)

# --- F6 within-person effect of each nutrient change (forest) + repeatability
def f_effects(T):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.28))
    ax = axes[0]
    names = [VAR_NAME[v] for v in VAR_ORDER if v != "C66 P22 F10.5 Fb0"]
    for k, nm in enumerate(names):
        for j, g in enumerate(GROUPS):
            e = E[(E.name == nm) & (E.group == g)]
            if e.empty: continue
            y = k + (j - 1) * 0.22
            ax.plot([e.lo.iat[0], e.hi.iat[0]], [y, y], color=GC(T)[g], lw=2)
            ax.plot(e.mean_diff.iat[0], y, "o", color=GC(T)[g], ms=7, mec=T["panel"], mew=1.5, label=g if k == 0 else None)
    ax.axvline(0, color=T["ink2"], lw=1)
    ax.set_yticks(range(len(names)), names); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ iAUC 0–120 min vs reference shake (mg/dL·min), within-person mean ± 95% CI")
    ax.set_title("Effect of changing one nutrient (reference = 66 g carb, 22 g protein, 10.5 g fat)"); ax.legend(fontsize=8.5, loc="lower right")
    ax = axes[1]
    v = "C66 P22 F10.5 Fb0"
    x = B[B.variant == v].groupby("sid").iauc120.apply(list)
    x = x[x.map(len) >= 2].map(lambda a: a[:2])
    mm = x.map(np.mean).sort_values()
    for i, (sid, m_) in enumerate(mm.items()):
        a = x[sid]
        ax.plot([i, i], [min(a), max(a)], color=T["grid"], lw=2.2, zorder=1)
        ax.plot([i, i], a, "o", color=GC(T)[grp[sid]], ms=5, mec=T["panel"], mew=1, zorder=2)
    ax.set_xticks([]); ax.set_xlabel("participants, sorted by mean response")
    ax.set_ylabel("iAUC 0–120 min (mg/dL·min)")
    ic = [i for i in S["shake_icc"] if i["variant"] == v][0]["icc"]
    ax.set_title(f"Same shake, two days: repeatability (ICC = {ic:.2f})"); ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=GC(T)[g], label=g) for g in GROUPS], fontsize=8.5, loc="upper left")
    return fig
save_fig(f_effects, "cg_effects", FIG)

# --- F7 lunches + iAUC vs carbs
def f_lunch(T):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.25))
    ax = axes[0]
    L_ = M[(M.meal == "lunch") & M.iauc120.notna()].copy()
    L_["lv"] = L_.apply(lambda r: f"{r.Calories:.0f} kcal · C{r.Carbs:g}/P{r.Protein:g}/F{r.Fat:g}", axis=1)
    top = L_.lv.value_counts(); top = top[top >= 30].index
    L_ = L_[L_.lv.isin(top)]
    order_l = L_.groupby("lv").Carbs.first().sort_values().index.tolist()
    for k, lv in enumerate(order_l):
        for j, g in enumerate(GROUPS):
            x = L_[(L_.lv == lv) & (L_.group == g)].iauc120
            if len(x) == 0: continue
            y = k + (j - 1) * 0.24
            ax.plot([x.quantile(.25), x.quantile(.75)], [y, y], color=GC(T)[g], lw=2.2)
            ax.plot(x.median(), y, "o", color=GC(T)[g], ms=6.5, mec=T["panel"], mew=1.3, label=g if k == 0 else None)
    ax.set_yticks(range(len(order_l)), order_l, fontsize=8.6); ax.grid(axis="y", visible=False)
    ax.set_xlabel("iAUC 0–120 min (mg/dL·min), median and IQR"); ax.set_title("The 10 standardized lunches (sorted by carbs)"); ax.legend(fontsize=8.5, loc="lower right")
    ax = axes[1]
    for g in GROUPS:
        q = fl[fl.group == g]
        ax.scatter(q.Carbs.clip(upper=150), q.iauc120, s=11, color=GC(T)[g], alpha=0.45, lw=0)
        qs = q.sort_values("Carbs"); k = np.array_split(np.arange(len(qs)), 6)
        ax.plot([qs.Carbs.iloc[i].median() for i in k], [qs.iauc120.iloc[i].median() for i in k], color=GC(T)[g], lw=2.4, marker="o", ms=5,
                mec=T["panel"], label=f"{g} (n={len(q)}, ρ = {S['free_iauc_vs_carbs_spearman'][g]:.2f})")
    ax.set_xlabel("carbohydrate in meal (g)"); ax.set_ylabel("iAUC 0–120 min (mg/dL·min)"); ax.set_ylim(0, np.nanpercentile(fl.iauc120, 98))
    ax.set_title("Free-living dinners & snacks (line = medians of 6 equal-count bins)"); ax.legend(fontsize=8.5, loc="upper left")
    return fig
save_fig(f_lunch, "cg_lunch_carbs", FIG)

# --- F8 cohort + correlation heatmap
def f_cohort(T):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.25))
    ax = axes[0]
    for g in GROUPS:
        b = bio[bio.group == g]
        ax.scatter(b.BMI, b.HOMA_IR, s=46, color=GC(T)[g], edgecolor=T["panel"], lw=1.5, label=f"{g} (n={len(b)})", zorder=3)
    ax.set_yscale("log"); ax.set_xlabel("BMI (kg/m²)"); ax.set_ylabel("HOMA-IR (log scale)"); ax.set_title("Insulin resistance vs BMI"); ax.legend(fontsize=8.5)
    ax = axes[1]
    cm_ = cgm_merge.rename(columns={"mean_g_dex": "CGM mean", "cv_dex": "CGM CV", "tir_dex": "TIR 70–180", "ref_iauc": "Shake iAUC", "HOMA_IR": "HOMA-IR",
                                     "FPG": "Fasting glucose", "Triglycerides": "TG", "Insulin": "Fasting insulin"})
    cols = ["Age", "BMI", "A1c", "Fasting glucose", "Fasting insulin", "HOMA-IR", "TG", "HDL", "LDL", "CGM mean", "CGM CV", "TIR 70–180", "Shake iAUC"]
    R = cm_[cols].corr(method="spearman").to_numpy()
    cmap = LinearSegmentedColormap.from_list("div", [T["s"][7], T["grid"], T["s"][0]])
    im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=50, ha="right", fontsize=8.6); ax.set_yticks(range(len(cols)), cols, fontsize=8.6)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and abs(R[i, j]) >= 0.4:
                ax.text(j, i, f"{R[i, j]:.1f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=7, color=T["panel"] if abs(R[i, j]) > 0.6 else T["ink"])
    ax.set_title("Spearman correlations (|ρ| ≥ 0.4 labelled)")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(colors=T["ink2"])
    return fig
save_fig(f_cohort, "cg_cohort", FIG)

# --- F9 gut + microbiome
def f_gut(T):
    fig = plt.figure(figsize=(14.5, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], hspace=0.45, wspace=0.32)
    ax = fig.add_subplot(gs[:, 0])
    gg = gut[gcols]
    cnt = pd.DataFrame({k: (gg == k).sum() for k in (1, 2, 3)})
    cnt = cnt.loc[cnt[3].sub(cnt[1]).sort_values().index]
    left = np.zeros(len(cnt))
    for k, lab, col in [(1, "Not optimal", T["s"][7]), (2, "Average", T["ref"]), (3, "Good", T["s"][0])]:
        ax.barh(range(len(cnt)), cnt[k], left=left, color=col, height=0.75, label=lab); left += cnt[k].to_numpy()
    ax.set_yticks(range(len(cnt)), [c if len(c) < 34 else c[:32] + "…" for c in cnt.index], fontsize=8)
    ax.set_xlabel("participants"); ax.set_title("Viome gut-health scores (22 tests)"); ax.legend(fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=1.8)
    ax.grid(axis="y", visible=False)
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(Xm.sum(0), bins=np.arange(-0.5, 46.5, 1), color=T["s"][0], lw=0)
    ax.set_yscale("log"); ax.set_xlabel("participants carrying the taxon"); ax.set_ylabel("taxa (log)"); ax.set_title(f"Prevalence of {Xm.shape[1]} taxa")
    ax = fig.add_subplot(gs[1, 1])
    for j, g in enumerate(GROUPS):
        x = rich[[grp.get(s) == g for s in rich.index]]
        ax.boxplot(x, positions=[j], widths=0.55, showfliers=False, patch_artist=True, boxprops=dict(facecolor=T["panel"], edgecolor=GC(T)[g]),
                   medianprops=dict(color=GC(T)[g], lw=2), whiskerprops=dict(color=T["ink2"]), capprops=dict(color=T["ink2"]))
        ax.scatter(np.full(len(x), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(x)), x, s=14, color=GC(T)[g], zorder=3, lw=0)
    ax.set_xticks(range(3), GROUPS); ax.set_ylabel("taxa detected"); ax.set_title(f"Richness (Kruskal p = {S['richness_kw_p']:.2f})"); ax.grid(axis="x", visible=False)
    ax = fig.add_subplot(gs[0, 2])
    for b_, mk in [("IDs 001–018", "o"), ("IDs 019–049", "s")]:
        sel = batch == b_
        ax.scatter(pcoa[sel, 0], pcoa[sel, 1], s=34, marker=mk, color=T["s"][4] if mk == "o" else T["s"][6], edgecolor=T["panel"], lw=1.2, label=b_)
    ve = S["pcoa_var_expl"]
    ax.set_xlabel(f"PCoA 1 ({ve[0] * 100:.0f}%)"); ax.set_ylabel(f"PCoA 2 ({ve[1] * 100:.0f}%)")
    ax.set_title("Raw taxa: two ID batches"); ax.legend(fontsize=8, loc="upper right")
    ax = fig.add_subplot(gs[1, 2])
    for g in GROUPS:
        sel = gl == g
        ax.scatter(pcoa_s[sel, 0], pcoa_s[sel, 1], s=34, marker="o", color=GC(T)[g], edgecolor=T["panel"], lw=1.2, label=g)
    vs = S["microbe_batches"]["pcoa_species_var"]
    ax.set_xlabel(f"PCoA 1 ({vs[0] * 100:.0f}%)"); ax.set_ylabel(f"PCoA 2 ({vs[1] * 100:.0f}%)")
    ax.set_title(f"Collapsed to {S['microbe_batches']['n_species']} species"); ax.legend(fontsize=8, loc="best")
    return fig
save_fig(f_gut, "cg_gut", FIG)

# --- F10 meal photo samples (before / after)
def photo_time(fn):
    m = re.search(r"PHOTO-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)", fn)
    return pd.Timestamp(*map(int, m.groups())) if m else None
def after_photo(sid, before_rel):
    pdir = os.path.join(ROOT, f"CGMacros-{sid}", "photos")
    files = sorted(os.listdir(pdir)); b = os.path.basename(before_rel)
    if b not in files: return None
    i = files.index(b)
    meal_paths = set(os.path.basename(p) for p in data[sid].loc[data[sid].meal.notna(), "Image path"].dropna())
    if i + 1 < len(files):
        nxt = files[i + 1]; t0, t1 = photo_time(b), photo_time(nxt)
        if nxt not in meal_paths and t0 and t1 and 0 <= (t1 - t0).total_seconds() <= 7200:
            return os.path.join(pdir, nxt)
    return None
picks = []
want = [("breakfast", "011"), ("lunch", "041"), ("lunch", "048"), ("dinner", "003"), ("dinner", "008"), ("snack", "006")]
for mt, sid in want:
    d = data[sid]; m = d[(d.meal == mt) & d["Image path"].notna()]
    for _, r in m.iterrows():
        a = after_photo(sid, r["Image path"])
        if a:
            picks.append((sid, r, os.path.join(ROOT, f"CGMacros-{sid}", r["Image path"]), a)); break
thumbs = {p: ImageOps.fit(ImageOps.exif_transpose(Image.open(p)).convert("RGB"), (300, 400)) for _, _, b, a in picks for p in (b, a)}
def f_photos(T):
    fig, axes = plt.subplots(1, 12, figsize=(16, 2.75), gridspec_kw=dict(wspace=0.04))
    for k, (sid, r, b, a) in enumerate(picks):
        for jj, (p, tag) in enumerate([(b, "before"), (a, "after")]):
            ax = axes[2 * k + jj]; ax.imshow(thumbs[p]); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s_ in ax.spines.values(): s_.set_visible(False)
            ax.set_title(tag, fontsize=8, color=T["ink2"], pad=2, loc="center", fontweight="normal")
        axes[2 * k].text(1.03, -0.05, f"#{sid} {r.meal} · {r.Calories:.0f} kcal\nC{r.Carbs:g} P{r.Protein:g} F{r.Fat:g} g", transform=axes[2 * k].transAxes,
                         ha="center", va="top", fontsize=8, color=T["ink"])
    return fig
save_fig(f_photos, "cg_photos", FIG, dpi=120)
print("figures written")
