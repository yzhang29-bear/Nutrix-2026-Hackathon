"""Type 1 diabetes track (MetaboNet, Type1Diabetes.parquet): the challenge protocol on people with T1D. For every person with
at least 5 meals on days 1-7 and meals on days 8-10, predict the day 8-10 post-meal curves from logged carbs (no photo),
pre-meal CGM, insulin (meal bolus and timing, insulin on board, basal), heart rate where recorded, time of day and the
person's first-week profile. The population model may learn from everyone else's data (all days); the test people
contribute only their days 1-7.
usage: python -I t1d_day8.py <t1d_events.npz> <out dir>
"""
import sys, os, json
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pmrf.pipeline import metrics

EV, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
z = np.load(EV, allow_pickle=True)
ev = pd.DataFrame(z["meta"]); pre = z["pre"].astype(float); post = z["post"].astype(float)
g0 = pre[:, -1]
d = post - g0[:, None]
ev["G0"] = g0
for k in (15, 30, 45, 60, 75, 90, 105, 120):
    ev[f"d{k}"] = d[:, k // 5]
for k in (30, 60, 90, 120):
    ev[f"G{k}"] = g0 + d[:, k // 5]
ev["dpeak"] = np.nanmax(d, 1); ev["Gmax"] = g0 + ev.dpeak
ev["ttp"] = np.nanargmax(np.nan_to_num(d, nan=-1e9), 1) * 5.0
ev["iauc"] = np.trapz(np.clip(np.nan_to_num(d), 0, None), dx=5, axis=1)
ev["slope30"] = (g0 - pre[:, -7]) / 30; ev["slope60"] = (g0 - pre[:, -13]) / 60; ev["sd60"] = np.nanstd(pre[:, -13:], 1)
ev["gmean2h"] = np.nanmean(pre, 1)
ev["tod_sin"], ev["tod_cos"] = np.sin(2 * np.pi * ev.hour / 24), np.cos(2 * np.pi * ev.hour / 24)
ev["bolus_per10g"] = ev.meal_bolus / (ev.carbs.clip(lower=1) / 10)
for m in ["AID", "SAP", "MDI"]:
    ev[f"mod_{m}"] = (ev.modality == m).astype(float)
ev["pid"] = ev.key
ok_rows = np.isfinite(ev.G0) & np.isfinite(ev.dpeak) & np.isfinite(ev.d120)
ev, CURV = ev[ok_rows].reset_index(drop=True), d[ok_rows.to_numpy()]

hist_n = ev[ev.day <= 7].groupby("pid").size()
test_ids = hist_n[hist_n >= 5].index.intersection(ev[ev.day.between(8, 10)].pid.unique())
is_test_person = ev.pid.isin(test_ids)
TE = ev[is_test_person & ev.day.between(8, 10)]
H_test = ev[is_test_person & (ev.day <= 7)]                                  # the test people's first week
OTHERS = ev[~is_test_person]                                                 # everyone else, all days
print(f"test people {len(test_ids)}, test meals {len(TE)} (day 8 only: {int((TE.day == 8).sum())}), "
      f"their history meals {len(H_test)}, other people {OTHERS.pid.nunique()} with {len(OTHERS)} meals", flush=True)


def profile(df, hist):
    """first-week descriptors per person; history rows get leave-one-out values"""
    g = hist.groupby("pid")
    tot = pd.DataFrame(dict(s_dpeak=g.dpeak.sum(), s_iauc=g.iauc.sum(), s_g0=g.G0.sum(), n=g.size()))
    out = df.join(tot, on="pid")
    own = df.index.isin(hist.index)
    n = out.n - own
    with np.errstate(all="ignore"):
        out["p_n"] = n
        out["p_dpeak"] = (out.s_dpeak - np.where(own, df.dpeak, 0)) / n
        out["p_iauc"] = (out.s_iauc - np.where(own, df.iauc, 0)) / n
        out["p_g0"] = (out.s_g0 - np.where(own, df.G0, 0)) / n
    med = hist.assign(r=hist.dpeak / (hist.carbs.clip(lower=5) / 10), b=hist.bolus_per10g).groupby("pid")[["r", "b"]].median()
    out["p_rise_per10g"] = out.pid.map(med.r); out["p_bolus_per10g"] = out.pid.map(med.b)
    return out.drop(columns=["s_dpeak", "s_iauc", "s_g0", "n"])


# history for the profile: each person's days 1-7 (other people: their days 1-7 too, so the profile means the same thing)
HIST = ev[ev.day <= 7]
TR = pd.concat([profile(OTHERS, HIST[~HIST.pid.isin(test_ids)]), profile(H_test, H_test)])
TEp = profile(TE, H_test)
BASE = ["G0", "slope30", "slope60", "sd60", "gmean2h", "tod_sin", "tod_cos", "hour", "carbs", "prev_carbs_6h", "hr_pre", "age"]
INS = ["meal_bolus", "bolus_lead", "iob", "basal", "bolus_per10g", "mod_AID", "mod_SAP", "mod_MDI"]
PROF = ["p_n", "p_dpeak", "p_iauc", "p_g0", "p_rise_per10g", "p_bolus_per10g"]
SETS = {"Population model (carbs + CGM context)": BASE, "Population model + insulin": BASE + INS,
        "Personalized (+ first-week profile)": BASE + PROF, "Personalized + insulin": BASE + INS + PROF}
YS = ["d15", "d30", "d45", "d60", "d75", "d90", "d105", "d120", "dpeak", "ttp", "iauc"]
LOGT = {"dpeak": 40, "iauc": 300}
PRM = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=32)
rng = np.random.default_rng(0)
TRs = TR.sample(n=min(len(TR), 250000), random_state=0) if len(TR) > 250000 else TR


def fit_predict(feats, y):
    tr = TRs[np.isfinite(TRs[y])]
    yt = np.log(tr[y].clip(lower=-LOGT[y] + 1) + LOGT[y]) if y in LOGT else tr[y]
    m = lgb.LGBMRegressor(**PRM, random_state=0).fit(tr[feats], yt)
    p = m.predict(TEp[feats])
    return (np.exp(p) - LOGT[y] if y in LOGT else p), m


def outputs(P):
    g0 = TEp.G0.to_numpy()
    o = {f"G{k}": g0 + P[f"d{k}"] for k in (30, 60, 90, 120)}
    o.update(Gmax=g0 + P["dpeak"], dpeak=P["dpeak"], ttp=np.clip(P["ttp"], 0, 120), iauc=np.clip(P["iauc"], 0, None))
    return o


preds, models = {}, {}
for name, feats in SETS.items():
    P = {}
    for y in YS:
        P[y], models[(name, y)] = fit_predict(feats, y)
    preds[name] = P
    print(name, "done", flush=True)
pm = TR.groupby(TR.hour // 6)[YS].mean()                                     # population mean by time-of-day block
preds["Population mean (by time of day)"] = {y: pm[y].reindex(TEp.hour // 6).to_numpy() for y in YS}

subsets = {"all": np.ones(len(TEp), bool), "day 8": (TEp.day == 8).to_numpy()}
RES = pd.concat([metrics(TEp, outputs(P), subsets).assign(model=k) for k, P in preds.items()], ignore_index=True)
pids = TEp.pid.to_numpy(); up = np.unique(pids); pos = {p: np.flatnonzero(pids == p) for p in up}
boots = [np.concatenate([pos[p] for p in rng.choice(up, len(up))]) for _ in range(1000)]
best = outputs(preds["Personalized + insulin"])
paired = {}
for ref in ["Population model + insulin", "Personalized (+ first-week profile)", "Population mean (by time of day)"]:
    o = outputs(preds[ref])
    paired[ref] = {}
    for t in ["G60", "dpeak", "iauc"]:
        e = np.abs(o[t] - TEp[t].to_numpy()) - np.abs(best[t] - TEp[t].to_numpy())
        b = [np.nanmean(e[ix]) for ix in boots]
        paired[ref][t] = dict(gain=float(np.nanmean(e)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))
gp = TEp.assign(p=best["dpeak"]).groupby("pid")[["p", "dpeak"]].mean()
person_r = float(stats.pearsonr(gp.p, gp.dpeak)[0])
imp = pd.Series(models[("Personalized + insulin", "dpeak")].booster_.feature_importance("gain"), index=SETS["Personalized + insulin"])
imp = (imp / imp.sum()).sort_values(ascending=False)

# ---------------------------------------------------------------- figures
def curve(P, i):
    tk = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120])
    return PchipInterpolator(tk, np.r_[0, [P[f"d{k}"][i] for k in tk[1:]]])(np.arange(0, 121, 5))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
def interpolated_run(c, n=6):
    """True if the curve has n or more consecutive steps on one straight line (a CGM gap filled linearly)"""
    flat = np.abs(np.diff(c, 2)) < 1e-6
    run = 0
    for f in flat:
        run = run + 1 if f else 0
        if run >= n - 1:
            return True
    return False

clean = np.array([not interpolated_run(CURV[i]) for i in TEp.index])
pick = rng.choice(np.flatnonzero((TEp.day == 8).to_numpy() & clean), 8, replace=False)
fig, axs = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
for ax, i in zip(axs.ravel(), pick):
    r = TEp.iloc[i]; t = np.arange(0, 121, 5)
    ax.plot(t, r.G0 + CURV[TEp.index[i]], color="#222", lw=2, label="observed")
    ax.plot(t, r.G0 + curve(preds["Personalized + insulin"], i), color="#2b6cb0", lw=2, label="personalized + insulin")
    ax.plot(t, r.G0 + curve(preds["Population model (carbs + CGM context)"], i), color="#9aa5b1", lw=1.5, ls="--", label="population, carbs only")
    ax.set_title(f"{r.src}, {r.carbs:.0f} g carbs, bolus {r.meal_bolus:.1f} U", fontsize=9)
for ax in axs[1]: ax.set_xlabel("minutes after meal")
for ax in axs[:, 0]: ax.set_ylabel("glucose (mg/dL)")
axs[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Type 1 diabetes (MetaboNet): randomly drawn day-8 meals, predicted from days 1-7", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "t1d_day8_examples.png"), dpi=150); plt.close(fig)

M = RES[RES.subset == "all"].pivot_table(index="model", columns="target", values="mae")
order = ["Population mean (by time of day)", "Population model (carbs + CGM context)", "Population model + insulin", "Personalized (+ first-week profile)", "Personalized + insulin"]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, t, lab in zip(axs, ["G60", "dpeak", "iauc"], ["glucose at 60 min", "peak rise", "iAUC 0-120 min"]):
    v = M.loc[order, t]
    ax.scatter(v.values, range(len(order)), color=["#9aa5b1"] * 3 + ["#63b3ed", "#2b6cb0"], s=60, zorder=3)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order if t == "G60" else [""] * len(order), fontsize=9)
    ax.set_title(f"{lab}: mean absolute error", fontsize=10); ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "t1d_day8_models.png"), dpi=150); plt.close(fig)

RES.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
summ = dict(test_people=int(len(test_ids)), test_meals=int(len(TEp)), test_meals_day8=int((TEp.day == 8).sum()),
            history_meals_test_people=int(len(H_test)), other_people=int(OTHERS.pid.nunique()), training_meals=int(len(TRs)),
            sources=TEp.src.value_counts().to_dict(), modality=TEp.modality.value_counts().to_dict(),
            mae=M.round(2).to_dict("index"), paired_gain_of_personalized_insulin=paired, person_level_r_dpeak=person_r,
            dpeak_feature_importance=imp.round(3).to_dict())
json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
print(M.loc[order, ["G30", "G60", "G90", "G120", "dpeak", "ttp", "iauc"]].round(1).to_string())
print(json.dumps(paired, indent=1)); print("person-level r (dpeak):", round(person_r, 2)); print(imp.head(8).round(3).to_string())
