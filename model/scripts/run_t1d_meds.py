"""Medication module check on T1D carb events: does insulin context (meal bolus, timing, insulin on board, basal) improve
day-8 predictions, with and without personalization? Same protocol: history = days 1-7, test = days 8-10.
usage: python -I run_t1d_meds.py <t1d_events.npz> <out dir>
"""
import sys, os, json
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats
from pmrf.pipeline import metrics

EV, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
z = np.load(EV, allow_pickle=True)
ev = pd.DataFrame(z["meta"]); pre = z["pre"].astype(float); post = z["post"].astype(float)
keep = ev.day.to_numpy() <= 10
ev, pre, post = ev[keep].reset_index(drop=True), pre[keep], post[keep]
g0 = pre[:, -1]; d = post - g0[:, None]
ev["G0"] = g0
for k, j in [(30, 6), (60, 12), (90, 18), (120, 24)]:
    ev[f"d{k}"] = d[:, j]; ev[f"G{k}"] = g0 + d[:, j]
ev["dpeak"] = np.nanmax(d, 1); ev["Gmax"] = g0 + ev.dpeak; ev["ttp"] = np.nanargmax(np.nan_to_num(d, nan=-1e9), 1) * 5.0
ev["iauc"] = np.trapz(np.clip(np.nan_to_num(d), 0, None), dx=5, axis=1)
ev["slope30"] = (g0 - pre[:, -7]) / 30; ev["slope60"] = (g0 - pre[:, -13]) / 60; ev["sd60"] = np.nanstd(pre[:, -13:], 1)
ev["tod_sin"], ev["tod_cos"] = np.sin(2 * np.pi * ev.hour / 24), np.cos(2 * np.pi * ev.hour / 24)
ev["bolus_per10g"] = ev.meal_bolus / (ev.carbs / 10)
for m in ["AID", "SAP", "MDI"]:
    ev[f"mod_{m}"] = (ev.modality == m).astype(float)
ev["pid"] = ev.key
H, T = ev[ev.day <= 7].copy(), ev[ev.day >= 8].copy()
ok = H.groupby("pid").size(); ok = ok[ok >= 5].index
H, T = H[H.pid.isin(ok)], T[T.pid.isin(ok)]

def profile(df, hist):
    g = hist.groupby("pid")
    s = pd.DataFrame(dict(p_dpeak=g.dpeak.mean(), p_iauc=g.iauc.mean(), p_n=g.size(), p_g0=g.G0.mean(),
                          p_dpeak_per10g=g.apply(lambda x: np.median(x.dpeak / (x.carbs / 10))),
                          p_bolus_per10g=g.apply(lambda x: np.nanmedian(x.bolus_per10g))))
    out = df.join(s, on="pid")
    if df is hist:   # leave-one-out for history rows
        n = out.p_n
        out["p_dpeak"] = (out.p_dpeak * n - out.dpeak) / (n - 1); out["p_iauc"] = (out.p_iauc * n - out.iauc) / (n - 1)
    return out
H, T = profile(H, H), profile(T, H)
BASE = ["carbs", "tod_sin", "tod_cos", "G0", "slope30", "slope60", "sd60", "prev_carbs_6h", "age", "mod_AID", "mod_SAP", "mod_MDI"]
INS = ["meal_bolus", "bolus_lead", "iob", "basal", "bolus_per10g"]
PERS = ["p_dpeak", "p_iauc", "p_n", "p_g0", "p_dpeak_per10g", "p_bolus_per10g"]
variants = {"population, no insulin": BASE, "population + insulin": BASE + INS, "personalized, no insulin": BASE + PERS,
            "personalized + insulin": BASE + INS + PERS}
res = []
subsets = {"all": np.ones(len(T), bool), "meal bolus given": (T.meal_bolus > 0).to_numpy(), "no meal bolus": (T.meal_bolus == 0).to_numpy()}
for name, feats in variants.items():
    out = {}
    for y in ["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"]:
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1, n_jobs=16, random_state=0).fit(H[feats], H[y])
        out[y] = m.predict(T[feats])
    o = {f"G{k}": T.G0.to_numpy() + out[f"d{k}"] for k in (30, 60, 90, 120)}
    o.update(Gmax=T.G0.to_numpy() + out["dpeak"], dpeak=out["dpeak"], ttp=np.clip(out["ttp"], 0, 120), iauc=np.clip(out["iauc"], 0, None))
    res.append(metrics(T, o, subsets).assign(model=name))
    print(name, "done", flush=True)
R = pd.concat(res, ignore_index=True)
R.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
info = dict(history_events=len(H), test_events=len(T), subjects=int(T.pid.nunique()), share_bolused=float((T.meal_bolus > 0).mean()),
            sources=T.src.value_counts().to_dict())
json.dump(info, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
print(info)
for sub in subsets:
    print("==", sub)
    print(R[R.subset == sub].pivot(index="model", columns="target", values="mae")[["G60", "G120", "dpeak", "iauc", "ttp"]].round(1).to_string())
    print(R[R.subset == sub].pivot(index="model", columns="target", values="r")[["G60", "dpeak", "iauc"]].round(2).to_string())
