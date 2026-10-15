"""Accuracy experiments for the LightGBM component, judged only on rolling inner validation
(history cut-offs at day 4, 5, 6; validate on CGMacros meals from the cut-off to day 7).
usage: python -I exp_accuracy.py <CGMacros dir> <BIG IDEAs dir> <embedding npz> <stage1 dir> <out dir>
"""
import sys, os, json, pickle, time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from pmrf.data import load_cgmacros, load_bigideas
from pmrf.features import person_profile
from pmrf.pipeline import build, photo_features
from pmrf.models.gbm import _fit_bag

DATA, BIG, EMBP, S1, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
FEATS, PRM = list(st["FEATS"]), dict(st["params"])
YS = ["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"]
persons = {**load_cgmacros(DATA), **load_bigideas(BIG)}

def prepare(hd):
    M, CURV, PRE, HRS, EMB = build(persons, DATA, EMBP, hd)
    M, mest, pca, E = photo_features(M, EMB)
    M = M.join(person_profile(persons, M, M.in_hist.to_numpy(), hd))
    M["source"] = np.where(M.pid.str.startswith("B"), "bigideas", "cgmacros")
    M["macro_from_log"] = 0.0
    return M

def fit_predict(tr, va, feats, y, params, seeds=(0, 1, 2), log=False):
    off = log if not isinstance(log, bool) else 100
    yt = np.log(tr[y] + off) if log else tr[y]
    m = _fit_bag(tr[feats], yt, seeds, params=params)
    p = m.predict(va[feats])
    if log:
        res = yt - m.predict(tr[feats])
        p = np.exp(p) * np.mean(np.exp(res)) - off
    return p

def oof(tr, feats, y, params):
    p = np.zeros(len(tr))
    for a, b in GroupKFold(5).split(tr, groups=tr.pid):
        p[b] = _fit_bag(tr.iloc[a][feats], tr.iloc[a][y], (0,), params=params).predict(tr.iloc[b][feats])
    return p

MONO_UP = {"est_carbs", "est_kcal", "carbs", "kcal"}
def mono_params(feats, y):
    if y not in ("dpeak", "iauc", "d30", "d60"):
        return PRM
    return dict(PRM, monotone_constraints=[1 if f in MONO_UP else 0 for f in feats], monotone_constraints_method="advanced")

rows = []
for hd in (4, 5, 6):
    M = prepare(hd)
    cg = M.source == "cgmacros"; big = M.source == "bigideas"
    TRc = M[cg & M.in_hist & M.label & (M.emb_idx >= 0)]
    VA = M[cg & (M.day > hd) & (M.day <= 7) & M.label & (M.emb_idx >= 0)]
    Bn = M[big & M.label].copy()
    Bl = Bn.copy()
    for c in ["carbs", "protein", "fat", "fiber"]:
        Bl[f"est_{c}"] = Bl[c]
    Bl["est_kcal"] = Bl.kcal; Bl["macro_from_log"] = 1.0
    ref = {y: np.mean(np.abs(VA[y] - TRc.groupby("meal_type")[y].mean().reindex(VA.meal_type).to_numpy())) for y in YS}
    print(f"cut-off day {hd}: train {len(TRc)} (+{len(Bn)} BIG IDEAs), validate {len(VA)} [{time.time() - T0:.0f}s]", flush=True)
    variants = {"base": (TRc, FEATS, None)}
    for off in (30, 100, 300, 1000):
        variants[f"log iAUC +{off}"] = (TRc, FEATS, {"iauc": off})
    for off in (10, 20, 40):
        variants[f"log dpeak +{off}"] = (TRc, FEATS, {"dpeak": off})
    variants["+ BIG IDEAs, log iAUC +300"] = (pd.concat([TRc, Bn]), FEATS, {"iauc": 300})
    for name, (tr, feats, logs) in variants.items():
        for y in (["dpeak", "iauc"] if name != "base" else YS):
            lg = (logs or {}).get(y, False)
            p = fit_predict(tr, VA, feats, y, PRM, log=lg)
            rows.append(dict(cutoff=hd, variant=name, target=y, mae=float(np.mean(np.abs(p - VA[y]))), rel=float(np.mean(np.abs(p - VA[y])) / ref[y]), n=len(VA)))
    continue
    # multiplicative personal calibration (scale = shrunken ratio observed / out-of-fold predicted on the person's history)
    for y in ["dpeak", "iauc", "d60"]:
        po = oof(TRc, FEATS, y, PRM); p = fit_predict(TRc, VA, FEATS, y, PRM)
        t = pd.DataFrame(dict(pid=TRc.pid.to_numpy(), o=TRc[y].to_numpy(), p=po)).groupby("pid").agg(o=("o", "sum"), p=("p", "sum"), n=("o", "size"))
        mean_p = max(np.mean(np.abs(po)), 1e-6)
        for c in (2, 5, 10, 20, 40):
            kappa = c * mean_p
            scale = ((t.o + kappa) / (t.p + kappa)).clip(0.5, 2.0)
            q = p * VA.pid.map(scale).fillna(1.0).to_numpy()
            rows.append(dict(cutoff=hd, variant=f"multiplicative calibration c={c}", target=y, mae=float(np.mean(np.abs(q - VA[y]))),
                             rel=float(np.mean(np.abs(q - VA[y])) / ref[y]), n=len(VA)))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "exp_accuracy2.csv"), index=False)
print(R.pivot_table(index="variant", columns=["target", "cutoff"], values="mae").round(1).to_string())
W = R.groupby(["variant", "target"]).apply(lambda g: np.average(g.mae, weights=g.n)).unstack()
print(W.round(2).to_string())
base = W.loc["base"]
print("\nchange vs base (%, weighted over cut-offs):")
print(((W / base - 1) * 100).round(1).to_string())
print(f"done [{time.time() - T0:.0f}s]")
