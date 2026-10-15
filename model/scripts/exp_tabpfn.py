"""TabPFN v2 (tabular foundation model) vs the final LightGBM, same features and rows, rolling inner validation
(history cut-off day 4, 5, 6; validate on CGMacros meals up to day 7). Reports LightGBM, TabPFN and their average.
Per-meal predictions are saved so later blends can reuse them.
usage: CUDA_VISIBLE_DEVICES=1 .venv-gpu/bin/python -I exp_tabpfn.py <CGMacros dir> <BIG IDEAs dir> <embeddings npz> <stage1 dir> <out dir>
"""
import sys, os, json, pickle, time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, torch
from tabpfn import TabPFNRegressor
from pmrf.data import load_cgmacros, load_bigideas
from pmrf.features import meal_table, person_profile
from pmrf.pipeline import photo_features, attach_embeddings
from pmrf.models.gbm import _fit_bag

DATA, BIG, EMBP, S1, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
FE, PRM, LOGT = list(st["FEATS"]), dict(st["params"]), dict(st.get("log_targets", {}))
YS = ["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"]
persons = {**load_cgmacros(DATA), **load_bigideas(BIG)}
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)


def prepare(hd):
    M, CURV, PRE, HRS = meal_table(persons, hd)
    M, EMB = attach_embeddings(M, EMBP, DATA)
    M["in_hist"] = M.day <= hd
    for mt in ["breakfast", "lunch", "dinner", "snack"]:
        M[f"is_{mt}"] = (M.meal_type == mt).astype(float)
    M, mest, pca, E = photo_features(M, EMB)
    return M.join(person_profile(persons, M, M.in_hist.to_numpy(), hd))


def lgbm(tr, te, y):
    off = LOGT.get(y)
    yt = np.log(tr[y] + off) if off is not None else tr[y]
    p = _fit_bag(tr[FE], yt, (0, 1, 2), params=PRM).predict(te[FE])
    return np.exp(p) - off if off is not None else p


def tabpfn(tr, te, y, n_est=8):
    m = TabPFNRegressor(device=dev, n_estimators=n_est, random_state=0, ignore_pretraining_limits=True)
    m.fit(tr[FE].to_numpy(np.float32), tr[y].to_numpy(np.float32))
    return m.predict(te[FE].to_numpy(np.float32), output_type="median")


rows, preds = [], []
for hd in (4, 5, 6):
    M = prepare(hd)
    cg, ph = ~M.pid.str.startswith("B"), M.emb_idx >= 0
    TR = M[(cg & M.in_hist & M.label & ph) | (~cg & M.label)]
    VA = M[cg & (M.day > hd) & (M.day <= 7) & M.label & ph]
    print(f"cut-off {hd}: train {len(TR)}, validate {len(VA)} [{time.time() - T0:.0f}s]", flush=True)
    for y in YS:
        t = time.time(); pl = lgbm(TR, VA, y); tl = time.time() - t
        t = time.time(); pt = tabpfn(TR, VA, y); tt = time.time() - t
        for name, p in [("LightGBM (current)", pl), ("TabPFN v2", pt), ("mean(LightGBM, TabPFN)", 0.5 * pl + 0.5 * pt)]:
            rows.append(dict(cutoff=hd, model=name, target=y, n=len(VA), mae=float(np.mean(np.abs(p - VA[y])))))
        preds.append(pd.DataFrame(dict(cutoff=hd, target=y, meal=VA.index, pid=VA.pid.to_numpy(), obs=VA[y].to_numpy(), lgbm=pl, tabpfn=pt)))
        print(f"  {y}: LightGBM {rows[-3]['mae']:.2f} ({tl:.0f}s)  TabPFN {rows[-2]['mae']:.2f} ({tt:.0f}s)  mean {rows[-1]['mae']:.2f}", flush=True)

R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "inner_validation.csv"), index=False)
pd.concat(preds).to_csv(os.path.join(OUT, "predictions.csv"), index=False)
W = R.groupby(["model", "target"]).apply(lambda g: np.average(g.mae, weights=g.n)).unstack()[YS]
pc = R.pivot_table(index=["model", "target"], columns="cutoff", values="mae")
wins = {m: {y: int((pc.loc[(m, y)] < pc.loc[("LightGBM (current)", y)]).sum()) for y in YS} for m in W.index}
summ = dict(mae=W.round(2).to_dict("index"), change_pct=((W / W.loc["LightGBM (current)"] - 1) * 100).round(1).to_dict("index"),
            cutoffs_better_than_lightgbm=wins)
json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(((W / W.loc["LightGBM (current)"] - 1) * 100).round(1).to_string()); print(json.dumps(wins))
print(f"done [{time.time() - T0:.0f}s]")
