"""Final pipeline, stage 1: features for the inner split (history = days 1-5, validation = days 6-7) and the outer split
(history = days 1-7, test = day 8+); LightGBM configuration chosen on the inner validation only; final bagged model.
usage: python -I final_gbm.py <CGMacros dir> <embedding npz> <out dir> [BIG IDEAs dir]
v4: BIG IDEAs participants join the training data (all their days; no photos), and Δpeak / iAUC are modelled on a log scale
(offsets 40 and 300, chosen on rolling inner validation in scripts/exp_accuracy*.py).
"""
import sys, os, json, time, pickle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd
from pmrf.data import load_cgmacros, load_bigideas
from pmrf.features import person_profile
from pmrf.pipeline import build, photo_features, feature_sets, to_challenge_outputs, metrics
from pmrf.models.gbm import GBMResponse, _fit_bag

DATA, EMBP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
BIG = sys.argv[4] if len(sys.argv) > 4 else None
LOGT = {"dpeak": 40, "iauc": 300} if BIG else {}
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
persons = load_cgmacros(DATA)
if BIG:
    persons.update(load_bigideas(BIG))

def prepare(hd):
    M, CURV, PRE, HRS, EMB = build(persons, DATA, EMBP, hd)
    M, mest, pca, E = photo_features(M, EMB)
    M = M.join(person_profile(persons, M, M.in_hist.to_numpy(), hd))
    M["source"] = np.where(M.pid.str.startswith("B"), "bigideas", "cgmacros")
    return dict(M=M, CURV=CURV, PRE=PRE, HRS=HRS, E=E, mest=mest, pca=pca)

inner, outer = prepare(5), prepare(7)
print(f"prepared [{time.time() - T0:.0f}s]", flush=True)
Mi, Mo = inner["M"], outer["M"]
ph_i, ph_o = Mi.emb_idx >= 0, Mo.emb_idx >= 0
cg_i, cg_o = Mi.source == "cgmacros", Mo.source == "cgmacros"
# extra people (BIG IDEAs) are never validated or tested on, so all their labelled meals are training data
TRi = Mi[(cg_i & Mi.in_hist & Mi.label & ph_i) | (~cg_i & Mi.label)]; VAi = Mi[cg_i & Mi.day.isin([6, 7]) & Mi.label & ph_i]
TR = Mo[(cg_o & Mo.in_hist & Mo.label & ph_o) | (~cg_o & Mo.label)].copy(); TE = Mo[cg_o & (~Mo.in_hist) & Mo.label & ph_o].copy()
seen = TR[TR.source == "cgmacros"].assign(rec=TR.meal_type + TR.carbs.astype(str) + TR.protein.astype(str) + TR.fat.astype(str)).groupby("pid").rec.apply(set)
TE["seen"] = [(r.meal_type in ("breakfast", "lunch")) and (r.meal_type + str(r.carbs) + str(r.protein) + str(r.fat)) in seen.get(r.pid, set())
              for r in TE.itertuples()]
print(f"inner train {len(TRi)} val {len(VAi)} | outer train {len(TR)} test {len(TE)}", flush=True)

# ------------------------------------------------------------------ choose the configuration on the inner validation
FEATS = feature_sets()["personalized"]
YS = ["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"]
TRi_cg = TRi[TRi.source == "cgmacros"]
ref = {y: np.mean(np.abs(VAi[y] - TRi_cg.groupby("meal_type")[y].mean().reindex(VAi.meal_type).to_numpy())) for y in YS}
GRID = {"A": dict(num_leaves=7, n_estimators=300, min_child_samples=20),
        "B": dict(num_leaves=15, n_estimators=400, min_child_samples=20),
        "C": dict(num_leaves=15, n_estimators=800, learning_rate=0.02, min_child_samples=30),
        "D": dict(num_leaves=31, n_estimators=400, min_child_samples=40),
        "E": dict(num_leaves=7, n_estimators=800, learning_rate=0.02, min_child_samples=10),
        "F": dict(num_leaves=15, n_estimators=400, min_child_samples=20, colsample_bytree=0.5)}
scores = {}
for name, prm in GRID.items():
    sc = []
    m = GBMResponse(FEATS, targets=YS, personal_offset=False, params=prm, log_targets=LOGT, quantiles=(0.1, 0.9))
    for y in YS:
        if y in LOGT:
            yt = np.log(TRi[y] + LOGT[y]); p = np.exp(_fit_bag(TRi[FEATS], yt, (0,), params=prm).predict(VAi[FEATS])) - LOGT[y]
        else:
            p = _fit_bag(TRi[FEATS], TRi[y], (0,), params=prm).predict(VAi[FEATS])
        sc.append(np.mean(np.abs(p - VAi[y])) / ref[y])
    scores[name] = float(np.mean(sc))
    print(f"config {name}: relative val MAE {scores[name]:.4f}", flush=True)
best = min(scores, key=scores.get)
# personal offset on/off, also decided on the inner validation
off_scores = {}
for off in (False, True):
    for lp in (["median", "mean"] if LOGT else ["median"]):
        m = GBMResponse(FEATS, targets=YS, personal_offset=off, params=GRID[best], log_targets=LOGT, log_point=lp).fit(TRi)
        P = m.predict(VAi)
        off_scores[(off, lp)] = float(np.mean([np.mean(np.abs(P[y]["pred"] - VAi[y])) / ref[y] for y in YS]))
use_off, log_point = min(off_scores, key=off_scores.get)
print(f"best config {best}, personal offset {use_off}, log point {log_point} ({off_scores})", flush=True)

# inner-validation predictions of the chosen model (for blend weights later)
m_in = GBMResponse(FEATS, personal_offset=use_off, params=GRID[best], seeds=(0, 1, 2), log_targets=LOGT, log_point=log_point).fit(TRi)
P_val = m_in.predict(VAi)
# final model on days 1-7, applied once to day 8+
m_out = GBMResponse(FEATS, personal_offset=use_off, params=GRID[best], seeds=(0, 1, 2), log_targets=LOGT, log_point=log_point).fit(TR)
P_te = m_out.predict(TE, contributions=True)
subsets = {"all": np.ones(len(TE), bool), "unseen": ~TE.seen.to_numpy()}
R = metrics(TE, to_challenge_outputs(TE, P_te), subsets).assign(model="final_gbm")
print(R[R.subset == "unseen"].set_index("target")[["mae", "r"]].round(2).T.to_string(), flush=True)
pickle.dump(dict(inner=inner, outer=outer, TRi=TRi, VAi=VAi, TR=TR, TE=TE, FEATS=FEATS, config=best, params=GRID[best], use_offset=use_off,
                 grid_scores=scores, offset_scores={str(k): v for k, v in off_scores.items()}, log_targets=LOGT, log_point=log_point, P_val=P_val, P_te=P_te, gbm_final=m_out), open(os.path.join(OUT, "stage1.pkl"), "wb"))
print(f"stage 1 done [{time.time() - T0:.0f}s]")
