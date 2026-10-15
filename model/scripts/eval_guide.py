"""Check the extra outputs on day-8+ meals: (1) are the high-peak probabilities calibrated, (2) does the personal carb budget
separate meals that stay at or below 140 mg/dL from those that do not?
usage: python -I eval_guide.py <stage1 dir> <blend dir> <demo dir with bundle.pkl> <out json>
"""
import sys, os, json, pickle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, brier_score_loss
from pmrf.guide import exceed_prob, person_guide, person_guide_measured
from pmrf.predict import Predictor

S1, BL, DEMO, OUT = sys.argv[1:5]
torch.set_num_threads(8)
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
fp = pickle.load(open(os.path.join(BL, "final_preds.pkl"), "rb"))
TE = st["TE"].loc[fp["te_index"]]
res = {}
out = dict(curve=fp["curve"], band=(fp["band_lo"], fp["band_hi"]), dpeak=fp["blend"]["dpeak"], ttp=fp["blend"]["ttp"])
P = pd.DataFrame([exceed_prob(g0, out, i) for i, g0 in enumerate(TE.G0.to_numpy())])
for t in ("140", "180"):
    y = (TE.Gmax.to_numpy() > float(t)).astype(int); p = P[t].to_numpy()
    bins = pd.cut(p, [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
    cal = pd.DataFrame(dict(p=p, y=y, b=bins)).groupby("b", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    res[f"exceed_{t}"] = dict(n=int(len(y)), prevalence=float(y.mean()), auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                              brier_baseline=float(brier_score_loss(y, np.full(len(y), y.mean()))),
                              calibration=[dict(bin=str(k), **{c: float(v) for c, v in r.items()}) for k, r in cal.iterrows()])
pred = Predictor(pickle.load(open(os.path.join(DEMO, "bundle.pkl"), "rb")))
EFFECTS = pickle.load(open(os.path.join(os.path.dirname(DEMO.rstrip("/")), "guide", "shake_effects.pkl"), "rb"))
budgets, guides = {}, {}
for pid in sorted(TE.pid.unique()):
    g = person_guide_measured(pred.b["person_state"][pid]["hist"], EFFECTS, pid) or person_guide(pred, pid)
    guides[pid] = g
    if g.get("available"):
        budgets[pid] = g["carb_budget_g_per_meal"]
T = TE.assign(budget=TE.pid.map(budgets)).dropna(subset=["budget"])
within = T.carbs <= T.budget
ok30 = T.dpeak <= 50
res["carb_budget"] = dict(people=int(T.pid.nunique()), meals=int(len(T)), median_budget_g=float(np.median(list(budgets.values()))),
                          meals_within_budget=int(within.sum()), target_rise=50, share_rise_le_target_within=float(ok30[within].mean()), share_rise_le_target_over=float(ok30[~within].mean()),
                          rise_within=float(T.dpeak[within].mean()), rise_over=float(T.dpeak[~within].mean()))
json.dump(res, open(OUT, "w"), indent=1)
pickle.dump(guides, open(os.path.join(os.path.dirname(OUT), "guides.pkl"), "wb"))
print(json.dumps({k: (v if k == "carb_budget" else {kk: vv for kk, vv in v.items() if kk != "calibration"}) for k, v in res.items()}, indent=1))
