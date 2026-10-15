"""Inner-validation check (days 1-5 -> 6-7): do the transformer's direct heads beat the values derived from its median curve?
usage: python -I exp_tf_heads.py <stage1 dir> <pretrained.pt> <out json>
"""
import sys, os, json, pickle, time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, torch
from pmrf.models.icl_transformer import (ICLTransformer, EpisodeData, numeric_block, train, predict, predict_scalars, curve_to_targets, NUM_COLS)

S1, PT, OUT = sys.argv[1:4]
torch.set_num_threads(16)
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
KNOWN = [NUM_COLS.index(c) for c in ["k_carbs", "k_protein", "k_fat", "k_fiber"]]
prep = st["inner"]; cg = prep["M"].copy()
cg["k_carbs"], cg["k_protein"], cg["k_fat"], cg["k_fiber"] = cg.carbs, cg.protein, cg.fat, cg.fiber
for c in ["carbs", "protein", "fat", "fiber"]:
    cg[f"e_{c}"] = cg[f"est_{c}"]
cg["e_carbs_sd"] = cg.est_carbs_sd
for mt in ["breakfast", "lunch", "dinner", "snack"]:
    cg[f"mt_{mt}"] = cg[f"is_{mt}"]
lab = cg.label.to_numpy()
resp = np.where(lab[:, None], prep["CURV"], np.nan)
scal = np.where(lab[:, None], cg[["dpeak", "iauc", "ttp"]].to_numpy(float), np.nan)
has = (cg.emb_idx.to_numpy() >= 0).astype(float)
ed = EpisodeData(numeric_block(cg), prep["PRE"], resp, prep["E"], has, cg.pid.to_numpy(), scalars=scal)
pid, day = cg.pid.to_numpy(), cg.day.to_numpy()
okm = lab & (has > 0)
L5 = {p: np.flatnonzero(okm & (pid == p) & (day <= 5)) for p in np.unique(pid)}
q = np.concatenate(list(L5.values()))
va = st["VAi"].index.to_numpy(); ctx = [L5[pid[i]] for i in va]
VA = st["VAi"]
res = {}
for w in (0.0, 1.0):
    curves_all, sc_all = [], []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        m = ICLTransformer(img_dim=prep["E"].shape[1]); m.load_state_dict(torch.load(PT), strict=False)
        val = lambda mm: float(np.nanmean(np.abs(predict(mm, ed, va, ctx, KNOWN)[:, :, 1] - prep["CURV"][va])))
        m, h = train(m, ed, q, lambda i: L5[pid[i]][L5[pid[i]] != i], 1500, lr=1e-4, batch=64, kmax=30, mask_query_known=KNOWN,
                     val=val, val_every=50, log_every=0, seed=seed, scalar_weight=w)
        curves_all.append(predict(m, ed, va, ctx, KNOWN)); sc_all.append(predict_scalars(m, ed, va, ctx, KNOWN))
    P = np.mean(curves_all, 0); Sc = np.mean(sc_all, 0)
    ct = curve_to_targets(P[:, :, 1], VA.G0.to_numpy())
    r = {f"curve:{t}": float(np.mean(np.abs(ct[t] - VA[t]))) for t in ["G60", "dpeak", "iauc", "ttp"]}
    if w > 0:
        r.update({f"head:{t}": float(np.mean(np.abs(Sc[:, j, 1] - VA[t]))) for j, t in enumerate(["dpeak", "iauc", "ttp"])})
    r["curve_mae_all_points"] = float(np.nanmean(np.abs(P[:, :, 1] - prep["CURV"][va])))
    res[f"scalar_weight={w}"] = r
    print(w, json.dumps({k: round(v, 2) for k, v in r.items()}), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
