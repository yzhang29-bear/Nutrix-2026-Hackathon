"""Pretrain the in-context transformer on T1D carb events, fine-tune on CGMacros days 1-7, evaluate on day 8+.
usage: python -I run_transformer.py <t1d_events.npz> <v1 results dir> <out dir> [pretrain_steps] [finetune_steps]
"""
import sys, os, json, time, pickle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, torch
from pmrf.models.icl_transformer import ICLTransformer, EpisodeData, numeric_block, train, predict, curve_to_targets, NUM_COLS, pinball, CURVE_SCALE
from pmrf.pipeline import metrics, to_challenge_outputs

EV, V1, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PRE_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
FT_STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(16); torch.manual_seed(0)
T0 = time.time()
KNOWN = [NUM_COLS.index(c) for c in ["k_carbs", "k_protein", "k_fat", "k_fiber"]]

# ------------------------------------------------------------------ T1D events -> tokens
z = np.load(EV, allow_pickle=True)
ev = pd.DataFrame(z["meta"]); pre_raw = z["pre"].astype(float); post_raw = z["post"].astype(float)
g0 = pre_raw[:, -1]
ev["k_carbs"] = ev.carbs
ev["tod_sin"], ev["tod_cos"] = np.sin(2 * np.pi * ev.hour / 24), np.cos(2 * np.pi * ev.hour / 24)
ev["G0"] = g0
ev["slope30"] = (g0 - pre_raw[:, -7]) / 30; ev["slope60"] = (g0 - pre_raw[:, -13]) / 60
ev["sd60"] = np.nanstd(pre_raw[:, -13:], 1)
ev["prev_carbs"] = ev.prev_carbs_6h
for c in ["hr_pre"]:
    ev[c] = np.nan                     # absolute HR without a resting reference; not comparable with CGMacros
t1 = EpisodeData(numeric_block(ev), pre_raw - g0[:, None], post_raw - g0[:, None], np.zeros((len(ev), 512), np.float32),
                 np.zeros(len(ev)), ev.key.to_numpy())
keys = ev.key.to_numpy()
uk = np.unique(keys); rng = np.random.default_rng(0)
val_keys = set(rng.choice(uk, max(20, len(uk) // 20), replace=False))
by_key = pd.Series(np.arange(len(ev))).groupby(keys).apply(lambda s: s.to_numpy()).to_dict()
is_val = np.array([k in val_keys for k in keys])
tr_q = np.flatnonzero(~is_val); va_q = rng.choice(np.flatnonzero(is_val), 3000, replace=False)
hist_t1 = lambda i: by_key[keys[i]][by_key[keys[i]] != i]
va_ctx = [rng.choice(hist_t1(i), min(32, len(hist_t1(i))), replace=False) for i in va_q]
print(f"T1D events {len(ev)}, subjects {len(uk)}, val subjects {len(val_keys)}  [{time.time() - T0:.0f}s]", flush=True)

def val_t1(model):
    p = predict(model, t1, va_q, va_ctx) / CURVE_SCALE
    y = t1.resp[va_q]; m = t1.resp_mask[va_q]
    return float(pinball(torch.tensor(p), y, m))

# ------------------------------------------------------------------ CGMacros tokens
A = pickle.load(open(os.path.join(V1, "arrays.pkl"), "rb"))
M, CURV, PRE, EMB = A["M"], A["curves"], A["pre"], A["emb"]
TE = pd.read_pickle(os.path.join(V1, "test_meals.pkl"))
cg = M.copy()
cg["k_carbs"], cg["k_protein"], cg["k_fat"], cg["k_fiber"] = cg.carbs, cg.protein, cg.fat, cg.fiber
for c in ["carbs", "protein", "fat", "fiber"]:
    cg[f"e_{c}"] = cg[f"est_{c}"]
cg["e_carbs_sd"] = cg.est_carbs_sd
for mt in ["breakfast", "lunch", "dinner", "snack"]:
    cg[f"mt_{mt}"] = cg[f"is_{mt}"]
resp = np.where(cg.label.to_numpy()[:, None], CURV, np.nan)
has_img = (cg.emb_idx.to_numpy() >= 0).astype(float)
cm = EpisodeData(numeric_block(cg), PRE, resp, EMB, has_img, cg.pid.to_numpy())
lab = cg.label.to_numpy() & (has_img > 0)
hist = cg.in_hist.to_numpy()
pid = cg.pid.to_numpy(); day = cg.day.to_numpy()
pool = {p: np.flatnonzero(lab & hist & (pid == p)) for p in np.unique(pid)}
early = {p: np.flatnonzero(lab & hist & (pid == p) & (day <= 5)) for p in np.unique(pid)}
ft_q = np.flatnonzero(lab & hist & (day <= 5))
ft_hist = lambda i: early[pid[i]][early[pid[i]] != i]
va_q_cg = np.flatnonzero(lab & hist & (day >= 6))
va_ctx_cg = [early[pid[i]] for i in va_q_cg]
te_idx = TE.index.to_numpy()
te_ctx = [pool[pid[i]] for i in te_idx]

def val_cg(model):
    p = predict(model, cm, va_q_cg, va_ctx_cg, KNOWN)[:, :, 1]
    return float(np.nanmean(np.abs(p - CURV[va_q_cg])))

def evaluate(model, name):
    P = predict(model, cm, te_idx, te_ctx, KNOWN)
    out = curve_to_targets(P[:, :, 1], TE.G0.to_numpy())
    subsets = {"all": np.ones(len(TE), bool), "unseen": ~TE.seen.to_numpy()}
    cover = float(np.nanmean((CURV[te_idx] >= P[:, :, 0]) & (CURV[te_idx] <= P[:, :, 2])))
    return metrics(TE, out, subsets).assign(model=name), out, P, cover

results, outs = [], {}
# ------------------------------------------------------------------ (a) CGMacros only, from scratch
torch.manual_seed(0)
scratch = ICLTransformer(img_dim=EMB.shape[1])
scratch, h_s = train(scratch, cm, ft_q, ft_hist, FT_STEPS, lr=5e-4, batch=64, kmax=30, mask_query_known=KNOWN, val=val_cg, val_every=250, log_every=0)
r, o, P_s, cov_s = evaluate(scratch, "transformer_scratch"); results.append(r); outs["transformer_scratch"] = o
print(f"scratch done, best val {min(v for _, _, v in h_s):.2f} [{time.time() - T0:.0f}s]", flush=True)

# ------------------------------------------------------------------ (b) pretrain on T1D, then fine-tune
torch.manual_seed(0)
model = ICLTransformer(img_dim=EMB.shape[1])
t = time.time()
model, h_p = train(model, t1, tr_q, hist_t1, PRE_STEPS, lr=5e-4, batch=128, kmax=32, mask_query_known=None, val=val_t1, val_every=1000, log_every=1000)
torch.save(model.state_dict(), os.path.join(OUT, "pretrained_t1d.pt"))
print(f"pretrain {PRE_STEPS} steps in {time.time() - t:.0f}s", flush=True)
model, h_f = train(model, cm, ft_q, ft_hist, FT_STEPS, lr=2e-4, batch=64, kmax=30, mask_query_known=KNOWN, val=val_cg, val_every=250, log_every=0)
r, o, P_p, cov_p = evaluate(model, "transformer_pretrained"); results.append(r); outs["transformer_pretrained"] = o
torch.save(model.state_dict(), os.path.join(OUT, "finetuned_cgmacros.pt"))
print(f"finetune done, best val {min(v for _, _, v in h_f):.2f} [{time.time() - T0:.0f}s]", flush=True)

# ------------------------------------------------------------------ ensemble with the gradient-boosted model
gbm = A["preds"]["personalized_gbm"]
ens = {k: (np.asarray(gbm[k]) + np.asarray(o[k])) / 2 for k in o}
subsets = {"all": np.ones(len(TE), bool), "unseen": ~TE.seen.to_numpy()}
results.append(metrics(TE, ens, subsets).assign(model="ensemble_gbm_transformer")); outs["ensemble_gbm_transformer"] = ens
RES = pd.concat(results, ignore_index=True)
RES.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
print(RES[RES.subset == "unseen"].pivot(index="model", columns="target", values="mae").round(1).to_string())
print(RES[RES.subset == "unseen"].pivot(index="model", columns="target", values="r").round(2).to_string())
json.dump(dict(curve_coverage80_scratch=cov_s, curve_coverage80_pretrained=cov_p, pretrain_hist=h_p, finetune_hist=h_f, scratch_hist=h_s),
          open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
pickle.dump(dict(outs=outs, P_pretrained=P_p, P_scratch=P_s, te_idx=te_idx), open(os.path.join(OUT, "preds.pkl"), "wb"))
print(f"done [{time.time() - T0:.0f}s]")
