"""Final pipeline, stage 2 (v4: transformer with direct peak/iAUC/time-to-peak heads; for each outcome the source
(curve or head) and the blend weight with LightGBM are chosen on the inner validation, as is the head loss weight). fine-tune the T1D-pretrained in-context transformer (3 seeds) on the inner split to choose the step
count and blend weights, refit on days 1-7, blend with the stage-1 LightGBM, calibrate the curve band, evaluate once on day 8+.
usage: python -I final_blend.py <stage1 dir> <pretrained.pt> <chronos dir> <out dir> [max_finetune_steps] [lrs, e.g. 5e-5,1e-4,2e-4] [val_every]
"""
import sys, os, json, time, pickle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, torch
from scipy import stats
from scipy.interpolate import PchipInterpolator
from pmrf.models.icl_transformer import ICLTransformer, EpisodeData, numeric_block, train, predict, predict_scalars, curve_to_targets, NUM_COLS, NUM_COLS_BY_VERSION
COLS_VERSION = os.environ.get("PMRF_NUM_COLS", "v1")          # v2 = v1 + age, sex, BMI (needs a v2-pretrained checkpoint)
COLS = NUM_COLS_BY_VERSION[COLS_VERSION]
from pmrf.pipeline import to_challenge_outputs, metrics, enforce_consistency
from pmrf.features import TARGETS

S1, PRE_PT, CHR, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
MAX_STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
LRS = [float(x) for x in sys.argv[6].split(",")] if len(sys.argv) > 6 else [1e-4]
WS = [0.0, 1.0]                       # loss weight of the direct heads
VAL_EVERY = int(sys.argv[7]) if len(sys.argv) > 7 else 250
SEEDS = (0, 1, 2)
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(16)
T0 = time.time()
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
KNOWN = [NUM_COLS.index(c) for c in ["k_carbs", "k_protein", "k_fat", "k_fiber"]]

def episode_data(prep):
    cg = prep["M"].copy()
    cg["k_carbs"], cg["k_protein"], cg["k_fat"], cg["k_fiber"] = cg.carbs, cg.protein, cg.fat, cg.fiber
    for c in ["carbs", "protein", "fat", "fiber"]:
        cg[f"e_{c}"] = cg[f"est_{c}"]
    cg["e_carbs_sd"] = cg.est_carbs_sd
    for mt in ["breakfast", "lunch", "dinner", "snack"]:
        cg[f"mt_{mt}"] = cg[f"is_{mt}"]
    resp = np.where(cg.label.to_numpy()[:, None], prep["CURV"], np.nan)
    has = (cg.emb_idx.to_numpy() >= 0).astype(float)
    scal = np.where(cg.label.to_numpy()[:, None], cg[["dpeak", "iauc", "ttp"]].to_numpy(float), np.nan)
    return cg, EpisodeData(numeric_block(cg, COLS), prep["PRE"], resp, prep["E"], has, cg.pid.to_numpy(), scalars=scal)

def lib(cg, max_day):
    m = cg.label.to_numpy() & (cg.emb_idx.to_numpy() >= 0) & (cg.day.to_numpy() <= max_day)
    pid = cg.pid.to_numpy()
    return {p: np.flatnonzero(m & (pid == p)) for p in np.unique(pid)}

def fresh(seed):
    torch.manual_seed(seed)
    m = ICLTransformer(img_dim=st["outer"]["E"].shape[1], n_num=len(COLS))
    m.load_state_dict(torch.load(PRE_PT, map_location="cpu"), strict=False)
    return m

# ------------------------------------------------------------------ inner: queries days 1-5, validate on days 6-7 (context = days 1-5)
cg_i, ed_i = episode_data(st["inner"])
L5 = lib(cg_i, 5); pid_i = cg_i.pid.to_numpy()
q_i = np.concatenate(list(L5.values()))
va_i = st["VAi"].index.to_numpy()
ctx_va = [L5[pid_i[i]] for i in va_i]
CURV_i = st["inner"]["CURV"]
val_fn = lambda mm: float(np.nanmean(np.abs(predict(mm, ed_i, va_i, ctx_va, KNOWN)[:, :, 1] - CURV_i[va_i])))
zero_shot_val = val_fn(fresh(0))
print(f"pretrained only (no fine-tuning): val curve MAE {zero_shot_val:.2f}", flush=True)
VAi = st["VAi"]
gb_val = to_challenge_outputs(VAi, st["P_val"])
SCAL_T = {"dpeak": 0, "iauc": 1, "ttp": 2}

def choose(Pv, Sv):
    """per outcome: transformer source (curve or head) and blend weight with LightGBM, chosen on the inner validation"""
    tfc = curve_to_targets(Pv[:, :, 1], VAi.G0.to_numpy())
    tfh = {}
    if Sv is not None:
        tfh = {"dpeak": Sv[:, 0, 1], "iauc": Sv[:, 1, 1], "ttp": Sv[:, 2, 1]}
        tfh["Gmax"] = VAi.G0.to_numpy() + tfh["dpeak"]
    src, W_, rel = {}, {}, []
    for t in TARGETS:
        best = None
        for sname, cand in [("curve", tfc[t])] + ([("head", tfh[t])] if (t in tfh and Sv is not None) else []):
            for w in np.round(np.arange(0, 1.01, 0.1), 1):
                e = np.mean(np.abs(w * cand + (1 - w) * gb_val[t] - VAi[t].to_numpy()))
                if best is None or e < best[0]:
                    best = (e, sname, float(w))
        src[t], W_[t] = best[1], best[2]
        rel.append(best[0] / np.mean(np.abs(gb_val[t] - VAi[t].to_numpy())))
    return src, W_, float(np.mean(rel))

lr_runs = {}
for lr in LRS:
    for wsc in WS:
        vc, vs, bsteps, bvals, hists = [], [], [], [], []
        for s in SEEDS:
            m = fresh(s)
            m, h = train(m, ed_i, q_i, lambda i: L5[pid_i[i]][L5[pid_i[i]] != i], MAX_STEPS, lr=lr, batch=64, kmax=30,
                         mask_query_known=KNOWN, val=val_fn, val_every=VAL_EVERY, log_every=0, seed=s, scalar_weight=wsc)
            bs = min(h, key=lambda x: x[2]); bsteps.append(bs[0]); bvals.append(bs[2]); hists.append(h)
            vc.append(predict(m, ed_i, va_i, ctx_va, KNOWN)); vs.append(predict_scalars(m, ed_i, va_i, ctx_va, KNOWN))
            print(f"inner lr {lr:g} head weight {wsc:g} seed {s}: best step {bs[0]}, val curve MAE {bs[2]:.2f} [{time.time() - T0:.0f}s]", flush=True)
        Pv_, Sv_ = np.mean(vc, 0), np.mean(vs, 0)
        src_, W__, score = choose(Pv_, Sv_ if wsc > 0 else None)
        lr_runs[(lr, wsc)] = dict(curves=vc, scal=vs, steps=bsteps, vals=bvals, hists=hists, score=score, src=src_, W=W__,
                                  ens_val=float(np.nanmean(np.abs(Pv_[:, :, 1] - CURV_i[va_i]))))
        print(f"lr {lr:g} head weight {wsc:g}: blended relative val error {score:.4f} (curve MAE {lr_runs[(lr, wsc)]['ens_val']:.2f}) sources {src_}", flush=True)
LR, WSC = min(lr_runs, key=lambda k: lr_runs[k]["score"])
Pv = np.mean(lr_runs[(LR, WSC)]["curves"], 0); Sv = np.mean(lr_runs[(LR, WSC)]["scal"], 0)
best_steps = lr_runs[(LR, WSC)]["steps"]
steps_final = int(np.median(best_steps))
SRC, W = lr_runs[(LR, WSC)]["src"], lr_runs[(LR, WSC)]["W"]

w_curve = float(np.mean([W[t] for t in ["G30", "G60", "G90", "G120"]]))
print("chosen:", dict(lr=LR, head_weight=WSC, sources=SRC, weights=W), flush=True)

def gbm_curve(P, n):
    """GBM change at 15-min steps -> 5-min curve (shape-preserving interpolation through 0 at t=0)"""
    tk = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120])
    pts = np.column_stack([np.zeros(n)] + [P[f"d{k}"]["pred"] for k in tk[1:]])
    return np.stack([PchipInterpolator(tk, row)(np.arange(0, 121, 5)) for row in pts])

# curve band: transformer quantile offsets around the blended median, scaled to 80% coverage on the inner validation
cb_val = w_curve * Pv[:, :, 1] + (1 - w_curve) * gbm_curve(st["P_val"], len(VAi))
lo_off, hi_off = Pv[:, :, 0] - Pv[:, :, 1], Pv[:, :, 2] - Pv[:, :, 1]
yv = CURV_i[va_i]
def cover(sc):
    ok = np.isfinite(yv)
    return np.mean(((yv >= cb_val + sc * lo_off) & (yv <= cb_val + sc * hi_off))[ok])
band_scale = float(min(np.arange(0.5, 3.01, 0.05), key=lambda sc: abs(cover(sc) - 0.8)))
print(f"band scale {band_scale:.2f} -> val coverage {cover(band_scale):.3f}", flush=True)

# ------------------------------------------------------------------ outer: fine-tune on days 1-7 with the chosen step count, predict day 8+
cg_o, ed_o = episode_data(st["outer"])
L7 = lib(cg_o, 7); pid_o = cg_o.pid.to_numpy()
q_o = np.concatenate(list(L7.values()))
TE = st["TE"]; te = TE.index.to_numpy()
ctx_te = [L7[pid_o[i]] for i in te]
te_curves, te_scal, models = [], [], []
for s in SEEDS:
    m = fresh(s)
    m, _ = train(m, ed_o, q_o, lambda i: L7[pid_o[i]][L7[pid_o[i]] != i], steps_final, lr=LR, batch=64, kmax=30,
                 mask_query_known=KNOWN, log_every=0, seed=s, schedule_steps=MAX_STEPS, scalar_weight=WSC)   # same LR trajectory as validated
    te_curves.append(predict(m, ed_o, te, ctx_te, KNOWN)); te_scal.append(predict_scalars(m, ed_o, te, ctx_te, KNOWN)); models.append(m.state_dict())
    print(f"outer seed {s} done [{time.time() - T0:.0f}s]", flush=True)
Pt = np.mean(te_curves, 0)
for s, sd in zip(SEEDS, models):
    torch.save(sd, os.path.join(OUT, f"transformer_seed{s}.pt"))

tf_te = curve_to_targets(Pt[:, :, 1], TE.G0.to_numpy())
St = np.mean(te_scal, 0)
heads_te = {"dpeak": St[:, 0, 1], "iauc": St[:, 1, 1], "ttp": St[:, 2, 1]}
heads_te["Gmax"] = TE.G0.to_numpy() + heads_te["dpeak"]
for t, sname in SRC.items():
    if sname == "head":
        tf_te[t] = heads_te[t]
gb_te = to_challenge_outputs(TE, st["P_te"])
blend = {t: W[t] * tf_te[t] + (1 - W[t]) * gb_te[t] for t in TARGETS}
curve = w_curve * Pt[:, :, 1] + (1 - w_curve) * gbm_curve(st["P_te"], len(TE))
blend = enforce_consistency(blend, TE.G0.to_numpy(), curve)          # peak >= every predicted point
band_lo, band_hi = curve + band_scale * (Pt[:, :, 0] - Pt[:, :, 1]), curve + band_scale * (Pt[:, :, 2] - Pt[:, :, 1])
CURV_o = st["outer"]["CURV"][te]
ok = np.isfinite(CURV_o)
curve_cov = float(np.mean(((CURV_o >= band_lo) & (CURV_o <= band_hi))[ok]))

# ------------------------------------------------------------------ evaluation with person-level bootstrap
subsets = {"all": np.ones(len(TE), bool), "unseen": ~TE.seen.to_numpy()}
A = pickle.load(open(os.path.join(ROOT_DIR, "results/v1_mobileclip/arrays.pkl"), "rb"))
TEv1 = pd.read_pickle(os.path.join(ROOT_DIR, "results/v1_mobileclip/test_meals.pkl"))
assert (TEv1.index == TE.index).all()
chron = pickle.load(open(os.path.join(CHR, "preds.pkl"), "rb"))
cands = {"Population mean (by meal type)": A["preds"]["population_mean"], "Chronos-2 zero-shot + carb covariate": chron["chronos2_with_carb_covariate"],
         "Population model (photo + context)": A["preds"]["population_gbm"], "Personalized LightGBM": gb_te,
         "Personalized transformer (T1D-pretrained)": tf_te, "Final ensemble": blend}
RES = pd.concat([metrics(TE, o, subsets).assign(model=k) for k, o in cands.items()], ignore_index=True)
rng = np.random.default_rng(0)
pids = TE.pid.to_numpy(); up = np.unique(pids); pos = {p: np.flatnonzero(pids == p) for p in up}
boots = [np.concatenate([pos[p] for p in rng.choice(up, len(up))]) for _ in range(1000)]
CI = []
for k, o in cands.items():
    for t in TARGETS:
        for sub, mask in subsets.items():
            e = np.abs(np.asarray(o[t]) - TE[t].to_numpy())
            b = [e[ix][mask[ix]].mean() for ix in boots]
            CI.append(dict(model=k, target=t, subset=sub, lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5)))
RES = RES.merge(pd.DataFrame(CI), on=["model", "target", "subset"])
# paired improvement of the final ensemble over the population model
paired = {}
for t in ["G60", "dpeak", "iauc"]:
    e1 = np.abs(np.asarray(cands["Population model (photo + context)"][t]) - TE[t].to_numpy())
    e2 = np.abs(np.asarray(blend[t]) - TE[t].to_numpy())
    d = [(e1[ix] - e2[ix]).mean() for ix in boots]
    paired[t] = dict(mean=float((e1 - e2).mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
pl = {}
for k in ["Population model (photo + context)", "Final ensemble"]:
    g = TE.assign(p=np.asarray(cands[k]["dpeak"])).groupby("pid")[["p", "dpeak"]].mean()
    pl[k] = float(stats.pearsonr(g.p, g.dpeak)[0])
RES.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
summ = dict(num_cols=COLS_VERSION, finetune_lr=LR, head_weight=WSC, tf_sources=SRC, zero_shot_val=zero_shot_val,
            lr_search={str(k[0]): dict(steps=v["steps"], vals=v["vals"], ens_val=v["ens_val"]) for k, v in lr_runs.items() if k[1] == WSC},
            search={f"lr={k[0]:g},head={k[1]:g}": dict(score=v["score"], ens_val=v["ens_val"], steps=v["steps"]) for k, v in lr_runs.items()},
            finetune_curves={str(k[0]): v["hists"] for k, v in lr_runs.items() if k[1] == WSC}, blend_weights=W, curve_weight=w_curve, band_scale=band_scale, curve_band_coverage_test=curve_cov,
            finetune_best_steps=best_steps, finetune_steps_final=steps_final, paired_gain_vs_population=paired, person_level_r_dpeak=pl,
            gbm_config=st["config"], gbm_params=st["params"], gbm_offset=st["use_offset"], gbm_grid_scores=st["grid_scores"],
            n=dict(train=len(st["TR"]), test=len(TE), unseen=int((~TE.seen).sum()), persons=int(TE.pid.nunique()), inner_train=len(st["TRi"]), inner_val=len(VAi)))
json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
pickle.dump(dict(curve=curve, band_lo=band_lo, band_hi=band_hi, blend=blend, tf=tf_te, gbm=gb_te, Pt=Pt, St=St, te_index=te), open(os.path.join(OUT, "final_preds.pkl"), "wb"))
print(RES[RES.subset == "unseen"].pivot(index="model", columns="target", values="mae")[TARGETS].round(1).to_string())
print(RES[RES.subset == "unseen"].pivot(index="model", columns="target", values="r")[TARGETS].round(2).to_string())
print(json.dumps(summ, indent=0, default=float))
print(f"done [{time.time() - T0:.0f}s]")
