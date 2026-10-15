"""Nutrition v2 (pmrf/nutrition.py) and the glucose model on top of it, rolling inner validation (history cut-off day 4, 5, 6;
validate on CGMacros meals up to day 7). Nutrition: all validation photos with logged nutrients; glucose: labelled meals.
usage: CUDA_VISIBLE_DEVICES=1 .venv-gpu/bin/python -I exp_nutrition_v2.py <CGMacros dir> <BIG IDEAs dir> <embeddings npz> <stage1 dir> <out dir>
"""
import sys, os, json, pickle, time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd
from pmrf.data import load_cgmacros, load_bigideas
from pmrf.features import meal_table, person_profile
from pmrf.pipeline import photo_features, attach_embeddings
from pmrf.models.gbm import _fit_bag
from pmrf.nutrition import NutritionStacker, NutritionGate, vlm_value, NUTRIENTS, ZS_MODELS

DATA, BIG, EMBP, S1, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
FE, PRM, LOGT = list(st["FEATS"]), dict(st["params"]), dict(st.get("log_targets", {}))
YS = ["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"]
CURVE_YS = ["d30", "d60", "d90", "d120"]
persons = {**load_cgmacros(DATA), **load_bigideas(BIG)}
R_ = os.path.join(ROOT_DIR, "results")
ZS_FILES = dict(qwen=f"{R_}/qwen3vl/zeroshot.jsonl", mistral=f"{R_}/vlm_mistral-large-3-675b/zeroshot.jsonl",
                llama=f"{R_}/vlm_llama4-maverick-17b/zeroshot.jsonl", nova=f"{R_}/vlm_nova-pro/zeroshot.jsonl")


def jl(p):
    return {d["rel"]: (d.get("parsed") or {}) for d in map(json.loads, open(p))} if os.path.exists(p) else {}


ZS = {m: jl(p) for m, p in ZS_FILES.items()}
FS = {c: jl(f"{R_}/qwen3vl/fewshot_c{c}.jsonl") for c in (4, 5, 6, 7)}
ORD = dict(form={"solid": 0, "mixed": 0.5, "liquid": 1}, glycemic_index={"low": 0, "medium": 1, "high": 2}, portion={"small": 0, "medium": 1, "large": 2})
QZ = ["qz_sugar", "qz_liquid", "qz_sugary_drink", "qz_gi", "qz_refined", "qz_fried", "qz_whole_grain", "qz_veg", "qz_portion"]


def add_vlm(M, cut):
    M = M.copy()
    rel = [os.path.relpath(p, DATA) if isinstance(p, str) else None for p in M.photo]
    for m in ZS_MODELS:
        d = [ZS[m].get(r, {}) if r else {} for r in rel]
        for k in NUTRIENTS:
            M[f"vz_{m}_{k}"] = [vlm_value(x, f"{k}_g") for x in d]
    q = [ZS["qwen"].get(r, {}) if r else {} for r in rel]
    b_ = lambda v: float(v) if isinstance(v, bool) else np.nan
    M["qz_sugar"] = [vlm_value(x, "sugar_g") for x in q]
    M["qz_liquid"] = M["vz_liquid"] = [ORD["form"].get(str(x.get("form")).lower(), np.nan) for x in q]
    M["qz_sugary_drink"] = [b_(x.get("sugary_drink")) for x in q]
    M["qz_gi"] = [ORD["glycemic_index"].get(str(x.get("glycemic_index")).lower(), np.nan) for x in q]
    M["qz_refined"] = [vlm_value(x, "refined_carb_share") for x in q]
    M["qz_fried"] = [b_(x.get("fried")) for x in q]
    M["qz_whole_grain"] = [b_(x.get("whole_grain")) for x in q]
    M["qz_veg"] = [vlm_value(x, "vegetable_share") for x in q]
    M["qz_portion"] = M["vz_portion"] = [ORD["portion"].get(str(x.get("portion")).lower(), np.nan) for x in q]
    f = [FS[cut].get(r, {}) if r else {} for r in rel]
    for k in NUTRIENTS:
        M[f"vf_{k}"] = [vlm_value(x, f"{k}_g") for x in f]
    M["vf_same"] = [np.nan if not x else float(x.get("same_as_example") not in (None, "null")) for x in f]
    M["own_sim"] = M["own_nn_sim"]
    return M


def prepare(hd):
    M, CURV, PRE, HRS = meal_table(persons, hd)
    M, EMB = attach_embeddings(M, EMBP, DATA)
    M["in_hist"] = M.day <= hd
    for mt in ["breakfast", "lunch", "dinner", "snack"]:
        M[f"is_{mt}"] = (M.meal_type == mt).astype(float)
    M, mest, pca, E = photo_features(M, EMB)
    M = M.join(person_profile(persons, M, M.in_hist.to_numpy(), hd))
    return add_vlm(M, hd)


def with_est(df, est):
    df = df.copy()
    for k in NUTRIENTS:
        df[f"est_{k}"] = est[k].reindex(df.index).to_numpy()
    df["est_kcal"] = 4 * df.est_carbs + 4 * df.est_protein + 9 * df.est_fat
    return df


def lgbm(tr, te, feats, y):
    off = LOGT.get(y)
    yt = np.log(tr[y] + off) if off is not None else tr[y]
    p = _fit_bag(tr[feats], yt, (0, 1, 2), params=PRM).predict(te[feats])
    return np.exp(p) - off if off is not None else p


def tabpfn(tr, te, feats, y):
    import torch
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu", n_estimators=8, random_state=0, ignore_pretraining_limits=True)
    m.fit(tr[feats].to_numpy(np.float32), tr[y].to_numpy(np.float32))
    return m.predict(te[feats].to_numpy(np.float32), output_type="median")


nut_rows, rows, GATE_ROWS = [], [], []
for hd in (4, 5, 6):
    M = prepare(hd)
    cg, ph = ~M.pid.str.startswith("B"), M.emb_idx >= 0
    Hn = M[cg & M.in_hist & ph & M.carbs.notna()]
    Vn = M[cg & (M.day > hd) & (M.day <= 7) & ph & M.carbs.notna()]
    ns = NutritionStacker().fit(Hn)
    est_v = ns.predict(Vn)
    GATE_ROWS.append(Vn.assign(cutoff=hd))
    srcs = {"ours (current)": Vn[[f"est_{k}" for k in NUTRIENTS]].set_axis(NUTRIENTS, axis=1),
            "VLM median (4 zero-shot)": pd.DataFrame({k: Vn[[f"vz_{m}_{k}" for m in ZS_MODELS]].median(1) for k in NUTRIENTS}),
            "Qwen few-shot": Vn[[f"vf_{k}" for k in NUTRIENTS]].set_axis(NUTRIENTS, axis=1),
            "nutrition v2 (learned combination)": est_v,
            "mean(ours, Qwen few-shot)": pd.DataFrame({k: Vn[[f"est_{k}", f"vf_{k}"]].mean(1) for k in NUTRIENTS}),
}
    for name, E_ in srcs.items():
        for sub, m in [("free-living (dinner, snack)", Vn.meal_type.isin(["dinner", "snack"])), ("standardized (breakfast, lunch)", Vn.meal_type.isin(["breakfast", "lunch"])),
                       ("all", Vn.meal_type.notna())]:
            r = dict(cutoff=hd, source=name, subset=sub, n=int(m.sum()))
            for k in ["carbs", "protein", "fat", "kcal"]:
                p = (4 * E_.carbs + 4 * E_.protein + 9 * E_.fat) if k == "kcal" else E_[k]
                o = (4 * Vn.carbs + 4 * Vn.protein + 9 * Vn.fat) if k == "kcal" else Vn[k]
                ok = m & p.notna() & o.notna()
                r[k] = float(np.mean(np.abs(p[ok] - o[ok])))
            nut_rows.append(r)
    # glucose models: history rows get the stacker's out-of-fold estimates, validation rows its predictions
    TR = M[(cg & M.in_hist & M.label & ph) | (~cg & M.label)]
    VA = M[cg & (M.day > hd) & (M.day <= 7) & M.label & ph]
    est_tr = ns.oof.reindex(TR.index)
    for k in NUTRIENTS:                                   # BIG IDEAs rows (no photos) keep NaN, as in the current model
        est_tr[k] = est_tr[k].where(TR.emb_idx >= 0)
    TRn, VAn = with_est(TR, est_tr), with_est(VA, ns.predict(VA))
    print(f"cut-off {hd}: nutrition train {len(Hn)} / val {len(Vn)}; glucose train {len(TR)} / val {len(VA)} [{time.time() - T0:.0f}s]", flush=True)
    mean_fs = lambda D: pd.DataFrame({k: D[[f"est_{k}", f"vf_{k}"]].mean(1) for k in NUTRIENTS}, index=D.index)
    mtr = mean_fs(TR); mtr.loc[TR.emb_idx < 0] = np.nan            # BIG IDEAs rows (no photo) keep NaN
    TRb, VAb = with_est(TR, mtr), with_est(VA, mean_fs(VA))
    P = {}
    for y in YS:
        P[("A0 current", y)] = lgbm(TR, VA, FE, y)
        P[("B4 nutrition = mean(ours, Qwen few-shot)", y)] = lgbm(TRb, VAb, FE, y)
        if y in CURVE_YS:
            P[("T1 A0 + TabPFN (30-120 min)", y)] = 0.5 * P[("A0 current", y)] + 0.5 * tabpfn(TR, VA, FE, y)
            P[("T2 B4 + TabPFN (30-120 min)", y)] = 0.5 * P[("B4 nutrition = mean(ours, Qwen few-shot)", y)] + 0.5 * tabpfn(TRb, VAb, FE, y)
        else:
            P[("T1 A0 + TabPFN (30-120 min)", y)] = P[("A0 current", y)]
            P[("T2 B4 + TabPFN (30-120 min)", y)] = P[("B4 nutrition = mean(ours, Qwen few-shot)", y)]
    for (v, y), p in P.items():
        rows.append(dict(cutoff=hd, variant=v, target=y, n=len(VA), mae=float(np.mean(np.abs(p - VA[y])))))
    print("  dpeak:", {v: round(float(np.mean(np.abs(p - VA['dpeak']))), 2) for (v, y), p in P.items() if y == "dpeak"}, f"[{time.time() - T0:.0f}s]", flush=True)

G = pd.concat(GATE_ROWS)
gate = NutritionGate().fit(G)                         # thresholds chosen on the pooled validation rows
print("gate thresholds (own-photo similarity):", {k: round(float(v), 3) for k, v in gate.t.items()}, flush=True)
EG = gate.predict(G)
for hd in (4, 5, 6):
    Vn = G[G.cutoff == hd]; E_ = EG[G.cutoff == hd]
    for sub, m in [("free-living (dinner, snack)", Vn.meal_type.isin(["dinner", "snack"])), ("standardized (breakfast, lunch)", Vn.meal_type.isin(["breakfast", "lunch"])),
                   ("all", Vn.meal_type.notna())]:
        r = dict(cutoff=hd, source="nutrition gate (familiar -> history, new -> vision)", subset=sub, n=int(m.sum()))
        for k in ["carbs", "protein", "fat", "kcal"]:
            p = (4 * E_.carbs + 4 * E_.protein + 9 * E_.fat) if k == "kcal" else E_[k]
            o = (4 * Vn.carbs + 4 * Vn.protein + 9 * Vn.fat) if k == "kcal" else Vn[k]
            ok = m & p.notna() & o.notna()
            r[k] = float(np.mean(np.abs(p[ok] - o[ok])))
        nut_rows.append(r)
json.dump({k: float(v) for k, v in gate.t.items()}, open(os.path.join(OUT, "gate_thresholds.json"), "w"))
N = pd.DataFrame(nut_rows); N.to_csv(os.path.join(OUT, "nutrition_validation_v2.csv"), index=False)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "glucose_validation_v2.csv"), index=False)
NW = N.groupby(["subset", "source"]).apply(lambda g: pd.Series({k: np.average(g[k], weights=g.n) for k in ["carbs", "protein", "fat", "kcal"]}))
W = R.groupby(["variant", "target"]).apply(lambda g: np.average(g.mae, weights=g.n)).unstack()[YS]
pc = R.pivot_table(index=["variant", "target"], columns="cutoff", values="mae")
wins = {v: {y: int((pc.loc[(v, y)] < pc.loc[("A0 current", y)]).sum()) for y in YS} for v in W.index}
summ = dict(nutrition_mae=NW.round(1).reset_index().to_dict("records"), glucose_mae=W.round(2).to_dict("index"),
            glucose_change_pct=((W / W.loc["A0 current"] - 1) * 100).round(1).to_dict("index"), cutoffs_better=wins)
json.dump(summ, open(os.path.join(OUT, "summary_v2.json"), "w"), indent=1)
print(NW.round(1).to_string()); print(((W / W.loc["A0 current"] - 1) * 100).round(1).to_string()); print(json.dumps(wins))
print(f"done [{time.time() - T0:.0f}s]")
