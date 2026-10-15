"""Do Qwen3-VL photo features and the Shanghai T1DM/T2DM cohort improve the response model? Final stage-1 LightGBM
configuration, rolling inner validation only (history cut-off day 4, 5, 6; validate on CGMacros meals up to day 7).
Also reports photo-nutrition accuracy (ours vs Qwen) on the validation photos of cut-off 5.
usage: python -I exp_qwen_shanghai.py <CGMacros dir> <BIG IDEAs dir> <Shanghai root> <shanghai nutrients.jsonl> <embeddings npz>
                                       <qwen dir> <stage1 dir> <out dir>
"""
import sys, os, json, pickle, time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd
from pmrf.data import load_cgmacros, load_bigideas, load_shanghai_meals
from pmrf.features import meal_table, person_profile
from pmrf.pipeline import photo_features, attach_embeddings
from pmrf.models.gbm import _fit_bag

DATA, BIG, SH, SHN, EMBP, QD, S1, OUT = sys.argv[1:9]
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
FE, PRM, LOGT = list(st["FEATS"]), dict(st["params"]), dict(st.get("log_targets", {}))
YS = ["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"]
cg_p, big_p, sh_p = load_cgmacros(DATA), load_bigideas(BIG), load_shanghai_meals(SH, SHN)
persons = {**cg_p, **big_p, **sh_p}
print(f"people: CGMacros {len(cg_p)}, BIG IDEAs {len(big_p)}, Shanghai {len(sh_p)} sessions "
      f"({sum(len(p.meals) for p in sh_p.values())} meals, {sum(p.meals.carbs.notna().sum() for p in sh_p.values())} with nutrients) "
      f"[{time.time() - T0:.0f}s]", flush=True)


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


ZS = {d["rel"]: d["parsed"] or {} for d in jl(os.path.join(QD, "zeroshot.jsonl"))}
FS = {c: {d["rel"]: d["parsed"] or {} for d in jl(os.path.join(QD, f"fewshot_c{c}.jsonl"))} for c in (4, 5, 6, 7)}
f_ = lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan
b_ = lambda v: float(v) if isinstance(v, bool) else np.nan
ORD = dict(form={"solid": 0, "mixed": 0.5, "liquid": 1}, glycemic_index={"low": 0, "medium": 1, "high": 2}, portion={"small": 0, "medium": 1, "large": 2})
QZ = ["qz_carbs", "qz_sugar", "qz_protein", "qz_fat", "qz_fiber", "qz_kcal", "qz_liquid", "qz_sugary_drink", "qz_gi", "qz_refined",
      "qz_fried", "qz_whole_grain", "qz_veg", "qz_portion"]
QF = ["qf_carbs", "qf_protein", "qf_fat", "qf_fiber", "qf_same", "qf_conf"]


def add_qwen(M, cut):
    rel = M.photo.map(lambda p: os.path.relpath(p, DATA) if isinstance(p, str) else None)
    z = [ZS.get(r, {}) if r else {} for r in rel]
    f = [FS[cut].get(r, {}) if r else {} for r in rel]
    M = M.copy()
    for k in ["carbs", "sugar", "protein", "fat", "fiber", "kcal"]:
        M[f"qz_{k}"] = [f_(d.get(f"{k}_g", d.get(k))) for d in z]
    M["qz_liquid"] = [ORD["form"].get(str(d.get("form")).lower(), np.nan) for d in z]
    M["qz_sugary_drink"] = [b_(d.get("sugary_drink")) for d in z]
    M["qz_gi"] = [ORD["glycemic_index"].get(str(d.get("glycemic_index")).lower(), np.nan) for d in z]
    M["qz_refined"] = [f_(d.get("refined_carb_share")) for d in z]
    M["qz_fried"] = [b_(d.get("fried")) for d in z]
    M["qz_whole_grain"] = [b_(d.get("whole_grain")) for d in z]
    M["qz_veg"] = [f_(d.get("vegetable_share")) for d in z]
    M["qz_portion"] = [ORD["portion"].get(str(d.get("portion")).lower(), np.nan) for d in z]
    for k in ["carbs", "protein", "fat", "fiber"]:
        M[f"qf_{k}"] = [f_(d.get(f"{k}_g")) for d in f]
    M["qf_same"] = [np.nan if not d else float(d.get("same_as_example") not in (None, "null")) for d in f]
    M["qf_conf"] = [f_(d.get("confidence")) for d in f]
    return M


def prepare(hd):
    M, CURV, PRE, HRS = meal_table(persons, hd)
    M, EMB = attach_embeddings(M, EMBP, DATA)
    M["in_hist"] = M.day <= hd
    for mt in ["breakfast", "lunch", "dinner", "snack"]:
        M[f"is_{mt}"] = (M.meal_type == mt).astype(float)
    M, mest, pca, E = photo_features(M, EMB)
    M = M.join(person_profile(persons, M, M.in_hist.to_numpy(), hd))
    M["source"] = np.select([M.pid.str.startswith("B"), M.pid.str.startswith("S")], ["bigideas", "shanghai"], "cgmacros")
    return add_qwen(M, hd)


def point_predict(tr, te, feats, y):
    off = LOGT.get(y)
    yt = np.log(tr[y] + off) if off is not None else tr[y]
    p = _fit_bag(tr[feats], yt, (0, 1, 2), params=PRM).predict(te[feats])
    return np.exp(p) - off if off is not None else p


def with_logged_nutrition(df, mask):
    df = df.copy()
    for c in ["carbs", "protein", "fat", "fiber"]:
        df.loc[mask, f"est_{c}"] = df.loc[mask, c]
    df.loc[mask, "est_kcal"] = df.loc[mask, "kcal"]
    df["macro_from_log"] = mask.astype(float)
    return df


def blend_est(df, w=0.5):
    df = df.copy()
    for c in ["carbs", "protein", "fat", "fiber"]:
        q = df[f"qf_{c}"]
        df[f"est_{c}"] = np.where(q.notna() & df[f"est_{c}"].notna(), (1 - w) * df[f"est_{c}"] + w * q, df[f"est_{c}"])
    df["est_kcal"] = 4 * df.est_carbs + 4 * df.est_protein + 9 * df.est_fat
    return df


rows, nut_rows = [], []
for hd in (4, 5, 6):
    M = prepare(hd)
    cg, big, sh, ph = M.source == "cgmacros", M.source == "bigideas", M.source == "shanghai", M.emb_idx >= 0
    base_tr = M[(cg & M.in_hist & M.label & ph) | (big & M.label)]
    sh_tr = M[sh & M.label]
    VA = M[cg & (M.day > hd) & (M.day <= 7) & M.label & ph]
    print(f"cut-off {hd}: train {len(base_tr)} (+{len(sh_tr)} Shanghai), validate {len(VA)} [{time.time() - T0:.0f}s]", flush=True)
    sh_nut = with_logged_nutrition(pd.concat([base_tr, sh_tr]), pd.concat([base_tr, sh_tr]).source.eq("shanghai") & pd.concat([base_tr, sh_tr]).carbs.notna())
    variants = {
        "A0 current model": (base_tr, VA, FE),
        "A1 + Shanghai (no nutrition)": (pd.concat([base_tr, sh_tr]), VA, FE),
        "A2 + Shanghai (LLM nutrition)": (sh_nut, VA.assign(macro_from_log=0.0), FE + ["macro_from_log"]),
        "B1 + Qwen few-shot nutrition": (base_tr, VA, FE + QF),
        "B2 + Qwen food attributes": (base_tr, VA, FE + QZ),
        "B3 + Qwen both": (base_tr, VA, FE + QF + QZ),
        "B4 nutrition = mean(ours, Qwen few-shot)": (blend_est(base_tr), blend_est(VA), FE),
        "C1 A1 + B3": (pd.concat([base_tr, sh_tr]), VA, FE + QF + QZ),
        "C2 A2 + B3": (sh_nut, VA.assign(macro_from_log=0.0), FE + ["macro_from_log"] + QF + QZ),
    }
    for name, (tr, va, feats) in variants.items():
        for y in YS:
            p = point_predict(tr, va, feats, y)
            rows.append(dict(cutoff=hd, variant=name, target=y, n=len(va), mae=float(np.mean(np.abs(p - va[y])))))
        print(f"  {name}: dpeak {rows[-3]['mae']:.2f} [{time.time() - T0:.0f}s]", flush=True)
    if hd == 5:
        for sub, m in [("standardized (breakfast, lunch)", VA.meal_type.isin(["breakfast", "lunch"])), ("free-living (dinner, snack)", VA.meal_type.isin(["dinner", "snack"]))]:
            v = VA[m]
            for c in ["carbs", "protein", "fat"]:
                for src, col in [("ours", f"est_{c}"), ("Qwen few-shot", f"qf_{c}"), ("Qwen zero-shot", f"qz_{c}")]:
                    ok = v[col].notna() & v[c].notna()
                    nut_rows.append(dict(subset=sub, nutrient=c, source=src, n=int(ok.sum()), mae=float(np.mean(np.abs(v.loc[ok, col] - v.loc[ok, c]))),
                                         bias=float(np.mean(v.loc[ok, col] - v.loc[ok, c])), r=float(np.corrcoef(v.loc[ok, col], v.loc[ok, c])[0, 1])))
                ok = v[f"qf_{c}"].notna()
                bl = 0.5 * v.loc[ok, f"est_{c}"] + 0.5 * v.loc[ok, f"qf_{c}"]
                nut_rows.append(dict(subset=sub, nutrient=c, source="mean(ours, Qwen few-shot)", n=int(ok.sum()), mae=float(np.mean(np.abs(bl - v.loc[ok, c]))),
                                     bias=float(np.mean(bl - v.loc[ok, c])), r=float(np.corrcoef(bl, v.loc[ok, c])[0, 1])))

R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "inner_validation.csv"), index=False)
N = pd.DataFrame(nut_rows); N.to_csv(os.path.join(OUT, "nutrition_validation.csv"), index=False)
W = R.groupby(["variant", "target"]).apply(lambda g: np.average(g.mae, weights=g.n)).unstack()[YS]
ref = W.loc["A0 current model"]
better = R.pivot_table(index=["variant", "target"], columns="cutoff", values="mae")
base_c = better.xs("A0 current model", level="variant")
wins = {v: int((better.xs(v, level="variant").loc["dpeak"] < base_c.loc["dpeak"]).sum()) for v in W.index}
summ = dict(mae=W.round(2).to_dict("index"), change_pct=((W / ref - 1) * 100).round(1).to_dict("index"),
            dpeak_cutoffs_better_than_A0=wins,
            nutrition=N.pivot_table(index=["subset", "nutrient"], columns="source", values="mae").round(1).reset_index().to_dict("records"))
json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
print(((W / ref - 1) * 100).round(1).to_string()); print("dpeak cut-offs better than A0:", wins)
print(N.pivot_table(index=["subset", "nutrient"], columns="source", values="mae").round(1).to_string())
print(f"done [{time.time() - T0:.0f}s]")
