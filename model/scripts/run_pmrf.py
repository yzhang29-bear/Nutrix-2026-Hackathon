"""Run the Day-8 protocol on CGMacros: vision + personalized response models, ablations, metrics, artifacts.
usage: python -I run_pmrf.py <CGMacros dir> <out dir> [embedding npz]
"""
import sys, os, json, time, pickle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd
from scipy import stats
from pmrf.data import load_cgmacros
from pmrf.features import person_profile, PROFILE_FEATURES, TARGETS
from pmrf.pipeline import build, photo_features, feature_sets, to_challenge_outputs, metrics, FEATURE_GROUPS
from pmrf.models.gbm import GBMResponse

DATA, OUT = sys.argv[1], sys.argv[2]
EMB = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT_DIR, "eda/cache/photo_mobileclip.npz")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
persons = load_cgmacros(DATA)
M, CURV, PRE, HRS, EMBM = build(persons, DATA, EMB)
print(f"meals {len(M)}, labelled {M.label.sum()}, with photo {(M.emb_idx >= 0).sum()}  [{time.time() - T0:.0f}s]", flush=True)

M, mest, pca, E = photo_features(M, EMBM)
prof = person_profile(persons, M, M.in_hist.to_numpy())
M = M.join(prof)
print(f"features ready [{time.time() - T0:.0f}s]", flush=True)

# --- photo -> macro quality on test photos
R = {}
test_ph = (~M.in_hist) & (M.emb_idx >= 0)
for c in ["carbs", "protein", "fat", "kcal"]:
    for sub, mask in [("all", test_ph), ("free-living", test_ph & M.meal_type.isin(["dinner", "snack"])),
                      ("standardized", test_ph & M.meal_type.isin(["breakfast", "lunch"]))]:
        o, p = M.loc[mask, c], M.loc[mask, f"est_{c}"]
        R[f"photo_{c}_{sub}"] = dict(n=int(mask.sum()), mae=float(np.mean(np.abs(o - p))), r=float(stats.pearsonr(o, p)[0]),
                                     median_ape=float(np.median(np.abs(o - p) / o.clip(lower=5)) * 100))
print(json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in R.items() if "carbs" in k or "kcal" in k}), flush=True)

# --- training / test sets
TR = M[M.in_hist & M.label & (M.emb_idx >= 0)].copy()
TE = M[(~M.in_hist) & M.label & (M.emb_idx >= 0)].copy()
seen = TR.assign(rec=TR.meal_type + TR.carbs.astype(str) + TR.protein.astype(str) + TR.fat.astype(str)).groupby("pid").rec.apply(set)
TE["seen"] = [(r.meal_type in ("breakfast", "lunch")) and (r.meal_type + str(r.carbs) + str(r.protein) + str(r.fat)) in seen.get(r.pid, set())
              for r in TE.itertuples()]
subsets = {"all": np.ones(len(TE), bool), "unseen": ~TE.seen.to_numpy()}
print(f"train {len(TR)} test {len(TE)} unseen {int((~TE.seen).sum())}", flush=True)

FS = feature_sets()
results, preds = [], {}
# baseline: population mean change by meal type
P0 = {}
for y in ["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"]:
    P0[y] = dict(pred=TE.meal_type.map(TR.groupby("meal_type")[y].mean()).to_numpy())
out0 = to_challenge_outputs(TE, P0); preds["population_mean"] = out0
results.append(metrics(TE, out0, subsets).assign(model="population_mean"))

variants = {
    "population_gbm": (FS["population"], False),
    "personalized_gbm": (FS["personalized"], False),
    "personalized_gbm_offset": (FS["personalized"], True),
    "oracle_macros": (FS["oracle_macros"], True),
}
models = {}
for name, (feats, off) in variants.items():
    t = time.time()
    m = GBMResponse(feats, personal_offset=off).fit(TR)
    P = m.predict(TE, contributions=(name == "personalized_gbm_offset"))
    out = to_challenge_outputs(TE, P); preds[name] = out
    results.append(metrics(TE, out, subsets).assign(model=name)); models[name] = (m, P)
    print(f"{name}: fit+predict {time.time() - t:.0f}s", flush=True)

# --- modality ablations of the main model (drop one group at a time)
main_feats = FS["personalized"]
abl = {"− photo (no image, no macro estimates)": [f for f in main_feats if f not in FEATURE_GROUPS["Meal (from photo)"] + FEATURE_GROUPS["Similar meals: your history"] + FEATURE_GROUPS["Similar meals: other people"]],
       "− CGM before meal": [f for f in main_feats if f not in FEATURE_GROUPS["Glucose before meal"] + FEATURE_GROUPS["Glucose trend before meal"]],
       "− heart rate": [f for f in main_feats if f not in FEATURE_GROUPS["Heart rate"]],
       "− personal history": [f for f in main_feats if f not in FEATURE_GROUPS["Personal profile (days 1-7)"] + FEATURE_GROUPS["Similar meals: your history"]]}
for name, feats in abl.items():
    m = GBMResponse(feats, personal_offset=(name != "− personal history")).fit(TR)
    out = to_challenge_outputs(TE, m.predict(TE))
    results.append(metrics(TE, out, subsets).assign(model=name))
    print(f"ablation {name} done", flush=True)

RES = pd.concat(results, ignore_index=True)
RES.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
piv = RES[RES.subset == "unseen"].pivot(index="model", columns="target", values="mae")[TARGETS]
print(piv.round(1).to_string())
print(RES[RES.subset == "unseen"].pivot(index="model", columns="target", values="r")[TARGETS].round(2).to_string())

# --- person-level personalization, interval coverage
main_m, main_P = models["personalized_gbm_offset"]
extra = {}
for name in ["population_gbm", "personalized_gbm_offset"]:
    t_ = TE.assign(p=preds[name]["dpeak"]).groupby("pid")[["p", "dpeak"]].mean()
    extra[f"person_level_r_dpeak_{name}"] = float(stats.pearsonr(t_.p, t_.dpeak)[0])
for y in ["d60", "dpeak", "iauc"]:
    lo, hi = main_P[y]["lo"], main_P[y]["hi"]
    extra[f"coverage80_{y}"] = float(np.mean((TE[y] >= lo) & (TE[y] <= hi)))
    extra[f"width80_{y}"] = float(np.mean(hi - lo))
extra["photo"] = R
extra["shrinkage_k"] = main_m.k
extra["n"] = dict(train=len(TR), test=len(TE), unseen=int((~TE.seen).sum()), persons=int(TE.pid.nunique()))
json.dump(extra, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in extra.items() if k != "photo"}, indent=0, default=float))

# --- save everything the app / API needs
TE.to_pickle(os.path.join(OUT, "test_meals.pkl")); TR.to_pickle(os.path.join(OUT, "train_meals.pkl"))
pickle.dump(dict(curves=CURV, pre=PRE, hr=HRS, emb=E, M=M, main_P=main_P, preds=preds), open(os.path.join(OUT, "arrays.pkl"), "wb"))
pickle.dump(dict(gbm=main_m, macro_est=mest, pca=pca, features=main_feats), open(os.path.join(OUT, "model.pkl"), "wb"))
print(f"done [{time.time() - T0:.0f}s]")
