"""Modality ablations and image-encoder comparison with the final LightGBM configuration (outer split, unseen day-8+ meals).
usage: python -I final_ablations.py <CGMacros dir> <stage1 dir> <out dir>
"""
import sys, os, json, time, pickle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd
from scipy import stats
from pmrf.data import load_cgmacros, load_bigideas
from pmrf.features import person_profile
from pmrf.pipeline import build, photo_features, to_challenge_outputs, metrics, FEATURE_GROUPS
from pmrf.models.gbm import GBMResponse

DATA, S1, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
BIG = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(OUT, exist_ok=True)
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
FE, PRM, OFF = st["FEATS"], st["params"], st["use_offset"]
LOGT, LP = st.get("log_targets", {}), st.get("log_point", "median")
TR, TE = st["TR"], st["TE"]
unseen = ~TE.seen.to_numpy()
subsets = {"unseen": unseen}
rows = []

def run(feats, name, tr=TR, te=TE):
    m = GBMResponse(feats, targets=["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"], personal_offset=OFF, params=PRM, seeds=(0, 1, 2), log_targets=LOGT, log_point=LP).fit(tr)
    r = metrics(te, to_challenge_outputs(te, m.predict(te)), subsets).assign(model=name)
    rows.append(r)
    print(name, r.set_index("target").loc[["G60", "dpeak", "iauc"], "mae"].round(1).to_dict(), flush=True)

run(FE, "All modalities")
drop = {"− meal photo (image, estimated nutrition, similar meals)": FEATURE_GROUPS["Meal (from photo)"] + FEATURE_GROUPS["Similar meals: your history"] + FEATURE_GROUPS["Similar meals: other people"],
        "− CGM before the meal": FEATURE_GROUPS["Glucose before meal"] + FEATURE_GROUPS["Glucose trend before meal"],
        "− heart rate": FEATURE_GROUPS["Heart rate"],
        "− personal history (profile, own similar meals)": FEATURE_GROUPS["Personal profile (days 1-7)"] + FEATURE_GROUPS["Similar meals: your history"],
        "− meal type & time of day": FEATURE_GROUPS["Meal type & time"],
        "− previous meals": FEATURE_GROUPS["Previous meals"]}
for name, cols in drop.items():
    run([f for f in FE if f not in cols], name)
ABL = pd.concat(rows, ignore_index=True)
ABL.to_csv(os.path.join(OUT, "ablations.csv"), index=False)

# image encoders: same protocol, only the photo embedding changes
persons = load_cgmacros(DATA)
if BIG:
    persons.update(load_bigideas(BIG))
enc_rows, photo_rows = [], []
for name, npz in [("ResNet50 (ImageNet)", "eda/cache/photo_emb.npz"), ("MobileCLIP-S2", "eda/cache/photo_mobileclip.npz"), ("CLIP ViT-L/14", "eda/cache/photo_clip.npz")]:
    M, CURV, PRE, HRS, EMB = build(persons, DATA, os.path.join(ROOT_DIR, npz), 7)
    M, mest, pca, E = photo_features(M, EMB)
    M = M.join(person_profile(persons, M, M.in_hist.to_numpy(), 7))
    tr = M.loc[TR.index]; te = M.loc[TE.index].assign(seen=TE.seen.to_numpy())
    m = GBMResponse(FE, targets=["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"], personal_offset=OFF, params=PRM, seeds=(0, 1, 2), log_targets=LOGT, log_point=LP).fit(tr)
    r = metrics(te, to_challenge_outputs(te, m.predict(te)), subsets).assign(encoder=name)
    enc_rows.append(r)
    tp = (~M.in_hist) & (M.emb_idx >= 0)
    for sub, mask in [("free-living (dinner, snack)", tp & M.meal_type.isin(["dinner", "snack"])), ("standardized (breakfast, lunch)", tp & M.meal_type.isin(["breakfast", "lunch"]))]:
        for c in ["carbs", "protein", "fat", "kcal"]:
            o, p = M.loc[mask, c], M.loc[mask, f"est_{c}"]
            photo_rows.append(dict(encoder=name, subset=sub, nutrient=c, n=int(mask.sum()), mae=float(np.mean(np.abs(o - p))),
                                   r=float(stats.pearsonr(o, p)[0]), median_ape=float(np.median(np.abs(o - p) / o.clip(lower=5)) * 100)))
    print(name, r.set_index("target").loc[["G60", "dpeak", "iauc"], "mae"].round(1).to_dict(), flush=True)
pd.concat(enc_rows, ignore_index=True).to_csv(os.path.join(OUT, "encoders_glucose.csv"), index=False)
pd.DataFrame(photo_rows).to_csv(os.path.join(OUT, "encoders_photo.csv"), index=False)
print(pd.DataFrame(photo_rows).pivot_table(index=["encoder", "subset"], columns="nutrient", values="r").round(2).to_string())
