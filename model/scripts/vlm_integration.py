"""Does adding the VLM's nutrition reading (Qwen2.5-VL-3B, zero-shot) improve glucose prediction?
Decided on the inner split (days 1-5 -> 6-7); the day-8+ result is reported alongside.
usage: python -I vlm_integration.py <stage1 dir> <vlm csv> <out json>
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from pmrf.models.gbm import GBMResponse
from pmrf.pipeline import to_challenge_outputs, metrics

S1, CSV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
V = pd.read_csv(CSV)
VF = ["vlm_carbs", "vlm_protein", "vlm_fat", "vlm_fiber", "vlm_kcal"]
for a, b in zip(["carbs_g", "protein_g", "fat_g", "fiber_g", "kcal"], VF):
    V[b] = pd.to_numeric(V[a], errors="coerce")
V = V.drop_duplicates("photo").set_index("photo")[VF]

def add(df):
    return df.join(V, on="photo")

FE = st["FEATS"]; PRM = st["params"]; OFF = st["use_offset"]
YS = ["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"]
res = {}
for split, tr, te in [("inner (days 1-5 -> 6-7)", st["TRi"], st["VAi"]), ("test (days 1-7 -> 8+)", st["TR"], st["TE"])]:
    tr, te = add(tr), add(te)
    cover = float(np.mean(tr[VF[0]].notna())), float(np.mean(te[VF[0]].notna()))
    subsets = {"all": np.ones(len(te), bool)}
    if "seen" in te:
        subsets["unseen"] = ~te.seen.to_numpy()
    for name, feats in [("current model", FE), ("+ VLM nutrition", FE + VF)]:
        m = GBMResponse(feats, targets=YS, personal_offset=OFF, params=PRM, seeds=(0, 1, 2)).fit(tr)
        r = metrics(te, to_challenge_outputs(te, m.predict(te)), subsets)
        res[f"{split} | {name}"] = {f"{x.subset}:{x.target}": round(x.mae, 2) for x in r.itertuples()}
        print(split, name, "coverage", cover, r[r.subset == r.subset.iloc[-1]].set_index("target").mae.round(1).loc[["G60", "dpeak", "iauc"]].to_dict(), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
