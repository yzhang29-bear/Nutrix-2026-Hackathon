"""Compare zero-shot VLM nutrition (Qwen2.5-VL-3B) with the CLIP-based estimator on the same day-8+ test photos.
usage: python -I compare_vlm.py <stage1 dir> <vlm csv> <out json>
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy import stats

S1, CSV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
Mo = st["outer"]["M"]
T = Mo[(~Mo.in_hist) & (Mo.emb_idx >= 0)].copy()
V = pd.read_csv(CSV)
VC = {"carbs_g": "vlm_carbs", "protein_g": "vlm_protein", "fat_g": "vlm_fat", "fiber_g": "vlm_fiber", "kcal": "vlm_kcal"}
for c in VC:
    V[c] = pd.to_numeric(V[c], errors="coerce")
V = V.rename(columns=VC).drop_duplicates("photo")
T = T.merge(V[["photo", "dish", "seconds"] + list(VC.values())], on="photo", how="inner")
res = dict(n=len(T), parsed=float(T.vlm_carbs.notna().mean()), seconds_median=float(T.seconds.median()),
           vlm_bias_carbs=float((T.vlm_carbs - T.carbs).median()), clip_bias_carbs=float((T.est_carbs - T.carbs).median()))
for nut, vc, cc in [("carbs", "vlm_carbs", "est_carbs"), ("protein", "vlm_protein", "est_protein"), ("fat", "vlm_fat", "est_fat"), ("kcal", "vlm_kcal", "est_kcal")]:
    for sub, mask in [("free-living", T.meal_type.isin(["dinner", "snack"])), ("standardized", T.meal_type.isin(["breakfast", "lunch"])), ("all", np.ones(len(T), bool))]:
        q = T[mask & T[vc].notna()]
        o = q[nut]
        for name, p in [("vlm", q[vc]), ("clip", q[cc]), ("average", (q[vc] + q[cc]) / 2)]:
            res[f"{nut}|{sub}|{name}"] = dict(n=len(q), mae=float(np.mean(np.abs(p - o))), r=float(stats.pearsonr(p, o)[0]),
                                             median_ape=float(np.median(np.abs(p - o) / o.clip(lower=5)) * 100))
json.dump(res, open(OUT, "w"), indent=1)
print(pd.DataFrame({k: v for k, v in res.items() if isinstance(v, dict)}).T.round(2).to_string())
print({k: v for k, v in res.items() if not isinstance(v, dict)})
