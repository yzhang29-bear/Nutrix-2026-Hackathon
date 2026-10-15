"""Extract isolated carb events with insulin context from the harmonized T1D parquet (sources that log carbs).
usage: python -I t1d_events.py <parquet> <out.npz>
Per event: subject key, time, study day, carbs, pre-meal CGM (-120..0 min, 5-min), post-meal curve (0..120 min),
meal bolus, bolus lead time, insulin on board, basal rate, heart rate (where recorded), delivery modality, age.
"""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq

F, OUT = sys.argv[1], sys.argv[2]
SRCS = ["Loop", "ReplaceBG", "PEDAP", "CTR3", "AZT1D", "HUPA-UCM", "BrisT1D", "T1D-UOM", "OhioT1DM"]
COLS = ["id", "date", "CGM", "basal", "bolus", "carbs", "heartrate", "age", "insulin_delivery_modality"]
PRE, POST = 24, 24          # samples at 5 min
TAU = 90.0
rec = []
for s in SRCS:
    d = pq.read_table(F, columns=COLS, filters=[("source_file", "==", s)], read_dictionary=["id", "insulin_delivery_modality"]).to_pandas()
    d = d.sort_values(["id", "date"], kind="stable").reset_index(drop=True)
    for sid, idx in d.groupby("id", observed=True, sort=False).indices.items():
        x = d.iloc[idx]
        g = x.CGM.to_numpy(float); cb = x.carbs.to_numpy(float); bo = np.nan_to_num(x.bolus.to_numpy(float)); ba = x.basal.to_numpy(float)
        hr = x.heartrate.to_numpy(float)
        tn = x.date.to_numpy().astype("int64") // 10 ** 9
        t0 = tn[0]
        age = x.age.dropna().iloc[0] if x.age.notna().any() else np.nan
        mod = x.insulin_delivery_modality.dropna().iloc[0] if x.insulin_delivery_modality.notna().any() else None
        ev = np.flatnonzero(cb >= 10)
        for i in ev:
            if i - PRE - 96 < 0 or i + POST >= len(x): continue
            if tn[i + POST] - tn[i - PRE] != (PRE + POST) * 300: continue          # contiguous 5-min grid
            if np.nansum(cb[i + 1:i + POST + 1]) >= 5: continue                   # isolated: no other carbs within 2 h after
            post = g[i:i + POST + 1]; pre = g[i - PRE:i + 1]
            if np.isnan(g[i]) or np.mean(np.isfinite(g[i - 6:i + POST + 1])) < 0.9: continue
            mins = (np.arange(i - 96, i + 13) - i) * 5                             # -480 .. +60 min
            b = bo[i - 96:i + 13]
            around = (mins >= -15) & (mins <= 15) & (b > 0)
            prior = (mins < -15) & (b > 0)
            rec.append(dict(src=s, key=f"{s}:{sid}", t=int(tn[i]), day=int((tn[i] - t0) // 86400) + 1, carbs=float(cb[i]),
                            prev_carbs_6h=float(np.nansum(cb[i - 72:i])), hour=float(pd.Timestamp(tn[i], unit="s").hour),
                            meal_bolus=float(b[around].sum()), bolus_lead=float(-mins[around][np.argmax(b[around])]) if around.any() else np.nan,
                            iob=float(np.sum(b[prior] * np.exp(mins[prior] / TAU))), basal=float(np.nanmean(ba[i - 12:i + 1])) if np.isfinite(ba[i - 12:i + 1]).any() else np.nan,
                            hr_pre=float(np.nanmean(hr[i - 6:i + 1])) if np.isfinite(hr[i - 6:i + 1]).any() else np.nan,
                            age=float(age), modality=mod, pre=pre.astype(np.float32), post=post.astype(np.float32)))
    print(s, len(rec), flush=True)
E = pd.DataFrame(rec)
np.savez_compressed(OUT, meta=E.drop(columns=["pre", "post"]).to_records(index=False), pre=np.stack(E.pre.to_numpy()), post=np.stack(E.post.to_numpy()))
print("saved", len(E))
