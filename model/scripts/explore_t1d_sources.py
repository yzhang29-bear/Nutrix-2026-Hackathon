import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
f = sys.argv[1]
cols = ["source_file", "id", "date", "CGM", "basal", "bolus", "insulin", "carbs", "heartrate", "steps",
        "age", "weight", "height", "insulin_delivery_modality", "cgm_device"]
srcs = pq.read_table(f, columns=["source_file"]).column(0).unique().to_pylist()
pd.set_option("display.width", 250)
rows = []
for s in srcs:
    d = pq.read_table(f, columns=cols, filters=[("source_file", "==", s)]).to_pandas()
    sorted_ok = (d.groupby("id", sort=False)["date"].apply(lambda x: x.is_monotonic_increasing)).mean()
    d = d.sort_values(["id", "date"])
    dt = d.groupby("id")["date"].diff().dt.total_seconds().div(60)
    span = d.groupby("id")["date"].agg(lambda x: (x.max() - x.min()).total_seconds() / 86400)
    days = len(d) * dt.median() / 1440
    rows.append(dict(src=s, ids=d.id.nunique(), rows=len(d), sorted=round(sorted_ok, 2),
                     dt_med=dt.median(), dt_mode=dt.mode().iloc[0], dt_p_not5=round((dt != 5).mean(), 3), dup_ts=int((dt == 0).sum()),
                     span_med_d=round(span.median(), 1), cgm_nn=round(d.CGM.notna().mean(), 3),
                     cgm_lt40=round((d.CGM < 40).mean(), 4), cgm_gt400=round((d.CGM > 400).mean(), 4),
                     basal_nn=round(d.basal.notna().mean(), 2), basal_med=d.basal.median(), basal_p99=d.basal.quantile(.99),
                     bolus_nz=round((d.bolus > 0).mean(), 4), bolus_neg=int((d.bolus < 0).sum()),
                     ins_eq=round(np.nanmean(np.isclose(d.insulin, d.basal.fillna(0) + d.bolus.fillna(0))), 3),
                     carbs_nn=round(d.carbs.notna().mean(), 2), carbs_nz=round((d.carbs > 0).mean(), 4),
                     TDD=round(d.groupby("id").insulin.sum().div(span.clip(lower=1)).median(), 1),
                     carbs_day=round(d.groupby("id").carbs.sum().div(span.clip(lower=1)).median(), 0),
                     hr=round(d.heartrate.notna().mean(), 3), steps=round(d.steps.notna().mean(), 3),
                     age_med=d.groupby("id").age.first().median(), wt_med=d.groupby("id").weight.first().median(),
                     ht_med=d.groupby("id").height.first().median(),
                     mod=d.insulin_delivery_modality.value_counts().index[0] if d.insulin_delivery_modality.notna().any() else None,
                     dev=d.cgm_device.value_counts().index[0] if d.cgm_device.notna().any() else None,
                     dmin=str(d.date.min())[:10], dmax=str(d.date.max())[:10]))
    print(rows[-1], flush=True)
    del d
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
print(pd.DataFrame(rows).to_string())
