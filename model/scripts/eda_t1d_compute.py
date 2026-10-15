"""T1D harmonized parquet EDA, compute pass (one source at a time to bound memory).
usage: python -I eda_t1d_compute.py <parquet> <out dir>
writes t1d_subjects.csv, t1d_arrays.npz, t1d_stats.json
"""
import sys, os, json, re
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc

F, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
NUM = ["date", "CGM", "basal", "bolus", "insulin", "carbs", "heartrate", "steps"]
STATIC = ["age", "age_of_diagnosis", "gender", "ethnicity", "weight", "height", "insulin_delivery_modality",
          "insulin_delivery_algorithm", "insulin_delivery_device", "cgm_device", "treatment_group",
          "is_pregnant", "subject_split_across_traintest"]
srcs = sorted(pq.read_table(F, columns=["source_file"]).column(0).unique().to_pylist())

GBINS = np.arange(0, 505, 5)            # CGM histogram bins (mg/dL)
GAP_EDGES = [1, 2, 4, 7, 13, 25, 289, 10 ** 9]   # missing-run lengths in samples (5 min each)
GAP_LABELS = ["5 min", "10-15 min", "20-30 min", "35-60 min", "1-2 h", "2-24 h", ">24 h"]
PP = np.arange(-6, 37)                  # -30 .. +180 min in 5-min steps

subj_rows, hourly, pp_curves, pp_meta = [], [], [], []
cgm_hist = {}
gap_counts = {s: np.zeros(len(GAP_LABELS)) for s in srcs}
gap_minutes = {s: np.zeros(len(GAP_LABELS)) for s in srcs}
nonint = {}
avail = {}
carb_hour = {s: np.zeros(24) for s in srcs}
carb_sizes = {}
bolus_carb = []

def runs(mask):
    """lengths of consecutive True runs"""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)

for s in srcs:
    t = pq.read_table(F, columns=["id"] + NUM + STATIC, filters=[("source_file", "==", s)],
                      read_dictionary=["id"] + [c for c in STATIC if c not in ("age", "age_of_diagnosis", "weight", "height", "is_pregnant", "subject_split_across_traintest")])
    d = t.to_pandas()
    del t
    d = d.sort_values(["id", "date"], kind="stable").reset_index(drop=True)
    n = len(d)
    avail[s] = {c: float(d[c].notna().mean()) for c in NUM[1:] + STATIC}
    avail[s]["carbs>0"] = float((d.carbs > 0).mean())
    g = d.CGM.to_numpy()
    gv = g[~np.isnan(g)]
    cgm_hist[s] = np.histogram(np.clip(gv, 0, 504.9), bins=GBINS)[0]
    nonint[s] = float(np.mean(np.abs(gv - np.round(gv)) > 1e-6)) if len(gv) else np.nan
    for sid, idx in d.groupby("id", observed=True, sort=False).indices.items():
        x = d.iloc[idx]
        cg = x.CGM.to_numpy()
        days = len(x) * 5 / 1440
        ok = ~np.isnan(cg)
        v = cg[ok]
        r = dict(source=s, id=str(sid), key=f"{s}:{sid}", rows=len(x), days=days,
                 span_days=(x.date.iloc[-1] - x.date.iloc[0]).total_seconds() / 86400,
                 start=str(x.date.iloc[0])[:10], end=str(x.date.iloc[-1])[:10],
                 cgm_cov=ok.mean(), cgm_days=ok.sum() * 5 / 1440)
        if len(v) >= 288:   # at least one day of CGM
            r.update(mean_g=v.mean(), sd_g=v.std(), cv=v.std() / v.mean() * 100,
                     tir=np.mean((v >= 70) & (v <= 180)) * 100, tbr70=np.mean(v < 70) * 100, tbr54=np.mean(v < 54) * 100,
                     tar180=np.mean(v > 180) * 100, tar250=np.mean(v > 250) * 100, gmi=3.31 + 0.02392 * v.mean(),
                     cgm_lt40=np.mean(v < 40) * 100, cgm_gt400=np.mean(v > 400) * 100)
            hr_ = x.date.dt.hour.to_numpy()[ok]
            med = np.full(24, np.nan)
            for h in range(24):
                vv = v[hr_ == h]
                if len(vv) >= 24: med[h] = np.median(vv)
            hourly.append(med)
        else:
            hourly.append(np.full(24, np.nan))
        r["hourly_idx"] = len(hourly) - 1
        # CGM gaps: runs of missing samples inside the subject's record
        L = runs(~ok)
        if len(L):
            b = np.digitize(L, GAP_EDGES) - 1
            np.add.at(gap_counts[s], b, 1)
            np.add.at(gap_minutes[s], b, L * 5)
        # insulin / carbs per day (only meaningful where the source logs them)
        ins, bas, bol, cb = x.insulin.to_numpy(), x.basal.to_numpy(), x.bolus.to_numpy(), x.carbs.to_numpy()
        tn = x.date.to_numpy().astype("int64") // 10 ** 9   # seconds
        hrs = x.date.dt.hour.to_numpy()
        r.update(ins_cov=np.mean(~np.isnan(ins)), tdd=np.nansum(ins) / days if np.any(~np.isnan(ins)) else np.nan,
                 basal_day=np.nansum(bas) / days if np.any(~np.isnan(bas)) else np.nan,
                 bolus_day=np.nansum(np.clip(bol, 0, None)) / days if np.any(~np.isnan(bol)) else np.nan,
                 bolus_neg=int(np.sum(bol < 0)),
                 carb_entries_day=np.sum(cb > 0) / days, carbs_day=np.nansum(cb) / days if np.any(cb > 0) else np.nan,
                 hr_cov=x.heartrate.notna().mean(), steps_cov=x.steps.notna().mean())
        for c in STATIC:
            vals = x[c].dropna()
            r[c] = vals.iloc[0] if len(vals) else None
            if c in ("age", "weight", "height", "insulin_delivery_modality", "cgm_device", "treatment_group"):
                r[c + "_nvals"] = vals.nunique()
        r["is_pregnant"] = bool(x.is_pregnant.fillna(False).any())
        r["split_flag"] = bool(x.subject_split_across_traintest.any())
        # carb events: timing, size, bolus pairing, postprandial CGM
        ci = np.flatnonzero(cb > 0)
        if len(ci):
            np.add.at(carb_hour[s], hrs[ci], 1)
            carb_sizes.setdefault(s, []).append(cb[ci])
            for i in ci:
                if cb[i] < 20 or i + PP[-1] >= len(x) or i + PP[0] < 0: continue
                w = slice(i + PP[0], i + PP[-1] + 1)
                if tn[i + PP[-1]] - tn[i + PP[0]] != (PP[-1] - PP[0]) * 300: continue
                other = cb[i + 1:i + PP[-1] + 1]
                if np.nansum(other) >= 5: continue
                prior = cb[max(0, i - 36):i]
                if np.nansum(prior) >= 5: continue
                seg = cg[w]
                if np.mean(np.isnan(seg)) > 0.1 or np.isnan(cg[i]): continue
                bwin = bol[max(0, i - 3):i + 4]
                pp_curves.append(seg - cg[i])
                pp_meta.append((s, f"{s}:{sid}", cb[i], np.nansum(np.clip(bwin, 0, None)), cg[i], hrs[i]))
        subj_rows.append(r)
    print(s, n, "subjects", d.id.nunique(), flush=True)
    del d

S = pd.DataFrame(subj_rows)
S.to_csv(os.path.join(OUT, "t1d_subjects.csv"), index=False)
pm = pd.DataFrame(pp_meta, columns=["source", "key", "carbs", "bolus_pm15", "g0", "hour"])
pm.to_csv(os.path.join(OUT, "t1d_pp_meta.csv"), index=False)
np.savez_compressed(os.path.join(OUT, "t1d_arrays.npz"), hourly=np.array(hourly), pp=np.array(pp_curves),
                    gbins=GBINS, cgm_hist=np.array([cgm_hist[s] for s in srcs]), srcs=np.array(srcs),
                    gap_counts=np.array([gap_counts[s] for s in srcs]), gap_minutes=np.array([gap_minutes[s] for s in srcs]),
                    carb_hour=np.array([carb_hour[s] for s in srcs]),
                    carb_sizes=np.concatenate([np.concatenate(v) for v in carb_sizes.values()]),
                    carb_sizes_src=np.concatenate([[k] * sum(len(a) for a in v) for k, v in carb_sizes.items()]))
json.dump(dict(avail=avail, nonint=nonint, gap_labels=GAP_LABELS), open(os.path.join(OUT, "t1d_stats.json"), "w"), indent=1)
print("done", len(S), "subjects;", len(pp_curves), "clean carb events")
