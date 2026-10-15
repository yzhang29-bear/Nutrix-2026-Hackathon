"""Zero-shot Chronos-2 (time-series foundation model) on the same day-8+ test meals.
usage: python -I run_chronos.py <CGMacros dir> <v1 results dir> <out dir>
Context = previous 24 h of Dexcom on a 5-min grid. Variant B adds a carbohydrate covariate (past meals + the test meal at t=0,
using the photo-estimated carbs, as known-future information).
"""
import sys, os, json, time, pickle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, torch
from chronos import Chronos2Pipeline
from pmrf.data import load_cgmacros
from pmrf.pipeline import metrics
from pmrf.models.icl_transformer import curve_to_targets

DATA, V1, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(16)
persons = load_cgmacros(DATA)
TE = pd.read_pickle(os.path.join(V1, "test_meals.pkl"))
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
H = 288   # 24 h of 5-min context
inputs_a, inputs_b = [], []
for r in TE.itertuples():
    p = persons[r.pid]
    t = pd.Timestamp(r.time).floor("min")
    idx = t + pd.to_timedelta(np.arange(-H + 1, 1) * 5, unit="min")
    g = p.cgm.reindex(idx).interpolate(limit_direction="both").to_numpy(float)
    carb_past = np.zeros(H)
    mt = pd.to_datetime(p.meals.time)
    for mtime, c in zip(mt, p.meals.carbs):
        k = int(round((mtime - idx[0]).total_seconds() / 300))
        if 0 <= k < H - 1 and mtime < t:
            carb_past[k] += c
    fut = np.zeros(24); fut[0] = r.est_carbs if np.isfinite(r.est_carbs) else 0
    inputs_a.append(torch.tensor(g, dtype=torch.float32))
    inputs_b.append(dict(target=g, past_covariates=dict(carbs=carb_past), future_covariates=dict(carbs=fut)))
res, outs = [], {}
subsets = {"all": np.ones(len(TE), bool), "unseen": ~TE.seen.to_numpy()}
for name, inp in [("chronos2_zero_shot", inputs_a), ("chronos2_with_carb_covariate", inputs_b)]:
    t0 = time.time()
    fc = pipe.predict(inp, prediction_length=24, batch_size=64)
    ms = (time.time() - t0) / len(inp) * 1000
    q = np.stack([f.numpy().reshape(-1, 24) for f in fc])          # (n, n_quantiles, 24)
    qi = pipe.quantiles.index(0.5) if hasattr(pipe, "quantiles") else q.shape[1] // 2
    med = q[:, qi, :]
    g0 = TE.G0.to_numpy()
    curve = np.concatenate([np.zeros((len(TE), 1)), med - g0[:, None]], 1)   # 0..120 min
    out = curve_to_targets(curve, g0)
    outs[name] = out
    res.append(metrics(TE, out, subsets).assign(model=name, ms_per_meal=ms))
    print(name, f"{ms:.0f} ms per meal", flush=True)
R = pd.concat(res, ignore_index=True)
R.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
print(R[R.subset == "unseen"].pivot(index="model", columns="target", values="mae").round(1).to_string())
print(R[R.subset == "unseen"].pivot(index="model", columns="target", values="r").round(2).to_string())
pickle.dump(outs, open(os.path.join(OUT, "preds.pkl"), "wb"))
