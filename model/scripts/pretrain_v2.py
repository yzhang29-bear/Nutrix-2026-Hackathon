"""Pretrain the in-context transformer with v2 inputs (v1 + age, sex, BMI) on the MetaboNet T1D meal events.
Same events, steps, learning rate and validation split as scripts/run_transformer.py; runs on the GPU when PMRF_DEVICE=cuda.
usage: PMRF_DEVICE=cuda python -I pretrain_v2.py <t1d_events.npz> <person info csv> <out dir> [steps]
"""
import sys, os, json, time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, torch
from pmrf.models.icl_transformer import ICLTransformer, EpisodeData, numeric_block, train, predict, pinball, CURVE_SCALE, NUM_COLS_V2, DEVICE

EV, INFO, OUT = sys.argv[1:4]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 30000
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
T0 = time.time()
z = np.load(EV, allow_pickle=True)
ev = pd.DataFrame(z["meta"]); pre_raw = z["pre"].astype(float); post_raw = z["post"].astype(float)
g0 = pre_raw[:, -1]
ev["k_carbs"] = ev.carbs
ev["tod_sin"], ev["tod_cos"] = np.sin(2 * np.pi * ev.hour / 24), np.cos(2 * np.pi * ev.hour / 24)
ev["G0"] = g0
ev["slope30"] = (g0 - pre_raw[:, -7]) / 30; ev["slope60"] = (g0 - pre_raw[:, -13]) / 60
ev["sd60"] = np.nanstd(pre_raw[:, -13:], 1)
ev["prev_carbs"] = ev.prev_carbs_6h
ev["hr_pre"] = np.nan                    # absolute HR without a resting reference; not comparable with CGMacros (as in v1)
info = pd.read_csv(INFO).set_index("key")
h_in = info.height * 12.0                                     # MetaboNet: height in feet, weight in pounds
bmi = (703.0 * info.weight / h_in ** 2).where(lambda b: b.between(10, 70))
ev["sex_female"] = ev.key.map(info.gender.map({"Female": 1.0, "Male": 0.0}))
ev["bmi"] = ev.key.map(bmi)
print(f"events {len(ev)}; with sex {ev.sex_female.notna().mean():.0%}, BMI {ev.bmi.notna().mean():.0%}, age {ev.age.notna().mean():.0%}; device {DEVICE}", flush=True)
t1 = EpisodeData(numeric_block(ev, NUM_COLS_V2), pre_raw - g0[:, None], post_raw - g0[:, None], np.zeros((len(ev), 512), np.float32),
                 np.zeros(len(ev)), ev.key.to_numpy())
keys = ev.key.to_numpy()
uk = np.unique(keys); rng = np.random.default_rng(0)
val_keys = set(rng.choice(uk, max(20, len(uk) // 20), replace=False))
by_key = pd.Series(np.arange(len(ev))).groupby(keys).apply(lambda s: s.to_numpy()).to_dict()
is_val = np.array([k in val_keys for k in keys])
tr_q = np.flatnonzero(~is_val); va_q = rng.choice(np.flatnonzero(is_val), 3000, replace=False)
hist_t1 = lambda i: by_key[keys[i]][by_key[keys[i]] != i]
va_ctx = [rng.choice(hist_t1(i), min(32, len(hist_t1(i))), replace=False) for i in va_q]


def val_t1(model):
    p = predict(model, t1, va_q, va_ctx) / CURVE_SCALE
    return float(pinball(torch.tensor(p), t1.resp[va_q], t1.resp_mask[va_q]))


model = ICLTransformer(img_dim=512, n_num=len(NUM_COLS_V2))
t = time.time()
model, hist = train(model, t1, tr_q, hist_t1, STEPS, lr=5e-4, batch=128, kmax=32, mask_query_known=None, val=val_t1, val_every=1000, log_every=1000)
print(f"pretrain {STEPS} steps in {time.time() - t:.0f}s, best val {min(h[2] for h in hist):.4f}", flush=True)
torch.save(model.state_dict(), os.path.join(OUT, "pretrained_t1d_v2.pt"))
json.dump(dict(steps=STEPS, num_cols="v2", hist=hist, seconds=time.time() - T0), open(os.path.join(OUT, "pretrain_v2.json"), "w"), indent=1)
print("saved", os.path.join(OUT, "pretrained_t1d_v2.pt"))
