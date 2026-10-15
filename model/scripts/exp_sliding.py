"""History window for later days: fixed (days 1-7), sliding (the last 7 days before the meal's day) or growing (all days
before it). The models are not retrained; only the person's history state changes. Evaluated on labelled CGMacros
meals of days 8-10 (day 8 is identical for all three).
usage: python -I exp_sliding.py <CGMacros dir> <model store dir> <stage1.pkl> <out dir>
"""
import sys, os, json, pickle, time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, torch
from pmrf.data import load_cgmacros, Person
from pmrf.store import load_store
from pmrf.predict import Predictor
from pmrf.state import build_state, open_image

DATA, STORE, S1, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(16)
T0 = time.time()
pred = Predictor(load_store(STORE), threads=16)
persons = load_cgmacros(DATA)
TE = pickle.load(open(S1, "rb"))["TE"]
TE = TE[TE.day.between(8, 10)]
cache, rows = {}, []


def state_for(pid, start, ndays):
    """history = days start .. start+ndays-1 (original numbering): shift day 1 to `start`"""
    k = (pid, start, ndays)
    if k not in cache:
        p = persons[pid]
        q = Person(pid, p.cgm, p.hr, p.meals, p.meds, p.t0 + pd.Timedelta(days=start - 1), p.meta)
        cache[k] = build_state(q, pred, open_image, ndays)
    return cache[k]


for n, (pid, g) in enumerate(TE.groupby("pid")):
    p = persons[pid]
    for i, r in g.iterrows():
        d = int(r.day)
        windows = {"fixed (days 1-7)": (1, 7), "sliding (last 7 days)": (d - 7, 7), "growing (days 1 to d-1)": (1, d - 1)}
        for name, (start, nd) in windows.items():
            st = state_for(pid, start, nd)
            out = pred.predict_meal(p, r.photo, r.time, r.meal_type, st)[0]
            rows.append(dict(pid=pid, meal=int(i), day=d, seen=bool(r.seen), window=name, n_hist=len(st["hist"]),
                             dpeak=float(out["dpeak"][0]), G60=float(out["G60"][0]), iauc=float(out["iauc"][0]), ttp=float(out["ttp"][0]),
                             obs_dpeak=r.dpeak, obs_G60=r.G60, obs_iauc=r.iauc, obs_ttp=r.ttp))
    print(f"{n + 1}/{TE.pid.nunique()} people [{time.time() - T0:.0f}s]", flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
for t in ["dpeak", "G60", "iauc", "ttp"]:
    D[f"ae_{t}"] = (D[t] - D[f"obs_{t}"]).abs()
summ = {}
for sub, m in [("days 9-10, all", D.day >= 9), ("days 9-10, unseen recipes", (D.day >= 9) & ~D.seen), ("day 8 (identical)", D.day == 8)]:
    summ[sub] = D[m].groupby("window")[["ae_dpeak", "ae_G60", "ae_iauc", "ae_ttp"]].mean().round(2).to_dict("index")
    summ[sub]["n_meals_per_window"] = int(m.sum() / 3)
rng = np.random.default_rng(0)
m9 = D[D.day >= 9]
piv = m9.pivot_table(index="meal", columns="window", values="ae_dpeak"); pids = m9.groupby("meal").pid.first().reindex(piv.index).to_numpy()
up = np.unique(pids); pos = {q: np.flatnonzero(pids == q) for q in up}
for alt in ["sliding (last 7 days)", "growing (days 1 to d-1)"]:
    dlt = (piv[alt] - piv["fixed (days 1-7)"]).to_numpy()
    b = [np.nanmean(dlt[np.concatenate([pos[q] for q in rng.choice(up, len(up))])]) for _ in range(1000)]
    summ[f"dpeak change vs fixed, {alt}"] = dict(mean=round(float(np.nanmean(dlt)), 2), lo=round(float(np.percentile(b, 2.5)), 2), hi=round(float(np.percentile(b, 97.5)), 2))
summ["history meals per state (median)"] = D.groupby("window").n_hist.median().to_dict()
json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
