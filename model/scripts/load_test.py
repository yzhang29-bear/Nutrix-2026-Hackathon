"""Load test: concurrent /predict calls on the imported cohort's demo meals; reports throughput and latency percentiles.
usage: python -I load_test.py <url> <concurrency> <requests> <out.json>
"""
import sys, os, json, time, pickle, threading
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, httpx
from concurrent.futures import ThreadPoolExecutor

URL, CONC, N, OUT = sys.argv[1].rstrip("/"), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
st = pickle.load(open(os.path.join(ROOT_DIR, "results", "final", "stage1.pkl"), "rb"))
Mo, TE = st["outer"]["M"], st["TE"]
DATA = os.path.join(ROOT_DIR, "cgmacros", "CGMacros")
reqs = [(Mo.loc[i].pid, dict(photo_key=os.path.relpath(Mo.loc[i].photo, DATA), meal_time=str(Mo.loc[i].time), meal_type=Mo.loc[i].meal_type,
                             log_meal=False)) for i in TE.index]
local = threading.local()


def one(k):
    if not hasattr(local, "c"):
        local.c = httpx.Client(base_url=URL, timeout=120)
    pid, body = reqs[k % len(reqs)]
    t = time.perf_counter()
    r = local.c.post(f"/v1/people/{pid}/predict", json=body)
    return (time.perf_counter() - t) * 1000, r.status_code


t0 = time.perf_counter()
with ThreadPoolExecutor(CONC) as ex:
    res = list(ex.map(one, range(N)))
wall = time.perf_counter() - t0
lat = np.array([r[0] for r in res]); ok = np.array([r[1] == 200 for r in res])
out = dict(url=URL, concurrency=CONC, requests=N, errors=int((~ok).sum()), throughput_per_s=round(N / wall, 2),
           latency_ms=dict(p50=round(float(np.percentile(lat, 50))), p90=round(float(np.percentile(lat, 90))), p99=round(float(np.percentile(lat, 99)))))
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out))
