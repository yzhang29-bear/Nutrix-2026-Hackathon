"""Does a DINOv2 photo encoder beat MobileCLIP-S2? Same LightGBM configuration as the final model (stage 1); only the photo
embedding changes. Decision metric: rolling inner validation (history cut-off at day 4, 5, 6; validate on CGMacros meals up to
day 7). The day-8+ test numbers are reported for comparison with the earlier encoder table, not used to choose.
usage: python -I exp_encoder_dino.py <CGMacros dir> <BIG IDEAs dir> <stage1 dir> <out dir> name=npz[+npz] ...
"""
import sys, os, json, pickle, time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd
from pmrf.data import load_cgmacros, load_bigideas
from pmrf.features import meal_table, person_profile
from pmrf.pipeline import photo_features, attach_embeddings, to_challenge_outputs, metrics
from pmrf.models.gbm import _fit_bag

DATA, BIG, S1, OUT = sys.argv[1:5]
ENC = [a.split("=", 1) for a in sys.argv[5:]]
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
FE, PRM, LOGT = list(st["FEATS"]), dict(st["params"]), dict(st.get("log_targets", {}))
YS = ["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"]
persons = {**load_cgmacros(DATA), **load_bigideas(BIG)}


def load_npz(spec):
    """one npz, or several joined with '+' (each part L2-normalised, concatenated, renormalised)"""
    parts = [np.load(os.path.join(ROOT_DIR, p), allow_pickle=True) for p in spec.split("+")]
    if len(parts) == 1:
        return parts[0]["path"], parts[0]["emb"]
    key = [{p: i for i, p in enumerate(z["path"])} for z in parts]
    paths = [p for p in parts[0]["path"] if all(p in k for k in key)]
    E = np.concatenate([z["emb"][[k[p] for p in paths]] for z, k in zip(parts, key)], 1)
    return np.array(paths), (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)


def point_predict(tr, te, y):
    off = LOGT.get(y)
    yt = np.log(tr[y] + off) if off is not None else tr[y]
    p = _fit_bag(tr[FE], yt, (0, 1, 2), params=PRM).predict(te[FE])
    return np.exp(p) - off if off is not None else p


TABLES = {hd: meal_table(persons, hd) for hd in (4, 5, 6, 7)}
print(f"meal tables [{time.time() - T0:.0f}s]", flush=True)
inner_rows, test_rows = [], []
for name, spec in ENC:
    paths, emb = load_npz(spec)
    tmp = os.path.join(OUT, f"_emb_{abs(hash(spec)) % 10 ** 8}.npz")
    np.savez(tmp, path=paths, emb=emb)
    for hd, (M0, CURV, PRE, HRS) in TABLES.items():
        M, EMB = attach_embeddings(M0, tmp, DATA)
        M["in_hist"] = M.day <= hd
        for mt in ["breakfast", "lunch", "dinner", "snack"]:
            M[f"is_{mt}"] = (M.meal_type == mt).astype(float)
        M, mest, pca, E = photo_features(M, EMB)
        M = M.join(person_profile(persons, M, M.in_hist.to_numpy(), hd))
        cg, ph = ~M.pid.str.startswith("B"), M.emb_idx >= 0
        TR = M[(cg & M.in_hist & M.label & ph) | (~cg & M.label)]
        if hd < 7:
            VA = M[cg & (M.day > hd) & (M.day <= 7) & M.label & ph]
            for y in YS:
                p = point_predict(TR, VA, y)
                inner_rows.append(dict(encoder=name, cutoff=hd, target=y, n=len(VA), mae=float(np.mean(np.abs(p - VA[y])))))
        else:
            TE = M[cg & ~M.in_hist & M.label & ph]
            P = {y: dict(pred=point_predict(TR, TE, y)) for y in YS}
            test_rows.append(metrics(TE, to_challenge_outputs(TE, P), {"test (day 8+)": np.ones(len(TE), bool)}).assign(encoder=name))
            tp = (~M.in_hist) & ph & cg
            for c in ["carbs", "protein", "fat", "kcal"]:
                o, q = M.loc[tp, c], M.loc[tp, f"est_{c}"]
                test_rows.append(pd.DataFrame([dict(encoder=name, subset="photo nutrition (day 8+)", target=c, n=int(tp.sum()),
                                                    mae=float(np.mean(np.abs(o - q))), r=float(np.corrcoef(o, q)[0, 1]))]))
        print(name, "cut-off", hd, f"[{time.time() - T0:.0f}s]", flush=True)
    os.remove(tmp)

I = pd.DataFrame(inner_rows)
I.to_csv(os.path.join(OUT, "inner_validation.csv"), index=False)
T = pd.concat(test_rows, ignore_index=True)
T.to_csv(os.path.join(OUT, "test.csv"), index=False)
ref = ENC[0][0]
piv = I.pivot_table(index="encoder", columns="target", values="mae", aggfunc="mean")
summ = dict(reference=ref, inner_mean_mae=piv.round(2).to_dict("index"),
            inner_change_vs_reference_pct=((piv / piv.loc[ref] - 1) * 100).round(1).to_dict("index"),
            test_mae=T[T.subset == "test (day 8+)"].pivot_table(index="encoder", columns="target", values="mae").round(2).to_dict("index"),
            photo_nutrition_test_mae=T[T.subset != "test (day 8+)"].pivot_table(index="encoder", columns="target", values="mae").round(1).to_dict("index"))
json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
