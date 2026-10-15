"""Single-meal inference: photo + recent CGM/HR + the person's history -> API JSON.
The demo page and the future REST API both use `build_response`, so they share one contract (see docs/api_contract.md).
"""
import time
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator
from PIL import Image

from .features import meal_table, person_profile, CURVE_T, TARGETS
from .pipeline import N_PCA, FEATURE_GROUPS, enforce_consistency
from .vision import PhotoEncoder, FoodTagger, FOOD_PROMPTS, _knn, MACROS
from .models.icl_transformer import ICLTransformer, EpisodeData, numeric_block, predict as tf_predict, predict_scalars, curve_to_targets, NUM_COLS, NUM_COLS_BY_VERSION
import dataclasses
from .guide import exceed_prob

API_VERSION = "0.1"
KNOWN = [NUM_COLS.index(c) for c in ["k_carbs", "k_protein", "k_fat", "k_fiber"]]
WHAT_IF = [("As photographed", dict()), ("Half the carbs", dict(carbs=0.5)), ("Add 20 g protein", dict(protein_add=20)),
           ("Add 10 g fiber", dict(fiber_add=10)), ("Double portion", dict(carbs=2.0, protein=2.0, fat=2.0, fiber=2.0))]


def _r(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def confidence(row, n_hist, hist_types, hist_carbs_max, photo_sim):
    flags = []
    if n_hist < 10:
        flags.append(f"Only {n_hist} usable meals in the 7-day history")
    if row["meal_type"] not in hist_types:
        flags.append(f"No {row['meal_type']} in the 7-day history")
    if np.isfinite(row.get("est_carbs", np.nan)) and row["est_carbs"] > hist_carbs_max:
        flags.append("More carbohydrate than any meal in the history")
    if np.isfinite(photo_sim) and photo_sim < 0.6:
        flags.append("Photo looks unlike any meal the model has seen")
    if not np.isfinite(row.get("slope30", np.nan)):
        flags.append("Gap in CGM just before the meal")
    level = "high" if not flags else ("medium" if len(flags) == 1 else "low")
    return level, flags


def group_contributions(contrib_row, features):
    out = []
    for g, cols in FEATURE_GROUPS.items():
        v = float(sum(contrib_row[c] for c in cols if c in features))
        if any(c in features for c in cols):
            out.append(dict(factor=g, mgdl=round(v, 1)))
    return sorted(out, key=lambda d: -abs(d["mgdl"]))


def build_response(pid, row, pred, band, contrib, base, profile, pop_avg_rise, similar, what_if, conf, tags, latency=None, observed=None,
                   probs=None):
    """Assemble the API JSON for one meal. `pred` holds the 8 challenge outputs (+ 'curve'), `band` the 80% curve band."""
    t = list(CURVE_T.astype(int))
    g0 = float(row["G0"])
    resp = dict(api_version=API_VERSION, person_id=pid,
                meal=dict(time=str(row["time"]), meal_type=row["meal_type"], photo_tags=[dict(label=l, p=_r(p, 2)) for l, p in tags],
                          estimated_nutrition=dict(carbs_g=_r(row["est_carbs"]), protein_g=_r(row["est_protein"]), fat_g=_r(row["est_fat"]),
                                                   fiber_g=_r(row["est_fiber"]), kcal=_r(row["est_kcal"], 0),
                                                   carbs_uncertainty_g=_r(row.get("est_carbs_sd", np.nan))),
                          similar_past_meals=similar),
                pre_meal_state=dict(glucose_mgdl=_r(g0), trend_mgdl_per_min=_r(row.get("slope30", np.nan), 2),
                                    glucose_vs_24h_median_mgdl=_r(row.get("g0_rel24", np.nan)),
                                    heart_rate_vs_resting_bpm=_r(row.get("hr_pre", np.nan)),
                                    minutes_since_last_meal=_r(row.get("prev_gap", np.nan), 0)),
                prediction=dict(curve=dict(minutes=t, glucose_mgdl=[_r(g0 + v) for v in pred["curve"]],
                                           lower80=[_r(g0 + v) for v in band[0]], upper80=[_r(g0 + v) for v in band[1]]),
                                glucose_at_min={str(k): _r(pred[f"G{k}"]) for k in (30, 60, 90, 120)},
                                peak_glucose_mgdl=_r(pred["Gmax"]), peak_rise_mgdl=_r(pred["dpeak"]),
                                time_to_peak_min=_r(pred["ttp"], 0), iauc_0_120_mgdl_min=_r(pred["iauc"], 0),
                                prob_peak_above_mgdl={k: _r(v, 3) for k, v in (probs or {}).items()}),
                explanation=dict(target="peak_rise_mgdl", population_baseline_mgdl=_r(base), contributions=contrib),
                personalization=dict(history_meals=int(profile.get("p_n", 0) or 0), your_avg_peak_rise_mgdl=_r(profile.get("p_dpeak", np.nan)),
                                     population_avg_peak_rise_mgdl=_r(pop_avg_rise),
                                     your_rise_per_10g_carbs_mgdl=_r(profile.get("p_dpeak_per10g", np.nan)),
                                     your_mean_glucose_mgdl=_r(profile.get("p_g_mean", np.nan)), your_resting_hr_bpm=_r(profile.get("p_hr_rest", np.nan))),
                confidence=dict(level=conf[0], flags=conf[1]),
                what_if=what_if)
    if latency is not None:
        resp["latency_ms"] = {k: _r(v) for k, v in latency.items()}
    if observed is not None:
        resp["observed"] = observed
    return resp


class Predictor:
    """Loads the trained components and predicts one meal end-to-end."""

    def __init__(self, bundle, encoder=None, threads=8):
        self.b = bundle
        self.enc = encoder or PhotoEncoder(*bundle["encoder"], threads=threads, preprocess=bundle.get("encoder_preprocess"))
        self.tagger = FoodTagger(bundle["text_emb"] if "text_emb" in bundle else self.enc.encode_texts(FOOD_PROMPTS))
        self.tf = []
        self.cols = NUM_COLS_BY_VERSION[bundle.get("num_cols", "v1")]      # transformer input columns (v1 = v4 model, v2 = v7)
        for sd in bundle["transformer_states"]:
            m = ICLTransformer(img_dim=bundle["img_dim"], n_num=len(self.cols)); m.load_state_dict(sd, strict=False); m.eval(); self.tf.append(m)

    def predict_meal(self, person, photo, meal_time, meal_type, state=None, logged_kcal=None):
        """person: pmrf.data.Person with CGM/HR up to the meal; photo: PIL image or path. The person's history state
        (features, curves and photo embeddings of days 1-7, profile; built once by pmrf.state.build_state) is passed in,
        or taken from the bundle for the development cohort."""
        b, lat = self.b, {}
        state = state if state is not None else b["person_state"][person.pid]
        H, prof = state["hist"], state["profile"]
        t0 = time.perf_counter()
        img = Image.open(photo) if isinstance(photo, str) else photo
        e, lat["photo_encoder"] = self.enc.timed_encode(img)
        t = time.perf_counter()
        # people of the development cohort are part of the fitted photo library (the trained path); new users bring their own
        own = None if state.get("in_library") else state.get("own_lib")
        est, sd, simp, _ = b["macro_est"].predict(e[None], np.array([person.pid]), own=own)
        tags = self.tagger.tags(e[None])[0]
        lat["nutrition_and_tags"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        p = person
        mt_ = pd.Timestamp(meal_time)
        recent = p.meals[(pd.to_datetime(p.meals.time) < mt_) & (pd.to_datetime(p.meals.time) >= mt_ - pd.Timedelta("12h"))]
        meals = pd.concat([recent, pd.DataFrame([dict(time=mt_, meal_type=meal_type, carbs=np.nan, protein=np.nan, fat=np.nan,
                                                      fiber=np.nan, kcal=np.nan, photo=None,
                                                      calories_logged=np.nan if logged_kcal is None else float(logged_kcal))])], ignore_index=True)
        tmp = dataclasses.replace(p, meals=meals)          # keeps activity and person info
        M, _, PRE, _ = meal_table({p.pid: tmp})
        k = M.index[M.time == mt_][-1]
        row, pre = M.loc[k].copy(), PRE[k]
        for j, c in enumerate(MACROS):
            row[f"est_{c}"] = est[0, j]
        row["est_kcal"] = 4 * est[0, 0] + 4 * est[0, 1] + 9 * est[0, 2]
        row["est_carbs_sd"], row["photo_sim_pop"] = sd[0, 0], simp[0]
        pc = b["pca"].transform(e[None])[0]
        for i in range(N_PCA):
            row[f"img{i}"] = pc[i]
        for m_ in ["breakfast", "lunch", "dinner", "snack"]:
            row[f"is_{m_}"] = float(meal_type == m_)
        mp, _, _, _ = _knn(e[None], b["lib_emb"], b["lib_y"], 10, (b["lib_pid"] == p.pid)[None])
        row["pop_nn_dpeak"], row["pop_nn_iauc"] = mp[0, 0], mp[0, 1]
        HE = np.stack(H.emb.to_numpy()); HY = H[["dpeak", "iauc", "d60"]].to_numpy(float)
        mo, _, so, idx = _knn(e[None], HE, HY, 3)
        row["own_nn_dpeak"], row["own_nn_iauc"], row["own_nn_d60"], row["own_nn_sim"] = mo[0, 0], mo[0, 1], mo[0, 2], so[0]
        for c, v in prof.items():
            row[c] = v
        lat["features"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        X = pd.DataFrame([row])
        out = self._predict_rows(X, pre[None], H, e[None])
        lat["response_models"] = (time.perf_counter() - t) * 1000
        lat["total"] = (time.perf_counter() - t0) * 1000
        return out, X, pre, e, tags, lat, idx[0]

    def _predict_rows(self, X, pre, H, emb, contributions=True):
        """blend of bagged LightGBM and the in-context transformer for already-featurized query rows"""
        b = self.b
        P = b["gbm"].predict(X, contributions=["dpeak"] if contributions else False)
        g = {f"G{k}": X.G0.to_numpy() + P[f"d{k}"]["pred"] for k in (30, 60, 90, 120)}
        g.update(Gmax=X.G0.to_numpy() + P["dpeak"]["pred"], dpeak=P["dpeak"]["pred"], ttp=np.clip(P["ttp"]["pred"], 0, 120), iauc=np.clip(P["iauc"]["pred"], 0, None))
        # transformer episode: query + the person's history meals with their observed curves
        qdf = X.copy()
        for c in ["carbs", "protein", "fat", "fiber"]:
            qdf[f"e_{c}"] = qdf[f"est_{c}"]; qdf[f"k_{c}"] = np.nan
        qdf["e_carbs_sd"] = qdf.est_carbs_sd
        for mt in ["breakfast", "lunch", "dinner", "snack"]:
            qdf[f"mt_{mt}"] = qdf[f"is_{mt}"]
        hdf = H.copy()
        for c in ["carbs", "protein", "fat", "fiber"]:
            hdf[f"e_{c}"] = hdf[f"est_{c}"]; hdf[f"k_{c}"] = hdf[c]
        hdf["e_carbs_sd"] = hdf.est_carbs_sd
        for mt in ["breakfast", "lunch", "dinner", "snack"]:
            hdf[f"mt_{mt}"] = hdf[f"is_{mt}"]
        num = np.concatenate([numeric_block(qdf, self.cols), numeric_block(hdf, self.cols)])
        preseq = np.concatenate([pre, np.stack(H.pre.to_numpy())])
        resp = np.concatenate([np.full((len(qdf), len(CURVE_T)), np.nan), np.stack(H.curve.to_numpy())])
        img = np.concatenate([emb, np.stack(H.emb.to_numpy())])
        ed = EpisodeData(num, preseq, resp, img, np.ones(len(img)), np.zeros(len(img)))
        ctx = [np.arange(len(qdf), len(img))] * len(qdf)
        Pt = np.mean([tf_predict(m, ed, np.arange(len(qdf)), ctx, KNOWN) for m in self.tf], 0)
        tfo = curve_to_targets(Pt[:, :, 1], X.G0.to_numpy())
        src = b.get("tf_sources", {})
        if any(v == "head" for v in src.values()):
            St = np.mean([predict_scalars(m, ed, np.arange(len(qdf)), ctx, KNOWN) for m in self.tf], 0)
            heads = {"dpeak": St[:, 0, 1], "iauc": St[:, 1, 1], "ttp": St[:, 2, 1]}
            heads["Gmax"] = X.G0.to_numpy() + heads["dpeak"]
            for t, sname in src.items():
                if sname == "head":
                    tfo[t] = heads[t]
        W = b["blend_weights"]
        out = {k: W[k] * tfo[k] + (1 - W[k]) * g[k] for k in g}
        tk = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120])
        gc = np.stack([PchipInterpolator(tk, np.r_[0, [P[f"d{k}"]["pred"][i] for k in tk[1:]]])(CURVE_T) for i in range(len(X))])
        wc = b["curve_weight"]
        out["curve"] = wc * Pt[:, :, 1] + (1 - wc) * gc
        sc = b["band_scale"]
        out["band"] = (out["curve"] + sc * (Pt[:, :, 0] - Pt[:, :, 1]), out["curve"] + sc * (Pt[:, :, 2] - Pt[:, :, 1]))
        out.update(enforce_consistency(out, X.G0.to_numpy(), out["curve"]))
        if contributions:
            out["contrib"] = P["dpeak"]["contrib"]; out["base"] = P["dpeak"]["base"]
        return out

    def what_if(self, X, pre, H, emb):
        """all scenarios in one batch (rows are predicted independently, so this equals one call per scenario)"""
        Zs = []
        for name, mod in WHAT_IF:
            Z = X.copy()
            for c in ["carbs", "protein", "fat", "fiber"]:
                if c in mod:
                    Z[f"est_{c}"] = Z[f"est_{c}"] * mod[c]
                if f"{c}_add" in mod:
                    Z[f"est_{c}"] = Z[f"est_{c}"] + mod[f"{c}_add"]
            Z["est_kcal"] = 4 * Z.est_carbs + 4 * Z.est_protein + 9 * Z.est_fat
            Zs.append(Z)
        k = len(Zs)
        o = self._predict_rows(pd.concat(Zs, ignore_index=True), np.repeat(pre, k, 0), H, np.repeat(emb, k, 0), contributions=False)
        g0 = X.G0.iloc[0]
        return [dict(scenario=name, peak_rise_mgdl=_r(o["dpeak"][i]), peak_glucose_mgdl=_r(o["Gmax"][i]), iauc_0_120_mgdl_min=_r(o["iauc"][i], 0),
                     curve_mgdl=[_r(g0 + v) for v in o["curve"][i]]) for i, (name, _) in enumerate(WHAT_IF)]

    def respond(self, person, photo, meal_time, meal_type, state=None, photo_ref=None, logged_kcal=None):
        """Full API response for one meal (prediction, explanation, similar past meals, confidence, what-ifs, P(peak > x)).
        photo_ref(history_row) -> the identifier returned for a similar past meal (default: its photo_key column)."""
        state = state if state is not None else self.b["person_state"][person.pid]
        H = state["hist"]
        out, X, pre, e, tags, lat, sidx = self.predict_meal(person, photo, meal_time, meal_type, state, logged_kcal=logged_kcal)
        t = time.perf_counter()
        wi = self.what_if(X, pre[None], H, e[None])
        lat["what_if"] = (time.perf_counter() - t) * 1000
        sim = []
        for j in np.atleast_1d(sidx)[:3]:
            h = H.iloc[int(j)]
            sim.append(dict(photo_key=photo_ref(h) if photo_ref else h.get("photo_key"), time=str(h.time), meal_type=h.meal_type,
                            logged_carbs_g=_r(h.carbs), observed_peak_rise_mgdl=_r(h.dpeak), similarity=_r(float(H.emb.iloc[int(j)] @ e), 2)))
        hist_types = set(H.meal_type); hmax = H.carbs.max() if len(H) else 0
        conf = confidence(X.iloc[0].to_dict(), len(H), hist_types, hmax, float(X.photo_sim_pop.iloc[0]))
        contrib = group_contributions(out["contrib"].iloc[0], self.b["gbm"].features)
        pr = {t: float(out[t][0]) for t in TARGETS}; pr["curve"] = out["curve"][0]
        probs = exceed_prob(float(X.G0.iloc[0]), out, 0)
        resp = build_response(person.pid, X.iloc[0].to_dict(), pr, (out["band"][0][0], out["band"][1][0]), contrib, float(out["base"][0]),
                              state["profile"], self.b["pop_avg_rise"], sim, wi, conf, tags, latency=lat, probs=probs)
        return resp, dict(out=out, X=X, pre=pre, emb=e)
