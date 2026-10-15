"""Nutrition v2: combine our photo estimator (CLIP retrieval + ridge, strong on meals the person has eaten before) with
large vision-language models on AWS Bedrock (better on new, free-living meals): Qwen3-VL few-shot (calibrated on the
person's own logged meals) and four zero-shot readings (Qwen3-VL, Mistral Large 3, Llama 4 Maverick, Nova Pro).
A small LightGBM per nutrient learns, from the context of the meal (how similar it is to the person's own earlier meals,
how much the models disagree, meal type), how much to trust each source. Trained on history meals with out-of-fold
inputs only, so its training inputs look like those at prediction time.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

NUTRIENTS = ["carbs", "protein", "fat", "fiber"]
ZS_MODELS = ["qwen", "mistral", "llama", "nova"]


def vlm_value(d, key):
    """value from a parsed VLM JSON (numbers, or {value: ...})"""
    v = d.get(key) if isinstance(d, dict) else None
    if isinstance(v, dict):
        v = v.get("value")
    if isinstance(v, str):
        try:
            v = float(v.split()[0].replace("g", ""))
        except ValueError:
            v = None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan


def source_columns(df, k):
    """the per-meal inputs for nutrient k: log1p of every source, their spread, and context"""
    zs = np.column_stack([df[f"vz_{m}_{k}"].to_numpy(float) for m in ZS_MODELS])
    with np.errstate(all="ignore"):
        med = np.nanmedian(zs, 1)
        sd = np.nanstd(np.log1p(np.clip(zs, 0, None)), 1)
    X = pd.DataFrame({
        "ours": np.log1p(np.clip(df[f"est_{k}"].to_numpy(float), 0, None)),
        "fewshot": np.log1p(np.clip(df[f"vf_{k}"].to_numpy(float), 0, None)),
        "zs_median": np.log1p(np.clip(med, 0, None)),
        "zs_spread": sd,
        **{f"zs_{m}": np.log1p(np.clip(df[f"vz_{m}_{k}"].to_numpy(float), 0, None)) for m in ZS_MODELS},
        "own_sim": df["own_sim"].to_numpy(float),          # max similarity to the person's own earlier photos
        "pop_sim": df["photo_sim_pop"].to_numpy(float),
        "fs_same": df["vf_same"].to_numpy(float),          # few-shot model says: same food as an earlier meal
        "liquid": df["vz_liquid"].to_numpy(float), "portion": df["vz_portion"].to_numpy(float),
        "is_breakfast": df["is_breakfast"].to_numpy(float), "is_lunch": df["is_lunch"].to_numpy(float),
        "is_dinner": df["is_dinner"].to_numpy(float), "is_snack": df["is_snack"].to_numpy(float),
    }, index=df.index)
    return X


PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=8)


class NutritionStacker:
    def __init__(self, seeds=(0, 1, 2)):
        self.seeds = seeds

    def fit(self, H):
        """H: history meals with photos and logged nutrients (+ source columns). Stores out-of-fold predictions for H."""
        self.models, self.oof = {}, pd.DataFrame(index=H.index)
        for k in NUTRIENTS:
            X, y = source_columns(H, k), np.log1p(H[k].clip(lower=0))
            ok = y.notna().to_numpy()
            self.models[k] = [lgb.LGBMRegressor(**PARAMS, random_state=s).fit(X[ok], y[ok]) for s in self.seeds]
            oof = np.full(len(H), np.nan)
            for a, b in GroupKFold(5).split(X, groups=H.pid):
                a = a[ok[a]]
                ms = [lgb.LGBMRegressor(**PARAMS, random_state=s).fit(X.iloc[a], y.iloc[a]) for s in self.seeds]
                oof[b] = np.mean([m.predict(X.iloc[b]) for m in ms], 0)
            self.oof[k] = np.clip(np.expm1(oof), 0, None)
        return self

    def predict(self, df):
        out = pd.DataFrame(index=df.index)
        for k in NUTRIENTS:
            X = source_columns(df, k)
            out[k] = np.clip(np.expm1(np.mean([m.predict(X) for m in self.models[k]], 0)), 0, None)
        return out


# ---------------------------------------------------------------- gated combination (what ships)
def candidates(df, k):
    """history-based (ours + Qwen few-shot) and vision-based (median of zero-shot VLMs + Qwen few-shot) estimates"""
    zs = df[[f"vz_{m}_{k}" for m in ZS_MODELS]].median(1)
    hist = df[[f"est_{k}", f"vf_{k}"]].mean(1)
    vis = pd.concat([zs, df[f"vf_{k}"]], axis=1).mean(1)
    return hist, vis


class NutritionGate:
    """Per nutrient: if the photo is similar to the person's own earlier meals (own_sim >= threshold) use the
    history-based estimate, otherwise the vision-based one. The threshold is the one with the lowest MAE on the history
    meals (inputs there are out-of-fold), searched over quantiles of own_sim."""

    def fit(self, H):
        self.t = {}
        for k in NUTRIENTS:
            hist, vis = candidates(H, k)
            o, s = H[k], H["own_sim"]
            ok = o.notna() & hist.notna() & vis.notna()
            best = (np.inf, None)
            for t in [-np.inf] + list(np.nanquantile(s[ok], np.linspace(0.05, 0.95, 19))) + [np.inf]:
                p = np.where(s[ok].fillna(-1) >= t, hist[ok], vis[ok])
                e = float(np.mean(np.abs(p - o[ok])))
                if e < best[0]:
                    best = (e, t)
            self.t[k] = best[1]
        return self

    def predict(self, df):
        out = pd.DataFrame(index=df.index)
        for k in NUTRIENTS:
            hist, vis = candidates(df, k)
            use_hist = df["own_sim"].fillna(-1) >= self.t[k]
            p = np.where(use_hist, hist, vis)
            p = np.where(np.isfinite(p), p, np.where(hist.notna(), hist, np.where(vis.notna(), vis, df[f"est_{k}"])))
            out[k] = np.clip(p, 0, None)
        return out
