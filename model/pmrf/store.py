"""Versioned model store in stable, non-pickle formats (what the deployed service loads).

Layout of <store>/<version>/:
  manifest.json            versions of the libraries it was exported with, configuration, blend weights, SHA-256 of every file
  encoder/                 MobileCLIP-S2 weights (safetensors) + food-tag text embeddings
  gbm/<target>/*.txt       LightGBM text models: point and 10 %/90 % quantile models, one per seed
  macro/                   nutrition estimator: ridge coefficients (npz), stacker LightGBM text models, photo library (npz)
  pca.npz, retrieval.npz   image PCA; population library of observed responses
  transformer/seed*.safetensors
  risk/models.json         HbA1c and T2D-risk linear models (imputer medians, scaler, coefficients) + reference data
  guide/effects.json       measured nutrient effects from the CGMacros standardized meals

`export_store` is run once on our side from the training bundle; `load_store` rebuilds a bundle-like dict whose
objects predict exactly like the trained ones (checked by scripts/golden_test.py).
"""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb

from .models.gbm import GBMResponse, _Bag
from .vision import MacroEstimator

STORE_FORMAT = 1


# ------------------------------------------------------------------ small stand-ins for fitted sklearn objects
class LinearPipe:
    """[median imputer] -> standard scaler -> linear model, reproducing sklearn's arithmetic (dtype handling included)."""

    def __init__(self, mean, scale, coef, intercept, impute=None, logistic=False):
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.coef, self.intercept = np.asarray(coef, float), np.asarray(intercept, float)
        self.impute = None if impute is None else np.asarray(impute, float)
        self.logistic = logistic

    def _decision(self, X):
        X = np.array(X, dtype=np.result_type(np.asarray(X).dtype, np.float32), copy=True)
        if self.impute is not None:
            X = X.astype(float)
            bad = ~np.isfinite(X)
            X[bad] = np.broadcast_to(self.impute, X.shape)[bad]
        X -= self.mean
        X /= self.scale
        return X @ self.coef.T + self.intercept

    def predict(self, X):
        return self._decision(X)

    def predict_proba(self, X):
        d = np.ravel(self._decision(X))
        p = 1.0 / (1.0 + np.exp(-d))
        return np.column_stack([1 - p, p])

    def to_json(self):
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist(), coef=self.coef.tolist(), intercept=self.intercept.tolist(),
                    impute=None if self.impute is None else self.impute.tolist(), logistic=self.logistic)

    @classmethod
    def from_json(cls, d):
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"], d.get("impute"), d.get("logistic", False))

    @classmethod
    def from_sklearn(cls, pipe):
        steps = dict(pipe.steps)
        imp = next((s for s in steps.values() if type(s).__name__ == "SimpleImputer"), None)
        sc = next(s for s in steps.values() if type(s).__name__ == "StandardScaler")
        mdl = pipe.steps[-1][1]
        logistic = type(mdl).__name__.startswith("LogisticRegression")
        if imp is not None:
            assert imp.strategy in ("median", "mean") and len(imp.statistics_) == len(sc.mean_), "imputer dropped a column"
        return cls(sc.mean_, sc.scale_, mdl.coef_, mdl.intercept_, None if imp is None else imp.statistics_, logistic)


class PCAProj:
    def __init__(self, mean, components):
        self.mean_, self.components_ = np.asarray(mean), np.asarray(components)

    def transform(self, X):          # sklearn PCA._transform without whitening
        Xt = np.asarray(X) @ self.components_.T
        Xt -= self.mean_.reshape(1, -1) @ self.components_.T
        return Xt


class _BoosterModel:
    """LightGBM Booster loaded from text, with the LGBMRegressor.predict call signature used by _Bag."""

    def __init__(self, path):
        self.booster = lgb.Booster(model_file=path)

    def predict(self, X, pred_contrib=False):
        return self.booster.predict(X, pred_contrib=pred_contrib)


# ------------------------------------------------------------------ export
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_booster(m, path):
    (m.booster_ if hasattr(m, "booster_") else m.booster).save_model(path)


def export_store(bundle, out_dir, version, encoder_weights, text_emb, risk_models=None, guide_effects=None, notes=""):
    import torch
    from safetensors.torch import save_file
    import open_clip, sklearn, scipy
    os.makedirs(out_dir, exist_ok=True)
    for d in ["encoder", "gbm", "macro", "transformer", "risk", "guide"]:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    g = bundle["gbm"]
    gbm_cfg = dict(features=g.features, targets=g.targets, quantiles=list(g.quantiles), seeds=list(g.seeds), log_targets=g.log_targets,
                   log_point=g.log_point, cqr=g.cqr, smear=getattr(g, "smear", {}), personal_offset=bool(g.personal_offset))
    assert not g.personal_offset, "personal offsets are not exported (the shipped model does not use them)"
    for y in g.targets:
        os.makedirs(os.path.join(out_dir, "gbm", y), exist_ok=True)
        for s, m in zip(g.seeds, g.point[y].models):
            _save_booster(m, os.path.join(out_dir, "gbm", y, f"point_seed{s}.txt"))
        for q, bag in zip(g.quantiles, g.qmodels[y]):
            for s, m in zip(g.seeds, bag.models):
                _save_booster(m, os.path.join(out_dir, "gbm", y, f"q{int(round(q * 100)):02d}_seed{s}.txt"))
    me = bundle["macro_est"]
    ridge = {}
    for j, p in enumerate(me.ridge):
        lp = LinearPipe.from_sklearn(p)
        ridge.update({f"mean{j}": lp.mean, f"scale{j}": lp.scale, f"coef{j}": lp.coef, f"intercept{j}": lp.intercept})
    np.savez(os.path.join(out_dir, "macro", "ridge.npz"), **ridge)
    for j, m in enumerate(me.stack):
        _save_booster(m, os.path.join(out_dir, "macro", f"stack{j}.txt"))
    np.savez(os.path.join(out_dir, "macro", "library.npz"), emb=me.LE, y=me.LY, pid=me.Lpid.astype(str))
    np.savez(os.path.join(out_dir, "pca.npz"), mean=bundle["pca"].mean_, components=bundle["pca"].components_)
    np.savez(os.path.join(out_dir, "retrieval.npz"), emb=bundle["lib_emb"], y=bundle["lib_y"], pid=bundle["lib_pid"].astype(str))
    for s, sd in enumerate(bundle["transformer_states"]):
        save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(out_dir, "transformer", f"seed{s}.safetensors"))
    shutil.copyfile(encoder_weights, os.path.join(out_dir, "encoder", "open_clip_model.safetensors"))
    np.save(os.path.join(out_dir, "encoder", "food_tag_text_emb.npy"), text_emb)
    if risk_models is not None:
        rm = risk_models
        risk = dict(version=rm["version"], created=rm["created"],
                    a1c={k: v for k, v in rm["a1c"].items() if k != "model"} | dict(model=LinearPipe.from_sklearn(rm["a1c"]["model"]).to_json()),
                    t2d={k: v for k, v in rm["t2d"].items() if k != "models"} |
                        dict(models={n: {k: v for k, v in m.items() if k != "model"} | dict(model=LinearPipe.from_sklearn(m["model"]).to_json())
                                     for n, m in rm["t2d"]["models"].items()}),
                    literature=rm["literature"], t2d_tier_reference=rm.get("t2d_tier_reference"))
        json.dump(risk, open(os.path.join(out_dir, "risk", "models.json"), "w"), indent=1, default=_json_default)
    if guide_effects is not None:
        r7 = guide_effects["r7"]
        eff = dict(pop={k: float(v) for k, v in guide_effects["pop"].items()},
                   r7={stat: {pid: {rec: (None if not np.isfinite(v) else float(v)) for rec, v in row.items()}
                              for pid, row in r7[stat].iterrows()} for stat in ["mean", "size"]})
        json.dump(eff, open(os.path.join(out_dir, "guide", "effects.json"), "w"), indent=1)
    from .vision import FOOD_PROMPTS
    import timm
    manifest = dict(store_format=STORE_FORMAT, version=version, created=datetime.now(timezone.utc).isoformat(timespec="seconds"), notes=notes,
                    libraries=dict(lightgbm=lgb.__version__, torch=torch.__version__.split("+")[0], open_clip_torch=open_clip.__version__,
                                   timm=timm.__version__, numpy=np.__version__, pandas=pd.__version__, scikit_learn=sklearn.__version__,
                                   scipy=scipy.__version__),
                    encoder=dict(model=bundle["encoder"][0], pretrained_tag=bundle["encoder"][1], weights="encoder/open_clip_model.safetensors",
                                 preprocess=open_clip.get_pretrained_cfg(*bundle["encoder"]), food_prompts=FOOD_PROMPTS),
                    gbm=gbm_cfg, macro=dict(k_pop=me.k_pop, k_pers=me.k_pers, n_macros=len(me.ridge)),
                    transformer=dict(img_dim=int(bundle["img_dim"]), seeds=len(bundle["transformer_states"]), num_cols=bundle.get("num_cols", "v1")),
                    blend=dict(weights=bundle["blend_weights"], curve_weight=bundle["curve_weight"], band_scale=bundle["band_scale"],
                               tf_sources=bundle.get("tf_sources", {})),
                    pop_avg_rise=float(bundle["pop_avg_rise"]))
    files = {}
    for root, _, fs in os.walk(out_dir):
        for f in fs:
            os.chmod(os.path.join(root, f), 0o644)
    for root, _, fs in os.walk(out_dir):
        for f in sorted(fs):
            p = os.path.join(root, f)
            if f != "manifest.json":
                files[os.path.relpath(p, out_dir)] = _sha(p)
    manifest["files"] = dict(sorted(files.items()))
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1, default=_json_default)
    return manifest


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Series,)):
        return o.to_dict()
    raise TypeError(type(o))


# ------------------------------------------------------------------ load
def load_store(path, verify=True):
    """-> bundle-like dict for pmrf.predict.Predictor (without person states, which live in the person store)."""
    from safetensors.torch import load_file
    man = json.load(open(os.path.join(path, "manifest.json")))
    assert man["store_format"] == STORE_FORMAT, f"store format {man['store_format']} not supported"
    if verify:
        for rel, h in man["files"].items():
            if _sha(os.path.join(path, rel)) != h:
                raise ValueError(f"checksum mismatch: {rel}")
    gc = man["gbm"]
    g = GBMResponse(gc["features"], targets=gc["targets"], quantiles=tuple(gc["quantiles"]), personal_offset=False, seeds=gc["seeds"],
                    log_targets=gc["log_targets"], log_point=gc["log_point"])
    g.point, g.qmodels, g.cqr, g.smear, g.offsets = {}, {}, dict(gc["cqr"]), dict(gc.get("smear", {})), {}
    for y in g.targets:
        d = os.path.join(path, "gbm", y)
        g.point[y] = _Bag([_BoosterModel(os.path.join(d, f"point_seed{s}.txt")) for s in g.seeds])
        g.qmodels[y] = [_Bag([_BoosterModel(os.path.join(d, f"q{int(round(q * 100)):02d}_seed{s}.txt")) for s in g.seeds]) for q in g.quantiles]
    mc = man["macro"]
    me = MacroEstimator(k_pop=mc["k_pop"], k_pers=mc["k_pers"])
    r = np.load(os.path.join(path, "macro", "ridge.npz"))
    me.ridge = [LinearPipe(r[f"mean{j}"], r[f"scale{j}"], r[f"coef{j}"], r[f"intercept{j}"]) for j in range(mc["n_macros"])]
    me.stack = [_BoosterModel(os.path.join(path, "macro", f"stack{j}.txt")) for j in range(mc["n_macros"])]
    lib = np.load(os.path.join(path, "macro", "library.npz"))
    me.LE, me.LY, me.Lpid = lib["emb"], lib["y"], lib["pid"].astype(object)
    pz = np.load(os.path.join(path, "pca.npz"))
    rv = np.load(os.path.join(path, "retrieval.npz"))
    tf = [load_file(os.path.join(path, "transformer", f"seed{s}.safetensors")) for s in range(man["transformer"]["seeds"])]
    risk = os.path.join(path, "risk", "models.json")
    eff = os.path.join(path, "guide", "effects.json")
    return dict(manifest=man, encoder=(man["encoder"]["model"], os.path.join(path, man["encoder"]["weights"])),
                encoder_preprocess=man["encoder"]["preprocess"], text_emb=np.load(os.path.join(path, "encoder", "food_tag_text_emb.npy")),
                macro_est=me, pca=PCAProj(pz["mean"], pz["components"]), gbm=g, transformer_states=tf, img_dim=man["transformer"]["img_dim"],
                blend_weights=man["blend"]["weights"], curve_weight=man["blend"]["curve_weight"], band_scale=man["blend"]["band_scale"],
                tf_sources=man["blend"]["tf_sources"], num_cols=man["transformer"].get("num_cols", "v1"), lib_emb=rv["emb"], lib_y=rv["y"], lib_pid=rv["pid"].astype(object),
                pop_avg_rise=man["pop_avg_rise"], person_state={},
                risk_models=load_risk_json(risk) if os.path.exists(risk) else None,
                guide_effects=load_effects_json(eff) if os.path.exists(eff) else None)


def load_risk_json(path):
    d = json.load(open(path))
    d["a1c"]["model"] = LinearPipe.from_json(d["a1c"]["model"])
    for m in d["t2d"]["models"].values():
        m["model"] = LinearPipe.from_json(m["model"])
    d["a1c"]["residuals"] = np.asarray(d["a1c"]["residuals"], float)
    return d


def load_effects_json(path):
    d = json.load(open(path))
    r7 = pd.concat({stat: pd.DataFrame.from_dict(d["r7"][stat], orient="index").astype(float) for stat in ["mean", "size"]}, axis=1)
    return dict(pop=d["pop"], r7=r7)
