"""Meal photo understanding: CLIP-family encoder, zero-shot food tags, and a macro estimator that blends
population regression with retrieval from a population library and the person's own food library."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MACROS = ["carbs", "protein", "fat", "fiber"]

FOOD_PROMPTS = ["a protein shake in a shaker bottle", "a burrito wrapped in foil", "a burrito bowl with rice and beans", "a salad",
                "a sandwich", "a pizza slice", "pasta", "a bowl of soup", "rice with meat", "tacos", "a hamburger", "fried chicken",
                "eggs and toast", "oatmeal or cereal", "fresh fruit", "a bag of chips", "cookies or pastries", "yogurt",
                "nuts", "a cup of coffee", "a soda or juice", "vegetables", "steak or grilled meat", "fish or seafood",
                "noodles", "a smoothie", "cheese and crackers", "a candy bar", "ice cream", "a plate of mixed home-cooked food"]


class PhotoEncoder:
    """OpenCLIP image/text encoder. MobileCLIP-S2 is the default deployment model (fast on CPU/phone)."""

    def __init__(self, model="MobileCLIP-S2", pretrained="datacompdr", threads=8, preprocess=None):
        """pretrained: an OpenCLIP tag or a local weights file; preprocess: the tag's preprocessing config (mean, std,
        interpolation, resize_mode), needed when loading from a file so images are prepared exactly as for the tag"""
        import torch
        import open_clip
        torch.set_num_threads(threads)
        self.torch = torch
        kw = {}
        if preprocess:
            kw = dict(image_mean=preprocess.get("mean"), image_std=preprocess.get("std"),
                      image_interpolation=preprocess.get("interpolation"), image_resize_mode=preprocess.get("resize_mode"))
        self.model, _, self.pre = open_clip.create_model_and_transforms(model, pretrained=pretrained, **kw)
        self.model.eval()
        self.tok = open_clip.get_tokenizer(model)
        self.name = f"{model}/{pretrained}"

    def encode_images(self, images):
        from PIL import ImageOps
        with self.torch.inference_mode():
            x = self.torch.stack([self.pre(ImageOps.exif_transpose(im).convert("RGB")) for im in images])
            e = self.model.encode_image(x)
            return (e / e.norm(dim=-1, keepdim=True)).numpy()

    def encode_texts(self, texts):
        with self.torch.inference_mode():
            e = self.model.encode_text(self.tok([f"a photo of {t}" for t in texts]))
            return (e / e.norm(dim=-1, keepdim=True)).numpy()

    def timed_encode(self, image):
        t = time.perf_counter()
        e = self.encode_images([image])
        return e[0], (time.perf_counter() - t) * 1000


class FoodTagger:
    """Zero-shot food labels from text prompts (for display and explanation only)."""

    def __init__(self, text_emb, prompts=FOOD_PROMPTS):
        self.T, self.prompts = text_emb, prompts

    def tags(self, emb, k=3, temp=100.0):
        z = temp * emb @ self.T.T
        p = np.exp(z - z.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        top = np.argsort(-p, -1)[..., :k]
        return [[(self.prompts[j], float(p[i, j])) for j in top[i]] for i in range(len(emb))]


def _knn(Q, L, Ly, k, exclude=None):
    """similarity-weighted mean of library targets over the top-k cosine neighbours; exclude[i] = boolean mask of library rows to skip"""
    S = Q @ L.T
    if exclude is not None:
        S = np.where(exclude, -np.inf, S)
    k = min(k, L.shape[0])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k] if L.shape[0] > k else np.tile(np.arange(L.shape[0]), (len(Q), 1))
    s = np.take_along_axis(S, idx, 1)
    w = np.where(np.isfinite(s), np.exp(20 * (s - np.nanmax(np.where(np.isfinite(s), s, -1), 1, keepdims=True))), 0)
    w = w / np.clip(w.sum(1, keepdims=True), 1e-9, None)
    y = Ly[idx]                                               # n x k x m
    mean = np.einsum("nk,nkm->nm", w, np.nan_to_num(y))
    var = np.einsum("nk,nkm->nm", w, (np.nan_to_num(y) - mean[:, None, :]) ** 2)
    sd = np.sqrt(np.clip(var, 0, None))
    sd[sd < 1e-3] = 0.0          # identical neighbours: exact zero, so float noise in the embedding cannot flip a tree split
    smax = np.where(np.isfinite(s), s, np.nan).max(1) if np.isfinite(s).any() else np.full(len(Q), np.nan)
    return mean, sd, smax, idx


class MacroEstimator:
    """photo embedding -> carbs/protein/fat/fiber (grams). Blend of (1) ridge on the embedding, (2) kNN over other people's
    labelled photos, (3) kNN over the same person's history photos; a small GBM stacker learns how much to trust each."""

    def __init__(self, k_pop=10, k_pers=3):
        self.k_pop, self.k_pers = k_pop, k_pers

    def _base(self, E, pid, ridge_models, LE, LY, Lpid, self_idx=None, own=None):
        """own: optional (E_own, Y_own, self_index) = the person's own labelled photos kept outside the library (new users);
        self_index[i] >= 0 excludes that own photo for query i (the query itself)."""
        R = np.column_stack([np.expm1(m.predict(E)) for m in ridge_models])
        ex_pop = (pid[:, None] == Lpid[None, :])
        mp, sp, smp, _ = _knn(E, LE, LY, self.k_pop, ex_pop)
        if own is None:
            ex_pers = ~ex_pop
            if self_idx is not None:
                ex_pers = ex_pers.copy(); ex_pers[np.arange(len(E)), self_idx] = True
            OE, OY = LE, LY
        else:
            OE, OY, oself = own
            ex_pers = np.zeros((len(E), len(OE)), bool)
            if oself is not None:
                oself = np.asarray(oself)
                ok = oself >= 0
                ex_pers[np.flatnonzero(ok), oself[ok]] = True
            pad = self.k_pers - len(OE)
            if pad > 0 and len(OE):
                # the fitted library path takes the top k over a library where other people's rows are excluded, so a
                # person with fewer than k photos gets excluded neighbours (max similarity = NaN); the stacker was
                # trained on that, so the separate own library is padded with excluded rows to give identical features
                OE = np.vstack([OE, np.zeros((pad, OE.shape[1]), OE.dtype)])
                OY = np.vstack([OY, np.zeros((pad, OY.shape[1]))])
                ex_pers = np.hstack([ex_pers, np.ones((len(E), pad), bool)])
        has = (~ex_pers).any(1) if ex_pers.shape[1] else np.zeros(len(E), bool)
        if has.any():
            mq, sq, smq, _ = _knn(E, OE, OY, self.k_pers, ex_pers)
        else:
            mq, sq, smq = np.full((len(E), LY.shape[1]), np.nan), np.full((len(E), LY.shape[1]), np.nan), np.full(len(E), np.nan)
        mq[~has] = np.nan; sq[~has] = np.nan; smq = np.where(has, smq, np.nan)
        return R, mp, sp, smp, mq, sq, smq

    def _stack_X(self, R, mp, sp, smp, mq, sq, smq, j):
        return np.column_stack([np.log1p(R[:, j]), np.log1p(mp[:, j]), sp[:, j], smp, np.log1p(mq[:, j]), sq[:, j], smq])

    def fit(self, E, Y, pid):
        E, Y, pid = np.asarray(E), np.asarray(Y, float), np.asarray(pid)
        self.LE, self.LY, self.Lpid = E, Y, pid
        mk = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
        self.ridge = []
        for j in range(Y.shape[1]):
            ok = np.isfinite(Y[:, j])
            self.ridge.append(mk().fit(E[ok], np.log1p(Y[ok, j])))
        # out-of-fold base predictions (persons held out) to train the stacker honestly
        oof = [np.full((len(E), Y.shape[1]), np.nan) for _ in range(2)] + [np.full(len(E), np.nan)] + \
              [np.full((len(E), Y.shape[1]), np.nan) for _ in range(2)] + [np.full(len(E), np.nan)]
        Rr = np.full((len(E), Y.shape[1]), np.nan)
        for a, b in GroupKFold(5).split(E, groups=pid):
            rm = []
            for j in range(Y.shape[1]):
                ok = np.isfinite(Y[a, j])
                rm.append(mk().fit(E[a][ok], np.log1p(Y[a][ok, j])))
            Rr[b] = np.column_stack([np.expm1(m.predict(E[b])) for m in rm])
            # population kNN from training folds only; personal kNN from the same person's other photos (all folds)
            mp, sp, smp, _ = _knn(E[b], E[a], Y[a], self.k_pop, pid[b][:, None] == pid[a][None, :])
            oof[0][b], oof[1][b], oof[2][b] = mp, sp, smp
        self_idx = np.arange(len(E))
        _, _, _, _, mq, sq, smq = self._base(E, pid, self.ridge, E, Y, pid, self_idx)
        oof[3], oof[4], oof[5] = mq, sq, smq
        self.stack = []
        for j in range(Y.shape[1]):
            X = self._stack_X(Rr, oof[0], oof[1], oof[2], oof[3], oof[4], oof[5], j)
            ok = np.isfinite(Y[:, j])
            self.stack.append(lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=20,
                                                subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=8).fit(X[ok], np.log1p(Y[ok, j])))
        self.oof_pred = self._stack_predict(Rr, *oof)
        self.oof_unc = oof[1]
        return self

    def _stack_predict(self, R, mp, sp, smp, mq, sq, smq):
        return np.column_stack([np.expm1(self.stack[j].predict(self._stack_X(R, mp, sp, smp, mq, sq, smq, j))) for j in range(len(self.stack))])

    def predict(self, E, pid, own=None):
        """returns macro estimates (n x 4), neighbour spread (n x 4), max similarity to population and to own library.
        own = (E_own, Y_own): the person's labelled photos when they are not part of the fitted library (new users)."""
        E, pid = np.asarray(E), np.asarray(pid)
        R, mp, sp, smp, mq, sq, smq = self._base(E, pid, self.ridge, self.LE, self.LY, self.Lpid,
                                                 own=None if own is None else (own[0], own[1], None))
        return np.clip(self._stack_predict(R, mp, sp, smp, mq, sq, smq), 0, None), sp, smp, smq

    def predict_history(self, E_own, Y_own, pid):
        """estimates for a new person's own history photos, each excluding itself from the personal neighbours (as the
        out-of-fold estimates used for history meals in training); returns est, neighbour spread, max population similarity"""
        E_own = np.asarray(E_own)
        p = np.full(len(E_own), pid, dtype=object)
        R, mp, sp, smp, mq, sq, smq = self._base(E_own, p, self.ridge, self.LE, self.LY, self.Lpid,
                                                 own=(E_own, np.asarray(Y_own, float), np.arange(len(E_own))))
        return self._stack_predict(R, mp, sp, smp, mq, sq, smq), sp, smp

    def neighbours(self, e, pid, k=3):
        """indices into the library of the person's most similar past meals (for 'your similar meals' in the app)"""
        own = np.flatnonzero(self.Lpid == pid)
        if len(own) == 0:
            return [], []
        s = self.LE[own] @ e
        o = np.argsort(-s)[:k]
        return own[o].tolist(), s[o].tolist()
