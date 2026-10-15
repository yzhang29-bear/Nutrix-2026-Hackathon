"""Gradient-boosted response model: one LightGBM per target (glucose change at 15-min steps, peak rise, time to peak, iAUC),
quantile models for an 80% interval, a shrunken per-person offset learned from the person's own history, and TreeSHAP
contributions for every prediction."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

DELTA_TARGETS = ["d15", "d30", "d45", "d60", "d75", "d90", "d105", "d120"]
SCALAR_TARGETS = ["dpeak", "ttp", "iauc"]
ALL_TARGETS = DELTA_TARGETS + SCALAR_TARGETS


DEFAULT_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20)


def _lgb(objective="l2", alpha=None, n_jobs=16, params=None, seed=0):
    kw = dict(DEFAULT_PARAMS, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=n_jobs, random_state=seed)
    kw.update(params or {})
    if objective == "quantile":
        kw.update(objective="quantile", alpha=alpha)
    return lgb.LGBMRegressor(**kw)


class _Bag:
    """average of LightGBM models trained with different seeds; keeps TreeSHAP contributions additive"""

    def __init__(self, models):
        self.models = models

    def predict(self, X, pred_contrib=False):
        return np.mean([m.predict(X, pred_contrib=pred_contrib) for m in self.models], 0)


def _fit_bag(X, y, seeds, **kw):
    return _Bag([_lgb(seed=s, **kw).fit(X, y) for s in seeds])


class GBMResponse:
    def __init__(self, features, targets=ALL_TARGETS, quantiles=(0.1, 0.9), personal_offset=True, n_jobs=16, params=None, seeds=(0,),
                 log_targets=None, log_point="median"):
        """log_targets: {target: offset} modelled as log(y + offset) (skewed, non-negative outcomes such as dpeak, iauc);
        log_point: 'median' back-transforms the log prediction directly, 'mean' applies Duan's smearing factor."""
        self.features, self.targets, self.quantiles = list(features), list(targets), quantiles
        self.personal_offset, self.n_jobs, self.params, self.seeds = personal_offset, n_jobs, params, tuple(seeds)
        self.log_targets, self.log_point = dict(log_targets or {}), log_point

    def _fwd(self, y, v):
        return np.log(np.asarray(v, float) + self.log_targets[y]) if y in self.log_targets else np.asarray(v, float)

    def _inv(self, y, v, point=False):
        if y not in self.log_targets:
            return v
        f = self.smear.get(y, 1.0) if (point and self.log_point == "mean") else 1.0
        return np.exp(v) * f - self.log_targets[y]

    def _oof(self, df, y, folds=5):
        p = np.zeros(len(df))
        for a, b in GroupKFold(folds).split(df, groups=df.pid):
            p[b] = _lgb(n_jobs=self.n_jobs, params=self.params).fit(df.iloc[a][self.features], df.iloc[a][y]).predict(df.iloc[b][self.features])
        return p

    @staticmethod
    def _offset(pids, res, k):
        t = pd.DataFrame(dict(pid=pids, res=res)).groupby("pid").res.agg(["sum", "count"])
        return t["sum"] / (t["count"] + k)

    def fit(self, TR):
        self.point, self.qmodels, self.offsets, self.k, self.cqr, self.res_sd, self.smear = {}, {}, {}, {}, {}, {}, {}
        TR = TR.copy()
        for y in self.targets:
            yt = pd.Series(self._fwd(y, TR[y]), index=TR.index)
            self.point[y] = _fit_bag(TR[self.features], yt, self.seeds, n_jobs=self.n_jobs, params=self.params)
            self.qmodels[y] = [_fit_bag(TR[self.features], yt, self.seeds, objective="quantile", alpha=q, n_jobs=self.n_jobs, params=self.params)
                               for q in self.quantiles]
            if y in self.log_targets:
                self.smear[y] = float(np.mean(np.exp(yt.to_numpy() - self.point[y].predict(TR[self.features]))))
            oof = self._inv(y, self._oof(TR.assign(**{y: yt}), y), point=True)
            res = TR[y].to_numpy() - oof
            # shrinkage strength for the personal offset, chosen on a temporal split inside the history (days 1-5 -> 6-7)
            if self.personal_offset:
                early, late = TR.day <= 5, TR.day >= 6
                errs = {}
                for k in [1, 2, 4, 8, 16, 32, 1e9]:
                    off = self._offset(TR.pid[early], res[early.to_numpy()], k)
                    errs[k] = np.mean(np.abs(res[late.to_numpy()] - TR.pid[late].map(off).fillna(0).to_numpy()))
                self.k[y] = min(errs, key=errs.get)
                self.offsets[y] = self._offset(TR.pid, res, self.k[y])
                res_after = res - TR.pid.map(self.offsets[y]).fillna(0).to_numpy()
            else:
                res_after = res
            # conformalised quantile regression on out-of-fold quantile predictions -> calibrated 80% interval
            qlo = np.zeros(len(TR)); qhi = np.zeros(len(TR))
            for a, b in GroupKFold(5).split(TR, groups=TR.pid):
                yt_a = self._fwd(y, TR.iloc[a][y])
                ql = _lgb("quantile", self.quantiles[0], self.n_jobs, self.params).fit(TR.iloc[a][self.features], yt_a)
                qh = _lgb("quantile", self.quantiles[1], self.n_jobs, self.params).fit(TR.iloc[a][self.features], yt_a)
                qlo[b], qhi[b] = self._inv(y, ql.predict(TR.iloc[b][self.features])), self._inv(y, qh.predict(TR.iloc[b][self.features]))
            shift = TR.pid.map(self.offsets[y]).fillna(0).to_numpy() if self.personal_offset else 0
            score = np.maximum(qlo + shift - TR[y].to_numpy(), TR[y].to_numpy() - (qhi + shift))
            level = self.quantiles[1] - self.quantiles[0]
            self.cqr[y] = float(np.quantile(score, min(1, np.ceil((len(score) + 1) * level) / len(score))))
            self.res_sd[y] = pd.Series(res_after).groupby(TR.pid.to_numpy()).std()
        return self

    def predict(self, TE, contributions=False):
        """contributions: True for every target, or a list of targets (TreeSHAP is the slow part at serving time)"""
        out = {}
        for y in self.targets:
            X = TE[self.features]
            off = TE.pid.map(self.offsets[y]).fillna(0).to_numpy() if self.personal_offset else np.zeros(len(TE))
            p = self._inv(y, self.point[y].predict(X), point=True) + off
            lo = self._inv(y, self.qmodels[y][0].predict(X)) + off - self.cqr[y]
            hi = self._inv(y, self.qmodels[y][1].predict(X)) + off + self.cqr[y]
            out[y] = dict(pred=p, lo=np.minimum(lo, p), hi=np.maximum(hi, p), offset=off)
            if contributions is True or (contributions and y in contributions):
                c = self.point[y].predict(X, pred_contrib=True)       # n x (features + bias), TreeSHAP (in model space)
                C, base = c[:, :-1], c[:, -1]
                if y in self.log_targets:
                    # log-space attributions rescaled so they sum to (prediction - baseline) in mg/dL; shares are preserved
                    base_mg = self._inv(y, base, point=True)
                    tot = C.sum(1, keepdims=True)
                    C = np.where(np.abs(tot) > 1e-9, C / np.where(np.abs(tot) > 1e-9, tot, 1) * (p - off - base_mg)[:, None], 0.0)
                    base = base_mg
                out[y]["contrib"] = pd.DataFrame(C, columns=self.features, index=TE.index)
                out[y]["base"] = base
        return out
