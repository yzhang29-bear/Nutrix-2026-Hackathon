"""Per-person data for the deployed service: raw signals (CGM, heart rate, meals, medication) and the person state built
once from the first week (history meals with their features, curves and photo embeddings; personal profile; the person's
own labelled photos for the nutrition estimator).

Storage is a directory tree (local disk in development, an EFS volume on AWS), one folder per person:
  people/<pid>/meta.json
  people/<pid>/signals/{cgm,hr,meals,meds}.parquet
  people/<pid>/state/{hist.parquet, arrays.npz, profile.json}
Every write goes to a temporary file first and is then renamed, so a reader never sees a half-written file.
"""
import io
import json
import os
import re
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image

from .data import Person, MED_COLUMNS, MEAL_TYPES
from .features import meal_table, cgm_profile, meal_profile, PROFILE_FEATURES
from .vision import MACROS, _knn

SIGNAL_COLUMNS = dict(cgm=["timestamp", "glucose_mgdl"], hr=["timestamp", "hr_bpm"],
                      meals=["time", "meal_type", "carbs", "protein", "fat", "fiber", "kcal", "calories_logged", "photo_key", "source"],
                      meds=["time", "med_class", "dose"], activity=["timestamp", "mets", "active_kcal"])
SIGNAL_KEYS = dict(cgm="timestamp", hr="timestamp", meals="time", meds="time", activity="timestamp")


def person_info(meta):
    """person-level model inputs from the stored profile (age, sex, BMI); missing = NaN"""
    p = (meta or {}).get("person") or {}
    sex = p.get("sex")
    return dict(age=float(p["age"]) if p.get("age") is not None else np.nan,
                sex_female=1.0 if sex == "female" else (0.0 if sex == "male" else np.nan),
                bmi=float(p["bmi"]) if p.get("bmi") is not None else np.nan)
CGM_INTERP_LIMIT_MIN = 6          # fill at most 6 missing minutes between two sensor readings (5-min sensors)
PID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
ARRAY_COLS = ["emb", "curve", "pre"]
STATE_FORMAT = 2          # 2: history meals carry every model input (needed by the model-based diet guide)


def check_pid(pid):
    if not PID_RE.match(pid or ""):
        raise ValueError("person id must be 1-64 characters: letters, digits, '_', '-', '.'")
    return pid


def _atomic_write(path, write):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.{threading.get_ident()}"
    write(tmp)
    os.replace(tmp, path)


def _naive(ts):
    """timestamps -> timezone-naive pandas datetimes (local wall-clock time of the person, as in the training data)"""
    t = pd.to_datetime(ts)
    if getattr(t.dt if isinstance(t, pd.Series) else t, "tz", None) is not None:
        t = t.dt.tz_localize(None) if isinstance(t, pd.Series) else t.tz_localize(None)
    return t


def assemble_person(pid, t0, cgm, hr, meals, meds, fill_gap_min=CGM_INTERP_LIMIT_MIN, activity=None, info=None):
    """raw signals -> Person on a 1-min grid (CGM linear between readings across gaps of up to fill_gap_min missing minutes;
    0 = the readings are already on a 1-min grid, as in CGMacros; HR per minute)"""
    c = pd.Series(cgm.glucose_mgdl.to_numpy(float), index=_naive(cgm.timestamp).dt.floor("min")).groupby(level=0).mean().sort_index()
    h = (pd.Series(hr.hr_bpm.to_numpy(float), index=_naive(hr.timestamp).dt.floor("min")).groupby(level=0).mean().sort_index()
         if len(hr) else pd.Series(dtype=float))
    t0 = pd.Timestamp(t0) if t0 is not None else c.index[0]
    start = min([t0.floor("min")] + [s.index[0] for s in (c, h) if len(s)])
    end = max([s.index[-1] for s in (c, h) if len(s)])
    grid = pd.date_range(start, end, freq="min")
    cg = c.reindex(grid)
    if fill_gap_min:
        cg = cg.interpolate(limit=int(fill_gap_min), limit_area="inside")
    hg = h.reindex(grid)
    m = meals.copy()
    m["time"] = _naive(m.time)
    m = pd.DataFrame(dict(time=m.time.to_numpy(), meal_type=m.meal_type.to_numpy(), carbs=m.carbs.to_numpy(float),
                          protein=m.protein.to_numpy(float), fat=m.fat.to_numpy(float), fiber=m.fiber.to_numpy(float),
                          kcal=m.kcal.to_numpy(float), photo=m.photo_key.to_numpy(),
                          calories_logged=(m.calories_logged.to_numpy(float) if "calories_logged" in m else np.full(len(m), np.nan))))
    md = meds.rename(columns={})[MED_COLUMNS].copy() if len(meds) else pd.DataFrame(columns=MED_COLUMNS)
    if len(md):
        md["time"] = _naive(md.time)
    act = akc = None
    if activity is not None and len(activity):              # METs and active kcal per minute (no filling across gaps)
        a = activity.copy(); a.index = _naive(a.timestamp).dt.floor("min")
        act = pd.to_numeric(a.mets, errors="coerce").groupby(level=0).mean().reindex(grid)
        akc = pd.to_numeric(a.active_kcal, errors="coerce").groupby(level=0).mean().reindex(grid)
    return Person(pid=pid, cgm=cg, hr=hg, meals=m.sort_values("time").reset_index(drop=True), meds=md, t0=t0,
                  act=act, act_kcal=akc, info=info or {})


class PersonStore:
    def __init__(self, root):
        self.root = root
        self._cache, self._lock = {}, threading.Lock()

    def _dir(self, pid, *parts):
        return os.path.join(self.root, "people", check_pid(pid), *parts)

    def exists(self, pid):
        return os.path.exists(self._dir(pid, "meta.json"))

    def has_state(self, pid):
        return os.path.exists(self._dir(pid, "state", "profile.json"))

    def meta(self, pid):
        return json.load(open(self._dir(pid, "meta.json")))

    def write_meta(self, pid, **kw):
        m = self.meta(pid) if self.exists(pid) else dict(pid=pid, created=_now())
        m.update(kw, updated=_now())
        _atomic_write(self._dir(pid, "meta.json"), lambda p: json.dump(m, open(p, "w"), indent=1, default=str))
        return m

    # ---------------------------------------------------------------- signals
    def read_signal(self, pid, kind):
        p = self._dir(pid, "signals", f"{kind}.parquet")
        return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame(columns=SIGNAL_COLUMNS[kind])

    def append_signal(self, pid, kind, df, replace=False):
        """append rows (deduplicated on the timestamp; a new row replaces an old one at the same time)"""
        df = df.reindex(columns=SIGNAL_COLUMNS[kind])
        key = SIGNAL_KEYS[kind]
        df[key] = _naive(df[key])
        with self._lock:
            old = pd.DataFrame(columns=SIGNAL_COLUMNS[kind]) if replace else self.read_signal(pid, kind)
            parts = [x for x in (old, df) if len(x)]
            new = pd.concat(parts, ignore_index=True) if parts else df
            if kind != "meds":
                new = new.drop_duplicates(key, keep="last")
            new = new.sort_values(key).reset_index(drop=True)
            _atomic_write(self._dir(pid, "signals", f"{kind}.parquet"), lambda p: new.to_parquet(p, index=False))
        return len(new)

    def person(self, pid, until=None):
        """Person from stored signals; until: drop signals after this time (a prediction never sees later data)"""
        meta = self.meta(pid)
        s = {k: self.read_signal(pid, k) for k in SIGNAL_COLUMNS}
        if until is not None:
            u = pd.Timestamp(until)
            for k in s:
                key = SIGNAL_KEYS[k]
                s[k] = s[k][_naive(s[k][key]) <= u] if len(s[k]) else s[k]
        if not len(s["cgm"]):
            raise ValueError("no CGM readings stored for this person")
        return assemble_person(pid, meta.get("t0"), s["cgm"], s["hr"], s["meals"], s["meds"],
                               fill_gap_min=meta.get("cgm_fill_gap_min", CGM_INTERP_LIMIT_MIN), activity=s["activity"], info=person_info(meta))

    # ---------------------------------------------------------------- state
    def save_state(self, pid, state):
        H = state["hist"].copy()
        arrays = {c: np.stack(H[c].to_numpy()) if len(H) else np.zeros((0, 0)) for c in ARRAY_COLS}
        E, Y = state["own_lib"]
        arrays.update(own_emb=np.asarray(E, np.float32), own_y=np.asarray(Y, float))
        H = H.drop(columns=ARRAY_COLS)
        for c in H.columns:
            if H[c].dtype == object and c not in ("pid", "meal_type", "photo", "photo_key", "group", "source"):
                H[c] = pd.to_numeric(H[c], errors="coerce")
        d = self._dir(pid, "state")
        _atomic_write(os.path.join(d, "hist.parquet"), lambda p: H.to_parquet(p))
        _atomic_write(os.path.join(d, "arrays.npz"), lambda p: np.savez(open(p, "wb"), **arrays))
        prof = {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in state["profile"].items()}
        extra = {k: state[k] for k in ("summary", "in_library") if k in state}
        _atomic_write(os.path.join(d, "profile.json"), lambda p: json.dump(dict(profile=prof, state_format=STATE_FORMAT, **extra), open(p, "w"),
                                                                           indent=1, default=str))
        with self._lock:
            self._cache.pop(pid, None)

    def load_state(self, pid):
        d = self._dir(pid, "state")
        stamp = os.path.getmtime(os.path.join(d, "profile.json"))
        with self._lock:
            hit = self._cache.get(pid)
            if hit and hit[0] == stamp:
                return hit[1]
        H = pd.read_parquet(os.path.join(d, "hist.parquet"))
        a = np.load(os.path.join(d, "arrays.npz"))
        for c in ARRAY_COLS:
            H[c] = list(a[c]) if len(H) else []
        pj = json.load(open(os.path.join(d, "profile.json")))
        prof = {k: (np.nan if v is None else v) for k, v in pj["profile"].items()}
        st = dict(hist=H, profile=prof, own_lib=(a["own_emb"], a["own_y"]), summary=pj.get("summary"), in_library=bool(pj.get("in_library")),
                  state_format=int(pj.get("state_format", 1)))
        with self._lock:
            self._cache[pid] = (stamp, st)
            if len(self._cache) > 512:
                self._cache.pop(next(iter(self._cache)))
        return st


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ building the person state from the first week
def build_state(person, predictor, load_photo, hist_days=7):
    """person: Person with (at least) the first `hist_days` days of signals and meals (logged nutrition + photo_key);
    load_photo(photo_key) -> PIL image. Mirrors the training pipeline (pmrf.pipeline.photo_features and
    pmrf.features.person_profile) for one person."""
    b = predictor.b
    M, CURV, PRE, _ = meal_table({person.pid: person}, hist_days)
    if not len(M):
        raise ValueError("no meals in the history")
    hist = ((M.day >= 1) & (M.day <= hist_days)).to_numpy()   # the window starts at t0 (day 1); earlier data is context only
    has = np.array([isinstance(k, str) and len(k) > 0 for k in M.photo]) & hist
    E = np.zeros((len(M), b["img_dim"]), np.float32)
    idx = np.flatnonzero(has)
    for s in range(0, len(idx), 16):
        ims = [load_photo(M.photo.iloc[i]) for i in idx[s:s + 16]]
        E[idx[s:s + 16]] = predictor.enc.encode_images(ims)
    Y = M.loc[has, MACROS].to_numpy(float)
    est, sp, smp = (b["macro_est"].predict_history(E[has], Y, person.pid) if has.any() else
                    (np.zeros((0, len(MACROS))), np.zeros((0, len(MACROS))), np.zeros(0)))
    for j, c in enumerate(MACROS):
        M[f"est_{c}"] = np.nan
        M.loc[has, f"est_{c}"] = est[:, j]
    M["est_kcal"] = 4 * M.est_carbs + 4 * M.est_protein + 9 * M.est_fat
    M["est_carbs_sd"], M["photo_sim_pop"] = np.nan, np.nan
    if has.any():
        M.loc[has, "est_carbs_sd"], M.loc[has, "photo_sim_pop"] = sp[:, 0], smp
    for mt in MEAL_TYPES:
        M[f"is_{mt}"] = (M.meal_type == mt).astype(float)
    M["in_hist"] = hist
    keep = has & M.label.to_numpy()
    # remaining model inputs of the history meals (needed when they are re-predicted, e.g. by the diet guide), as in
    # pmrf.pipeline.photo_features / pmrf.features.person_profile: image PCA, responses of similar meals (other people;
    # the person's own other meals) and the leave-one-out personal profile
    P = b["pca"].transform(E)
    for i in range(P.shape[1]):
        M[f"img{i}"] = np.where(has, P[:, i], np.nan)
    for c in ["pop_nn_dpeak", "pop_nn_iauc", "own_nn_dpeak", "own_nn_iauc", "own_nn_d60", "own_nn_sim"]:
        M[c] = np.nan
    if has.any():
        q = np.flatnonzero(has)
        mp, _, _, _ = _knn(E[q], b["lib_emb"], b["lib_y"], 10, (b["lib_pid"] == person.pid)[None, :].repeat(len(q), 0))
        M.loc[M.index[q], "pop_nn_dpeak"], M.loc[M.index[q], "pop_nn_iauc"] = mp[:, 0], mp[:, 1]
        lib = np.flatnonzero(keep)
        if len(lib):
            LY = M.iloc[lib][["dpeak", "iauc", "d60"]].to_numpy(float)
            ex = q[:, None] == lib[None, :]
            mo, _, so, _ = _knn(E[q], E[lib], LY, 3, ex)
            none = ex.all(1)
            mo[none] = np.nan; so = np.where(none, np.nan, so)
            for c, v in [("own_nn_dpeak", mo[:, 0]), ("own_nn_iauc", mo[:, 1]), ("own_nn_d60", mo[:, 2]), ("own_nn_sim", so)]:
                M.loc[M.index[q], c] = v
    cgp = cgm_profile(person, hist_days)
    HL = M[hist & M.label.to_numpy()]
    loo = [dict(meal_profile(HL[HL.index != i]), **cgp) for i in M.index]
    for c in PROFILE_FEATURES:
        M[c] = [d.get(c, np.nan) for d in loo]
    H = M[keep].copy()
    H["emb"] = list(E[keep]); H["curve"] = list(CURV[keep]); H["pre"] = list(PRE[keep])
    H["photo_key"] = H.photo
    prof = meal_profile(M[hist & M.label.to_numpy()])
    prof.update(cgm_profile(person, hist_days))
    prof = {k: prof.get(k, np.nan) for k in PROFILE_FEATURES}
    summary = dict(history_meals=int(hist.sum()), usable_meals=int(keep.sum()), meals_with_photo=int(has.sum()),
                   meal_types={k: int(v) for k, v in M[keep].meal_type.value_counts().items()},
                   cgm_days=float(np.isfinite(person.cgm[person.day(person.cgm.index) <= hist_days]).sum() / 1440))
    warn = []
    if keep.sum() < 10:
        warn.append(f"only {int(keep.sum())} usable meals (photo, logged nutrition and 2 h of CGM without another meal); 10 or more recommended")
    if summary["cgm_days"] < 5:
        warn.append(f"only {summary['cgm_days']:.1f} days of CGM in the first week")
    summary["warnings"] = warn
    return dict(hist=H, profile=prof, own_lib=(E[has], Y), summary=summary)


def open_image(data):
    """bytes or file path -> PIL image (EXIF orientation is applied by the encoder)"""
    im = Image.open(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    im.load()
    return im
