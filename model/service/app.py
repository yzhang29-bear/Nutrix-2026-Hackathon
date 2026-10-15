"""PMRF prediction service (FastAPI). Contract: docs/api_contract.md. Deployment: DEPLOY.md.

Configuration (environment variables):
  PMRF_MODEL_DIR        local model store version directory (default /models/current)
  PMRF_MODEL_URI        s3://bucket/prefix/<version>/ : downloaded to PMRF_MODEL_DIR at start-up if that directory is empty
  PMRF_DATA_DIR         person data (signals + states), e.g. an EFS mount (default /data)
  PMRF_PHOTO_BUCKET     S3 bucket for meal photos; without it photos are files under PMRF_PHOTO_DIR (default $PMRF_DATA_DIR/photos)
  PMRF_THREADS          CPU threads for the models (default 2)
  PMRF_MAX_CONCURRENT   predictions computed at the same time per container (default 2)
  PMRF_API_KEY          optional shared key checked in the x-api-key header (API Gateway normally handles sign-in)
  AWS_REGION            region for S3
"""
import io
import json
import logging
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pmrf.data import _meal_type_from_time, MEAL_TYPES
from pmrf.features import MED_CLASSES
from pmrf.guide import person_guide_measured, person_guide_group
from pmrf.predict import Predictor, API_VERSION, _r
from pmrf.risk import metabolic_profile, app_summary
from pmrf.state import PersonStore, build_state, check_pid, open_image
from pmrf.store import load_store
from service.storage import PhotoStore, fetch_model

LOG = logging.getLogger("pmrf")
logging.basicConfig(level=logging.INFO, format="%(message)s")
CFG = dict(model_dir=os.environ.get("PMRF_MODEL_DIR", "/models/current"), model_uri=os.environ.get("PMRF_MODEL_URI"),
           data_dir=os.environ.get("PMRF_DATA_DIR", "/data"), photo_bucket=os.environ.get("PMRF_PHOTO_BUCKET"),
           threads=int(os.environ.get("PMRF_THREADS", "2")), max_concurrent=int(os.environ.get("PMRF_MAX_CONCURRENT", "2")),
           api_key=os.environ.get("PMRF_API_KEY"), region=os.environ.get("AWS_REGION"))
CFG["photo_dir"] = os.environ.get("PMRF_PHOTO_DIR", os.path.join(CFG["data_dir"], "photos"))
MAX_PHOTO_BYTES = 15 * 1024 * 1024
S = {}                                  # loaded singletons


@asynccontextmanager
async def lifespan(app):
    t = time.perf_counter()
    import torch
    torch.set_num_threads(CFG["threads"])
    if CFG["model_uri"] and not os.path.exists(os.path.join(CFG["model_dir"], "manifest.json")):
        fetch_model(CFG["model_uri"], CFG["model_dir"], CFG["region"])
    bundle = load_store(CFG["model_dir"])
    S["pred"] = Predictor(bundle, threads=CFG["threads"])
    S["version"] = bundle["manifest"]["version"]
    S["people"] = PersonStore(CFG["data_dir"])
    S["photos"] = PhotoStore(root=CFG["photo_dir"], bucket=CFG["photo_bucket"], region=CFG["region"])
    S["sem"] = threading.BoundedSemaphore(CFG["max_concurrent"])
    S["profile_cache"] = {}
    from PIL import Image
    S["pred"].enc.encode_images([Image.new("RGB", (256, 256))])          # warm-up
    LOG.info(json.dumps(dict(event="ready", model_version=S["version"], load_s=round(time.perf_counter() - t, 2),
                             photos=S["photos"].describe(), data_dir=CFG["data_dir"], threads=CFG["threads"])))
    yield


app = FastAPI(title="PMRF post-meal glucose API", version=API_VERSION, lifespan=lifespan)


@app.middleware("http")
async def log_and_auth(request: Request, call_next):
    t = time.perf_counter()
    if CFG["api_key"] and request.url.path.startswith("/v1/") and request.headers.get("x-api-key") != CFG["api_key"]:
        return JSONResponse(status_code=401, content=dict(detail="missing or wrong x-api-key"))
    resp = await call_next(request)
    route = request.scope.get("route")
    LOG.info(json.dumps(dict(event="request", method=request.method, path=getattr(route, "path", request.url.path), status=resp.status_code,
                             ms=round((time.perf_counter() - t) * 1000, 1))))
    return resp


@app.exception_handler(ValueError)
async def _bad_request(request, exc):
    return JSONResponse(status_code=422, content=dict(detail=str(exc)))


@app.exception_handler(FileNotFoundError)
async def _not_found(request, exc):
    return JSONResponse(status_code=404, content=dict(detail=f"not found: {exc}"))


# ------------------------------------------------------------------ request bodies
class CGMReading(BaseModel):
    timestamp: str
    glucose_mgdl: float = Field(ge=20, le=600)


class HRReading(BaseModel):
    timestamp: str
    hr_bpm: float = Field(ge=20, le=250)


class MealIn(BaseModel):
    timestamp: str
    meal_type: Literal["breakfast", "lunch", "dinner", "snack"]
    photo_key: Optional[str] = None
    carbs_g: Optional[float] = Field(None, ge=0)
    protein_g: Optional[float] = Field(None, ge=0)
    fat_g: Optional[float] = Field(None, ge=0)
    fiber_g: Optional[float] = Field(None, ge=0)
    kcal: Optional[float] = Field(None, ge=0, le=5000, description="calories the user logged for the meal (optional)")


class ActivityReading(BaseModel):
    """one activity reading (per minute, or per interval): METs, or a 0-3 intensity level, and/or active kcal per minute"""
    timestamp: str
    mets: Optional[float] = Field(None, ge=0.5, le=25)
    level: Optional[int] = Field(None, ge=0, le=3, description="0 sedentary, 1 light, 2 moderate, 3 vigorous; used when mets is missing")
    active_kcal: Optional[float] = Field(None, ge=0, le=60, description="kcal per minute")


LEVEL_TO_METS = {0: 1.2, 1: 2.5, 2: 4.5, 3: 7.0}          # as used for the CGMacros participants with a level only


class MedIn(BaseModel):
    timestamp: str
    med_class: Literal[tuple(MED_CLASSES)]
    dose: float = Field(ge=0)


class PersonInfo(BaseModel):
    age: Optional[float] = None
    bmi: Optional[float] = None
    sex: Optional[Literal["female", "male"]] = None
    has_diabetes: Optional[bool] = None


class HistoryIn(BaseModel):
    start_time: Optional[str] = Field(None, description="start of day 1 (default: first CGM reading)")
    hist_days: int = Field(7, ge=3, le=14)
    cgm: List[CGMReading]
    hr: List[HRReading] = []
    meals: List[MealIn]
    meds: List[MedIn] = []
    activity: List[ActivityReading] = []
    person: Optional[PersonInfo] = None


class SignalsIn(BaseModel):
    cgm: List[CGMReading] = []
    hr: List[HRReading] = []
    meals: List[MealIn] = Field([], description="meals eaten without a prediction (nutrition optional)")
    meds: List[MedIn] = []
    activity: List[ActivityReading] = []


class PredictIn(BaseModel):
    photo_key: str
    meal_time: Optional[str] = Field(None, description="local time of the meal; default: the photo's EXIF time")
    meal_type: Optional[Literal["breakfast", "lunch", "dinner", "snack"]] = Field(None, description="default: from the time of day")
    recent_cgm: List[CGMReading] = Field([], description="optional: CGM readings to store before predicting")
    recent_hr: List[HRReading] = []
    recent_activity: List[ActivityReading] = []
    logged_kcal: Optional[float] = Field(None, ge=0, le=5000, description="calories the user entered for this meal (optional; used by the demo model)")
    log_meal: bool = Field(True, description="add this meal (estimated nutrition) to the person's meal log")


class UploadUrlIn(BaseModel):
    content_type: Literal["image/jpeg", "image/png", "image/heic", "image/webp"] = "image/jpeg"


# ------------------------------------------------------------------ helpers
def _ts(s):
    t = pd.Timestamp(s)
    return t.tz_localize(None) if t.tzinfo is not None else t       # keep the local wall-clock time


def _frames(cgm=(), hr=(), meals=(), meds=(), source="logged", activity=()):
    out = {}
    if cgm:
        out["cgm"] = pd.DataFrame(dict(timestamp=[_ts(r.timestamp) for r in cgm], glucose_mgdl=[r.glucose_mgdl for r in cgm]))
    if hr:
        out["hr"] = pd.DataFrame(dict(timestamp=[_ts(r.timestamp) for r in hr], hr_bpm=[r.hr_bpm for r in hr]))
    if meals:
        rows = []
        for m in meals:
            c, p, f = (np.nan if v is None else v for v in (m.carbs_g, m.protein_g, m.fat_g))
            rows.append(dict(time=_ts(m.timestamp), meal_type=m.meal_type, carbs=c, protein=p, fat=f,
                             fiber=np.nan if m.fiber_g is None else m.fiber_g, kcal=4 * c + 4 * p + 9 * f,
                             calories_logged=np.nan if m.kcal is None else m.kcal, photo_key=m.photo_key, source=source))
        out["meals"] = pd.DataFrame(rows)
    if meds:
        out["meds"] = pd.DataFrame(dict(time=[_ts(r.timestamp) for r in meds], med_class=[r.med_class for r in meds], dose=[r.dose for r in meds]))
    if activity:
        out["activity"] = pd.DataFrame(dict(timestamp=[_ts(r.timestamp) for r in activity],
                                            mets=[r.mets if r.mets is not None else LEVEL_TO_METS.get(r.level, np.nan) for r in activity],
                                            active_kcal=[np.nan if r.active_kcal is None else r.active_kcal for r in activity]))
    return out


def _clean(o):
    """JSON-safe: numpy scalars -> Python, NaN/inf -> null"""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


def _require(pid, state=True):
    check_pid(pid)
    st = S["people"]
    if not st.exists(pid):
        raise HTTPException(404, f"unknown person {pid}: upload the first week with POST /v1/people/{pid}/history")
    if state and not st.has_state(pid):
        raise HTTPException(409, f"person {pid} has no history state yet: POST /v1/people/{pid}/history")


def _load_photo(key):
    try:
        return open_image(S["photos"].get(key))
    except FileNotFoundError:
        raise HTTPException(404, f"photo not found: {key}")
    except (OSError, SyntaxError):
        raise HTTPException(422, f"not a readable image: {key}")


def _exif_time(img):
    try:
        ex = img.getexif()
        v = ex.get_ifd(0x8769).get(36867) or ex.get(306)
        return pd.Timestamp.strptime(v, "%Y:%m:%d %H:%M:%S") if v else None
    except Exception:
        return None


# ------------------------------------------------------------------ endpoints
@app.get("/healthz")
def healthz():
    return dict(status="ok", model_version=S.get("version"), api_version=API_VERSION)


@app.get("/v1/version")
def version():
    m = S["pred"].b["manifest"]
    return dict(api_version=API_VERSION, model_version=m["version"], created=m["created"], notes=m.get("notes"), libraries=m["libraries"])


@app.post("/v1/people/{pid}/photos")
async def upload_photo(pid: str, file: UploadFile = File(...)):
    check_pid(pid)
    data = await file.read()
    if len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(413, "photo larger than 15 MB")
    try:
        open_image(data)
    except Exception:
        raise HTTPException(422, "not a readable image")
    ct = file.content_type if file.content_type in ("image/jpeg", "image/png", "image/heic", "image/webp") else "image/jpeg"
    key = S["photos"].put(S["photos"].new_key(pid, ct), data, ct)
    return dict(photo_key=key)


@app.post("/v1/people/{pid}/photos/upload-url")
def upload_url(pid: str, body: UploadUrlIn):
    check_pid(pid)
    key = S["photos"].new_key(pid, body.content_type)
    try:
        url = S["photos"].upload_url(key, body.content_type)
    except NotImplementedError as e:
        raise HTTPException(501, str(e))
    return dict(photo_key=key, upload_url=url, method="PUT", headers={"Content-Type": body.content_type}, expires_in_s=900)


@app.post("/v1/people/{pid}/history")
def history(pid: str, body: HistoryIn):
    check_pid(pid)
    if not body.cgm:
        raise HTTPException(422, "cgm readings are required")
    fr = _frames(body.cgm, body.hr, body.meals, body.meds, source="history", activity=body.activity)
    missing = [m.photo_key for m in body.meals if m.photo_key]
    missing = [k for k in missing if not _photo_exists(k)]
    if missing:
        raise HTTPException(404, dict(detail="photos not found", photo_keys=missing[:20]))
    t0 = _ts(body.start_time) if body.start_time else fr["cgm"].timestamp.min()
    st = S["people"]
    st.write_meta(pid, t0=str(t0), hist_days=body.hist_days, person=(body.person.model_dump() if body.person else {}),
                  model_version=S["version"])
    for kind in ("cgm", "hr", "meals", "meds", "activity"):
        st.append_signal(pid, kind, fr.get(kind, pd.DataFrame()), replace=True)
    person = st.person(pid)
    t = time.perf_counter()
    with S["sem"]:
        state = build_state(person, S["pred"], _load_photo, body.hist_days)
    st.save_state(pid, state)
    S["profile_cache"].pop(pid, None)
    return _clean(dict(person_id=pid, model_version=S["version"], build_ms=round((time.perf_counter() - t) * 1000), **state["summary"],
                       profile={k: v for k, v in state["profile"].items()}))


def _photo_exists(key):
    try:
        S["photos"].get(key)
        return True
    except (FileNotFoundError, ValueError):
        return False


@app.post("/v1/people/{pid}/signals")
def signals(pid: str, body: SignalsIn):
    _require(pid, state=False)
    fr = _frames(body.cgm, body.hr, body.meals, body.meds, source="logged", activity=body.activity)
    counts = {k: S["people"].append_signal(pid, k, v) for k, v in fr.items()}
    S["profile_cache"].pop(pid, None)
    return dict(person_id=pid, stored=counts)


@app.post("/v1/people/{pid}/predict")
def predict(pid: str, body: PredictIn):
    _require(pid)
    img = _load_photo(body.photo_key)
    mt = _ts(body.meal_time) if body.meal_time else _exif_time(img)
    if mt is None:
        raise HTTPException(422, "meal_time is required (the photo has no EXIF time)")
    meal_type = body.meal_type or _meal_type_from_time(mt, 1000.0)
    st = S["people"]
    if body.recent_cgm or body.recent_hr or body.recent_activity:
        for k, v in _frames(body.recent_cgm, body.recent_hr, activity=body.recent_activity).items():
            st.append_signal(pid, k, v)
    person = st.person(pid, until=mt)
    recent = person.cgm[(person.cgm.index >= mt - pd.Timedelta("30min")) & (person.cgm.index <= mt)]
    if not np.isfinite(recent.to_numpy(float)).any():
        raise HTTPException(409, "no CGM reading in the 30 minutes before the meal: upload recent CGM (signals or recent_cgm)")
    state = st.load_state(pid)
    with S["sem"]:
        resp, extra = S["pred"].respond(person, img, mt, meal_type, state=state, logged_kcal=body.logged_kcal)
    resp["model_version"] = S["version"]
    if body.log_meal:
        X = extra["X"].iloc[0]
        st.append_signal(pid, "meals", pd.DataFrame([dict(time=mt, meal_type=meal_type, carbs=X.est_carbs, protein=X.est_protein, fat=X.est_fat,
                                                          fiber=X.est_fiber, kcal=X.est_kcal, photo_key=body.photo_key, source="predicted",
                                                          calories_logged=np.nan if body.logged_kcal is None else body.logged_kcal)]))
    return _clean(resp)


@app.get("/v1/people/{pid}/profile")
def profile(pid: str):
    _require(pid)
    st = S["people"]
    state = st.load_state(pid)
    stamp = (id(state), os.path.getmtime(st._dir(pid, "signals", "cgm.parquet")))
    hit = S["profile_cache"].get(pid)
    if hit and hit[0] == stamp:
        return hit[1]
    meta = st.meta(pid)
    pred, H, prof = S["pred"], state["hist"], state["profile"]
    b = pred.b
    pers = dict(history_meals=int(prof.get("p_n") or 0), your_avg_peak_rise_mgdl=_r(prof.get("p_dpeak", np.nan)),
                population_avg_peak_rise_mgdl=_r(b["pop_avg_rise"]), your_rise_per_10g_carbs_mgdl=_r(prof.get("p_dpeak_per10g", np.nan)),
                your_mean_glucose_mgdl=_r(prof.get("p_g_mean", np.nan)), your_resting_hr_bpm=_r(prof.get("p_hr_rest", np.nan)))
    eff = b.get("guide_effects")
    gd = None
    if eff is not None and state.get("in_library") and pid in eff["r7"].index:
        gd = person_guide_measured(H, eff, pid)
        if gd is not None:
            gd["personalized"] = True
    if gd is None and eff is not None:
        gd = person_guide_group(H, eff)
    if gd is None:
        gd = dict(available=False, reason="this model version has no nutrient effects")
    for sp in gd.get("biggest_spikes", []):
        sp["photo_key"] = sp.pop("photo_path", None)
    mp = None
    if b.get("risk_models"):
        person = st.person(pid)
        days = int(meta.get("hist_days", 7))
        g = person.cgm[person.day(person.cgm.index) <= days].dropna()
        info = meta.get("person") or {}
        d = metabolic_profile(g, age=info.get("age"), bmi=info.get("bmi"), sex=info.get("sex"), has_diabetes=info.get("has_diabetes"),
                              days=days, models=b["risk_models"])
        mp = app_summary(d, b["risk_models"].get("t2d_tier_reference"))
    out = _clean(dict(api_version=API_VERSION, model_version=S["version"], person_id=pid, personalization=pers, diet_guide=gd,
                      metabolic_profile=mp))
    S["profile_cache"][pid] = (stamp, out)
    return out
