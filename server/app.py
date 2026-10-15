"""Entry point for the AWS deployment: the model team's service (model/service/app.py) with person data on
DynamoDB + S3 instead of files under /data. Nothing in model/ is changed; the store is swapped at start-up.

/predict fills a missing logged_kcal with the calories estimated from the photo (see fill_logged_kcal), and also returns `observed` (the real CGM curve and outcomes after the meal) when the store already holds CGM
for the two hours after it, e.g. for onboarded study participants; see server/observed.py.

GET /v1/people/{pid}/history/days and /history/days/{YYYY-MM-DD} serve the stored data per calendar day for the
app's history screen (server/history_view.py).

Env: PMRF_TABLE (DynamoDB table), PMRF_STATE_BUCKET (S3 bucket for first-week states), AWS_REGION,
plus every PMRF_* variable the base service reads (see model/DEPLOY.md).
"""
import json
import os
from contextlib import asynccontextmanager

import boto3
import pandas as pd
from fastapi import HTTPException, Request
from botocore.config import Config
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from service import app as base
from server.history_view import day_view, days_overview
from server.mobile_gateway import verify_mobile_request
from service.storage import check_key
from server.observed import CGM_AFTER_MIN, observed_outcome
from server.store import DynamoPersonStore

_base_lifespan = base.app.router.lifespan_context


def fill_logged_kcal(predictor):
    """The v7 model was trained with the logged meal calories always present; LightGBM reads a missing value as 0 kcal and
    under-predicts the peak. When /predict has no logged_kcal, use the calories estimated from the photo instead (on the
    24 day-8 test meals: peak-rise MAE 18.7 mg/dL, vs 18.6 with the logged calories and 19.4 with none)."""
    rows = predictor._predict_rows

    def _predict_rows(X, *a, **kw):
        if "meal_kcal_logged" in X and X.meal_kcal_logged.isna().any():
            X = X.assign(meal_kcal_logged=X.meal_kcal_logged.fillna(X.est_kcal))
        return rows(X, *a, **kw)

    predictor._predict_rows = _predict_rows


@asynccontextmanager
async def lifespan(app):
    async with _base_lifespan(app):
        store = DynamoPersonStore(os.environ["PMRF_TABLE"], os.environ["PMRF_STATE_BUCKET"], region=os.environ.get("AWS_REGION"))
        base.S["people"] = store
        fill_logged_kcal(base.S["pred"])
        base.LOG.info(json.dumps(dict(event="person_store", store=store.describe())))
        yield


PREDICT = "/v1/people/{pid}/predict"
base.app.router.routes = [r for r in base.app.router.routes if not (isinstance(r, APIRoute) and r.path == PREDICT)]


@base.app.post(PREDICT)
def predict(pid: str, body: base.PredictIn):
    """The model team's /predict, plus `observed` when the CGM after the meal is already stored. The prediction itself
    still only sees data up to the meal time."""
    resp = base.predict(pid, body)
    mt = pd.Timestamp(resp["meal"]["time"])
    person = base.S["people"].person(pid, until=mt + pd.Timedelta(minutes=CGM_AFTER_MIN))
    obs = observed_outcome(person, mt, resp["prediction"])
    if obs is not None:
        resp["observed"] = base._clean(obs)
    return resp


PHOTO_URL_TTL_S = 3600
_photo_s3 = {}


def _photo_url(key):
    """presigned GET for a meal photo (the task role may read photos/* only), or None without an S3 photo bucket"""
    bucket = base.CFG["photo_bucket"]
    if not bucket or not key.startswith("photos/"):
        return None
    check_key(key)
    if "c" not in _photo_s3:
        _photo_s3["c"] = boto3.client("s3", region_name=base.CFG["region"], config=Config(signature_version="s3v4"))
    return _photo_s3["c"].generate_presigned_url("get_object", Params=dict(Bucket=bucket, Key=key), ExpiresIn=PHOTO_URL_TTL_S)


@base.app.get("/v1/people/{pid}/history/days")
def history_days(pid: str):
    """dates with stored data, each with a glucose summary and meal / medication counts (history screen day picker)"""
    base._require(pid, state=False)
    return base._clean(days_overview(base.S["people"], pid))


@base.app.get("/v1/people/{pid}/history/days/{date}")
def history_day(pid: str, date: str):
    """one calendar day: CGM, 5-min heart rate, meals with photo URL and observed response, medication"""
    base._require(pid, state=False)
    return base._clean(day_view(base.S["people"], pid, date, photo_url=_photo_url))


base.app.router.lifespan_context = lifespan


class MobilePredictionIn(BaseModel):
    log_meal: bool = True
    meal_time: str
    meal_type: str
    photo_key: str


@base.app.middleware("http")
async def authenticate_mobile_routes(request: Request, call_next):
    """Verify Clerk before the base app checks its private x-api-key.

    In Starlette the last-added middleware is outermost. The key is injected
    only into this in-process request after Clerk verification; it is never
    exposed to the mobile client.
    """
    if request.url.path.startswith("/v1/mobile/"):
        try:
            request.state.mobile_identity = verify_mobile_request(request)
        except HTTPException as error:
            return JSONResponse(status_code=error.status_code, content={"detail": error.detail})
        headers = [(key, value) for key, value in request.scope["headers"] if key != b"x-api-key"]
        headers.append((b"x-api-key", base.CFG["api_key"].encode()))
        request.scope["headers"] = headers
    return await call_next(request)


# Register after the auth middleware: Starlette runs the last-added middleware first,
# letting CORS answer browser preflight requests before token validation.
cors_origins = [origin.strip() for origin in os.environ.get("MOBILE_CORS_ORIGINS", "").split(",") if origin.strip()]
if cors_origins:
    base.app.add_middleware(
        CORSMiddleware,
        allow_credentials=False,
        allow_headers=["Authorization", "Content-Type"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_origins=cors_origins,
    )


def mobile_identity(request: Request):
    return request.state.mobile_identity


@base.app.get("/v1/mobile/readiness")
def mobile_readiness(request: Request):
    identity = mobile_identity(request)
    store = base.S["people"]
    if not store.exists(identity.person_id):
        return dict(status="collecting_baseline", daysCollected=0, requiredDays=7)
    if not store.has_state(identity.person_id):
        return dict(status="processing_baseline", daysCollected=0, requiredDays=7)
    return dict(status="ready", daysCollected=7, requiredDays=7)


@base.app.post("/v1/mobile/photos/upload-url")
def mobile_upload_url(request: Request):
    identity = mobile_identity(request)
    result = base.upload_url(identity.person_id, base.UploadUrlIn(content_type="image/jpeg"))
    return dict(photoKey=result["photo_key"], uploadUrl=result["upload_url"], headers=result["headers"])


@base.app.post("/v1/mobile/predictions")
def mobile_prediction(request: Request, body: MobilePredictionIn):
    if body.meal_type not in base.MEAL_TYPES:
        raise ValueError("meal_type must be breakfast, lunch, dinner, or snack")
    identity = mobile_identity(request)
    return predict(
        identity.person_id,
        base.PredictIn(
            photo_key=body.photo_key,
            meal_time=body.meal_time,
            meal_type=body.meal_type,
            log_meal=body.log_meal,
        ),
    )
app = base.app
