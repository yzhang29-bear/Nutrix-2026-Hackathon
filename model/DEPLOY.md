# Deploying the PMRF prediction service on AWS

The service predicts the 0–120 min glucose response to a meal from its photo, the person's recent CGM and heart rate,
and their first week of data. It is one CPU-only container (FastAPI). API fields: `docs/api_contract.md`.

## What is in the package

| Path | What |
|---|---|
| `deploy/Dockerfile`, `deploy/requirements.txt`, `.dockerignore` | Image build (Python 3.11 slim, CPU PyTorch, pinned libraries) |
| `service/` | Web service: `app.py` (endpoints), `storage.py` (photos on disk or S3, model download from S3) |
| `pmrf/` | Model code; `pmrf/store.py` loads a model version, `pmrf/state.py` stores per-person data and builds the first-week state |
| `deploy/model_store/v4-20261010/` | Model version in stable formats: LightGBM text models, numpy arrays, safetensors weights, JSON; `manifest.json` lists library versions and SHA-256 of every file (checked at start-up). No pickles |
| `scripts/golden_test.py`, `tests/` | Acceptance tests (below) |

## Run it locally

The model files are not in git (466 MB; they live in S3). From this folder (`model/` in the team repo):

```bash
AWS_PROFILE=<profile> python -I scripts/fetch_model_store.py      # -> deploy/model_store/v7-20261010 (checksums verified)
docker build -f deploy/Dockerfile -t pmrf-api:v7-20261010 .
mkdir -p /tmp/pmrf-data && chmod 777 /tmp/pmrf-data
docker run --rm --cpus 2 --memory 4g -p 8080:8080 -v /tmp/pmrf-data:/data pmrf-api:v7-20261010
curl localhost:8080/healthz          # {"status":"ok","model_version":"v4-20261010",...}
```
Interactive API docs: `http://localhost:8080/docs`.

## AWS architecture

```
App ──HTTPS──> API Gateway (HTTP API, Cognito JWT authorizer) ──VPC link──> ALB ──> ECS Fargate service (this image)
App ──presigned PUT──> S3 bucket (meal photos)                                       │ reads photos (S3)
                                                                                     │ reads/writes person data (EFS at /data)
                                                                                     │ logs (CloudWatch)
Offline training ──> S3 (model versions) ──> new image or PMRF_MODEL_URI ──> rolling deploy
```

| Component | Setting |
|---|---|
| ECS Fargate task | 2 vCPU, 4 GB, Linux x86_64, port 8080, health check `GET /healthz` (start period 60 s). One uvicorn worker per task |
| EFS | Mounted at `/data` (person data: signals and first-week state, about 1–3 MB per person). Encryption at rest on; access point with uid/gid 10001 |
| S3 | Photos bucket (SSE on, block public access, CORS allowing `PUT` from the app); optional `models/` prefix for model versions |
| IAM task role | `s3:GetObject`, `s3:PutObject` on `arn:aws:s3:::<bucket>/photos/*`; `s3:GetObject` on `models/*` if models are loaded from S3; EFS client mount/write |
| API Gateway | HTTP API with a Cognito JWT authorizer; routes `/v1/*` and `/healthz`; payload limit 10 MB (a first-week upload is 1–2 MB) |
| Logs | `awslogs` driver to CloudWatch. The service logs one JSON line per request (method, route, status, ms) and never logs glucose values or photos |
| Scaling | Target tracking on CPU 60 %; minimum 1 task (2 for availability). One task serves about 3.5 predictions/s |

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PMRF_DATA_DIR` | `/data` | Person data directory (EFS mount) |
| `PMRF_PHOTO_BUCKET` | — | S3 bucket for photos; without it photos are files under `PMRF_PHOTO_DIR` (`$PMRF_DATA_DIR/photos`) |
| `PMRF_MODEL_DIR` | `/models/current` | Model version baked into the image |
| `PMRF_MODEL_URI` | — | `s3://<bucket>/models/<version>/`: downloaded at start-up into `PMRF_MODEL_DIR` when that directory is empty (set `PMRF_MODEL_DIR=/tmp/model`) |
| `PMRF_THREADS` | `2` | CPU threads for the models (= vCPUs) |
| `PMRF_MAX_CONCURRENT` | `2` | Predictions computed at the same time per task |
| `PMRF_API_KEY` | — | Optional shared key (`x-api-key` header) when the service is reachable without API Gateway |
| `AWS_REGION` | — | Region of the S3 bucket |

### Model versions

| Version | S3 | What changed |
|---|---|---|
| `v7-20261010` (current, default in the image) | `s3://nutrix-models/models/v7-20261010/` | Model A also uses activity before the meal (METs, active kcal), age, sex, BMI and the logged meal calories. The transformer also uses age, sex and BMI and was re-pretrained on MetaboNet on a GPU. Accuracy on new day-8 meals is the same as v4 within noise, and the API accepts the new inputs (`docs/api_contract.md`, v0.4) |
| `v4-20261010` | `s3://nutrix-models/models/v4-20261010/` | Photo, CGM, heart rate, time, medication and first-week inputs |

Single-meal (API) predictions of v7 match the batch evaluation to a median of 0.03 mg/dL. 4 of 176 meals differ by up to 1.6 mg/dL in peak rise, because the batch run used the GPU and the service uses the CPU.

### Model artifacts in S3

Each version is uploaded to `s3://nutrix-models/models/<version>/`, for example `v7-20261010` (us-west-2; 115 files, 466 MB; bucket encrypted,
private, versioned) and was verified by downloading it back and checking every file against `manifest.json`.
To load it from S3 instead of the copy in the image: `PMRF_MODEL_URI=s3://nutrix-models/models/v4-20261010/`,
`PMRF_MODEL_DIR=/tmp/model`, `AWS_REGION=us-west-2`.

### Releasing a new model version

1. On our side: retrain, then `python -I scripts/export_model_store.py results deploy/model_store <version>`.
2. Either rebuild the image with `--build-arg MODEL_VERSION=<version>`, or upload it with `python -I scripts/upload_model_store.py deploy/model_store/<version> nutrix-models` (files first, manifest last, never overwrites a version) and set `PMRF_MODEL_URI`.
3. Roll the ECS service. People's first-week states are tied to the model version that built them: after a model change, rebuild them by replaying `/history` (the raw signals are kept in `/data`).

## How the app uses it

1. **Photos**: `POST /v1/people/{id}/photos/upload-url` gives a presigned S3 URL and a `photo_key`; the app `PUT`s the JPEG there. (`POST /v1/people/{id}/photos` with a multipart file also works.)
2. **Week 1**: `POST /v1/people/{id}/history` with CGM, heart rate, meals (time, type, `photo_key`, logged nutrition) and medication. The service builds the person's state (about 2–3 s).
3. **After week 1**: `POST /v1/people/{id}/signals` with new CGM and heart-rate readings (batches from the CGM or health-app sync), or attach `recent_cgm` to each prediction.
4. **Each meal**: `POST /v1/people/{id}/predict` with `photo_key`, `meal_time` (local time) and optionally `meal_type`. Requires a CGM reading in the 30 min before the meal.
5. **Profile screen**: `GET /v1/people/{id}/profile` (diet guide and metabolic profile).

Times are the person's local wall-clock time; an offset in the timestamp is ignored (the local part is kept).

## Acceptance tests (all passed on 2026-10-10)

| Test | Command | Result |
|---|---|---|
| Golden: report predictions over HTTP (cohort imported with `scripts/import_cohort.py`) | `golden_test.py <url> imported results/demo/demo_data.json out.json` | 176 of 176 meals identical to the report (curve, band, peak, iAUC, what-ifs, explanation, flags), in the container |
| New users from raw data (`/history` → `/signals` → `/predict`, 44 people, 402 day-8+ meals) | `golden_test.py <url> onboard ...` | Unseen-meal peak-rise error 19.8 mg/dL, the same as the batch evaluation; median difference per meal 0.05 mg/dL |
| End-to-end flow incl. photo upload, auth and error cases | `tests/test_api_flow.py <url> <CGMacros dir>` | 13 of 13 checks |
| S3 photos, presigned upload, model download (mocked S3) | `tests/test_s3_storage.py deploy/model_store/v4-20261010` | passed |
| Load (container limited to 2 vCPU / 4 GB) | `scripts/load_test.py <url> 1 120 out.json` | 1 client: median 252 ms, p99 374 ms, 3.8 predictions/s; 4 clients: same throughput, median 1.2 s. Memory about 1.0 GB, start-up 3 s |

## Privacy and security

- If real users' data is processed: use HIPAA-eligible services only (ECS Fargate, EFS, S3, API Gateway, Cognito and CloudWatch are) and sign a BAA with AWS. Encrypt EFS and S3 at rest; TLS terminates at API Gateway.
- The container runs as a non-root user; photo keys are validated (no path traversal); request bodies are validated (types, ranges).
- Person ids should be pseudonymous app ids, not names or emails.

## Known limits

- Person data on EFS is written file-by-file with atomic renames. Uploads for the same person should come from one client at a time; for many writers per person, move signals to DynamoDB.
- The diet guide is personal only for people with standardized test meals in week 1 (the development cohort). New users get their own biggest week-1 spikes with the group-average effect of each change, measured in those standardized meals, and no carb budget (`personalized: false`). Model counterfactuals were checked as an alternative and under-react to carbs about five-fold, so they are not used. A personal guide for new users would need one standardized test meal in week 1.
- Predictions need a CGM reading within 30 min before the meal (409 otherwise).
- The model was developed on CGMacros (45 adults, Dexcom G6, Libre). Retrain on the official challenge data before release.
