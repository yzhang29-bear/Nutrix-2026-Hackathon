# PMRF API reference

Every endpoint of the deployed prediction service, for the app. Testing with curl: [testing_api.md](testing_api.md).
Infrastructure: [architecture.md](architecture.md). Field-by-field meaning of the `/predict` and `/profile` responses:
[`model/docs/api_contract.md`](../model/docs/api_contract.md) (the model contract; this page notes where the
deployed service differs).

## Conventions

| | |
| --- | --- |
| Base URL | `https://8la1c1eazd.execute-api.us-west-2.amazonaws.com` |
| Auth | Header `x-api-key: <key>` on every `/v1/*` call; key in Secrets Manager `pmrf/api-key`. `/healthz` is open |
| Person id | `{pid}`: 1–64 characters `A-Z a-z 0-9 _ . -`, chosen by the app |
| Times | ISO 8601 in the person's local wall-clock time. A UTC offset in a timestamp is dropped, not converted |
| Units | Glucose mg/dL (divide by 18.0 for mmol/L), heart rate bpm, nutrition grams |
| Missing values | `null` |
| Limits | 10 requests/s, burst 20 (API Gateway, 429 above it); photos up to 15 MB |
| Interactive docs | `GET /docs` (FastAPI, same key) |

## Typical flow

1. **Onboard** a person once: upload each first-week meal photo, then `POST /history` with a week of CGM, heart rate,
   meals and medication. This builds the person's state; predictions need it.
2. **Keep data current**: send new CGM and heart rate with `POST /signals` (or attach `recent_cgm` to `/predict`).
3. **Predict a meal**: get an upload URL, `PUT` the photo to S3, then `POST /predict` with the `photo_key`.
4. **Screens**: history from `GET /history/days` and `/history/days/{date}`, profile from `GET /profile`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | [`/healthz`](#get-healthz) | Liveness (no key) |
| `GET` | [`/v1/version`](#get-v1version) | API and model version |
| `POST` | [`/v1/people/{pid}/photos/upload-url`](#post-photosupload-url) | Presigned S3 upload for a meal photo |
| `POST` | [`/v1/people/{pid}/photos`](#post-photos) | Upload a photo through the API instead |
| `POST` | [`/v1/people/{pid}/history`](#post-history) | First week of data; builds the person's state |
| `POST` | [`/v1/people/{pid}/signals`](#post-signals) | New readings and meals after week 1 |
| `POST` | [`/v1/people/{pid}/predict`](#post-predict) | Predicted 0–120 min glucose response to a meal photo |
| `GET` | [`/v1/people/{pid}/profile`](#get-profile) | Personalization, diet guide, metabolic profile |
| `GET` | [`/v1/people/{pid}/history/days`](#get-historydays) | Dates with stored data and a summary of each |
| `GET` | [`/v1/people/{pid}/history/days/{date}`](#get-historydaysdate) | One day of CGM, heart rate, meals and medication |

The last two and the `observed` field of `/predict` come from the deployment layer (`server/`); everything else is the
model service (`model/service/app.py`).

### GET /healthz

`{"status": "ok", "model_version": "v7-20261010", "api_version": "0.1"}`

### GET /v1/version

`{api_version, model_version, created, notes, libraries}`.

### POST photos/upload-url

`POST /v1/people/{pid}/photos/upload-url`

| Body | |
| --- | --- |
| `content_type` | `image/jpeg` (default), `image/png`, `image/heic` or `image/webp` |

Response `{photo_key, upload_url, method: "PUT", headers, expires_in_s: 900}`. `PUT` the image bytes to `upload_url`
with the returned `Content-Type` header, then use `photo_key` in `/history`, `/signals` or `/predict`. The photo goes
straight to S3, not through the API.

### POST photos

`POST /v1/people/{pid}/photos`, multipart form with field `file` → `{photo_key}`. Simpler than the presigned upload;
413 over 15 MB, 422 if the file is not a readable image.

### POST history

`POST /v1/people/{pid}/history`. Replaces anything stored for the person and builds their state (a few seconds).

| Body | |
| --- | --- |
| `cgm[]` (required) | `{timestamp, glucose_mgdl}`, 20–600 |
| `hr[]` | `{timestamp, hr_bpm}`, 20–250 |
| `meals[]` (required) | `{timestamp, meal_type, photo_key, carbs_g, protein_g, fat_g, fiber_g, kcal}`; `meal_type` is `breakfast`, `lunch`, `dinner` or `snack`; `kcal` is the calories the user logged. A meal counts as an example for the model when it has a photo, logged nutrition and 2 h of CGM without another meal |
| `meds[]` | `{timestamp, med_class, dose}` |
| `activity[]` | `{timestamp, mets, level, active_kcal}`, one per minute or per interval. `mets` 0.5–25 (1 = rest, about 3 = walking), or `level` 0–3 (sedentary, light, moderate, vigorous) when only a level is known; `active_kcal` is kcal per minute. Only activity before a meal is used |
| `start_time` | Start of day 1; default the first CGM reading |
| `hist_days` | Days of history, 3–14, default 7 |
| `person` | `{age, bmi, sex: "female" \| "male", has_diabetes}`. Age, sex and BMI are model inputs (from v7) and feed the metabolic profile. Send them: a missing `sex` is read as male |

Response `{person_id, model_version, build_ms, history_meals, usable_meals, meals_with_photo, meal_types, cgm_days,
warnings[], profile}`. Show `warnings` (fewer than 10 usable meals or 5 days of CGM lowers accuracy). 404 lists photo
keys that do not exist.

### POST signals

`POST /v1/people/{pid}/signals` with any of `cgm[]`, `hr[]`, `meals[]` (eaten without a prediction; nutrition
optional), `meds[]`, `activity[]`, same item shapes as `/history`. Appends; CGM, heart rate or a meal at an already stored timestamp
replaces the old row.
Response `{person_id, stored: {kind: total rows}}`.

### POST predict

`POST /v1/people/{pid}/predict`

| Body | |
| --- | --- |
| `photo_key` (required) | From a photo upload |
| `meal_time` | Default: the photo's EXIF time (422 if it has none) |
| `meal_type` | Default: from the time of day |
| `logged_kcal` | Send it whenever the user enters calories for this meal, 0–5000. The v7 model was trained with it always present; when it is missing, the deployed service uses the calories estimated from the photo instead (`meal.estimated_nutrition.kcal`; about as accurate on the day-8 test meals) |
| `recent_cgm[]`, `recent_hr[]`, `recent_activity[]` | Readings to store before predicting |
| `log_meal` | Default `true`: adds the meal with its estimated nutrition to the log, so later predictions see it as the previous meal. Use `false` for "what would this do" checks |

The response is the contract in `model/docs/api_contract.md`: `meal` (photo tags, estimated nutrition, similar past
meals), `pre_meal_state`, `prediction` (curve with 80% band, glucose at 30/60/90/120 min, peak, time to peak, iAUC,
probability of exceeding 140/180), `explanation`, `personalization`, `confidence`, `what_if[]`, `latency_ms`, plus
`model_version`.

Deployed differences: a missing `logged_kcal` is filled from the photo (above), and `observed` (listed as demo-only in the contract) is returned when CGM for the two hours after
`meal_time` is already stored, e.g. for onboarded study participants: the real `curve`, `glucose_at_min`,
`pre_meal_glucose_mgdl`, `peak_glucose_mgdl`, `peak_rise_mgdl`, `time_to_peak_min`, `iauc_0_120_mgdl_min`,
`cgm_coverage` and `share_of_curve_inside_band80`. Absent for a live meal.

409 when there is no CGM reading in the 30 minutes before the meal, or the person has no state yet.

### GET profile

`GET /v1/people/{pid}/profile` → `{api_version, model_version, person_id, personalization, diet_guide,
metabolic_profile}`. Field meanings in the contract's "Profile endpoint" section. Computed on each call and cached
until new CGM arrives. Show `metabolic_profile.note` next to the numbers; none of it is a diagnosis.

### GET history/days

`GET /v1/people/{pid}/history/days`: every calendar date with stored data, oldest first. Use it for the day picker;
the first 7 dates are the person's first week.

```json
{
  "person_id": "cgmacros-003",
  "history_window": {"start": "2020-03-11T09:03:00", "end": "2020-03-18T09:03:00", "days": 7},
  "days": [
    {"date": "2020-03-11", "in_history_window": true, "cgm_readings": 826, "mean_glucose_mgdl": 134.4,
     "min_glucose_mgdl": 86.0, "max_glucose_mgdl": 213.0, "time_in_range_pct": 90.7,
     "hr_readings": 897, "meals": 2, "meds": 0}
  ]
}
```

| Field | |
| --- | --- |
| `history_window` | The model's first week: from `start` (day 1 of `/history`) for `days` days. It usually starts mid-day, so it does not line up with calendar dates |
| `days[].in_history_window` | The date overlaps the first week |
| `days[].time_in_range_pct` | Share of readings in 70–180 mg/dL |
| `days[].cgm_readings`, `hr_readings`, `meals`, `meds` | Stored rows on that date |

### GET history/days/{date}

`GET /v1/people/{pid}/history/days/2020-03-12` (`YYYY-MM-DD`), about 30 KB.

| Field | |
| --- | --- |
| `summary` | The same glucose summary as in `/history/days`, from every stored reading |
| `cgm[]` | `{t, mgdl}`, averaged per 5 minutes; `t` is the start of the 5-minute bin |
| `hr[]` | `{t, bpm}`, averaged per 5 minutes |
| `meals[]` | `time`, `meal_type`, `carbs_g`, `protein_g`, `fat_g`, `fiber_g`, `kcal`, `source`, `photo_key`, `photo_url`, `in_history_window`, `observed` |
| `meals[].source` | `history` (first-week upload), `logged` (`/signals`) or `predicted` (logged by `/predict`, nutrition estimated from the photo) |
| `meals[].photo_url` | Presigned S3 GET, valid 1 hour; `null` without a photo |
| `meals[].observed` | The measured response, same fields as `observed` in `/predict`; `null` without enough CGM around the meal. For first-week meals these are the examples the model personalizes from |
| `meds[]` | `{time, med_class, dose}` |

404 when nothing is stored on that date, 422 for a malformed date.

## Errors

The body is `{"detail": "..."}`, written to be readable.

| Status | Meaning |
| --- | --- |
| 401 | Missing or wrong `x-api-key` |
| 404 | Unknown person (no `/history` yet), photo key not found, or no data on that date |
| 409 | Person has no first-week state yet, or no CGM reading in the 30 min before the meal (send `/signals` or `recent_cgm`) |
| 413 | Photo over 15 MB |
| 422 | Invalid input: values out of range, wrong types, not an image, malformed date, no meal time |
| 429 | Throttled (10 requests/s, burst 20) |
| 503 / 504 | Service starting or overloaded; check CloudWatch logs `/ecs/pmrf` |
