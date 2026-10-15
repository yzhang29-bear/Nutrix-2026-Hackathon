# Testing the deployed API

Run from the repo root in Git Bash. Endpoints, request bodies and errors: [api.md](api.md). Architecture:
[architecture.md](architecture.md).

## 0. Setup

```bash
export API=https://8la1c1eazd.execute-api.us-west-2.amazonaws.com
export PMRF_API_KEY=$(aws secretsmanager get-secret-value --secret-id pmrf/api-key \
  --profile nutrix --region us-west-2 --query SecretString --output text)
```

`aws: command not found` in Git Bash: the shell was opened before the AWS CLI was installed. Run
`export PATH="/c/Program Files/Amazon/AWSCLIV2:$PATH"` (add it to `~/.bashrc` to keep it).
`ExpiredToken` means the event credentials expired: copy new ones from the Workshop Studio event page ("Get AWS CLI
credentials") into the `nutrix` profile with `aws configure set aws_access_key_id|aws_secret_access_key|aws_session_token
... --profile nutrix`, then run the export again.

## 1. Health check (no key needed)

```bash
curl $API/healthz
# {"status":"ok","model_version":"v7-20261010","api_version":"0.1"}
```

## 2. Full test (recommended)

```bash
PATH="$PWD/.venv/Scripts:$PATH" bash scripts/test_integration.sh $API
```

Needs `data/CGMacros` (`./ops/data/sync_data.sh pull`), AWS credentials (profile `nutrix`) and numpy, pandas, httpx,
boto3 in `.venv`. It first deletes the two test users (`ops/data/reset_person.py`), since person data persists in
DynamoDB + S3 between runs. Then it runs:

1. The end-to-end flow, 13 checks: photo upload → `/history` → `/signals` → `/predict` → `/profile`, plus
   auth and error cases.
2. The 5 unseen hospital meal photos for CGMacros participant 001 at day 8, 12:30, printed as a table. Full JSON:
   `tests/integration/out/hospital_meals.json`.

Result on 2026-10-10 (DynamoDB + S3 store): 13/13 checks passed; about 315 ms per prediction.

## 3. One prediction

```bash
./scripts/predict.sh <patient> <photo> [<meal time>] [--log]
./scripts/predict.sh cgmacros-003 day8-lunch          # the patient's real day-8 lunch: prediction + observed curve
./scripts/predict.sh cgmacros-003 korea_snuh_evening_meal 2020-03-18T12:30:00
```

Output of the hospital-photo call (no `observed` lines; a `day8-...` call adds them):

```
patient     cgmacros-003   meal time 2020-03-18 12:30:00 (lunch)
photo       photos/test/hospital-meals/korea_snuh_evening_meal.jpg
photo tags  a plate of mixed home-cooked food 54%, rice with meat 42%, yogurt 1%
nutrition   carbs 42 g (+/- 36), protein 16 g, fat 23 g, 436 kcal
glucose     before 113 -> peak 179 mg/dL (+66) at 56 min
at minutes  30: 135  60: 169  90: 157  120: 142
P(peak>140) 92%   P(peak>180) 49%
confidence  high
```

- `<patient>`: one of the onboarded patients (section 4).
- `<photo>`, one of:
  - `day8-<meal type>` (`day8-breakfast`, `day8-lunch`, `day8-dinner`, `day8-snack`: the first of that type) or
    `day8-<n>` (the n-th meal of the day): the patient's own day-8 meal photo, listed in
    `tests/integration/day8_meals.csv` with its time (section 4). Use these for predicted vs observed.
  - a hospital test image already in S3 (`korea_snuh_evening_meal`, `us_chicken_penne_greens`,
    `germany_cabbage_roll_mash`, `germany_greek_salad_pita`, `israel_hospital_meal`, stored under
    `s3://nutrix-input-images/photos/test/hospital-meals/`): foods the model has never seen, but not what the patient
    ate at that time, so there is nothing real to compare against.
  - any S3 key `photos/...`, or a local `.jpg` (uploaded first).
- `<meal time>`: local time within the patient's record (before `CGM until` in section 4), with CGM in the 30 min before.
  Taken from the list for `day8-...`; required otherwise.
- Add `--log` to record the meal in the patient's food log (`log_meal=true`: later predictions see it as the previous
  meal). Without it the call is a preview and changes nothing, so you can compare several photos at the same time.
- The API key is read from Secrets Manager unless `PMRF_API_KEY` is set. Errors print the service's reason.
  Full JSON: `tests/integration/out/predict_<patient>.json`.

The same call with curl (the photo is already in S3, so no upload):

```bash
# predict: cgmacros-003's real day-8 lunch (photo key and time from tests/integration/day8_meals.csv;
# the key changes when the patient is onboarded again)
curl -s -H "x-api-key: $PMRF_API_KEY" -H "content-type: application/json" \
  -d '{"photo_key":"photos/cgmacros-003/748bcafa1a7b460697ce9157b30f3e73.jpg","meal_time":"2020-03-18T13:34:00","log_meal":false}' \
  $API/v1/people/cgmacros-003/predict | python -m json.tool

# profile
curl -s -H "x-api-key: $PMRF_API_KEY" $API/v1/people/cgmacros-003/profile | python -m json.tool
```

Plot the glucose curve of the last `predict.sh` call (needs matplotlib, in `.venv`):

```bash
.venv/Scripts/python scripts/plot_prediction.py            # -> tests/integration/out/predict_<patient>.png
.venv/Scripts/python scripts/plot_prediction.py my.json out.png
```

It draws `prediction.curve` (`minutes` 0–120 in 5-min steps, `glucose_mgdl`, 80% band `lower80`–`upper80`), the
140 / 180 mg/dL lines with their probabilities, and the predicted peak. The peak is predicted on its own, not read off
the curve, so it usually sits above the curve's top: the curve is an average and smooths the peak out.

### Predicted vs observed

When the store already holds CGM for the two hours after `meal_time` (true for the onboarded patients in section 4,
whose later CGM is uploaded), the response also has `observed`: the real curve and outcomes after the meal, added by
`server/observed.py` with the same definitions the model was trained on. The prediction itself still only sees data
up to the meal time. Without that CGM (a live meal) the field is absent.

| Field | Meaning |
| --- | --- |
| `observed.curve` | `minutes` 0–120 (5-min steps) and the real `glucose_mgdl` |
| `observed.glucose_at_min`, `peak_glucose_mgdl`, `peak_rise_mgdl`, `time_to_peak_min`, `iauc_0_120_mgdl_min` | The same outcomes as `prediction`, measured |
| `observed.pre_meal_glucose_mgdl` | Glucose at the meal time |
| `observed.cgm_coverage` | Share of minutes 0–120 with a CGM reading (at least 0.9 in −30…120 min is required) |
| `observed.share_of_curve_inside_band80` | Share of the 25 curve points inside the predicted 80% band |

`predict.sh` prints the observed outcomes and `plot_prediction.py` draws the observed curve plus a predicted / observed
/ error table. The comparison is only meaningful when the photo is the meal actually eaten at that time, so use the
`day8-...` meals; a hospital test photo at an arbitrary time is compared with whatever the patient really ate.

What to look at in the prediction:

| Field | Meaning |
| --- | --- |
| `meal.photo_tags`, `meal.estimated_nutrition` | What the model recognised and its carb estimate |
| `prediction.curve` | 0–120 min glucose curve with the 80% band (`lower80`, `upper80`) |
| `prediction.peak_rise_mgdl`, `prediction.prob_peak_above_mgdl` | Peak rise and the chance of going above 140 / 180 mg/dL |
| `confidence.level`, `confidence.flags` | Confidence and plain-language warnings |

`demo-hospital-001` and `flowtest-user` are test users that `scripts/test_integration.sh` deletes and rebuilds on every
run; use the patients in section 4 for demos.

### History screen

```bash
# dates with data, oldest first, each with a glucose summary and meal / medication counts
curl -s -H "x-api-key: $PMRF_API_KEY" $API/v1/people/cgmacros-003/history/days | python -m json.tool
# one day: 5-min CGM and heart rate, meals with photo URL and observed response, medication
curl -s -H "x-api-key: $PMRF_API_KEY" $API/v1/people/cgmacros-003/history/days/2020-03-12 | python -m json.tool
```

Fields: [api.md](api.md#get-historydays). Checked on 2026-10-10 against `cgmacros-003`: 11 dates, the first 7 in the
history window; 2020-03-12 returns 288 CGM and 288 heart-rate points (30 KB), three first-week meals with observed
responses, and each `photo_url` downloads the JPEG.

## 4. Onboarded patients

Six CGMacros participants are stored as patients, two per group, with week 1 as history and all later CGM, heart
rate and logged meals uploaded, so predictions work at any time up to `cgm_until` and see that day's earlier meals as
previous meals (as in training):

| Patient | Group | Usable week-1 meals | CGM until |
| --- | --- | --- | --- |
| `cgmacros-002` | Healthy | 21 | 2019-11-25 08:08 |
| `cgmacros-004` | Healthy | 27 | 2023-09-21 07:48 |
| `cgmacros-007` | Prediabetes | 13 | 2023-11-10 16:10 |
| `cgmacros-008` | Prediabetes | 21 | 2023-02-09 08:57 |
| `cgmacros-003` | T2D | 22 | 2020-03-21 08:08 |
| `cgmacros-005` | T2D | 23 | 2020-08-27 10:37 |

Predict for one of them with `./scripts/predict.sh` (section 3).

Day-8 test meals: onboarding also uploads each patient's own day-8 meal photos that have enough CGM around them (a
reading at the meal time and at least 90% of minutes −30…+120, the training rule) and lists them in
`tests/integration/day8_meals.csv`: `patient`, `n` (order in the day), `meal_time`, `meal_type`, `photo_key`, logged
macros and the observed peak rise. 24 meals across the six patients: 002 has 4, 004 has 8, 007 none (its only day-8
meal lacks CGM), 008 has 2, 003 has 4, 005 has 6. Re-onboarding uploads the photos again under new keys and rewrites
those patients' rows.
Add or refresh patients (re-running replaces their history), optionally comparing one photo across all of them at
the same study day and time:

```bash
python -I ops/data/onboard_patients.py $API data/CGMacros 002 004 007 008 003 005 \
  --compare-photo tests/integration/hospital_meals/korea_snuh_evening_meal.jpg --day 8 --time 12:30
```

Same Korean hospital meal at day 8, 12:30 (2026-10-10): predicted rise from +31 mg/dL (`cgmacros-005`) to +66 mg/dL
(`cgmacros-003`); the group label alone does not order the responses, which is why the model is personalised.

## 5. A new person

A new person needs a first week of data before `/predict` works:

1. Upload each week-1 meal photo: `POST /v1/people/{id}/photos` (multipart `file`) → `photo_key`.
2. `POST /v1/people/{id}/history` with `cgm[]`, `hr[]`, `meals[]` (time, type, `photo_key`, logged nutrition) and
   optional `meds[]`. Builds the person's state in about 10 s.
3. Before each prediction, send CGM up to the meal time with `POST /v1/people/{id}/signals`, or attach `recent_cgm` to
   the `/predict` body.

`tests/integration/demo_hospital_meals.py` does all three from CGMacros data and is the easiest template.

## 6. Errors

Status codes and their meaning: [api.md](api.md#errors).
