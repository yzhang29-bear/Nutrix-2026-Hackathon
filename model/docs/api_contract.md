# Prediction API contract (v0.4)

This is the shape of the JSON the prediction service will return for one meal. The demo page reads exactly this
structure from `demo_data.json`, so UI work done against the demo carries over to the live API unchanged.
`results/demo/api_example_response.json` is a full real example.

## Endpoints (implemented in `service/app.py`, deployed as described in `DEPLOY.md`)

All `/v1` routes are behind the API gateway's sign-in (or an `x-api-key` header when `PMRF_API_KEY` is set).
Times are the person's local wall-clock time (ISO 8601); an offset in a timestamp is ignored. Interactive docs: `/docs`.

| Method | Path | Body → response |
|---|---|---|
| `POST` | `/v1/people/{person_id}/photos/upload-url` | `{content_type}` → `{photo_key, upload_url, method: "PUT", headers}`; the app uploads the JPEG directly to S3 |
| `POST` | `/v1/people/{person_id}/photos` | multipart `file` → `{photo_key}` (alternative to the presigned upload) |
| `POST` | `/v1/people/{person_id}/history` | First week: `cgm[] {timestamp, glucose_mgdl}`, `hr[] {timestamp, hr_bpm}`, `meals[] {timestamp, meal_type, photo_key, carbs_g, protein_g, fat_g, fiber_g}`, `meds[] {timestamp, med_class, dose}`, optional `start_time`, `hist_days` (7), `person {age, bmi, sex, has_diabetes}` → `{usable_meals, history_meals, meals_with_photo, meal_types, cgm_days, warnings[], profile}`. Replaces any earlier upload and builds the person's state (2–3 s) |
| `POST` | `/v1/people/{person_id}/signals` | New readings after week 1: `cgm[]`, `hr[]`, optional `meals[]` eaten without a prediction, `meds[]` → `{stored}` |
| `POST` | `/v1/people/{person_id}/predict` | `{photo_key, meal_time, meal_type?, recent_cgm[]?, recent_hr[]?, log_meal (true)}` → the JSON below. `meal_time` defaults to the photo's EXIF time and `meal_type` to the time of day; `log_meal` adds the meal (estimated nutrition) to the person's log so later predictions see it as the previous meal |
| `GET` | `/v1/people/{person_id}/profile` | Profile screen: `personalization`, `diet_guide` and `metabolic_profile` (below) |
| `GET` | `/healthz`, `/v1/version` | Liveness and model version |

**v0.4 (model v7-20261010) adds optional inputs.** Every one may be left out; the model then treats it as missing.
- `person {age, sex, bmi}` in `/history`: age in years, `sex` = `female`/`male`, BMI in kg/m².
- `activity[] {timestamp, mets, level, active_kcal}` in `/history` and `/signals`, and `recent_activity[]` in `/predict`. Send one reading per minute, or per interval. `mets` is the metabolic equivalent (1 = rest, about 3 = walking); give `level` 0–3 instead when only a level is known (0 sedentary, 1 light, 2 moderate, 3 vigorous). `active_kcal` is kcal per minute. Only activity before the meal is used.
- `meals[].kcal`: the calories the user logged for a meal.
- `logged_kcal` in `/predict`: the calories the user entered for this meal. The v7 demo model was trained with it; a real day-8 meal provides only the photo, so leave it out in that setting.

Errors: `401` missing key, `404` unknown person or photo, `409` no first-week state yet or no CGM reading in the 30 min
before the meal, `413` photo over 15 MB, `422` invalid input (message in `detail`).

## Response fields

| Field | Type | Meaning |
|---|---|---|
| `api_version` | string | `"0.1"` |
| `model_version` | string | Model store version that made the prediction, e.g. `v4-20261010` |
| `person_id` | string | |
| `meal.time`, `meal.meal_type` | string | Echo of the request |
| `meal.photo_tags[]` | `{label, p}` | Zero-shot food labels from the photo, top 3, `p` sums to ≤ 1 |
| `meal.estimated_nutrition` | object | `carbs_g`, `protein_g`, `fat_g`, `fiber_g`, `kcal`, `carbs_uncertainty_g` (spread of similar meals) |
| `meal.similar_past_meals[]` | object | Up to 3 of this person's history meals that look most alike: `photo_key`, `time`, `meal_type`, `logged_carbs_g`, `observed_peak_rise_mgdl`, `similarity` (0–1) |
| `pre_meal_state` | object | `glucose_mgdl`, `trend_mgdl_per_min` (last 30 min), `glucose_vs_24h_median_mgdl`, `heart_rate_vs_resting_bpm`, `minutes_since_last_meal` |
| `prediction.curve` | object | `minutes` = 0, 5, …, 120; `glucose_mgdl`, `lower80`, `upper80` (80% band) |
| `prediction.glucose_at_min` | object | Keys `"30"`, `"60"`, `"90"`, `"120"` → mg/dL |
| `prediction.peak_glucose_mgdl` | number | Maximum glucose in 0–120 min |
| `prediction.peak_rise_mgdl` | number | Peak minus pre-meal glucose |
| `prediction.time_to_peak_min` | number | |
| `prediction.iauc_0_120_mgdl_min` | number | Incremental area under the curve above the pre-meal level |
| `prediction.prob_peak_above_mgdl` | object | Keys `"140"`, `"180"` → probability (0–1) that the peak exceeds that level |
| `explanation.population_baseline_mgdl` | number | Average peak rise before any information about this meal or person |
| `explanation.contributions[]` | `{factor, mgdl}` | How much each factor pushed the peak-rise prediction up (+) or down (−), sorted by size. Factors: meal (from photo), meal type & time, glucose before meal, glucose trend before meal, previous meals, heart rate, medication, personal profile, similar meals (yours / other people) |
| `personalization` | object | `history_meals`, `your_avg_peak_rise_mgdl`, `population_avg_peak_rise_mgdl`, `your_rise_per_10g_carbs_mgdl`, `your_mean_glucose_mgdl`, `your_resting_hr_bpm` |
| `confidence.level` | `high` / `medium` / `low` | |
| `confidence.flags[]` | string | Plain-language reasons, e.g. "No snack in the 7-day history", "Gap in CGM just before the meal" |
| `what_if[]` | object | Scenarios (`As photographed`, `Half the carbs`, `Add 20 g protein`, `Add 10 g fiber`, `Double portion`) with `peak_rise_mgdl`, `peak_glucose_mgdl`, `iauc_0_120_mgdl_min`, `curve_mgdl` |
| `latency_ms` | object | `photo_encoder`, `nutrition_and_tags`, `features`, `response_models`, `total` (prediction without what-ifs), `what_if` |

Demo-only fields (not returned by the live API): `observed` (the real Dexcom curve and outcomes for the meal),
`meal.photo_id` / `meal_id` and `photo_id` of similar meals (keys into the demo's `images` map, base64 JPEG thumbnails;
the live API returns `photo_key` instead).

## UI notes

- Show the band, not only the line: the 80% band is calibrated (about 80% of real curves fall inside it).
- `confidence.flags` are written to be shown to users verbatim.
- `what_if` curves share the `minutes` grid of `prediction.curve`.
- Glucose is always mg/dL; divide by 18.0 for mmol/L.

## Profile endpoint (v0.2)

`GET /v1/people/{person_id}/profile` returns, in addition to `personalization`:

| Field | Type | Meaning |
|---|---|---|
| `diet_guide.method`, `diet_guide.personalized` | string, bool | `measured in your first-week standardized meals` (`personalized: true`) or `group-average effects ...` (`personalized: false`, for people without standardized meals of their own) |
| `diet_guide.carb_budget_g_per_meal` | number or null | Largest carbs per meal expected to keep the rise above the pre-meal level at or below `target_rise_mgdl`; null with `carb_budget_note` when the person has no standardized meal of their own |
| `diet_guide.target_rise_mgdl` | number | 50 by default (a rise target does not depend on the sensor's overall offset) |
| `diet_guide.measured_effects` | object | `low carb`, `+protein`, `+fat` → `{you_mgdl, adjusted_mgdl, group_mgdl}`: the person's own week-1 contrast, the value used after adjusting toward the group, and the group average |
| `diet_guide.predicted_rise_per_10g_carbs_mgdl` | number | Model-based sensitivity |
| `diet_guide.carb_curve[]` | `{carbs_g, mean_predicted_rise_mgdl, mean_predicted_peak_mgdl}` | For a chart of carbs vs expected rise |
| `diet_guide.biggest_spikes[]` | object | Up to 3 first-week meals with the largest rise: `photo_key` (`photo_id` in the demo), `meal_type`, `logged_carbs_g`, `observed_rise_mgdl`, `best_change`, `options[]` (`change`, `predicted_rise_mgdl`, `change_vs_as_eaten_mgdl`) |
| `diet_guide.rise_per_10g_by_time_of_day_mgdl` | object | `morning`, `evening` when there are enough meals |
| `metabolic_profile.est_hba1c` (+ `_lo`, `_hi`) | number | HbA1c estimated from the first week of CGM, % |
| `metabolic_profile.gmi` | number | Standard CGM-to-HbA1c formula, for reference |
| `metabolic_profile.status` | `{label, prob}` | Most likely range (normal / prediabetes / diabetes) and its probability; never show as a diagnosis |
| `metabolic_profile.gri_zone` | string | Glycemia Risk Index zone A–E (literature-derived) |
| `metabolic_profile.t2d_risk_tier` | string or null | Relative future type 2 diabetes risk among people without diabetes: "lower than most", "about average", "higher than most". Absolute probabilities are not shown because they are not calibrated for Dexcom data |
| `metabolic_profile.note` | string | Plain-language caveat to show next to the numbers |

All diet guidance and risk numbers are model-based estimates for decision support, not medical advice; show the `note`.
