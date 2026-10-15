# PRD: Predict My Glucose Response

Oct 10, 2026 · @Fiona

We will ship a person-specific glucose response predictor that learns from 7 days of CGM, wearable, meal and medication data and forecasts a person's glucose curve on day 8 after a new meal or activity, with an uncertainty band and the top drivers of each prediction. AWS hosts the data, models and API; Microsoft supplies a second, independent meal-photo estimator; ElevenLabs adds voice logging and spoken results; Anakin grounds nutrition in real menu data; Vlad Health supplies medication reference data; Cigna and Sevaro shape the payer and clinician pathway. Target: a working end-to-end demo for RU HealthHack 2026, Wellness Reimagined Challenge 3, on Sunday, October 11.

## Problem and users

The same meal can send one person's glucose soaring and barely move another's. Carb counting ignores what actually drives the difference: baseline glucose and its trend, recent activity and heart rate, time of day, the prior meal, and medication timing. People find out after they eat; we want them to know before.

| User | What they need | Where they see it |
| --- | --- | --- |
| Adult with type 2 diabetes or prediabetes | Know whether a meal will spike them, and what would soften it | Patient app: photo or voice in, predicted curve out |
| Adult with type 1 diabetes on insulin | Anticipate post-meal highs and post-exercise lows | Patient app with hypoglycemia-risk warning |
| Clinician or diabetes educator | See this person's response profile and which levers matter most | Clinician dashboard |
| Payer care manager (Cigna use case) | Find members whose responses are worsening and target coaching | Cohort view (stretch) |
| Hackathon judges | Day-8 predicted vs observed, proof of personalization | Clinician dashboard, demo mode |

Core user stories:

1. As a patient, I snap a photo or say what I am about to eat and see my predicted 2-hour glucose curve with a confidence band.
2. As a patient, I ask "what if I walk 15 minutes after?" and hear how much the predicted peak drops.
3. As a person with type 1 diabetes, before a run I see my predicted glucose drop and my risk of going below 70 mg/dL.
4. As a clinician, I see this person's carb sensitivity next to the population average, their time-of-day pattern, and the top drivers of each prediction.
5. As a clinician, I see when the model is unsure and why: an unfamiliar food, a time of day never seen, or a medication change.

## Goals, non-goals and success metrics

Success means beating a population model on held-out day-8 data by a clear margin, with honest uncertainty, and showing it live.

Goals:

1. **Meal response.** Predict glucose at 30, 60, 90 and 120 min, the peak, the rise from pre-meal to peak, time to peak, and 2-hour incremental area under the curve (iAUC).
2. **Activity response.** Predict glucose at 30, 60 and 120 min after an activity starts, the lowest value, time to that low, and the probability of dropping below 70 mg/dL.
3. **Personalization.** Learn per-person parameters and show how much of the response is person-specific.
4. **Explanation.** Name the top drivers of every prediction.
5. **Uncertainty.** Give a band for every curve and flag predictions the model should not be trusted on.
6. **Any-format input.** Accept CGM, wearable, meal and activity data in whatever form it arrives: photos, voice, text logs, device exports, parquet.

Non-goals: insulin or medication dosing advice, regulatory claims, real-time closed-loop control, real patient data during the hackathon, and training foundation models from scratch.

| Metric | Definition | Target |
| --- | --- | --- |
| Horizon error | MAE in mg/dL at 30, 60, 90, 120 min on held-out day 8 | At least 15% lower than the population LightGBM at 60 and 90 min |
| Peak error | MAE of peak and of rise-to-peak, mg/dL | Lower than the population LightGBM |
| Time-to-peak error | MAE in minutes | Lower than the population LightGBM |
| iAUC error | Mean absolute percentage error of 2-hour iAUC | Lower than the population LightGBM |
| Personalization gain | (population MAE − personal MAE) ÷ population MAE | At least 15%, reported per cohort |
| Calibration | Share of observed points inside the 80% band | 75% to 85% |
| Photo-to-carbs error | MAE in grams vs CGMacros ground truth | Calibrated ensemble beats either raw vision model |
| Post-activity low detection | Sensitivity for glucose below 70 mg/dL within 2 hours (MetaboNet) | Reported with false-alarm rate |
| API latency | 95th percentile of /predict, photo included | Under 5 seconds |

All targets are fixed before anyone looks at day-8 results. Every table also reports two naive baselines: glucose stays flat at its pre-meal value, and the population average curve for the meal's carb range.

## Partner roles

AWS is the backbone; every other partner owns one bounded piece, so nothing is built twice across two clouds.

| Partner | What it owns | What we use | Team owner | Priority |
| --- | --- | --- | --- | --- |
| Amazon Web Services | Cloud, training, serving, meal-photo estimator #1 | S3 data lake; SageMaker AI notebooks and training jobs; a Bedrock vision model (Claude or Amazon Nova) for photo-to-nutrition; Lambda and API Gateway for /predict; Amplify to host both UIs; credits and AWS mentors through the [Rutgers–RWJBarnabas hub](https://press.aboutamazon.com/aws/2026/9/rutgers-and-rwjbarnabas-health-to-launch-innovation-hub-powered-by-amazon-web-services) | B (backend), C (nutrition) | Must |
| Microsoft | Meal-photo estimator #2, an independent second opinion | An Azure OpenAI vision model in [Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry-classic/foundry-models/concepts/models-sold-directly-by-azure); where it disagrees with Bedrock, the uncertainty band widens. Stretch: Azure Health Data Services FHIR export of predictions | C | Must |
| [ElevenLabs](https://elevenlabs.io/docs/overview) | Voice in, voice out | Scribe v2 speech-to-text turns "two tacos and a soda, then a 20-minute walk" into meal and activity events; text-to-speech or an ElevenAgents voice agent reads the prediction and answers what-if questions. Helps users with diabetic eye disease | UI-1 | Should |
| [Anakin](https://anakin.io/) | Nutrition grounded in real menus | When a user names a restaurant dish or brand, the Search API and URL Scraper pull the official nutrition page as JSON, which anchors the photo estimate. Free tier is 300 credits, so results are cached | C | Should |
| Vlad Health | Medication and device reference data | Assumed to be [Vlada Health](https://www.vladahealth.com/) (public healthcare data through one API or MCP endpoint): map free-text medications to drug class; pull CGM coverage and price context for the pitch | B, D | Stretch |
| Cigna | Payer pathway and outcome framing | Mentor review of the member use case, the outcomes a payer pays for (time in range, HbA1c), a care-manager cohort view, and a pilot design | D (pitch) | Mentor |
| [Sevaro](https://www.cbinsights.com/company/sevaro) | Clinician workflow and escalation | Sevaro runs virtual neurology care on its Synapse AI platform; its mentors review the clinician dashboard, alert thresholds and escalation path. Tie-in: hypoglycemia is a classic stroke mimic | UI-2 | Mentor |

We found no public company named Vlad Health; Vlada Health is the closest match. Confirm it in the #vlad-health channel, along with the credits, API keys and mentors each partner offers.

## Data

The two datasets cover opposite halves of the problem: MetaboNet brings scale, insulin and exercise for type 1 diabetes; CGMacros brings meal photos with nutrition ground truth for type 2, prediabetes and no diabetes. We train population priors on both and personalize on each person's own 7 days.

|  | [MetaboNet](https://rutgersconnect-my.sharepoint.com/:u:/g/personal/adv32_rwjms_rutgers_edu/IQA5DqU5MfzSSYNdwDrp-oLkAe-BYD0P2iil0ea5kWCkxUQ?e=Br1Eq8) | [CGMacros](https://rutgersconnect-my.sharepoint.com/:u:/g/personal/adv32_rwjms_rutgers_edu/IQCiN5qixNmsT4Jq8jPgXHb7AVIXMKoJCBrjclHhf4xrhBs?e=LfLaUw) |
| --- | --- | --- |
| People | 3,135 with type 1 diabetes, pooled from 21 studies (13 public) | 45 adults: 15 without diabetes, 16 prediabetes, 14 type 2 diabetes |
| Days per person | Varies widely; 1,228 patient-years with both CGM and insulin | About 10 consecutive days |
| CGM | Harmonized 5-minute grid | Dexcom G6 Pro (5-min) and FreeStyle Libre Pro (15-min), stored in 1-minute rows |
| Meals | Self-reported carbs only; sparse, and zero vs missing is ambiguous | Photo before and after each meal, plus calories, carbs, protein, fat, fiber and meal type |
| Activity and heart rate | Exercise entries, workout label and heart rate where the source study had them | Fitbit Sense heart rate and activity |
| Medication | Basal and bolus insulin | None; bio.csv has HbA1c, fasting glucose and insulin, lipids |
| Format | One parquet file, one row per person per timestamp | One CSV per person plus a photo folder; bio.csv, gut microbiome files |
| License | Public subset after login (JAEB data use agreement or CC BY 4.0, by source) | CC BY-NC-SA 4.0 (non-commercial) |
| What we use it for | Insulin and exercise effect priors; the activity task; scale for the population model | Photo-to-nutrition calibration; the meal task; type 2 and prediabetes personalization |

How we build day-8 samples:

1. **Canonicalize.** Convert every source to the canonical schema (see Functional requirements) on a 5-minute grid in mg/dL. Use Dexcom for CGMacros; fall back to Libre where Dexcom is missing.
2. **Filter.** Keep people with at least 8 days of at least 70% CGM coverage. For MetaboNet, start with a sample of about 200 people that have both carb and exercise entries, so iteration stays fast.
3. **Roll windows.** Days d−7 to d−1 are history and day d is the target. CGMacros yields up to 3 target days per person; MetaboNet yields many, capped per person so heavy users do not dominate.
4. **Cut events.** Meal windows run from 60 min before to 180 min after; activity windows from 30 min before to 180 min after. Flag any window with another meal or activity inside 2 hours.
5. **Split.** The last eligible day per person is the test set and is never used for tuning. Model choices use leave-one-day-out inside days 1–7, plus a leave-people-out split to test the population priors.
6. **Plug in official data.** If organizers release the synthetic Challenge 3 set, it enters through the same loader and the same split.

MetaboNet column names come from its [data dictionary](https://metabo-net.org/data-dictionary); confirm them against the parquet schema on first load.

## Functional requirements

Everything a user gives us, in any format, lands in one canonical schema, and one API serves both apps.

| ID | Requirement | Partner | Owner | Priority |
| --- | --- | --- | --- | --- |
| FR-1 | **Canonical schema.** Every input becomes rows in `series` (person, time, glucose, heart rate, steps or METs, insulin) on a 5-minute grid, or in `events` (person, time, type = meal, activity or medication, details, source, confidence) | — | B | Must |
| FR-2 | **CGM adapters.** MetaboNet parquet, CGMacros CSV, Dexcom Clarity and LibreView CSV exports; resample 1- and 15-minute data; convert mmol/L to mg/dL | AWS | B | Must |
| FR-3 | **Wearable adapters.** Fitbit export first, then Apple Health XML and generic heart-rate CSV; derive resting heart rate, heart rate above resting, steps and activity bouts | AWS | B | Must |
| FR-4 | **Meal from photo.** Two vision estimators each return carbs, protein, fat, fiber and kcal; few-shot with the 3 most similar training meals (CLIP); calibrate each nutrient on training meals; their spread becomes input uncertainty | AWS, Microsoft | C | Must |
| FR-5 | **Meal from text or voice.** Speech-to-text, LLM parse into items and portions, then nutrition from the official menu or brand page, else USDA FoodData Central | ElevenLabs, Anakin | C | Should |
| FR-6 | **Activity from log, voice or wearable.** Type, duration and intensity; detect unlogged bouts from heart rate and METs | ElevenLabs | B | Must |
| FR-7 | **Medication normalization.** Free-text names to drug class (metformin, sulfonylurea, GLP-1 agonist, SGLT2 inhibitor, rapid or long insulin), with dose and time | Vlad Health | B | Stretch |
| FR-8 | **Meal prediction.** POST /predict returns the 0–120 min curve every 15 min with an 80% band, plus all summary metrics | AWS | Yifei, B | Must |
| FR-9 | **Activity prediction.** Same endpoint returns the 0–180 min curve, lowest value, time to low and probability of going below 70 mg/dL | AWS | Yifei | Should |
| FR-10 | **What-if.** Re-run with a different time of day, a post-meal walk or a smaller portion | — | D | Should |
| FR-11 | **Drivers.** Per-prediction contributions (model terms and SHAP) and person parameters vs population | — | D | Must |
| FR-12 | **Confidence flags.** Carbs outside the person's seen range, high photo disagreement, a time of day never seen, a medication change, fewer than 5 clean meal windows | Microsoft | Yifei | Must |
| FR-13 | **Patient app.** Photo or voice in; curve, band, one-line explanation, what-if, spoken readout | ElevenLabs, AWS | UI-1 | Must |
| FR-14 | **Clinician dashboard.** 7-day overview, per-meal fits, day-8 predicted vs observed, person vs population, drivers, flags | Sevaro (review) | UI-2 | Must |
| FR-15 | **FHIR export.** Predictions written as FHIR Observations for EHR use | Microsoft | B | Stretch |
| FR-16 | **Care-manager cohort view.** Members ranked by worsening response | Cigna (review) | UI-2 | Stretch |

API contract, extended from the team workflow so meals and activities share one endpoint. Values are examples; the UI builds against this mock tonight.

```json
POST /predict
{
  "participant_id": "P03",
  "event": {
    "type": "meal",
    "time": "2026-10-11T08:05:00",
    "inputs": [
      {"kind": "photo", "uri": "s3://bucket/P03/day8_breakfast.jpg"},
      {"kind": "voice", "uri": "s3://bucket/P03/day8_note.m4a"}
    ]
  },
  "history_days": 7
}

Response
{
  "participant_id": "P03",
  "meal_estimate": {
    "carbs": 62, "protein": 25, "fat": 18, "fiber": 6, "kcal": 520,
    "by_source": {"bedrock_carbs": 58, "azure_carbs": 66, "anakin_carbs": null},
    "uncertainty": 0.18
  },
  "prediction": {"t": [0, 15, 30, 45, 60, 75, 90, 105, 120],
                 "mean": [112, 128, 151, 166, 170, 162, 148, 135, 124],
                 "lo": [], "hi": []},
  "summary": {"g30": 151, "g60": 170, "g90": 148, "g120": 124, "peak": 172,
              "delta_peak": 60, "time_to_peak": 55, "iauc_2h": 3900},
  "activity_summary": null,
  "person_params": {"carb_sensitivity": 1.9, "pop_mean": 1.2, "tau_min": 52},
  "drivers": [{"feature": "carbs", "contribution": 31},
              {"feature": "morning", "contribution": 9}],
  "confidence_flags": ["carbs above training range"]
}
```

## Model and evaluation

The main model is an interpretable physiological response function per person, fit with hierarchical Bayesian shrinkage from population priors, so the same fit yields the prediction, its uncertainty and the explanation.

The glucose change after an event at time 0:

```latex
\Delta G_i(t) = A_i \, C_{\mathrm{eff}} \, \frac{t}{\tau_i} \, e^{1 - t/\tau_i} \; - \; S_i \, U \, k_{\mathrm{ins}}(t) \; - \; E_i \, W \, k_{\mathrm{act}}(t) \; + \; s_0 \, t \, e^{-t/\lambda}
```

```latex
C_{\mathrm{eff}} = \mathrm{carbs} - \alpha \cdot \mathrm{fiber} \qquad A_i^{\mathrm{event}} = A_i \exp\left(\beta_{\mathrm{morning}} + \beta_{\mathrm{act}} \, a + \beta_{\mathrm{med}} \, m\right)
```

- A is carb sensitivity (mg/dL per gram) and τ is time to peak, which grows with fat and protein.
- S is insulin sensitivity and U the bolus units (type 1 only); E is activity sensitivity and W the activity dose (intensity × minutes).
- The last term carries the pre-meal slope forward and lets it fade. a is recent activity, m is medication in effect.
- Events closer than 2 hours apart add together, so the model is fit to the whole 7-day CGM trace, not isolated windows.

Model tiers, in build order:

1. **Baselines (Saturday morning).** Flat pre-meal glucose; population mean curve by carb range; population LightGBM, one model per target.
2. **Main model.** The response function above in NumPyro or PyMC: population priors from MetaboNet and CGMacros, person parameters from days 1–7.
3. **Residual booster.** A small LightGBM on the main model's errors, kept only if leave-one-day-out error drops.
4. **Fallback.** If the Bayesian fit has not converged by Saturday afternoon: per-person ridge least squares shrunk to the population mean, with bootstrap intervals.

LightGBM features: carbs, protein, fat, fiber, kcal; pre-meal glucose, its 30-minute slope and 2-hour mean; heart rate above resting and steps or METs over the prior 2 and 24 hours; time of day; minutes since the last meal and last medication; medication class and dose; insulin on board (type 1); the person's 7-day mean glucose, variability, time in range and fasting level; cohort.

The activity task uses the same function with the meal term off, trained on MetaboNet exercise events. Inputs: activity type, duration, heart rate above resting, insulin on board, pre-activity glucose and slope, time of day.

Uncertainty has two sources, and both reach the band: the posterior over person parameters, and the spread between the Bedrock and Azure nutrition estimates, sampled through the model.

Evaluation, on the held-out last day per person:

- Every metric in the Goals table, for the baselines and every model tier.
- Split by cohort: type 1, type 2, prediabetes, no diabetes.
- Run twice, with ground-truth macros and with photo-estimated macros, to separate vision error from physiology error.
- Personalization evidence: personal vs population error per cohort, the share of response variance explained by the person (intraclass correlation), and a "same meal, two people" chart.

## Architecture

&#91;embedded content: System architecture · 6 layers, 7 partners\]

Read top to bottom: partner-owned adapters turn any input into one S3 store, the person fit on SageMaker serves one endpoint, and every app calls that endpoint.

## Privacy, safety and compliance

The hackathon build uses research data only and gives no treatment advice; a real pilot would need business associate agreements and a regulatory review first.

- **Licenses.** CGMacros is CC BY-NC-SA 4.0, so it cannot train a commercial product; a Cigna pilot needs other training data or permission. MetaboNet's public subset follows each source's data use agreement. Raw data stays in a private S3 bucket and never enters the git repo.
- **No patient data this weekend.** Demo photos and voice notes come from the datasets or from team members.
- **Before real users.** Sign BAAs with AWS and Microsoft, and confirm whether ElevenLabs and Anakin can sign one before any real voice notes or meal data reach them.
- **Photos.** Strip EXIF location data before an image leaves our backend.
- **Safety.** The app never suggests insulin or medication doses. A predicted low below 70 mg/dL says "check your glucose and follow your care plan", and the clinician view shows the escalation path Sevaro's mentors review.
- **Regulatory.** We frame the product as wellness insight for patients and decision support for clinicians; anything that drives dosing would be software as a medical device and is out of scope.
- **Fairness.** Report error by cohort and, where the data has it, by age, sex and ethnicity.

## Delivery plan

&#91;embedded content: Delivery plan · 6 workstreams, 3 gates\]

If the main model slips, the baseline-backed API still carries the demo. Phases follow the team workflow; the event itself runs until Sunday 6 pm.

## Risks and open questions

The biggest risk is the main model not converging in time; the LightGBM baseline and the API must work by Saturday noon so there is always something to demo.

| Risk | Mitigation | Owner |
| --- | --- | --- |
| Bayesian fit not converging by Saturday afternoon | Switch to ridge fallback with bootstrap intervals; baseline already live | Yifei |
| Photo estimates too noisy | Two-model ensemble, calibration on training meals, wider band and a flag when estimators disagree | C |
| Two clouds slow the team down | AWS runs everything; Azure is a single API call; FHIR export drops first | B |
| MetaboNet carbs: zero vs missing is ambiguous | Count a meal only when carbs are above zero; drop windows with an unexplained rise | B |
| MetaboNet is too large to iterate on | Start with about 200 people; scale up only for the final population priors | B |
| Official synthetic data arrives late or in another format | Adapters write the canonical schema; develop on CGMacros and MetaboNet meanwhile | B |
| Overfitting to day 8 | Freeze models and targets before scoring the test day; never tune on it | D |
| Partner credits or keys not ready | Fall back to open models (open\_clip, local LightGBM) and browser TTS for the demo | C, UI-1 |

Open questions:

- [ ] Is #vlad-health Vlada Health, and what credits, API keys and mentors does each partner channel offer?
- [ ] Are MetaboNet and CGMacros now the official data, or will synthetic Challenge 3 data still arrive? When and in what format?
- [ ] How is day 8 scored: MAE per time point, iAUC, or a judged demo?
- [ ] Is the activity prediction scored, or only meals?
- [ ] May we call external LLM and web APIs during judging?
- [ ] Which MetaboNet columns hold exercise and heart rate? Check the data dictionary on first load.

## Sources

- Challenge 3 brief, "Predict My Glucose Response" (Wellness\_Reimagined\_Challenge3\_CGM\_Monitoring.docx) and team plan (Glucose\_Response\_Workflow.pdf, 2026-10-09)
- [MetaboNet paper (arXiv 2601.11505)](https://arxiv.org/html/2601.11505v1); Wolff MK et al., J Diabetes Sci Technol 2026, doi:10.1177/19322968261441637; [MetaboNet data dictionary](https://metabo-net.org/data-dictionary)
- [CGMacros on PhysioNet](https://physionet.org/content/cgmacros/); Das A et al., [Sci Data 12, 1557 (2025)](https://doi.org/10.1038/s41597-025-05851-7)
- [Rutgers Health Hackathon 2026 event page](https://www.rutgers.edu/index%2ephp/event/rutgers-health-hackathon)
- [Rutgers and RWJBarnabas Health innovation hub powered by AWS](https://press.aboutamazon.com/aws/2026/9/rutgers-and-rwjbarnabas-health-to-launch-innovation-hub-powered-by-amazon-web-services)
- [Amazon Bedrock supported models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [Microsoft Foundry models sold directly by Azure](https://learn.microsoft.com/en-us/azure/foundry-classic/foundry-models/concepts/models-sold-directly-by-azure)
- [ElevenLabs documentation overview](https://elevenlabs.io/docs/overview)
- [Anakin.io](https://anakin.io/)
- [Vlada Health](https://www.vladahealth.com/)
- [Sevaro company profile (CB Insights)](https://www.cbinsights.com/company/sevaro)
