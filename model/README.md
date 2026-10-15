# PMRF: personalized multimodal post-meal glucose model (Wellness Reimagined, Challenge #3)

Predicts the 0–120 min glucose response to a new meal from its photo, the CGM and heart rate before the meal, medication,
and the person's first 7 days. Technical report with an interactive prediction inspector: `site/index.html`
(published as a claude.ai artifact). Data exploration report: `eda/cgmacros_eda.html`.

In the team repo this folder is `model/`. Data, results and model binaries are not in git: the trained model version is in
`s3://nutrix-models/models/v4-20261010/` (`python -I scripts/fetch_model_store.py`); the data sets are listed below with
their sources. Setup: `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu`.

## Layout

| Path | What |
|---|---|
| `pmrf/data.py` | `Person` record + adapters: `load_cgmacros` (dev data), `load_challenge` (official data, layout documented in the docstring) |
| `pmrf/features.py` | Labels (G30–G120, peak, Δpeak, time to peak, iAUC) and pre-meal features, personal profile |
| `pmrf/vision.py` | MobileCLIP photo encoder, zero-shot food tags, nutrition estimator (ridge + population/personal retrieval + stacker) |
| `pmrf/models/gbm.py` | LightGBM response model: bagged seeds, quantiles + conformal band, personal offset, TreeSHAP |
| `pmrf/models/icl_transformer.py` | In-context personalization transformer (pretrain on T1D, fine-tune on CGMacros) |
| `pmrf/pipeline.py` | Day-8 protocol glue, feature sets, metrics |
| `pmrf/predict.py` | Single-meal `Predictor` and `build_response` = the API JSON (see `docs/api_contract.md`) |
| `pmrf/store.py`, `pmrf/state.py` | Model store in stable formats (export/load); per-person data store and first-week state builder |
| `service/` | FastAPI service (`app.py`) and photo/model storage (`storage.py`); deployment: `DEPLOY.md`, `deploy/Dockerfile` |
| `deploy/model_store/<version>/` | Exported model version loaded by the service (no pickles; checksums in `manifest.json`) |
| `scripts/` | Reproducible runs (below) |
| `results/` | Metrics, predictions, `demo/bundle.pkl` (all trained components for the API) |

## Data

| Dataset | Folder | Use |
|---|---|---|
| CGMacros (PhysioNet) | `cgmacros/CGMacros` | development and test set |
| BIG IDEAs Lab (PhysioNet 1.1.3; Dexcom, HR, food logs only; participant 003 skipped, malformed food log) | `data_ext/bigideas` | extra training people |
| Harmonized T1D collection | `Type1Diabetes.parquet` | transformer pretraining, medication check |
| ShanghaiT1DM/T2DM (figshare 10.6084/m9.figshare.20444397) | `data_ext/shanghai` | HbA1c and complication models |
| Colas et al. 2019, S1 File (PLoS One) | `data_ext/colas` | future T2D risk |

## Reproduce (CPU only, venv `.venv`)

```bash
P=".venv/bin/python -I"
$P scripts/photo_embed_clip.py cgmacros/CGMacros eda/cache/photo_mobileclip.npz MobileCLIP-S2 datacompdr
$P scripts/t1d_events.py Type1Diabetes.parquet eda/cache/t1d_events.npz
$P scripts/run_pmrf.py cgmacros/CGMacros results/v1_mobileclip eda/cache/photo_mobileclip.npz        # baselines reused below
$P scripts/run_chronos.py cgmacros/CGMacros results/v1_mobileclip results/chronos
$P scripts/run_transformer.py eda/cache/t1d_events.npz results/v1_mobileclip results/tf_v1 30000 4000   # T1D pretraining
$P scripts/final_gbm.py cgmacros/CGMacros eda/cache/photo_mobileclip.npz results/final data_ext/bigideas   # stage 1 (v4)
$P scripts/final_blend_v4.py results/final results/tf_v1/pretrained_t1d.pt results/chronos results/final_blend 1500 1e-4 50
$P scripts/final_ablations.py cgmacros/CGMacros results/final results/final data_ext/bigideas
$P scripts/run_t1d_meds.py eda/cache/t1d_events.npz results/t1d_meds
$P scripts/run_risk.py results/risk 2000                                                                  # metabolic profile & risk
$P scripts/nutrient_effects.py results/final results/guide                                               # diet-guide effects + check
$P scripts/export_demo.py cgmacros/CGMacros results/final results/final_blend results/demo
$P scripts/eval_guide.py results/final results/final_blend results/demo results/guide_eval.json
$P scripts/bench_encoders.py <photo.jpg> results/bench_encoders.json
$P scripts/make_site_figs.py results results/site_fig
$P scripts/build_site.py results site
```

Deployment package (after the steps above):

```bash
$P scripts/export_model_store.py results deploy/model_store v4-20261010        # stable-format model version
docker build -f deploy/Dockerfile -t pmrf-api:v4-20261010 .
$P scripts/import_cohort.py cgmacros/CGMacros results/demo/bundle.pkl <data dir> # golden test data (development cohort)
$P scripts/golden_test.py <url> imported results/demo/demo_data.json out.json   # 176 report predictions, identical
$P scripts/golden_test.py <url> onboard cgmacros/CGMacros results/final/stage1.pkl results/final_blend/final_preds.pkl out.json
PMRF_API_KEY=... $P tests/test_api_flow.py <url> cgmacros/CGMacros; $P tests/test_s3_storage.py deploy/model_store/v4-20261010
$P scripts/load_test.py <url> 1 120 out.json
```
Encoder check: `bash scripts/run_dino_compare.sh` (DINOv2 S/B/L vs MobileCLIP-S2, results in `results/encoder_dino/`).

Model v7 (current deployment; same accuracy as v4, more optional inputs for the app):

```bash
LOGGED_KCAL=1 bash scripts/run_v6.sh v6_demo                                     # model A with activity, age, sex, BMI, logged kcal
.venv/bin/python -I scripts/pretrain_v2.py eda/cache/t1d_events.npz eda/cache/t1d_person_info.csv results/tf_v2   # PMRF_DEVICE=cuda, .venv-gpu
bash scripts/run_v7.sh results/final_v6_demo v7_demo                              # transformer with age, sex, BMI (GPU) + blend
$P scripts/export_demo.py cgmacros/CGMacros results/final_v6_demo results/final_blend_v7_demo results/demo_v7
STORE_NOTES="v7 ..." $P scripts/export_model_store.py results deploy/model_store v7-20261010 demo_v7
```

All model choices use inner validation (days 1–5 → 6–7, rolling cut-offs in `scripts/exp_accuracy*.py`); the day-8+ test set is scored once.
Version history: v3 results are kept in `results/final_v3`, `results/final_blend_v3`, `results/demo_v3`.
