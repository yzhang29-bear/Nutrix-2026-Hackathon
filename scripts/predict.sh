#!/usr/bin/env bash
# Predict one patient's glucose response to a meal photo, against the deployed API, in one command.
#   ./scripts/predict.sh <patient> <photo> [<meal time YYYY-MM-DDTHH:MM:SS>] [--kcal <logged kcal>] [--log]
#   e.g. ./scripts/predict.sh cgmacros-003 day8-lunch
#        ./scripts/predict.sh cgmacros-003 korea_snuh_evening_meal 2020-03-18T12:30:00
# <photo> is one of:
#   - day8-<meal type> or day8-<n>: the patient's own day-8 meal (first of that type, or the n-th of the day) from
#     tests/integration/day8_meals.csv (written by ops/data/onboard_patients.py). The meal time comes from the list, and
#     the response includes the observed curve, so this is the one to use for predicted vs observed
#   - a test image name (already in S3 under photos/test/hospital-meals/): korea_snuh_evening_meal,
#     us_chicken_penne_greens, germany_cabbage_roll_mash, germany_greek_salad_pita, israel_hospital_meal
#   - an S3 key the service already has: photos/...
#   - a local .jpg file, uploaded first
# Prints a summary; the full JSON goes to tests/integration/out/predict_<patient>.json.
# --log records the meal in the patient's food log (log_meal=true); default is a preview that changes nothing.
# --kcal sends the calories the user logged for the meal (logged_kcal); without it the service uses the calories estimated
# from the photo. day8-* meals take them from the list.
# Env: API (default: the live API), PMRF_API_KEY (default: read from Secrets Manager pmrf/api-key)
set -euo pipefail
for d in "/c/Program Files/Amazon/AWSCLIV2"; do [ -d "$d" ] && PATH="$d:$PATH"; done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATIENT="${1:?usage: $0 <patient> <photo> <meal time> [--log]}"
PHOTO="${2:?photo missing: day8-<meal type>, a test image name, an S3 key photos/... or a local .jpg}"
shift 2
LOG=false; MEAL_TIME=""; KCAL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --log) LOG=true ;;
    --kcal) KCAL="${2:?--kcal needs a number}"; shift ;;
    *) MEAL_TIME="$1" ;;
  esac
  shift
done
API="${API:-https://8la1c1eazd.execute-api.us-west-2.amazonaws.com}"

if [ -z "${PMRF_API_KEY:-}" ]; then
  PMRF_API_KEY="$(aws secretsmanager get-secret-value --secret-id pmrf/api-key --profile "${AWS_PROFILE:-nutrix}" \
    --region us-west-2 --query SecretString --output text)" || { echo "could not read the API key (AWS credentials expired?)" >&2; exit 1; }
fi

# 1. photo key: a day-8 test meal, an existing S3 key, a local file to upload, or a test image name
if [[ "$PHOTO" == day8-* ]]; then
  read -r KEY LISTED_TIME LISTED_KCAL < <(python - "$ROOT/tests/integration/day8_meals.csv" "$PATIENT" "${PHOTO#day8-}" <<'EOF'
import csv, sys
path, patient, which = sys.argv[1:]
rows = [r for r in csv.DictReader(open(path, encoding="utf8")) if r["patient"] == patient]
pick = [r for r in rows if (r["n"] == which if which.isdigit() else r["meal_type"] == which)][:1]
if not pick:
    sys.exit(f"no day-8 meal '{which}' for {patient} in {path}; listed: " + ", ".join(f"{r['n']} {r['meal_type']} {r['meal_time']}" for r in rows))
print(pick[0]["photo_key"], pick[0]["meal_time"], pick[0].get("logged_kcal") or "-")
EOF
  )
  [ -n "${KEY:-}" ] || exit 1
  MEAL_TIME="${MEAL_TIME:-$(printf %s "$LISTED_TIME" | tr -d '[:cntrl:]')}"     # Windows Python ends lines with CRLF
  LISTED_KCAL="$(printf %s "${LISTED_KCAL:--}" | tr -d '[:cntrl:]')"
  [ "$LISTED_KCAL" = - ] || KCAL="${KCAL:-$LISTED_KCAL}"

elif [[ "$PHOTO" == photos/* ]]; then
  KEY="$PHOTO"
elif [ -f "$PHOTO" ]; then
  RESP="$(mktemp)"
  CODE="$(curl -s -o "$RESP" -w '%{http_code}' -H "x-api-key: $PMRF_API_KEY" -F "file=@$PHOTO;type=image/jpeg" \
    "$API/v1/people/$PATIENT/photos")"
  [ "$CODE" = 200 ] || { echo "photo upload failed ($CODE): $(cat "$RESP")" >&2; exit 1; }
  KEY="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['photo_key'])" "$RESP")"
else
  KEY="photos/test/hospital-meals/${PHOTO%.jpg}.jpg"
fi

[ -n "$MEAL_TIME" ] || { echo "meal time missing, e.g. 2020-03-18T12:30:00" >&2; exit 1; }

# 2. predict
mkdir -p "$ROOT/tests/integration/out"
JSON="$ROOT/tests/integration/out/predict_$PATIENT.json"
CODE="$(curl -s -o "$JSON" -w '%{http_code}' -H "x-api-key: $PMRF_API_KEY" -H "content-type: application/json" \
  -d "{\"photo_key\":\"$KEY\",\"meal_time\":\"$MEAL_TIME\",\"log_meal\":$LOG${KCAL:+,\"logged_kcal\":$KCAL}}" "$API/v1/people/$PATIENT/predict")"
[ "$CODE" = 200 ] || { echo "predict failed ($CODE): $(cat "$JSON")" >&2; exit 1; }

python - "$JSON" "$KEY" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf8"))
m, p, c = d["meal"], d["prediction"], d["confidence"]
n = m["estimated_nutrition"]
print(f"patient     {d['person_id']}   meal time {m['time']} ({m['meal_type']})")
print(f"model       {d.get('model_version')}")
print(f"photo       {sys.argv[2]}")
print("photo tags  " + ", ".join(f"{t['label']} {t['p']:.0%}" for t in m["photo_tags"]))
print(f"nutrition   carbs {n['carbs_g']:.0f} g (+/- {n['carbs_uncertainty_g']:.0f}), protein {n['protein_g']:.0f} g, fat {n['fat_g']:.0f} g, {n['kcal']:.0f} kcal")
print(f"glucose     before {d['pre_meal_state']['glucose_mgdl']:.0f} -> peak {p['peak_glucose_mgdl']:.0f} mg/dL "
      f"(+{p['peak_rise_mgdl']:.0f}) at {p['time_to_peak_min']:.0f} min")
print("at minutes  " + "  ".join(f"{k}: {v:.0f}" for k, v in p["glucose_at_min"].items()))
print(f"P(peak>140) {p['prob_peak_above_mgdl']['140']:.0%}   P(peak>180) {p['prob_peak_above_mgdl']['180']:.0%}")
print(f"confidence  {c['level']}" + (f" ({'; '.join(c['flags'])})" if c["flags"] else ""))
o = d.get("observed")
if o:
    print(f"observed    before {o['pre_meal_glucose_mgdl']:.0f} -> peak {o['peak_glucose_mgdl']:.0f} mg/dL (+{o['peak_rise_mgdl']:.0f}) "
          f"at {o['time_to_peak_min']:.0f} min, {o['share_of_curve_inside_band80']:.0%} of the curve inside the 80% band")
    print("at minutes  " + "  ".join(f"{k}: {v:.0f}" for k, v in o["glucose_at_min"].items()))
print(f"full JSON   {sys.argv[1]}")
EOF
