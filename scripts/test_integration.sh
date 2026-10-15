#!/usr/bin/env bash
# Integration tests against a running service (local container or the deployed API).
#   PMRF_API_KEY=... ./scripts/test_integration.sh <url>
# 0. reset the test users (needs AWS credentials, profile nutrix)
# 1. the model team's end-to-end flow (13 checks, CGMacros participant 001)
# 2. unseen hospital meal photos (tests/integration/hospital_meals/)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="${1:?usage: $0 <url>}"
DATA="${CGMACROS_DIR:-$ROOT/data/CGMacros}"
OUT="$ROOT/tests/integration/out"
mkdir -p "$OUT"
: "${PMRF_API_KEY:?set PMRF_API_KEY to the x-api-key of the service}"

# Test users start from scratch (person data persists in DynamoDB + S3 between runs)
python -I "$ROOT/ops/data/reset_person.py" flowtest-user demo-hospital-001
python -I "$ROOT/model/tests/test_api_flow.py" "$URL" "$DATA"
python -I "$ROOT/tests/integration/demo_hospital_meals.py" "$URL" "$DATA" \
  "$ROOT/tests/integration/hospital_meals" "$OUT/hospital_meals.json"
