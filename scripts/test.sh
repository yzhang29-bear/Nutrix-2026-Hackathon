#!/usr/bin/env bash
# Static checks, the same as CI: errors only (syntax, undefined names) on the service and the scripts deploys use.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ruff check --select E9,F63,F7,F82 model/pmrf model/service model/tests scripts ops tests server \
  model/scripts/{fetch_model_store,golden_test,load_test,import_cohort,check_store}.py
python -m compileall -q model/pmrf model/service ops tests server
echo "static checks passed"
