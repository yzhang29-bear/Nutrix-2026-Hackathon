#!/usr/bin/env bash
# Sync datasets between local data/ and S3.
#   ./ops/data/sync_data.sh pull   # S3 -> data/
#   ./ops/data/sync_data.sh push   # data/ -> S3
# Env: AWS_PROFILE (default nutrix)
set -euo pipefail

PROFILE="${AWS_PROFILE:-nutrix}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$ROOT/data"
META_DIR="cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0"

BUCKET="s3://nutrix-data"

aws_() { aws --profile "$PROFILE" --only-show-errors "$@"; }

case "${1:-pull}" in
  pull)
    mkdir -p "$DATA"
    aws_ s3 sync "$BUCKET/cgmacros/raw/CGMacros/" "$DATA/CGMacros/"
    aws_ s3 sync "$BUCKET/cgmacros/raw/physionet-1.0.0/" "$DATA/$META_DIR/"
    aws_ s3 cp "$BUCKET/t1d/raw/Type1Diabetes.parquet" "$DATA/Type1Diabetes.parquet"
    ;;
  push)
    aws_ s3 sync "$DATA/CGMacros/" "$BUCKET/cgmacros/raw/CGMacros/"
    aws_ s3 sync "$DATA/$META_DIR/" "$BUCKET/cgmacros/raw/physionet-1.0.0/"
    aws_ s3 cp "$DATA/Type1Diabetes.parquet" "$BUCKET/t1d/raw/Type1Diabetes.parquet"
    ;;
  *)
    echo "usage: $0 [pull|push]" >&2; exit 1 ;;
esac
