#!/usr/bin/env bash
# Build the PMRF service image (linux/amd64, as Fargate runs it). Fetch the model first: ./scripts/fetch_model.sh
#   ./scripts/build.sh [version]     # tags pmrf-api:<version> and pmrf-api:<git sha>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-${MODEL_VERSION:-v7-20261010}}"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"

[ -f "$ROOT/model/deploy/model_store/$VERSION/manifest.json" ] || { echo "model $VERSION missing: run ./scripts/fetch_model.sh $VERSION" >&2; exit 1; }

docker build --platform linux/amd64 -f "$ROOT/model/deploy/Dockerfile" \
  --build-arg MODEL_VERSION="$VERSION" -t "pmrf-api:$VERSION" -t "pmrf-api:$SHA" "$ROOT/model"
