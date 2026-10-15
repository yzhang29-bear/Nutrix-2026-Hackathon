#!/bin/bash
# Copy the code (allow-list only: no data, results, model binaries or credentials) into the team repo's model/ folder.
# usage: bash scripts/sync_to_nutrix.sh <Nutrix clone>
set -e
DST="$1/model"
mkdir -p "$DST"
rsync -a --delete --prune-empty-dirs \
  --exclude '__pycache__/' --exclude '*.pyc' \
  --include 'pmrf/***' --include 'service/***' --include 'scripts/***' --include 'tests/***' \
  --include 'docs/' --include 'docs/api_contract.md' \
  --include 'deploy/' --include 'deploy/Dockerfile' --include 'deploy/requirements.txt' \
  --include 'site/' --include 'site/template.html' \
  --include 'README.md' --include 'DEPLOY.md' --include '.dockerignore' --include 'requirements-dev.txt' \
  --exclude '*' ./ "$DST/"
cp deploy/gitignore.model "$DST/.gitignore"
echo "synced to $DST"
# UI showcase package (12 meals, photos + API JSON; CGMacros CC BY-NC-SA, attribution in its README) - approved for the public repo 2026-10-10
if [ -d results/best_subjects/ui_package ]; then
  rsync -a --delete results/best_subjects/ui_package/ "$DST/ui_showcase/"
  cp docs/ui_showcase_README.md "$DST/ui_showcase/README.md"
  echo "ui_showcase synced"
fi
