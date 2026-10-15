#!/bin/bash
# v6 final build: stage 1 (model A with the new inputs) then stage 2 (transformer fine-tune + blend), evaluated once on day 8+.
# usage: LOGGED_KCAL=0|1 bash scripts/run_v6.sh <tag>      (tag: v6 = challenge-compliant, v6_demo = with logged meal kcal)
set -e
P=".venv/bin/python -I"
$P scripts/final_gbm_v6.py cgmacros/CGMacros eda/cache/photo_mobileclip.npz results/final_$1 data_ext/bigideas
$P scripts/final_blend_v4.py results/final_$1 results/tf_v1/pretrained_t1d.pt results/chronos results/final_blend_$1 1500 1e-4 50
echo "done $1"
