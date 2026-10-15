#!/bin/bash
# v7 = transformer pretrained with v2 inputs (age, sex, BMI) on GPU, fine-tuned + blended on top of a stage-1 directory.
# usage: bash scripts/run_v7.sh <stage-1 dir> <out tag>
set -e
PMRF_NUM_COLS=v2 PMRF_DEVICE=cuda .venv-gpu/bin/python -I scripts/final_blend_v4.py $1 results/tf_v2/pretrained_t1d_v2.pt results/chronos results/final_blend_$2 1500 1e-4 50
echo "done $2"
