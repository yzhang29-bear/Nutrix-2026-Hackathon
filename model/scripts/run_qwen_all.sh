#!/bin/bash
# All Qwen3-VL calls for the CGMacros photos (resumable). usage: bash scripts/run_qwen_all.sh zeroshot|fewshot
P=".venv/bin/python -I"; A="results/final/stage1.pkl eda/cache/photo_mobileclip.npz cgmacros/CGMacros results/qwen3vl"
if [ "$1" = "zeroshot" ]; then
  $P scripts/qwen_bedrock.py $A zeroshot
else
  for c in 7 5 4 6; do $P scripts/qwen_bedrock.py $A fewshot $c; done
fi
echo "done $1"
