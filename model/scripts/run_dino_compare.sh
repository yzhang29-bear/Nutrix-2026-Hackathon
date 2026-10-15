#!/bin/bash
# DINOv2 photo encoders (S/B/L, CPU) + comparison against MobileCLIP-S2 with the final stage-1 configuration.
# usage (from the project root): bash scripts/run_dino_compare.sh
set -e
P=".venv/bin/python -I"
for v in s:vit_small_patch14_reg4_dinov2.lvd142m b:vit_base_patch14_reg4_dinov2.lvd142m l:vit_large_patch14_reg4_dinov2.lvd142m; do
  k=${v%%:*}; m=${v#*:}
  [ -f eda/cache/photo_dinov2_$k.npz ] || $P scripts/photo_embed_timm.py cgmacros/CGMacros eda/cache/photo_dinov2_$k.npz $m > eda/cache/photo_dinov2_$k.log 2>&1
done
$P scripts/exp_encoder_dino.py cgmacros/CGMacros data_ext/bigideas results/final results/encoder_dino \
  "MobileCLIP-S2=eda/cache/photo_mobileclip.npz" "DINOv2-S=eda/cache/photo_dinov2_s.npz" "DINOv2-B=eda/cache/photo_dinov2_b.npz" \
  "DINOv2-L=eda/cache/photo_dinov2_l.npz" "MobileCLIP-S2 + DINOv2-B=eda/cache/photo_mobileclip.npz+eda/cache/photo_dinov2_b.npz" \
  > results/encoder_dino.log 2>&1
echo done >> results/encoder_dino.log
