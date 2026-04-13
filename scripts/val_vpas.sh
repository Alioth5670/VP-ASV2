#!/usr/bin/env bash

set -euo pipefail

# Edit these values directly, then run: bash scripts/val_vpas.sh
CONFIG="outputs/vpas_dinov3_vitb16_soft_norm_product_noite/resolved_config.yaml"
CKPT="outputs/vpas_dinov3_vitb16_soft_norm_product_noite/last.pth"
SPLIT="test"
DEVICE="cuda:3"
SAVE_JSON="outputs/vpas_dinov3_vitb16_soft_norm_product_noite/test_outputs_random/metrics.json"

# Leave empty to use data.{split}_json from config.
JSON_PATHS=(
  "dataset/MVTec-AD/oneprompt_seed1-test.json"
  "dataset/MVTec-AD/oneprompt_seed3-test.json"
  "dataset/MVTec-AD/oneprompt_seed5-test.json"
  "dataset/MVTec-AD/oneprompt_seed7-test.json"
  "dataset/MVTec-AD/oneprompt_seed9-test.json"
)

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

CMD=(
  python val.py
  --config "$CONFIG"
  --ckpt "$CKPT"
  --split "$SPLIT"
  --device "$DEVICE"
  --save_json "$SAVE_JSON"
)

if (( ${#JSON_PATHS[@]} > 0 )); then
  CMD+=(--json_path "${JSON_PATHS[@]}")
fi

"${CMD[@]}"
