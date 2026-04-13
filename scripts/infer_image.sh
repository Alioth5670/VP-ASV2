#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/vpas/bin/python"  # Python解释器路径
CONFIG="$ROOT_DIR/outputs/vpas_dinov3_vitb16/resolved_config.yaml"   # 配置文件路径，可选: 其他yaml配置
CKPT="$ROOT_DIR/outputs/vpas_dinov3_vitb16/last.pth"               # 推理权重路径，可选: best.pth / last.pth / 其他权重
DEVICE="cuda:0"                                      # 推理设备，可选: cuda:0 / cuda:1 / cpu
OUTPUT_DIR="outputs/infer_images/vpas_dinov3_vitb162"  # 输出目录，可选: 留空或任意输出路径
INPUT_JSON=""                                        # 成对json路径，可选: prompt_path/query_path格式的json
PROMPT_IMAGE="../V1/outputs/infer_images/infer_image_vpas_dinov3_vitb16_norm_product2/prompt.png" # Prompt图像路径，可选: 任意单张图片路径
QUERY_IMAGE="../V1/outputs/infer_images/infer_image_vpas_dinov3_vitb16_norm_product2/query.png"   # Query图像路径，可选: 任意单张图片路径
QUERY_DIR=""                                         # Query文件夹路径，可选: 任意图片文件夹路径

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

if [[ -n "$INPUT_JSON" ]]; then
  if [[ ! -f "$INPUT_JSON" ]]; then
    echo "Input JSON not found: $INPUT_JSON" >&2
    exit 1
  fi
  if [[ -n "$PROMPT_IMAGE" || -n "$QUERY_IMAGE" || -n "$QUERY_DIR" ]]; then
    echo "When INPUT_JSON is set, leave PROMPT_IMAGE, QUERY_IMAGE, and QUERY_DIR empty" >&2
    exit 1
  fi
else
  if [[ -z "$PROMPT_IMAGE" ]]; then
    echo "Please set PROMPT_IMAGE in scripts/infer_image.sh" >&2
    exit 1
  fi
  if [[ -z "$QUERY_IMAGE" && -z "$QUERY_DIR" ]]; then
    echo "Please set QUERY_IMAGE or QUERY_DIR in scripts/infer_image.sh" >&2
    exit 1
  fi
  if [[ -n "$QUERY_IMAGE" && -n "$QUERY_DIR" ]]; then
    echo "Set only one of QUERY_IMAGE or QUERY_DIR" >&2
    exit 1
  fi
fi

CMD=("$PYTHON_BIN" "$ROOT_DIR/inference.py" --mode image --config "$CONFIG" --ckpt "$CKPT" --device "$DEVICE")

if [[ -n "$INPUT_JSON" ]]; then
  CMD+=(--input-json "$INPUT_JSON")
else
  CMD+=(--prompt-image "$PROMPT_IMAGE")
  if [[ -n "$QUERY_IMAGE" ]]; then
    CMD+=(--query-image "$QUERY_IMAGE")
  fi
  if [[ -n "$QUERY_DIR" ]]; then
    CMD+=(--query-dir "$QUERY_DIR")
  fi
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output-dir "$OUTPUT_DIR")
fi

cd "$ROOT_DIR"
echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
