#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/vpas/bin/python"  # Python解释器路径
CONFIG="$ROOT_DIR/config/dinov3/vpas_dinov3_vitb16.yaml"   # 配置文件路径，可选: 其他yaml配置
CKPT="$ROOT_DIR/outputs/test/best.pth"               # 推理权重路径，可选: best.pth / last.pth / 其他权重
DEVICE="cuda:0"                                      # 推理设备，可选: cuda:0 / cuda:1 / cpu
OUTPUT_DIR=""                                        # 输出目录，可选: 留空或任意输出路径
VIDEO_PATH=""                                        # 视频路径，可选: 任意单个视频文件路径
PROMPT_IMAGE=""                                      # 外部Prompt图像路径，可选: 留空或任意单张图片路径
VIDEO_PROMPT_MODE="first_n_mean"                     # 视频Prompt模式，可选: first_frame / first_n_mean / all_mean
VIDEO_PROMPT_FRAMES=8                                 # Prompt帧数，仅first_n_mean生效，可选: 任意正整数
VIDEO_MAX_FRAMES=-1                                   # 最多处理帧数，可选: -1=全部, 其他正整数
VIDEO_STRIDE=1                                        # 视频抽帧步长，可选: 1 / 2 / 4 / 其他正整数
SAVE_MASK_VIDEO=0                                     # 是否保存二值mask视频，可选: 0=不保存, 1=保存
SAVE_HEATMAP_VIDEO=0                                  # 是否保存热力图视频，可选: 0=不保存, 1=保存
EXPORT_FRAME_EVERY=0                                  # 每隔多少帧导出图片，可选: 0=不导出, 其他正整数
EXPORT_FRAME_INDICES=()                               # 指定导出的帧索引，可选: () / (0 30 60)

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
if [[ -z "$VIDEO_PATH" ]]; then
  echo "Please set VIDEO_PATH in scripts/infer_video.sh" >&2
  exit 1
fi

CMD=("$PYTHON_BIN" "$ROOT_DIR/inference.py" --mode video --config "$CONFIG" --ckpt "$CKPT" --device "$DEVICE" --video-path "$VIDEO_PATH" --video-prompt-mode "$VIDEO_PROMPT_MODE" --video-prompt-frames "$VIDEO_PROMPT_FRAMES" --video-max-frames "$VIDEO_MAX_FRAMES" --video-stride "$VIDEO_STRIDE")

if [[ -n "$PROMPT_IMAGE" ]]; then
  CMD+=(--prompt-image "$PROMPT_IMAGE")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output-dir "$OUTPUT_DIR")
fi
if [[ "$SAVE_MASK_VIDEO" == "1" ]]; then
  CMD+=(--save-mask-video)
fi
if [[ "$SAVE_HEATMAP_VIDEO" == "1" ]]; then
  CMD+=(--save-heatmap-video)
fi
if [[ "$EXPORT_FRAME_EVERY" != "0" ]]; then
  CMD+=(--export-frame-every "$EXPORT_FRAME_EVERY")
fi
if [[ ${#EXPORT_FRAME_INDICES[@]} -gt 0 ]]; then
  CMD+=(--export-frame-indices "${EXPORT_FRAME_INDICES[@]}")
fi

cd "$ROOT_DIR"
echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
