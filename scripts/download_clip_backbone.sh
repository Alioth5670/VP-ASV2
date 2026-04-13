#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${1:-CLIP_ViT-L/14}"
OUTPUT_PATH="${2:-}"

cd "$ROOT_DIR"

if [[ -n "$OUTPUT_PATH" ]]; then
  python -m models.backbone.clip.download "$MODEL_NAME" --output "$OUTPUT_PATH"
else
  python -m models.backbone.clip.download "$MODEL_NAME"
fi
