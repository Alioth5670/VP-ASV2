#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/vpas/bin/python"  # Python解释器路径
OUTPUTS_DIR="$ROOT_DIR/outputs"                      # 自动扫描目录，可选: 默认扫描outputs
KEYWORD="noite"                                      # 仅扫描包含该关键字的实验目录，可选: noite / soft / 其他
METRICS_SUBDIR="test_outputs_random"                         # 指定输出子目录，可选: test_outputs / test_outputs_best / test_outputs_random / 留空=全扫描
OUTPUT_PATH="$ROOT_DIR/outputs/yuque_metrics.html"  # 输出文件路径，可选: 留空则打印到终端
SPLIT_METRIC_TABLES=0                                 # 是否拆分成单独指标表格，可选: 0=否, 1=是
MEAN_STD=1                                            # 是否使用mean+-std格式，可选: 0=否, 1=是
METRICS_JSON=()                                       # 手动指定metrics.json路径，可选: () / (path1 path2)
NAMES=()                                              # 手动指定模型名，可选: () / (model1 model2)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$OUTPUTS_DIR" ]]; then
  echo "Outputs directory not found: $OUTPUTS_DIR" >&2
  exit 1
fi
if [[ ${#NAMES[@]} -gt 0 && ${#NAMES[@]} -ne ${#METRICS_JSON[@]} ]]; then
  echo "When NAMES is set, it must have the same length as METRICS_JSON" >&2
  exit 1
fi
for metrics_path in "${METRICS_JSON[@]}"; do
  if [[ ! -f "$metrics_path" ]]; then
    echo "Metrics file not found: $metrics_path" >&2
    exit 1
  fi
done

CMD=("$PYTHON_BIN" "$ROOT_DIR/utils/metrics_to_yuque.py")

if [[ ${#METRICS_JSON[@]} -gt 0 ]]; then
  CMD+=("${METRICS_JSON[@]}")
else
  CMD+=(--outputs-dir "$OUTPUTS_DIR" --keyword "$KEYWORD")
  if [[ -n "$METRICS_SUBDIR" ]]; then
    CMD+=(--metrics-subdir "$METRICS_SUBDIR")
  fi
fi

if [[ ${#NAMES[@]} -gt 0 ]]; then
  CMD+=(--names "${NAMES[@]}")
fi
if [[ "$SPLIT_METRIC_TABLES" == "1" ]]; then
  CMD+=(--split-metric-tables)
fi
if [[ "$MEAN_STD" == "1" ]]; then
  CMD+=(--mean-std)
fi
if [[ -n "$OUTPUT_PATH" ]]; then
  CMD+=(--output "$OUTPUT_PATH")
fi

cd "$ROOT_DIR"
echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
