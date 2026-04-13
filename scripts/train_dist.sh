#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/vpas/bin/python"    # Python解释器路径
CONFIG="$ROOT_DIR/config/dinov3/vpas_dinov3_vitb16.yaml"     # 配置文件路径，可选: 其他yaml配置
GPU_IDS="4,5,6,7"                                     # 使用的显卡编号，可选: "0" / "1" / "0,1" / "0,1,2,3"
BATCH_SIZE="32"                                        # 训练batch size(每卡)，多卡时总 batch = 卡数 × 每卡 batch
OUTPUT_DIR="$ROOT_DIR/outputs/vpas_dinov3_vitb16" # 输出目录，可选: 任意输出路径
NUM_WORKERS="8"                                       # DataLoader线程数，可选: 0 / 2 / 4 / 8
RESUME=""                                             # 断点恢复权重路径，可选: 留空或某个last.pth/best.pth
SEED="42"                                             # 随机种子，可选: 任意整数
DIST_BACKEND="nccl"                                   # 分布式后端，可选: nccl / gloo
NO_VAL=1                                              # 是否关闭验证，可选: 0=开启验证, 1=关闭验证
DISABLE_CUDNN=0                                       # 是否禁用cuDNN，可选: 0=不禁用, 1=禁用
NNODES=1                                              # 节点数，可选: 1(单机) / >1(多机)
NODE_RANK=0                                           # 当前节点编号，可选: 0 到 NNODES-1
MASTER_ADDR="127.0.0.1"                               # 主节点地址，可选: 127.0.0.1 或主节点IP
MASTER_PORT=29500                                     # 主节点端口，可选: 29500 / 29501 / 其他空闲端口

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi

if [[ -z "$GPU_IDS" ]]; then
  echo "GPU_IDS is empty" >&2
  exit 1
fi

NPROC_PER_NODE=$(GPU_IDS="$GPU_IDS" "$PYTHON_BIN" - <<'PY'
import os
ids = [x.strip() for x in os.environ["GPU_IDS"].split(",") if x.strip()]
print(len(ids))
PY
)

if [[ "$NPROC_PER_NODE" -lt 1 ]]; then
  echo "No valid GPU ids found in GPU_IDS=$GPU_IDS" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

TMP_CONFIG=""
cleanup() {
  if [[ -n "$TMP_CONFIG" && -f "$TMP_CONFIG" ]]; then
    rm -f "$TMP_CONFIG"
  fi
}
trap cleanup EXIT

if [[ "$NO_VAL" == "1" ]]; then
  TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/vpas_train_dist_noval_XXXXXX.yaml")"
  "$PYTHON_BIN" - <<PY
from pathlib import Path
import yaml

src = Path(r"$CONFIG")
dst = Path(r"$TMP_CONFIG")
with src.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
base = cfg.get("_base_")
if isinstance(base, str):
    cfg["_base_"] = str((src.parent / base).resolve())
elif isinstance(base, list):
    cfg["_base_"] = [str((src.parent / item).resolve()) if isinstance(item, str) else item for item in base]
train = cfg.get("train")
if not isinstance(train, dict):
    train = {}
    cfg["train"] = train
train["val_interval"] = -1
with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
  CONFIG="$TMP_CONFIG"
fi

CMD=(
  "$PYTHON_BIN" -m torch.distributed.run
  --nproc_per_node "$NPROC_PER_NODE"
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  "$ROOT_DIR/train.py"
  --config "$CONFIG"
  --distributed
  --dist_backend "$DIST_BACKEND"
  --output_dir "$OUTPUT_DIR"
  --num_workers "$NUM_WORKERS"
  --batch_size "$BATCH_SIZE"
  --seed "$SEED"
)

if [[ -n "$RESUME" ]]; then
  CMD+=(--resume "$RESUME")
fi

if [[ "$DISABLE_CUDNN" == "1" ]]; then
  export DISABLE_CUDNN=1
fi

cd "$ROOT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
