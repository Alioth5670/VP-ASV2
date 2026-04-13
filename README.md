# VPAS

VPAS 是一个基于 **Prompt 图像 vs Query 图像** 的变化分割项目。

给定一张参考图（Prompt）和一张待检测图（Query），模型输出与 Query 对齐的单通道变化掩码（异常区域）。

## 1. 功能概览

- 支持单卡/多卡训练
- 支持 `val/test` 评估并导出 `metrics.json`
- 支持图像对推理、批量 JSON 推理、视频推理
- 支持从实验目录自动汇总指标（`utils/metrics_to_yuque.py`）

## 2. 项目结构

```text
VPAS/
├── train.py                      # 训练入口
├── val.py                        # 验证/测试入口
├── inference.py                  # 推理入口（image/video）
├── config/                       # 配置文件（base + 各实验yaml）
├── data/                         # 数据集读取、增强、json构建
├── models/                       # 主模型、backbone、neck、loss
├── scripts/                      # 常用脚本（train/val/infer）
├── utils/                        # 配置、评估、分布式等工具
└── outputs/                      # 默认输出目录
```

## 3. 环境准备

推荐 Python 3.10+，CUDA 环境按你的 PyTorch 版本匹配。

### 3.1 创建环境（示例）

```bash
conda create -n vpas python=3.10 -y
conda activate vpas
```

### 3.2 安装依赖（示例）

```bash
# 先按你的 CUDA 版本安装 torch/torchvision
# 例如: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install numpy pillow opencv-python tqdm pyyaml scipy scikit-learn pycocotools tensorboard
```

说明：仓库当前未提供 `requirements.txt`，上面是根据代码 import 整理的最小依赖集合。

## 4. 数据准备

### 4.1 JSON 基本格式

训练/验证/测试都由 JSON 列表驱动。每个元素至少应包含：

```json
{
  "prompt_path": "/abs/path/to/prompt.png",
  "query_path": ["/abs/path/to/query_1.png", "/abs/path/to/query_2.png"],
  "mask_path": ["/abs/path/to/mask_1.png", "/abs/path/to/mask_2.png"],
  "annotation_path": "/abs/path/to/annotation.npy",
  "img_clsname": "optional_class_name"
}
```

说明：

- `prompt_path` 必填
- `query_path`、`mask_path` 建议一一对应（支持列表）
- 训练集的 object-change 分支会用到 `annotation_path`
- 测试集 `img_clsname` 可选，用于分类别指标汇总

### 4.2 生成 JSON（可选）

`data/make_dataset_json.py` 内置了可编辑 `CONFIG`，按注释修改后运行：

```bash
python data/make_dataset_json.py
```

## 5. 配置说明

主配置示例：`config/dinov3/vpas_dinov3_vitb16.yaml`

关键字段：

- `data.train_json / data.val_json / data.test_json`：数据清单路径
- `model.backbone.weights`：backbone 权重路径
- `train.batch_size / train.epochs / train.amp`：训练超参
- `inference.threshold`：推理/评估阈值

配置支持 `_base_` 继承（见 `config/base.yaml`）。

## 6. 训练

### 6.1 单卡训练

```bash
python train.py \
  --config config/dinov3/vpas_dinov3_vitb16.yaml \
  --device cuda:0 \
  --output_dir outputs/vpas_exp \
  --batch_size 8 \
  --num_workers 4
```

或直接修改并运行：

```bash
bash scripts/train_vpas.sh
```

### 6.2 多卡分布式训练

```bash
bash scripts/train_dist.sh
```

脚本里可设置 `GPU_IDS`、`BATCH_SIZE`、`OUTPUT_DIR`、`NO_VAL` 等。

## 7. 验证 / 测试

```bash
python val.py \
  --config outputs/your_exp/resolved_config.yaml \
  --ckpt outputs/your_exp/last.pth \
  --split test \
  --device cuda:0 \
  --save_json outputs/your_exp/test_outputs/metrics.json
```

如果需要对多份 JSON 分别跑并聚合统计：

```bash
python val.py \
  --config outputs/your_exp/resolved_config.yaml \
  --ckpt outputs/your_exp/last.pth \
  --split test \
  --json_path a.json b.json --json_path c.json \
  --save_json outputs/your_exp/test_outputs_random/metrics.json
```

也可以直接用：

```bash
bash scripts/val_vpas.sh
```

## 8. 推理

### 8.1 图像模式

单对图像：

```bash
python inference.py \
  --mode image \
  --config outputs/your_exp/resolved_config.yaml \
  --ckpt outputs/your_exp/last.pth \
  --device cuda:0 \
  --prompt-image path/to/prompt.png \
  --query-image path/to/query.png \
  --output-dir outputs/infer_image_demo
```

批量 JSON：

```bash
python inference.py \
  --mode image \
  --config outputs/your_exp/resolved_config.yaml \
  --ckpt outputs/your_exp/last.pth \
  --input-json path/to/pairs.json \
  --output-dir outputs/infer_json_demo
```

脚本方式：

```bash
bash scripts/infer_image.sh
```

### 8.2 视频模式

```bash
python inference.py \
  --mode video \
  --config outputs/your_exp/resolved_config.yaml \
  --ckpt outputs/your_exp/last.pth \
  --device cuda:0 \
  --video-path path/to/demo.mp4 \
  --video-prompt-mode first_n_mean \
  --video-prompt-frames 8 \
  --save-mask-video \
  --save-heatmap-video \
  --output-dir outputs/infer_video_demo
```

脚本方式：

```bash
bash scripts/infer_video.sh
```

## 9. 输出说明

训练输出目录（`output_dir`）常见文件：

- `last.pth`：最后一个 checkpoint
- `best.pth`：最佳指标 checkpoint
- `resolved_config.yaml`：训练时固化后的配置
- `train_log.txt` / `train_log.json`：日志
- `events.out.tfevents.*`：TensorBoard 日志

推理输出目录常见文件：

- `prompt.png` / `query.png`
- `mask.png`：二值掩码
- `heatmap.png`：概率热力图
- `overlay.png`：叠加可视化
- `prob.npy`：概率图数组

## 10. 指标汇总（可选）

将多个实验的 `metrics.json` 自动整理成 HTML：

```bash
bash scripts/metrics_to_yuque.sh
```

或直接：

```bash
python utils/metrics_to_yuque.py --outputs-dir outputs --keyword noite --metrics-subdir test_outputs_random --mean-std --output outputs/yuque_metrics.html
```

## 11. 常见问题

1. `Checkpoint model state_dict mismatch`

- 请确认 `--config` 与 `--ckpt` 来自同一实验
- 若 backbone 配置变化，旧权重可能无法直接加载

2. `Input size ... is not divisible by patch size ...`

- 调整 `data.output_size`（以及 val/test 对应 size）使其可被 patch size 整除

3. 测试指标异常低

- 检查 `inference.threshold` 或 `val.py --threshold`
- 检查测试 JSON 中 `query_path` 与 `mask_path` 是否严格对齐

## 12. 相关文档

- 架构说明：`PROJECT_ARCHITECTURE.md`
