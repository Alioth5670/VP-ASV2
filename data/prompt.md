请为我编写一个 PyTorch Dataset 类，用于训练视频异常分割模型 VPAS。
Dataset 的输入是一个 JSON 文件。

请严格按照以下规范，实现 Dataset 类、transform 逻辑、辅助函数等。

# =========================================================
# 🎯 1. Dataset 输出格式（必须严格一致）
# =========================================================

{
    "prompt": Tensor[3,H,W],   # Normal frame (visual prompt)
    "query":  Tensor[3,H,W],   # Real or synthetic anomaly image
    "mask":   Tensor[1,H,W]
}

要求：
- prompt/query/mask 的 spatial size 完全一致
- mask 的插值必须是 nearest-neighbor
- mask 值域必须为 {0,1}

# =========================================================
# 🎯 2. JSON 文件格式
# =========================================================

[
    {
        "id": "000001",
        "prompt_path": "/path/prompt.png",
        "query_path":  "/path/query.png",  # 若为 null，需 synthetic augment
        "mask_path":   "/path/mask.png"    # 若为 null，可来自 synthetic augment
    },
    ...
]

# =========================================================
# 🎯 3. Dataset 初始化接口（必须严格一致）
# =========================================================

class VPAnomalyTrainDataset(Dataset):
    def __init__(
        self,
        json_path,
        transform_prompt=None,    # ONLY prompt uses this (独立外观增强)
        transform_query=None,     # query 和 mask 用这个外观增强（不一致于 prompt）
        transform_geo=None,       # prompt、query 和 mask 使用一致几何变换
        anomaly_aug=None,
        return_mask=True,
    ):

# =========================================================
# 🎯 4. Transform 设计（关键点：mask 必须跟随 query）
# =========================================================

Dataset 必须使用三类 transform：

------------------------------------------------------------
(A) prompt 的独立外观增强：transform_prompt
------------------------------------------------------------
prompt 可以拥有不同于 query 的光照、色彩、轻微旋转等外观差异。
这些增强可以是：
- brightness / contrast / saturation / hue
- color jitter
- random noise
- motion blur
- small random rotation (≤3°) —— optional

重要：
> transform_prompt 只作用于 prompt，不作用于 query/mask。

------------------------------------------------------------
(B) query 与 mask 的外观增强：transform_query
------------------------------------------------------------

query 使用与 prompt 不同的采样增强，以模拟真实监控中摄像头差异。
mask 不受外观增强影响。

即：

query_aug, mask_aug = transform_query(query_raw, mask_raw)
prompt_aug = transform_prompt(prompt_raw)

------------------------------------------------------------
(C) prompt、query 与 mask 的几何变换：transform_geo
------------------------------------------------------------

关键逻辑：
> **所有几何变换必须对 query 与 mask 使用完全相同的参数。**

必须同步变换的操作：
- resize
- crop
- flip
- affine / rotation（若为几何）
- pad

mask 必须使用 nearest-neighbor 插值。

# =========================================================
# 🎯 5. 数据流流程（必须严格遵守）
# =========================================================

对每条样本：

1. 从 JSON 读取 prompt_path, query_path, mask_path
2. 加载 prompt_raw = load_image(prompt_path)

3. 若 query_path != None:
        query_raw = load_image(query_path)
        如果 mask_path != None :
            mask_raw = load_mask(mask_path)
        否则:
            mask_raw = zeros(1,h,w)
   否则:
       query_raw, mask_raw = anomaly_aug(prompt_raw)
   ⚠ anomaly_aug 必须在 transform 之前执行，作用于 raw 图像。

4. 外观增强阶段：
       prompt_aug = transform_prompt(prompt_raw)
       query_aug, mask_aug  = transform_query(query_raw, mask_raw)
       注意: mask_raw 不参与光照，噪声对比度等外观变化，但是要保证和query一致的轻微旋转

5. 几何对齐阶段（query/mask 必须同步）：
        prompt_final, query_final, mask_final = transform_geo(prompt_aug, query_aug, mask_aug)
        mask 缩放必须使用 nearest-neighbor 插值。



6. 返回字典：
{
    "prompt": prompt_final,
    "query":  query_final,
    "mask":   mask_final 
}

# =========================================================
# 🎯 6. 必须包含的辅助函数
# =========================================================

- load_image(path)
- load_mask(path)
- load_json_list(path)
- 你可以使用 torchvision.transforms 或自定义 transforms

# =========================================================
# 🎯 7. 输出要求
# =========================================================

请生成完整可运行的 PyTorch 代码，包括：

- 所有 import
- 完整的 VPASTrainDataset 类
- transform_prompt, transform_query, transform_geo 的设计
- 所有辅助函数
- 清晰的注释

必须保证：
> **mask 在几何变换阶段与 query 100% 同步。**

现在开始生成代码。