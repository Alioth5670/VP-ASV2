import math
import random
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from utils import get_section, to_dict, to_hw


def lerp_np(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x + w * (y - x)


def fade_np(t: np.ndarray) -> np.ndarray:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def rand_perlin_2d_np(shape: Tuple[int, int], res: Tuple[int, int]) -> np.ndarray:
    """Generate 2D Perlin noise in numpy."""
    h, w = int(shape[0]), int(shape[1])
    ry, rx = int(res[0]), int(res[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"shape must be positive, got {shape}")
    if ry <= 0 or rx <= 0:
        raise ValueError(f"res must be positive, got {res}")

    delta = (ry / h, rx / w)
    repeat_y = max(int(math.ceil(h / ry)), 1)
    repeat_x = max(int(math.ceil(w / rx)), 1)
    grid = np.mgrid[0:ry:delta[0], 0:rx:delta[1]].transpose(1, 2, 0) % 1
    grid = grid[:h, :w]

    angles = 2 * math.pi * np.random.rand(ry + 1, rx + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles))).astype(np.float32)

    g00 = gradients[:-1, :-1].repeat(repeat_y, 0).repeat(repeat_x, 1)[:h, :w]
    g10 = gradients[1:, :-1].repeat(repeat_y, 0).repeat(repeat_x, 1)[:h, :w]
    g01 = gradients[:-1, 1:].repeat(repeat_y, 0).repeat(repeat_x, 1)[:h, :w]
    g11 = gradients[1:, 1:].repeat(repeat_y, 0).repeat(repeat_x, 1)[:h, :w]

    n00 = np.sum(grid * g00, axis=2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, axis=2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, axis=2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, axis=2)

    t = fade_np(grid)
    n0 = lerp_np(n00, n10, t[:, :, 0])
    n1 = lerp_np(n01, n11, t[:, :, 0])
    return (np.sqrt(2.0) * lerp_np(n0, n1, t[:, :, 1])).astype(np.float32)


class AnomalyAug:
    """
    用于生成局部异常/局部变化的增强器。
    输入一张正常图像，输出：
      - augmented_image: 带局部变化的图像
      - anomaly_mask:    变化区域 mask, shape=(H, W, 1)
      - has_anomaly:     是否真的生成了异常, shape=(1,)
    """

    def __init__(
        self,
        anomaly_source_path: str,
        resize_shape: Optional[Tuple[int, int]] = None,
        perlin_scale: int = 6,
        min_perlin_scale: int = 0,
        mask_threshold: float = 0.5,
        no_anomaly_prob: float = 0.5,
        blend_beta_max: float = 0.8,
    ):
        if perlin_scale <= min_perlin_scale:
            raise ValueError(
                f"perlin_scale must be > min_perlin_scale, got {perlin_scale} and {min_perlin_scale}"
            )

        self.resize_shape = resize_shape
        self.perlin_scale = perlin_scale
        self.min_perlin_scale = min_perlin_scale
        self.mask_threshold = mask_threshold
        self.no_anomaly_prob = no_anomaly_prob
        self.blend_beta_max = blend_beta_max

        root = Path(anomaly_source_path)
        patterns = ('*/*.jpg', '*/*.jpeg', '*/*.png', '*.jpg', '*.jpeg', '*.png')
        self.anomaly_source_paths = []
        for pattern in patterns:
            self.anomaly_source_paths.extend(str(p) for p in sorted(root.glob(pattern)))

        if len(self.anomaly_source_paths) == 0:
            raise ValueError(f"No anomaly source images found in: {anomaly_source_path}")

    def sample_anomaly_source(self) -> str:
        idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        return self.anomaly_source_paths[idx]

    def load_anomaly_source(self, path: str, resize_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read anomaly source image: {path}")

        target_shape = resize_shape if resize_shape is not None else self.resize_shape
        if target_shape is not None:
            img = cv2.resize(img, dsize=(int(target_shape[1]), int(target_shape[0])))

        return img

    def _np_to_pil_rgb(self, img_bgr_or_rgb: np.ndarray, assume_bgr: bool = True) -> Image.Image:
        if img_bgr_or_rgb.dtype != np.uint8:
            img_bgr_or_rgb = np.clip(img_bgr_or_rgb, 0, 255).astype(np.uint8)

        if assume_bgr:
            img_rgb = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_bgr_or_rgb

        return Image.fromarray(img_rgb)

    def _pil_to_np_rgb_float(self, img_pil: Image.Image) -> np.ndarray:
        return np.array(img_pil).astype(np.float32) / 255.0

    def _apply_random_ops(self, img_pil: Image.Image) -> Image.Image:
        def op_gamma(img: Image.Image) -> Image.Image:
            gamma = random.uniform(0.5, 2.0)
            img_t = TF.pil_to_tensor(img).float() / 255.0
            img_t = torch.clamp(img_t, 0.0, 1.0)
            img_t = img_t ** gamma
            img_t = torch.clamp(img_t * 255.0, 0.0, 255.0).to(torch.uint8)
            return TF.to_pil_image(img_t)

        def op_brightness(img: Image.Image) -> Image.Image:
            brightness_factor = random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, brightness_factor)
            offset = random.randint(-30, 30)
            img_t = TF.pil_to_tensor(img).to(torch.int16)
            img_t = torch.clamp(img_t + offset, 0, 255).to(torch.uint8)
            return TF.to_pil_image(img_t)

        def op_sharpness(img: Image.Image) -> Image.Image:
            sharpness_factor = random.uniform(0.5, 2.0)
            return TF.adjust_sharpness(img, sharpness_factor)

        def op_hue_saturation(img: Image.Image) -> Image.Image:
            hue_factor = random.uniform(-0.15, 0.15)
            saturation_factor = random.uniform(0.5, 1.5)
            img = TF.adjust_hue(img, hue_factor)
            img = TF.adjust_saturation(img, saturation_factor)
            return img

        def op_solarize(img: Image.Image) -> Image.Image:
            threshold = random.randint(32, 128)
            return TF.solarize(img, threshold=threshold)

        def op_posterize(img: Image.Image) -> Image.Image:
            bits = random.randint(3, 7)
            return TF.posterize(img, bits)

        def op_invert(img: Image.Image) -> Image.Image:
            return TF.invert(img)

        def op_autocontrast(img: Image.Image) -> Image.Image:
            return TF.autocontrast(img)

        def op_equalize(img: Image.Image) -> Image.Image:
            return TF.equalize(img)

        def op_rotate(img: Image.Image) -> Image.Image:
            angle = random.uniform(-45.0, 45.0)
            return TF.rotate(img, angle=angle)

        ops = [
            op_gamma,
            op_brightness,
            op_sharpness,
            op_hue_saturation,
            op_solarize,
            op_posterize,
            op_invert,
            op_autocontrast,
            op_equalize,
            op_rotate,
        ]

        for op in random.sample(ops, 3):
            img_pil = op(img_pil)

        return img_pil

    def augment_anomaly_source(self, anomaly_source_img: np.ndarray) -> np.ndarray:
        img_pil = self._np_to_pil_rgb(anomaly_source_img, assume_bgr=True)
        img_pil = self._apply_random_ops(img_pil)
        return self._pil_to_np_rgb_float(img_pil)

    def rotate_mask(self, mask: np.ndarray) -> np.ndarray:
        angle = random.uniform(-90.0, 90.0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        rotated = TF.rotate(mask_t, angle=angle)
        return rotated.squeeze(0).numpy().astype(np.float32)

    def generate_perlin_mask(self, h: int, w: int) -> np.ndarray:
        perlin_scalex = 2 ** torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).item()
        perlin_scaley = 2 ** torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).item()

        perlin_noise = rand_perlin_2d_np((h, w), (perlin_scalex, perlin_scaley)).astype(np.float32)
        perlin_noise = self.rotate_mask(perlin_noise)

        mask = np.where(
            perlin_noise > self.mask_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise),
        ).astype(np.float32)

        return np.expand_dims(mask, axis=2)

    def apply(self, image: np.ndarray, anomaly_source_path: Optional[str] = None):
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        image = np.clip(image, 0.0, 1.0)
        h, w = image.shape[:2]

        if anomaly_source_path is None:
            anomaly_source_path = self.sample_anomaly_source()

        anomaly_source_img = self.load_anomaly_source(anomaly_source_path, resize_shape=(h, w))
        anomaly_img_augmented = self.augment_anomaly_source(anomaly_source_img)

        perlin_mask = self.generate_perlin_mask(h, w)
        texture_in_mask = anomaly_img_augmented * perlin_mask
        beta = torch.rand(1).item() * self.blend_beta_max

        augmented_image = (
            image * (1.0 - perlin_mask)
            + ((1.0 - beta) * texture_in_mask + beta * image * perlin_mask)
        ).astype(np.float32)

        if torch.rand(1).item() < self.no_anomaly_prob:
            return (
                image.copy(),
                np.zeros_like(perlin_mask, dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )

        anomaly_mask = perlin_mask.astype(np.float32)
        has_anomaly = 1.0 if np.sum(anomaly_mask) > 0 else 0.0

        augmented_image = anomaly_mask * augmented_image + (1.0 - anomaly_mask) * image
        augmented_image = np.clip(augmented_image, 0.0, 1.0).astype(np.float32)

        return (
            augmented_image,
            anomaly_mask,
            np.array([has_anomaly], dtype=np.float32),
        )

    def __call__(self, image: Any):
        """Dataset-facing API: returns PIL RGB query and PIL L mask."""
        if isinstance(image, Image.Image):
            image_np = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
        elif isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
            image_np = np.clip(image_np, 0.0, 1.0)
        elif isinstance(image, np.ndarray):
            image_np = image.astype(np.float32)
            if image_np.max() > 1.0:
                image_np = image_np / 255.0
            image_np = np.clip(image_np, 0.0, 1.0)
        else:
            raise TypeError(f'Unsupported image type: {type(image).__name__}')

        augmented_image, anomaly_mask, _ = self.apply(image_np)
        query_pil = Image.fromarray((augmented_image * 255.0).astype(np.uint8), mode='RGB')
        mask_2d = (anomaly_mask.squeeze(-1) > 0.5).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_2d, mode='L')
        return query_pil, mask_pil


AnomlyAug = AnomalyAug


def build_local_change_aug(cfg: Any) -> Optional[AnomalyAug]:
    aug_cfg = to_dict(get_section(cfg, 'local_change_aug', default={}, required=False), allow_empty=True)
    if not aug_cfg:
        return None

    enabled = bool(aug_cfg.get('enabled', True))
    if not enabled:
        return None

    source_path = aug_cfg.get('anomaly_source_path', None)
    if not source_path:
        raise KeyError("Missing 'local_change_aug.anomaly_source_path' in config.")

    resize_raw = aug_cfg.get('resize_shape', None)
    resize_shape = None if resize_raw is None else to_hw(resize_raw, default=(256, 256))

    return AnomalyAug(
        anomaly_source_path=str(source_path),
        resize_shape=resize_shape,
        perlin_scale=int(aug_cfg.get('perlin_scale', 6)),
        min_perlin_scale=int(aug_cfg.get('min_perlin_scale', 0)),
        mask_threshold=float(aug_cfg.get('mask_threshold', 0.5)),
        no_anomaly_prob=float(aug_cfg.get('no_anomaly_prob', 0.5)),
        blend_beta_max=float(aug_cfg.get('blend_beta_max', 0.8)),
    )
