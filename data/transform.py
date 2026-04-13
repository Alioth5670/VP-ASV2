import random
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from utils import get_section, to_2tuple_float, to_dict, to_hw


def _build_jitter(sec: dict, defaults: Tuple[float, float, float, float]) -> ColorJitter:
    return ColorJitter(
        brightness=float(sec.get("brightness", defaults[0])),
        contrast=float(sec.get("contrast", defaults[1])),
        saturation=float(sec.get("saturation", defaults[2])),
        hue=float(sec.get("hue", defaults[3])),
    )


class AppearanceTransform:
    """Appearance augmentation with optional mask-aware tiny rotation."""

    def __init__(
        self,
        jitter: Optional[ColorJitter] = None,
        jitter_p: float = 1.0,
        noise_std: float = 0.02,
        noise_p: float = 0.3,
        blur_p: float = 0.1,
        blur_kernel: int = 3,
        rot_deg: float = 3.0,
        rot_p: float = 0.3,
    ) -> None:
        self.jitter = jitter if jitter is not None else ColorJitter(0.25, 0.25, 0.25, 0.08)
        self.jitter_p = jitter_p
        self.noise_std = noise_std
        self.noise_p = noise_p
        self.blur_p = blur_p
        self.blur_kernel = blur_kernel
        self.rot_deg = rot_deg
        self.rot_p = rot_p

    def __call__(self, image: Image.Image, mask: Optional[Image.Image] = None):
        img = self.jitter(image) if random.random() < self.jitter_p else image

        if random.random() < self.blur_p:
            img = TF.gaussian_blur(img, kernel_size=self.blur_kernel)

        if self.noise_std > 0 and random.random() < self.noise_p:
            arr = np.asarray(img).astype(np.float32) / 255.0
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 1.0)
            img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")

        if random.random() < self.rot_p:
            angle = random.uniform(-self.rot_deg, self.rot_deg)
            img = TF.rotate(img, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            if mask is not None:
                mask = TF.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST, fill=0)

        if mask is None:
            return img
        return img, mask


class SynchronizedGeoTransform:
    """Apply same geometric params to prompt/query/mask; mask uses nearest interpolation."""

    def __init__(
        self,
        crop_size: Optional[Tuple[int, int]] = None,
        hflip_p: float = 0.5,
        affine_p: float = 0.3,
        affine_degrees: float = 5.0,
        affine_translate: Tuple[float, float] = (0.02, 0.02),
        affine_scale: Tuple[float, float] = (0.98, 1.02),
        affine_shear: float = 2.0,
    ) -> None:
        self.crop_size = crop_size
        self.hflip_p = hflip_p
        self.affine_p = affine_p
        self.affine_degrees = affine_degrees
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.affine_shear = affine_shear

    def __call__(
        self,
        prompt: Image.Image,
        query: Image.Image,
        mask: Image.Image,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:

        if random.random() < self.hflip_p:
            prompt = TF.hflip(prompt)
            query = TF.hflip(query)
            mask = TF.hflip(mask)

        if random.random() < self.affine_p:
            angle = random.uniform(-self.affine_degrees, self.affine_degrees)
            h, w = TF.get_image_size(prompt)[1], TF.get_image_size(prompt)[0]
            max_dx = self.affine_translate[0] * w
            max_dy = self.affine_translate[1] * h
            translate = (
                int(random.uniform(-max_dx, max_dx)),
                int(random.uniform(-max_dy, max_dy)),
            )
            scale = random.uniform(self.affine_scale[0], self.affine_scale[1])
            shear = [random.uniform(-self.affine_shear, self.affine_shear), 0.0]

            prompt = TF.affine(
                prompt,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            query = TF.affine(
                query,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        if self.crop_size is not None:
            ch, cw = self.crop_size
            h, w = TF.get_image_size(prompt)[1], TF.get_image_size(prompt)[0]
            if h < ch or w < cw:
                pad_h = max(0, ch - h)
                pad_w = max(0, cw - w)
                pad = [0, 0, pad_w, pad_h]
                prompt = TF.pad(prompt, pad, fill=0)
                query = TF.pad(query, pad, fill=0)
                mask = TF.pad(mask, pad, fill=0)

            i = random.randint(0, TF.get_image_size(prompt)[1] - ch)
            j = random.randint(0, TF.get_image_size(prompt)[0] - cw)
            prompt = TF.crop(prompt, i, j, ch, cw)
            query = TF.crop(query, i, j, ch, cw)
            mask = TF.crop(mask, i, j, ch, cw)

        return prompt, query, mask


def build_transforms(
    cfg: Any,
    split: str = "train",
) -> Tuple[Optional[AppearanceTransform], Optional[AppearanceTransform], SynchronizedGeoTransform]:
    transform_cfg = to_dict(get_section(cfg, "transform", default={}, required=False), allow_empty=True)
    split_l = str(split).lower()

    if split_l in {"val", "valid", "validation", "test"}:
        val_cfg = to_dict(transform_cfg.get("val", {}), allow_empty=True)
        geo_cfg = to_dict(val_cfg.get("geo", {}), allow_empty=True)
        crop_raw = geo_cfg.get("crop_size", None)
        crop_size = None if crop_raw is None else to_hw(crop_raw, default=(512, 512))

        geo_transform = SynchronizedGeoTransform(
            crop_size=crop_size,
            hflip_p=0.0,
            affine_p=0.0,
            affine_degrees=0.0,
            affine_translate=(0.0, 0.0),
            affine_scale=(1.0, 1.0),
            affine_shear=0.0,
        )
        return None, None, geo_transform

    prompt_cfg = to_dict(transform_cfg.get("prompt", {}), allow_empty=True)
    query_cfg = to_dict(transform_cfg.get("query", {}), allow_empty=True)
    geo_cfg = to_dict(transform_cfg.get("geo", {}), allow_empty=True)

    prompt_transform = AppearanceTransform(
        jitter=_build_jitter(prompt_cfg, defaults=(0.25, 0.25, 0.25, 0.08)),
        jitter_p=float(prompt_cfg.get("jitter_p", 1.0)),
        noise_std=float(prompt_cfg.get("noise_std", 0.02)),
        noise_p=float(prompt_cfg.get("noise_p", 0.3)),
        blur_p=float(prompt_cfg.get("blur_p", 0.1)),
        blur_kernel=int(prompt_cfg.get("blur_kernel", 3)),
        rot_deg=float(prompt_cfg.get("rot_deg", 3.0)),
        rot_p=float(prompt_cfg.get("rot_p", 0.3)),
    )

    query_transform = AppearanceTransform(
        jitter=_build_jitter(query_cfg, defaults=(0.25, 0.25, 0.25, 0.08)),
        jitter_p=float(query_cfg.get("jitter_p", 1.0)),
        noise_std=float(query_cfg.get("noise_std", 0.02)),
        noise_p=float(query_cfg.get("noise_p", 0.3)),
        blur_p=float(query_cfg.get("blur_p", 0.1)),
        blur_kernel=int(query_cfg.get("blur_kernel", 3)),
        rot_deg=float(query_cfg.get("rot_deg", 3.0)),
        rot_p=float(query_cfg.get("rot_p", 0.3)),
    )

    crop_raw = geo_cfg.get("crop_size", None)
    crop_size = None if crop_raw is None else to_hw(crop_raw, default=(512, 512))

    geo_transform = SynchronizedGeoTransform(
        crop_size=crop_size,
        hflip_p=float(geo_cfg.get("hflip_p", 0.5)),
        affine_p=float(geo_cfg.get("affine_p", 0.3)),
        affine_degrees=float(geo_cfg.get("affine_degrees", 5.0)),
        affine_translate=to_2tuple_float(geo_cfg.get("affine_translate", (0.02, 0.02)), (0.02, 0.02)),
        affine_scale=to_2tuple_float(geo_cfg.get("affine_scale", (0.98, 1.02)), (0.98, 1.02)),
        affine_shear=float(geo_cfg.get("affine_shear", 2.0)),
    )

    return prompt_transform, query_transform, geo_transform
