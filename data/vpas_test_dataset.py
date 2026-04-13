from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from utils import get_section, to_dict

from .vpas_dataset import _is_missing_path, load_image, load_json_list, load_mask


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


class VPASTestDataset(Dataset):
    """Test dataset that keeps only resize and normalization."""

    def __init__(
        self,
        json_path: str | Path,
        output_size=(512, 512),
        normalize: bool = True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        return_mask: bool = True,
    ) -> None:
        self.samples = self._flatten_samples(load_json_list(json_path))
        self.return_mask = bool(return_mask)

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if len(output_size) != 2:
            raise ValueError(f"output_size must be int or (H, W), got {output_size}")
        self.output_size = (int(output_size[0]), int(output_size[1]))

        self.normalize = bool(normalize)
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean/std must have 3 elements for RGB normalization")
        self.mean = [float(v) for v in mean]
        self.std = [float(v) for v in std]

    @staticmethod
    def _flatten_samples(raw_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flattened: List[Dict[str, Any]] = []
        for item_idx, item in enumerate(raw_samples):
            query_paths = _ensure_list(item.get("query_path"))
            mask_paths = _ensure_list(item.get("mask_path"))

            if len(query_paths) == 0:
                raise ValueError(f"Missing 'query_path' for test sample index {item_idx}.")

            for query_idx, query_path in enumerate(query_paths):
                if _is_missing_path(query_path):
                    raise ValueError(f"Invalid query_path at sample index {item_idx}, query index {query_idx}.")

                mask_path = mask_paths[query_idx] if query_idx < len(mask_paths) else None
                flattened.append(
                    {
                        "id": item.get("id", str(item_idx)),
                        "img_clsname": item.get("img_clsname"),
                        "prompt_path": item["prompt_path"],
                        "query_path": query_path,
                        "mask_path": mask_path,
                        "query_index": query_idx,
                    }
                )

        return flattened

    def __len__(self) -> int:
        return len(self.samples)

    def _zero_mask_like(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        return Image.new("L", (w, h), 0)

    def _load_mask_or_zeros(self, mask_path: Any, query_img: Image.Image) -> Image.Image:
        if _is_missing_path(mask_path):
            return self._zero_mask_like(query_img)

        mask_path = Path(mask_path)
        if not mask_path.exists():
            return self._zero_mask_like(query_img)
        return load_mask(mask_path)

    def _resize_image(self, img: Image.Image) -> Image.Image:
        target_h, target_w = self.output_size
        if img.size == (target_w, target_h):
            return img
        return TF.resize(img, (target_h, target_w), interpolation=InterpolationMode.BILINEAR)

    def _resize_mask(self, mask: Image.Image) -> Image.Image:
        target_h, target_w = self.output_size
        if mask.size == (target_w, target_h):
            return mask
        return TF.resize(mask, (target_h, target_w), interpolation=InterpolationMode.NEAREST)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int | None]:
        item = self.samples[index]

        prompt = self._resize_image(load_image(item["prompt_path"]))
        query = self._resize_image(load_image(item["query_path"]))
        mask = self._resize_mask(self._load_mask_or_zeros(item.get("mask_path"), query))

        prompt_t = TF.to_tensor(prompt)
        query_t = TF.to_tensor(query)
        mask_t = (TF.pil_to_tensor(mask).float() / 255.0 > 0.5).float()
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)
        elif mask_t.dim() == 3 and mask_t.shape[0] != 1:
            mask_t = mask_t[:1]

        if self.normalize:
            prompt_t = TF.normalize(prompt_t, mean=self.mean, std=self.std)
            query_t = TF.normalize(query_t, mean=self.mean, std=self.std)

        output: Dict[str, torch.Tensor | str | int | None] = {
            "prompt": prompt_t,
            "query": query_t,
            "mask": mask_t,
            "id": item["id"],
            "img_clsname": item.get("img_clsname"),
            "query_index": int(item["query_index"]),
            "prompt_path": str(item["prompt_path"]),
            "query_path": str(item["query_path"]),
            "mask_path": '' if _is_missing_path(item.get("mask_path")) else str(item["mask_path"]),
        }
        if not self.return_mask:
            output.pop("mask")
        return output


def build_test_dataset(
    cfg: Any,
    split: str = "test",
    return_mask: bool = True,
) -> VPASTestDataset:
    data_cfg = to_dict(get_section(cfg, "data", required=True))
    split_l = str(split).lower()

    json_key = f"{split_l}_json"
    json_path = data_cfg.get(json_key, data_cfg.get("json_path", None))
    if json_path is None:
        raise KeyError(f"Missing '{json_key}' (or 'data.json_path') in config.")

    output_size_key = f"{split_l}_output_size"
    output_size = data_cfg.get(output_size_key, data_cfg.get("val_output_size", data_cfg.get("output_size", (512, 512))))

    normalize_cfg = data_cfg.get("normalize", False)
    if isinstance(normalize_cfg, dict):
        normalize = bool(normalize_cfg.get("enabled", True))
        mean = normalize_cfg.get("mean", (0.485, 0.456, 0.406))
        std = normalize_cfg.get("std", (0.229, 0.224, 0.225))
    else:
        normalize = bool(normalize_cfg)
        mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
        std = data_cfg.get("std", (0.229, 0.224, 0.225))

    return VPASTestDataset(
        json_path=json_path,
        output_size=output_size,
        normalize=normalize,
        mean=mean,
        std=std,
        return_mask=return_mask,
    )
