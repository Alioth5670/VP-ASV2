import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from utils import get_section, to_dict
from pycocotools import mask as coco_mask_utils
from .transform import build_transforms

PngImagePlugin.MAX_TEXT_CHUNK = 10485760


def load_json_list(path: str | Path) -> List[Dict[str, Any]]:
    """Load a json file whose root is a list of sample dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON root must be a list.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Each item must be dict, got {type(item).__name__} at index {i}.")
        if "prompt_path" not in item:
            raise ValueError(f"Missing 'prompt_path' in item index {i}.")
    return data


def load_image(path: str | Path) -> Image.Image:
    """Load RGB image."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    with Image.open(p) as img:
        return img.convert("RGB")


def load_mask(path: str | Path) -> Image.Image:
    """Load single-channel mask."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mask not found: {p}")
    with Image.open(p) as m:
        return m.convert("L")


def _to_pil_rgb(x: Any) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, torch.Tensor):
        return TF.to_pil_image(x.detach().cpu()).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(x).__name__}")


def _to_pil_mask(x: Any) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("L")
    if isinstance(x, torch.Tensor):
        return TF.to_pil_image(x.detach().cpu()).convert("L")
    if isinstance(x, np.ndarray):
        return Image.fromarray(x)
    raise TypeError(f"Unsupported mask type: {type(x).__name__}")


def _is_missing_path(x: Any) -> bool:
    return x is None or x == "" or str(x).lower() == "null"

def single_coco_annotation_to_mask_image(annotation, image_shape_as_hw):
    """
    Converts a single object annotation which can be polygons, uncompressed RLE,
    or RLE to binary mask.
    """
    h, w = image_shape_as_hw
    segm = annotation["segmentation"]
    if type(segm) == list:
        rles = coco_mask_utils.frPyObjects(segm, h, w)
        rle = coco_mask_utils.merge(rles)
    elif type(segm["counts"]) == list:
        rle = coco_mask_utils.frPyObjects(segm, h, w)
    else:
        rle = annotation["segmentation"]
    m = coco_mask_utils.decode(rle)
    return m
def coco_annotations_to_mask_np_array(list_of_annotations, image_shape_as_hw):
    """
    Given a list of object annotations, returns a single binary mask.
    """
    mask = np.zeros(image_shape_as_hw, dtype=bool)
    for annotation in list_of_annotations:
        object_mask = single_coco_annotation_to_mask_image(annotation, image_shape_as_hw)
        mask = np.maximum(object_mask, mask)
    mask = _to_pil_mask(mask * 255)
    return mask

class VPAnomalyTrainDataset(Dataset):
    def __init__(
        self,
        json_path,
        transform_prompt=None,
        transform_query=None,
        transform_geo=None,
        local_change_aug=None,
        return_mask=True,
        output_size=(512, 512),
        normalize: bool = True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        pre_resize_to_output: bool = True,
    ):
        self.samples = load_json_list(json_path)
        self.transform_prompt = transform_prompt
        self.transform_query = transform_query
        self.transform_geo = transform_geo
        self.local_change_aug = local_change_aug
        self.return_mask = return_mask

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if len(output_size) != 2:
            raise ValueError(f"output_size must be int or (H, W), got {output_size}")
        self.output_size = (int(output_size[0]), int(output_size[1]))

        self.normalize = bool(normalize)
        self.pre_resize_to_output = bool(pre_resize_to_output)
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean/std must have 3 elements for RGB normalization")
        self.mean = [float(v) for v in mean]
        self.std = [float(v) for v in std]

    def __len__(self) -> int:
        return len(self.samples)

    def _zero_mask_like(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        return Image.new("L", (w, h), 0)

    def _pre_resize_image(self, img: Image.Image) -> Image.Image:
        if not self.pre_resize_to_output:
            return img
        target_h, target_w = self.output_size
        if img.size == (target_w, target_h):
            return img
        return TF.resize(img, (target_h, target_w), interpolation=InterpolationMode.BILINEAR)

    def _pre_resize_mask(self, mask: Image.Image) -> Image.Image:
        if not self.pre_resize_to_output:
            return mask
        target_h, target_w = self.output_size
        if mask.size == (target_w, target_h):
            return mask
        return TF.resize(mask, (target_h, target_w), interpolation=InterpolationMode.NEAREST)


    def _apply_transform_prompt(self, prompt: Image.Image) -> Image.Image:
        if self.transform_prompt is None:
            return prompt
        return self.transform_prompt(prompt)

    def _apply_transform_query(self, query: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.transform_query is None:
            return query, mask
        return self.transform_query(query, mask)

    def _apply_transform_geo(
        self,
        prompt: Image.Image,
        query: Image.Image,
        mask: Image.Image,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Apply geometric transform in PIL domain; mask keeps nearest interpolation."""
        if self.transform_geo is None:
            return prompt, query, mask
        return self.transform_geo(prompt, query, mask)

    def _to_tensor_triplet(
        self,
        prompt: Image.Image,
        query: Image.Image,
        mask: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_t = TF.to_tensor(prompt)
        query_t = TF.to_tensor(query)
        mask_t = (TF.pil_to_tensor(mask).float() / 255.0 > 0.5).float()
        return prompt_t, query_t, mask_t

    def _resize_triplet(
        self,
        prompt_t: torch.Tensor,
        query_t: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_h, target_w = self.output_size

        if prompt_t.shape[-2:] != (target_h, target_w):
            prompt_t = TF.resize(prompt_t, (target_h, target_w), interpolation=InterpolationMode.BILINEAR)
        if query_t.shape[-2:] != (target_h, target_w):
            query_t = TF.resize(query_t, (target_h, target_w), interpolation=InterpolationMode.BILINEAR)
        if mask_t.shape[-2:] != (target_h, target_w):
            mask_t = TF.resize(mask_t, (target_h, target_w), interpolation=InterpolationMode.NEAREST)

        return prompt_t, query_t, mask_t

    def _ensure_binary_mask(self, mask_t: torch.Tensor) -> torch.Tensor:
        mask_t = (mask_t > 0.5).float()
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)
        elif mask_t.dim() == 3 and mask_t.shape[0] != 1:
            mask_t = mask_t[:1]
        return mask_t

    def _normalize_pair(self, prompt_t: torch.Tensor, query_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.normalize:
            return prompt_t, query_t
        prompt_t = TF.normalize(prompt_t, mean=self.mean, std=self.std)
        query_t = TF.normalize(query_t, mean=self.mean, std=self.std)
        return prompt_t, query_t

    def add_random_objects(self, image, item_index):
        all_indices_except_current = list(range(item_index)) + list(
            range(item_index + 1, len(self.samples))
        )
        random_image_index = random.choice(all_indices_except_current)
        item = self.samples[random_image_index]

        original_image = load_image(item["prompt_path"])
        annotation_path = item["annotation_path"]
        annotations = np.load(annotation_path, allow_pickle=True)
        original_image_height, original_image_width = original_image.height, original_image.width
        original_image_mask = coco_annotations_to_mask_np_array(annotations, (original_image_height, original_image_width))
        target_size = image.size
        original_image_resized_to_current, annotations_resized = original_image.resize(target_size, Image.BILINEAR), original_image_mask.resize(target_size, Image.NEAREST)

        image, original_image_resized_to_current= np.array(image), np.array(original_image_resized_to_current)
        annotation_mask = np.array(annotations_resized) > 0

        image[annotation_mask] = original_image_resized_to_current[annotation_mask]
        image = Image.fromarray(image)
        # annotations_resized = Image.fromarray(annotations_resized).convert("L")
        return image, annotations_resized

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.samples[index]

        prompt_raw = load_image(item["prompt_path"])
        prompt_raw = self._pre_resize_image(prompt_raw)
        query_paths = item.get("query_path", [])
        mask_paths = item.get("mask_path", [])

        assert (len(query_paths)>0) or (len(mask_paths)>0), f"query_path or mask_path must be provided and equal, but got num_query_path={len(query_paths)} and num_mask_paths={len(mask_paths)}"

        if random.random() < 0.5:
            # do Object change
            if random.random() < 0.5:
                # inpaint
                query_index = random.randint(0, len(query_paths) - 1)
                query_path = query_paths[query_index]
                query_raw = load_image(query_path)
                query_raw = self._pre_resize_image(query_raw)
                mask_path = mask_paths[query_index]
                mask_raw = load_mask(mask_path)
                mask_raw = self._pre_resize_mask(mask_raw)
            else:
                query_raw, mask_raw = self.add_random_objects(prompt_raw, index)

            if random.random() < 0.5:
                query_raw, prompt_raw = prompt_raw, query_raw
            
        else:
            # do local change
            query_raw, mask_raw = self.local_change_aug(prompt_raw)
            query_raw = _to_pil_rgb(query_raw)
            mask_raw = _to_pil_mask(mask_raw)

        # 1) Geometric transform first, all in PIL domain.
        prompt_geo, query_geo, mask_geo = self._apply_transform_geo(prompt_raw, query_raw, mask_raw)

        # 2) Appearance transforms after geometry, still in PIL domain.
        prompt_aug = self._apply_transform_prompt(prompt_geo)
        query_aug, mask_aug = self._apply_transform_query(query_geo, mask_geo)

        # 3) Tensor/resize/mask-fix/normalize.
        prompt_final, query_final, mask_final = self._to_tensor_triplet(prompt_aug, query_aug, mask_aug)
        prompt_final, query_final, mask_final = self._resize_triplet(prompt_final, query_final, mask_final)
        mask_final = self._ensure_binary_mask(mask_final)
        prompt_final, query_final = self._normalize_pair(prompt_final, query_final)

        output = {
            "prompt": prompt_final,
            "query": query_final,
            "mask": mask_final,
        }
        if not self.return_mask:
            output.pop("mask")
        return output


def build_dataset(
    cfg: Any,
    split: str = "train",
    local_change_aug=None,
    return_mask: bool = True,
):
    """Build VPAS dataset from config.

    Expected keys:
    - data.{split}_json or data.json_path
    - data.{split}_output_size or data.output_size
    - data.normalize: bool OR dict(enabled/mean/std)
    - transform.* for build_transforms(cfg, split=...)
    """
    split_l = str(split).lower()
    if split_l == "test":
        from .vpas_test_dataset import build_test_dataset

        return build_test_dataset(cfg, split=split_l, return_mask=return_mask)

    # if split_l in {"val", "valid", "validation"}:
    #     local_change_aug = None

    data_cfg = to_dict(get_section(cfg, "data", required=True))

    json_key = f"{split}_json"
    json_path = data_cfg.get(json_key, data_cfg.get("json_path", None))
    if json_path is None:
        raise KeyError(f"Missing '{json_key}' (or 'data.json_path') in config.")

    output_size_key = f"{split}_output_size"
    output_size = data_cfg.get(output_size_key, data_cfg.get("output_size", (512, 512)))

    normalize_cfg = data_cfg.get("normalize", False)
    if isinstance(normalize_cfg, dict):
        normalize = bool(normalize_cfg.get("enabled", True))
        mean = normalize_cfg.get("mean", (0.485, 0.456, 0.406))
        std = normalize_cfg.get("std", (0.229, 0.224, 0.225))
    else:
        normalize = bool(normalize_cfg)
        mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
        std = data_cfg.get("std", (0.229, 0.224, 0.225))

    transform_prompt, transform_query, transform_geo = build_transforms(cfg, split=split)

    pre_resize_to_output = bool(data_cfg.get("pre_resize_to_output", True))

    dataset = VPAnomalyTrainDataset(
        json_path=json_path,
        transform_prompt=transform_prompt,
        transform_query=transform_query,
        transform_geo=transform_geo,
        local_change_aug=local_change_aug,
        return_mask=return_mask,
        output_size=output_size,
        normalize=normalize,
        mean=mean,
        std=std,
        pre_resize_to_output=pre_resize_to_output,
    )
    return dataset
