from __future__ import annotations

import copy
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .loss import SegmentationLoss
from .neck import FARM, PrototypeDiscrepancyFilter, build_neck
from utils import get_section, to_dict


class VPAS(nn.Module):
    def __init__(self, backbone: nn.Module, neck: FARM, **kwargs: Any) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_layers = int(kwargs.get("num_layers", 1))
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}.")
        if self.num_layers != len(backbone.interaction_indexes):
            raise ValueError(
                f"num_layers ({self.num_layers}) must equal interaction indexes "
                f"({len(backbone.interaction_indexes)})."
            )

        if isinstance(backbone.patch_size, tuple):
            self.patch_h, self.patch_w = backbone.patch_size
        else:
            self.patch_h = self.patch_w = backbone.patch_size

        self.layer_wise_neck = bool(kwargs.get("layer_wise_neck", True))
        if self.layer_wise_neck:
            self.neck = nn.ModuleList([copy.deepcopy(neck) for _ in range(self.num_layers)])
        else:
            self.neck = neck

        self.use_discrepancy_filter = bool(kwargs.get("use_discrepancy_filter", True))
        if not self.use_discrepancy_filter:
            raise ValueError("VPAS ProtoDiff requires use_discrepancy_filter=True.")

        embedding_dim = backbone.vision_transformer.embed_dim
        self.prototype_mode = str(kwargs.get("prototype_mode", "layerwise"))
        self.mask_fusion_mode = str(kwargs.get("mask_fusion_mode", "add"))
        if self.mask_fusion_mode not in {"last", "avg", "add"}:
            raise ValueError(f"Unknown mask_fusion_mode: {self.mask_fusion_mode}")

        self.discrepancy_filter = PrototypeDiscrepancyFilter(
            dim=embedding_dim,
            num_layers=self.num_layers,
            prototype_mode=self.prototype_mode,
        )

    @staticmethod
    def _extract_patch_tokens(feature: Any, expected_tokens: int) -> torch.Tensor:
        """
        Convert backbone output to patch token tensor [B, N, C] and ignore cls token.
        """
        if isinstance(feature, (tuple, list)):
            feature = feature[0]

        if feature.dim() == 4:
            b, c, h, w = feature.shape
            if h * w != expected_tokens:
                raise ValueError(
                    f"Unexpected feature map shape {tuple(feature.shape)} for expected token count {expected_tokens}."
                )
            return feature.flatten(2).transpose(1, 2).contiguous()

        if feature.dim() != 3:
            raise ValueError(f"Unexpected feature shape {tuple(feature.shape)}, expected [B,N,C] or [B,C,H,W].")

        b, n, c = feature.shape
        del b, c
        if n == expected_tokens + 1:
            return feature[:, 1:, :]
        if n == expected_tokens:
            return feature
        raise ValueError(
            f"Unexpected token count {n}, expected {expected_tokens} (or {expected_tokens + 1} with cls token)."
        )

    @staticmethod
    def _tokens_to_logit_map(token_logits: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        """
        token_logits: [B, N, 2] -> [B, 2, feat_h, feat_w]
        """
        if token_logits.dim() != 3:
            raise ValueError(f"Expected [B,N,2], got {tuple(token_logits.shape)}")

        b, n, c = token_logits.shape
        if c != 2:
            raise ValueError(f"Expected channel=2 for ProtoDiff logits, got {c}")
        expected_tokens = feat_h * feat_w
        if n != expected_tokens:
            raise ValueError(f"Unexpected token count {n}, expected {expected_tokens}.")

        return token_logits.transpose(1, 2).reshape(b, 2, feat_h, feat_w).contiguous()

    def _get_neck_layers(self) -> Sequence[nn.Module]:
        if isinstance(self.neck, nn.ModuleList):
            return self.neck
        return [self.neck] * self.num_layers

    def _fuse_layer_logits(self, layer_logits: list[torch.Tensor]) -> torch.Tensor:
        if not layer_logits:
            raise RuntimeError("No layer logits were produced.")

        if self.mask_fusion_mode == "last":
            return layer_logits[-1]

        stacked = torch.stack(layer_logits, dim=0)
        if self.mask_fusion_mode == "avg":
            return stacked.mean(dim=0)
        return stacked.sum(dim=0)

    def forward(self, prompt_image: torch.Tensor, query_image: torch.Tensor) -> torch.Tensor:
        query_h, query_w = query_image.shape[-2:]
        if query_h % self.patch_h != 0 or query_w % self.patch_w != 0:
            raise ValueError(
                f"Input size {(query_h, query_w)} is not divisible by patch size {(self.patch_h, self.patch_w)}."
            )
        feat_h, feat_w = query_h // self.patch_h, query_w // self.patch_w
        expected_tokens = feat_h * feat_w

        with torch.no_grad():
            prompt_features = self.backbone(prompt_image)
            query_features = self.backbone(query_image)

        neck_layers = self._get_neck_layers()
        layer_logits: list[torch.Tensor] = []

        for i, neck_layer in enumerate(neck_layers):
            prompt_patch = self._extract_patch_tokens(prompt_features[i], expected_tokens)
            query_patch = self._extract_patch_tokens(query_features[i], expected_tokens)

            # difference_tokens: [B, N, C]
            difference_tokens = neck_layer(prompt_patch, query_patch)

            # token_scores: [B, N, 2], channel-0=normal, channel-1=anomaly
            token_scores = self.discrepancy_filter(difference_tokens, layer_idx=i)
            token_probs = F.softmax(token_scores/0.07, dim=-1)

            # Keep output in logit space for segmentation loss compatibility.
            token_logits = torch.logit(token_probs.clamp(min=1e-6, max=1.0 - 1e-6))
            logit_map = self._tokens_to_logit_map(token_logits, feat_h, feat_w)

            if logit_map.shape[-2:] != (query_h, query_w):
                logit_map = F.interpolate(logit_map, size=(query_h, query_w), mode="bilinear", align_corners=False)
            layer_logits.append(logit_map)

        if self.training:
            # Training path: return per-layer logits for layer-wise supervision.
            return layer_logits
        return self._fuse_layer_logits(layer_logits)


def build_model(cfg: Any) -> tuple[VPAS, nn.Module]:
    model_cfg = to_dict(get_section(cfg, "model", required=True))
    backbone_cfg = to_dict(model_cfg.pop("backbone"))
    neck_cfg = to_dict(model_cfg.pop("neck"))
    model_cfg.pop("name", None)

    backbone_name = backbone_cfg.pop("name")
    neck_name = neck_cfg.pop("name")

    backbone = build_backbone(backbone_name, **backbone_cfg)
    neck = build_neck(neck_name, **neck_cfg)
    model = VPAS(backbone=backbone, neck=neck, **model_cfg)

    loss_cfg = to_dict(get_section(cfg, "loss", required=True))
    loss_name = loss_cfg.pop("name", "segmentation_loss")
    if loss_name != "segmentation_loss":
        raise ValueError(f"Unknown loss: {loss_name}. Only 'segmentation_loss' is supported.")

    criterion = SegmentationLoss(**loss_cfg)
    return model, criterion
