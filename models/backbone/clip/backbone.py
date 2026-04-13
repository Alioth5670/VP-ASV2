from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .config import CLIP_MODEL_CONFIGS
from .load import load_clip_state_dict
from .model import CLIPVisionTransformer


class CLIPImageBackbone(nn.Module):
    """
    Manual CLIP image encoder wrapper inspired by VisualAD's local CLIP build path.
    It keeps VPAS's expected interface and returns intermediate patch tokens.
    """

    def __init__(
        self,
        name: str,
        weight_path: str | None = None,
        pretrained_model_name: str | None = None,
        interaction_indexes: Iterable[int] | None = None,
        freeze: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if name not in CLIP_MODEL_CONFIGS:
            raise ValueError(f"Unknown CLIP backbone: {name}")

        if pretrained_model_name:
            print(
                "Ignoring pretrained_model_name for manual CLIP backbone; "
                "please provide a local checkpoint path in `weights` if you want pretrained parameters."
            )
        if local_files_only:
            print("local_files_only has no effect in the manual CLIP backbone implementation.")

        self.vision_transformer = CLIPVisionTransformer(CLIP_MODEL_CONFIGS[name])
        self.interaction_indexes = list(interaction_indexes or [])
        self.patch_size = self.vision_transformer.patch_size

        if weight_path is not None:
            print(f"Loading CLIP weights from {weight_path}")
            state_dict = load_clip_state_dict(weight_path)
            missing, unexpected = self.vision_transformer.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Missing CLIP keys: {missing}")
            if unexpected:
                print(f"Unexpected CLIP keys: {unexpected}")
        elif freeze:
            raise ValueError("Manual CLIP backbone requires local `weights` when freeze=True.")
        else:
            print(f"Training {name} from scratch")

        if freeze:
            self.vision_transformer.eval()
            for param in self.vision_transformer.parameters():
                param.requires_grad = False

    def _validate_interaction_indexes(self, num_hidden_layers: int) -> None:
        for layer_idx in self.interaction_indexes:
            if layer_idx < 0 or layer_idx >= num_hidden_layers:
                raise ValueError(
                    f"interaction index {layer_idx} is out of range for CLIP with "
                    f"{num_hidden_layers} hidden layers."
                )

    @staticmethod
    def _split_tokens(hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_token = hidden_state[:, 0, :]
        patch_tokens = hidden_state[:, 1:, :]
        return patch_tokens, cls_token

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        need_hidden_states = len(self.interaction_indexes) > 0
        outputs = self.vision_transformer(
            pixel_values=x,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )

        if not need_hidden_states:
            return [self._split_tokens(outputs["last_hidden_state"])]

        hidden_states = outputs["hidden_states"]
        if hidden_states is None:
            raise RuntimeError("CLIP vision model did not return hidden states.")

        num_hidden_layers = len(hidden_states) - 1
        self._validate_interaction_indexes(num_hidden_layers)
        return [self._split_tokens(hidden_states[layer_idx + 1]) for layer_idx in self.interaction_indexes]
