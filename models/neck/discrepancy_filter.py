from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeDiscrepancyFilter(nn.Module):
    """
    ProtoDiff: prototype-based discrepancy filtering on difference tokens.

    Inputs:
    - difference_tokens: [B, N, C]
    - layer_idx: optional layer index for layerwise prototype mode

    Outputs:
    - scores: [B, N, 2], where channel-0 is normal score and channel-1 is anomaly score
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        prototype_mode: str = "layerwise",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if prototype_mode not in {"shared", "layerwise"}:
            raise ValueError(f"Unknown prototype_mode: {prototype_mode}")

        self.prototype_mode = prototype_mode
        self.num_layers = int(num_layers)
        self.eps = float(eps)

        if self.prototype_mode == "shared":
            self.proto_normal = nn.Parameter(torch.randn(dim))
            self.proto_anomaly = nn.Parameter(torch.randn(dim))
        else:
            self.proto_normal = nn.ParameterList([nn.Parameter(torch.randn(dim)) for _ in range(self.num_layers)])
            self.proto_anomaly = nn.ParameterList([nn.Parameter(torch.randn(dim)) for _ in range(self.num_layers)])

    def _get_prototypes(self, layer_idx: Optional[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prototype_mode == "shared":
            return self.proto_normal, self.proto_anomaly

        if layer_idx is None:
            raise ValueError("layer_idx must be provided when prototype_mode='layerwise'.")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx out of range: {layer_idx}, expected [0, {self.num_layers - 1}]")
        return self.proto_normal[layer_idx], self.proto_anomaly[layer_idx]

    def forward(self, difference_tokens: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        if difference_tokens.dim() != 3:
            raise ValueError(f"Expected [B,N,C], got {tuple(difference_tokens.shape)}")

        # difference_tokens: [B, N, C]
        tokens = F.normalize(difference_tokens, dim=-1, eps=self.eps)
        proto_n, proto_a = self._get_prototypes(layer_idx)
        proto_n = F.normalize(proto_n, dim=0, eps=self.eps)  # [C]
        proto_a = F.normalize(proto_a, dim=0, eps=self.eps)  # [C]

        # Cosine similarity:
        # s_n, s_a: [B, N]
        s_n = torch.einsum("bnc,c->bn", tokens, proto_n)
        s_a = torch.einsum("bnc,c->bn", tokens, proto_a)

        # scores: [B, N, 2], [normal, anomaly]
        return torch.stack([s_n, s_a], dim=-1)
