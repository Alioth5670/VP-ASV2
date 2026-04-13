import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAlign(nn.Module):
    def __init__(self, tau: float = 0.07) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, prompt_feats: torch.Tensor, query_feats: torch.Tensor) -> torch.Tensor:
        # dist = -torch.cdist(query_feats, prompt_feats)
        dist = F.normalize(query_feats, dim=-1) @ F.normalize(prompt_feats, dim=-1).transpose(-1, -2)
        attn = torch.softmax(dist / self.tau, dim=-1)
        return attn @ prompt_feats


class HardAlign(nn.Module):
    def forward(self, prompt_feats: torch.Tensor, query_feats: torch.Tensor) -> torch.Tensor:
        # Nearest-neighbor match from each query token to one prompt token.
        with torch.no_grad():
            nn_idx = torch.cdist(query_feats, prompt_feats).argmin(dim=-1)

        gather_idx = nn_idx.unsqueeze(-1).expand(-1, -1, prompt_feats.size(-1))
        return prompt_feats.gather(dim=1, index=gather_idx)


class CrossAlign(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(dim)
        self.norm_prompt = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

        # VPAS uses [B, N, C] tensors, so keep batch as first dimension.
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ratio)),
            nn.GELU(),
            nn.Linear(int(dim * ratio), dim),
        )

    def forward(self, prompt_feats: torch.Tensor, query_feats: torch.Tensor) -> torch.Tensor:
        query_feats = self.norm_query(query_feats)
        prompt_feats = self.norm_prompt(prompt_feats)

        aligned_feats, _ = self.attn(query_feats, prompt_feats, prompt_feats)
        aligned_feats = self.norm_mlp(aligned_feats)
        return aligned_feats + self.ffn(aligned_feats)


class FARM(nn.Module):
    def __init__(self, align_type: str = "soft", res_type: str = "abs_diff", **kwargs) -> None:
        super().__init__()
        # Keep both "cat" and "concat" for config compatibility.
        self.res_type = "concat" if res_type == "cat" else res_type

        dim = kwargs.get("dim", 768)
        ratio = kwargs.get("ratio", 4.0)
        tau = kwargs.get("tau", 0.07)

        if align_type == "soft":
            self.align = SoftAlign(tau)
        elif align_type == "hard":
            self.align = HardAlign()
        elif align_type == "cross":
            self.align = CrossAlign(dim, ratio)
        else:
            raise ValueError(f"Unknown align_type: {align_type}")

        if self.res_type not in ["abs_diff", "concat", "norm_product"]:
            raise ValueError(f"Unknown res_type: {res_type}")
        if self.res_type == "concat":
            self.proj = nn.Linear(2 * dim, dim)


    def forward(self, prompt_feats: torch.Tensor, query_feats: torch.Tensor) -> torch.Tensor:
        # Align prompt information to query token positions.
        aligned_prompt_feats = self.align(prompt_feats, query_feats)

        if self.res_type == "abs_diff":
            return torch.abs(aligned_prompt_feats - query_feats)

        if self.res_type == "concat":
            res = torch.cat((aligned_prompt_feats, query_feats), dim=-1)
            return self.proj(res)

        norm_aligned = F.normalize(aligned_prompt_feats, dim=-1)
        norm_query = F.normalize(query_feats, dim=-1)
        return norm_aligned * norm_query
