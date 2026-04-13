from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLIPVisionConfig


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = embeddings.shape[1] - 1
        base_num_patches = self.num_positions - 1
        if num_patches == base_num_patches and height == width == self.image_size:
            return self.position_embedding(self.position_ids)

        class_pos_embed = self.position_embedding.weight[:1]
        patch_pos_embed = self.position_embedding.weight[1:]
        dim = embeddings.shape[-1]

        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        base_grid = int(math.sqrt(base_num_patches))
        if base_grid * base_grid != base_num_patches:
            raise ValueError(f"CLIP positional embedding patch count {base_num_patches} is not a square.")

        patch_pos_embed = patch_pos_embed.reshape(1, base_grid, base_grid, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(grid_h, grid_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, dim)
        return torch.cat([class_pos_embed.unsqueeze(0), patch_pos_embed], dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = self.interpolate_pos_encoding(embeddings, height, width)
        return embeddings + position_embedding.to(embeddings.dtype)


class CLIPAttention(nn.Module):
    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")

        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self._shape(self.q_proj(hidden_states), batch_size, seq_len)
        key_states = self._shape(self.k_proj(hidden_states), batch_size, seq_len)
        value_states = self._shape(self.v_proj(hidden_states), batch_size, seq_len)

        attn_weights = torch.matmul(query_states * self.scale, key_states.transpose(-1, -2))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__()
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        all_hidden_states = [hidden_states] if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        return hidden_states, tuple(all_hidden_states) if all_hidden_states is not None else None


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dim = config.hidden_size
        self.patch_size = (config.patch_size, config.patch_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.pre_layrnorm(self.embeddings(pixel_values))
        last_hidden_state, hidden_states_tuple = self.encoder(
            hidden_states,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = self.post_layernorm(last_hidden_state[:, 0, :])

        if not return_dict:
            return last_hidden_state, pooled_output, hidden_states_tuple

        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooled_output,
            "hidden_states": hidden_states_tuple,
        }
