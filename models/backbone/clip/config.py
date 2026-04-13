from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CLIPVisionConfig:
    image_size: int
    patch_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    layer_norm_eps: float = 1e-5


_MODELS = {
    "CLIP_ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "CLIP_ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "CLIP_ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "CLIP_ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


CLIP_MODEL_CONFIGS = {
    "CLIP_ViT-B/32": CLIPVisionConfig(224, 32, 768, 3072, 12, 12),
    "CLIP_ViT-B/16": CLIPVisionConfig(224, 16, 768, 3072, 12, 12),
    "CLIP_ViT-L/14": CLIPVisionConfig(224, 14, 1024, 4096, 24, 16),
    "CLIP_ViT-L/14@336px": CLIPVisionConfig(336, 14, 1024, 4096, 24, 16),
}


DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def is_clip_model_name(name: str) -> bool:
    return name in _MODELS


def get_clip_model_url(name: str) -> str:
    if name not in _MODELS:
        raise ValueError(f"Unknown CLIP model: {name}")
    return _MODELS[name]


def get_clip_checkpoint_filename(name: str) -> str:
    return Path(get_clip_model_url(name)).name


def get_default_checkpoint_path(name: str) -> Path:
    return DEFAULT_CHECKPOINT_DIR / get_clip_checkpoint_filename(name)
