from __future__ import annotations

from collections.abc import Mapping

import torch


def _extract_state_dict_from_mapping(checkpoint: Mapping[str, object]) -> dict[str, torch.Tensor]:
    for key in ("state_dict", "model_state_dict", "model"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            checkpoint = nested
            break

    state_dict = dict(checkpoint)
    if not all(isinstance(key, str) for key in state_dict):
        raise ValueError("Checkpoint keys must be strings.")
    return state_dict  # type: ignore[return-value]


def unwrap_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        return _extract_state_dict_from_mapping(checkpoint)

    if hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
        if isinstance(state_dict, Mapping):
            return dict(state_dict)

    raise ValueError("Unsupported CLIP checkpoint format.")


def strip_prefix_if_present(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


def convert_openai_visual_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith("visual."):
            continue

        new_key = key[len("visual.") :]
        if new_key == "class_embedding":
            converted["embeddings.class_embedding"] = value
            continue
        if new_key == "positional_embedding":
            converted["embeddings.position_embedding.weight"] = value
            continue
        if new_key == "conv1.weight":
            converted["embeddings.patch_embedding.weight"] = value
            continue
        if new_key == "ln_pre.weight":
            converted["pre_layrnorm.weight"] = value
            continue
        if new_key == "ln_pre.bias":
            converted["pre_layrnorm.bias"] = value
            continue
        if new_key == "ln_post.weight":
            converted["post_layernorm.weight"] = value
            continue
        if new_key == "ln_post.bias":
            converted["post_layernorm.bias"] = value
            continue

        if new_key.startswith("transformer.resblocks."):
            parts = new_key.split(".")
            layer_idx = parts[2]
            suffix = ".".join(parts[3:])
            prefix = f"encoder.layers.{layer_idx}."
            if suffix == "attn.in_proj_weight":
                q_weight, k_weight, v_weight = value.chunk(3, dim=0)
                converted[prefix + "self_attn.q_proj.weight"] = q_weight
                converted[prefix + "self_attn.k_proj.weight"] = k_weight
                converted[prefix + "self_attn.v_proj.weight"] = v_weight
                continue
            if suffix == "attn.in_proj_bias":
                q_bias, k_bias, v_bias = value.chunk(3, dim=0)
                converted[prefix + "self_attn.q_proj.bias"] = q_bias
                converted[prefix + "self_attn.k_proj.bias"] = k_bias
                converted[prefix + "self_attn.v_proj.bias"] = v_bias
                continue

            mapping = {
                "attn.out_proj.weight": prefix + "self_attn.out_proj.weight",
                "attn.out_proj.bias": prefix + "self_attn.out_proj.bias",
                "ln_1.weight": prefix + "layer_norm1.weight",
                "ln_1.bias": prefix + "layer_norm1.bias",
                "ln_2.weight": prefix + "layer_norm2.weight",
                "ln_2.bias": prefix + "layer_norm2.bias",
                "mlp.c_fc.weight": prefix + "mlp.fc1.weight",
                "mlp.c_fc.bias": prefix + "mlp.fc1.bias",
                "mlp.c_proj.weight": prefix + "mlp.fc2.weight",
                "mlp.c_proj.bias": prefix + "mlp.fc2.bias",
            }
            mapped_key = mapping.get(suffix)
            if mapped_key is not None:
                converted[mapped_key] = value
    return converted


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = strip_prefix_if_present(state_dict, "module.")

    if any(key.startswith("visual.") for key in state_dict):
        return convert_openai_visual_state_dict(state_dict)

    state_dict = strip_prefix_if_present(state_dict, "vision_model.")
    state_dict.pop("position_ids", None)
    state_dict.pop("embeddings.position_ids", None)
    return state_dict


def load_clip_state_dict(weight_path: str) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    except RuntimeError:
        checkpoint = torch.jit.load(weight_path, map_location="cpu")
    state_dict = unwrap_checkpoint(checkpoint)
    return normalize_state_dict(state_dict)
