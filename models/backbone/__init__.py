from .clip import CLIPImageBackbone
from .clip.config import is_clip_model_name
from .dinov3 import DINOv3Backbone


def build_backbone(backbone_name, **kwargs):
    if "dinov3" in backbone_name:
        weights = kwargs.get("weights", None)
        interaction_indexes = kwargs.get("interaction_indexes", [])
        freeze = kwargs.get("freeze", True)
        return DINOv3Backbone(
            backbone_name,
            weight_path=weights,
            interaction_indexes=interaction_indexes,
            freeze=freeze,
        )

    if is_clip_model_name(backbone_name):
        weights = kwargs.get("weights", None)
        pretrained_model_name = kwargs.get("pretrained_model_name", None)
        interaction_indexes = kwargs.get("interaction_indexes", [])
        freeze = kwargs.get("freeze", True)
        local_files_only = kwargs.get("local_files_only", False)
        return CLIPImageBackbone(
            backbone_name,
            weight_path=weights,
            pretrained_model_name=pretrained_model_name,
            interaction_indexes=interaction_indexes,
            freeze=freeze,
            local_files_only=local_files_only,
        )

    raise ValueError(f"Unknown backbone: {backbone_name}")
