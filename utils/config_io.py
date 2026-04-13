from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def save_resolved_config(cfg: Any, path: str | Path) -> Path:
    if hasattr(cfg, "to_dict"):
        data = cfg.to_dict()
    elif isinstance(cfg, dict):
        data = dict(cfg)
    else:
        raise TypeError(f"Unsupported config type: {type(cfg).__name__}")

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return save_path
