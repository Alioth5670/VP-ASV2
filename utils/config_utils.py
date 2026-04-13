from __future__ import annotations

from typing import Any, Tuple


def to_dict(cfg: Any, *, allow_empty: bool = False) -> dict[str, Any]:
    """Convert dict-like config object to plain dict."""
    if cfg is None:
        if allow_empty:
            return {}
        raise TypeError("cfg is None")

    if isinstance(cfg, dict):
        return dict(cfg)

    if hasattr(cfg, "to_dict"):
        out = cfg.to_dict()
        if not isinstance(out, dict):
            raise TypeError(f"to_dict() must return dict, got {type(out).__name__}")
        return out

    if allow_empty:
        return {}

    raise TypeError(f"Unsupported config type: {type(cfg).__name__}")


def get_section(cfg: Any, key: str, *, default: Any = None, required: bool = False) -> Any:
    """Read section by key from dict/config-object."""
    sentinel = object()

    if isinstance(cfg, dict):
        value = cfg.get(key, sentinel)
    elif hasattr(cfg, key):
        value = getattr(cfg, key)
    elif hasattr(cfg, "get"):
        value = cfg.get(key, sentinel)
    else:
        value = sentinel

    if value is sentinel:
        if required:
            raise KeyError(f"Missing required config section: {key}")
        return default

    return value


def to_hw(value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Expected int or (H, W), got {value}")


def to_2tuple_float(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        v = float(value)
        return (v, v)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"Expected scalar or 2-tuple, got {value}")
