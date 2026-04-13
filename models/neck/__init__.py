from __future__ import annotations

from typing import Any

from utils import to_dict

from .discrepancy_filter import PrototypeDiscrepancyFilter
from .farm import FARM


def build_neck(neck_name: Any = "farm", **kwargs: Any) -> FARM:
    """
    Supported call styles:
    1) build_neck("farm", align_type="hard", res_type="abs_diff", ...)
    2) build_neck({"name": "farm", "align_type": "hard", ...})
    3) build_neck(cfg.model.neck)
    """
    if isinstance(neck_name, str):
        name = neck_name
        params = dict(kwargs)
    else:
        params = to_dict(neck_name)
        name = params.pop("name", kwargs.pop("name", "farm"))
        params.update(kwargs)

    if name == "farm":
        return FARM(**params)

    raise ValueError(f"Unknown neck: {name}")


__all__ = ["FARM", "PrototypeDiscrepancyFilter", "build_neck"]
