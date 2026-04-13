from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigNode:
    def __init__(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict, got {type(data).__name__}")
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        if not key:
            return self._wrap(self._data)

        current: Any = self._data
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return self._wrap(current)

    def set(self, key: str, value: Any) -> None:
        if not key:
            raise ValueError("key must be non-empty")

        parts = key.split(".")
        current = self._data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def update(self, other: dict[str, Any]) -> None:
        self._data = self._merge_dict(self._data, other)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._data:
            return self._wrap(self._data[name])
        raise AttributeError(f"No such config key: {name}")

    def __getitem__(self, key: str) -> Any:
        sentinel = object()
        value = self.get(key, default=sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        sentinel = object()
        return self.get(key, default=sentinel) is not sentinel

    def __repr__(self) -> str:
        return f"ConfigNode({self._data!r})"

    def __str__(self) -> str:
        return repr(self._data)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return ConfigNode(value)
        return value

    @staticmethod
    def _merge_dict(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(extra, dict):
            raise TypeError(f"extra must be a dict, got {type(extra).__name__}")

        merged = deepcopy(base)
        for k, v in extra.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = ConfigNode._merge_dict(merged[k], v)
            else:
                merged[k] = v
        return merged


class ConfigParser(ConfigNode):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    @classmethod
    def _read_yaml(cls, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"Root of config file must be a mapping, got {type(data).__name__}")
        return data

    @classmethod
    def _load_with_inheritance(cls, path: Path, stack: list[Path] | None = None) -> dict[str, Any]:
        path = path.resolve()
        stack = stack or []
        if path in stack:
            chain = " -> ".join([str(p) for p in stack + [path]])
            raise ValueError(f"Circular config inheritance detected: {chain}")

        data = cls._read_yaml(path)
        base_spec = data.pop("_base_", None)
        if base_spec is None:
            return data

        if isinstance(base_spec, (str, Path)):
            base_list = [base_spec]
        elif isinstance(base_spec, list):
            base_list = base_spec
        else:
            raise TypeError(f"_base_ must be str or list, got {type(base_spec).__name__}")

        merged_base: dict[str, Any] = {}
        for base in base_list:
            base_path = (path.parent / str(base)).resolve()
            base_data = cls._load_with_inheritance(base_path, stack=stack + [path])
            merged_base = ConfigNode._merge_dict(merged_base, base_data)

        return ConfigNode._merge_dict(merged_base, data)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "ConfigParser":
        path = Path(file_path)
        data = cls._load_with_inheritance(path)
        return cls(data)
