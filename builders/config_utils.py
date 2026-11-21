"""Small config helpers used by builder modules."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML or a frozen JSON configuration, preserving JSON numeric types."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle) if Path(path).suffix == ".json" else yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""

    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_nested(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted config path such as ``optimizer.lr``."""

    current: Any = config
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def require_nested(config: Mapping[str, Any], path: str) -> Any:
    """Read a dotted config path and raise a clear error if it is missing."""

    sentinel = object()
    value = get_nested(config, path, default=sentinel)
    if value is sentinel:
        raise KeyError(f"Missing required config value: {path}")
    return value
