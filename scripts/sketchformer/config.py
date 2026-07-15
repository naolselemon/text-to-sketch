"""Shared config loading for Sketchformer training scripts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch

from builders.config_utils import deep_merge, get_nested, load_yaml


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from a script location."""

    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / "pyproject.toml").exists() and (path / "configs").exists():
            return path
    raise RuntimeError("Could not find project root with pyproject.toml and configs/")


def compose_training_config(
    config_path: str | Path,
    *,
    experiment: str | None = None,
) -> dict[str, Any]:
    """Compose the lightweight Hydra-style config files used by this repo."""

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = find_project_root(Path(__file__)) / config_path
    config_dir = config_path.parent

    root_config = load_yaml(config_path)
    composed: dict[str, Any] = {}

    for entry in root_config.get("defaults", []):
        if entry == "_self_":
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Unsupported defaults entry: {entry}")
        for group, name in entry.items():
            selected = experiment if group == "experiment" and experiment else name
            selected_config = load_yaml(config_dir / group / f"{selected}.yaml")
            if group == "experiment" and "inherits" in selected_config:
                parent_name = str(selected_config.pop("inherits"))
                parent_config = load_yaml(config_dir / group / f"{parent_name}.yaml")
                selected_config = deep_merge(parent_config, selected_config)
            composed[group] = selected_config

    root_without_defaults = {
        key: value for key, value in root_config.items() if key != "defaults"
    }
    composed = deep_merge(composed, root_without_defaults)

    overrides = get_nested(composed, "experiment.overrides", {})
    for section, section_override in overrides.items():
        composed[section] = deep_merge(composed.get(section, {}), section_override)

    _sync_token_dictionary_config(composed)
    return composed


def pin_anchored_v3_artifacts(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> Path | None:
    """Resolve ``current`` once so a running job cannot switch V3 artifacts."""

    if str(get_nested(config, "data.format.type")) != "anchored_v3":
        return None
    root_value = get_nested(config, "data.dataset.root")
    if not root_value:
        raise ValueError("Anchored V3 requires data.dataset.root")
    requested_root = resolve_path(project_root, root_value)
    resolved_root = requested_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Anchored V3 root is not a directory: {resolved_root}")

    artifact_manifest = resolved_root / "manifest.jsonl"
    required = (artifact_manifest, resolved_root / "codebook.npy")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Anchored V3 artifact is incomplete: " + ", ".join(missing)
        )

    dataset_config = config["data"]["dataset"]
    configured_manifest = dataset_config.get("manifest_path")
    if configured_manifest:
        resolved_manifest = resolve_path(project_root, configured_manifest).resolve(
            strict=True
        )
        if resolved_manifest != artifact_manifest:
            raise ValueError(
                "Anchored V3 data.dataset.manifest_path must resolve to the "
                f"pinned dataset manifest {artifact_manifest}, got {resolved_manifest}"
            )

    configured_manifest_file = Path(
        str(dataset_config.get("manifest_file", "manifest.jsonl"))
    )
    manifest_from_root = (
        configured_manifest_file
        if configured_manifest_file.is_absolute()
        else resolved_root / configured_manifest_file
    ).resolve(strict=True)
    if manifest_from_root != artifact_manifest:
        raise ValueError(
            "Anchored V3 data.dataset.manifest_file must identify the pinned "
            f"dataset manifest {artifact_manifest}, got {manifest_from_root}"
        )

    try:
        pinned_root: Path = resolved_root.relative_to(project_root.resolve())
    except ValueError:
        pinned_root = resolved_root
    dataset_config["root"] = str(pinned_root)
    dataset_config["manifest_file"] = "manifest.jsonl"
    dataset_config["manifest_path"] = str(pinned_root / "manifest.jsonl")
    config["data"]["format"]["token_dictionary"]["codebook_path"] = str(
        pinned_root / "codebook.npy"
    )
    model_tokens = get_nested(config, "model.input.token_dictionary")
    if isinstance(model_tokens, dict):
        model_tokens["codebook_path"] = str(pinned_root / "codebook.npy")
    return resolved_root


def configured_minimum_source_sketches(config: dict[str, Any]) -> int:
    """Return the positive cleaned-source minimum selected by the run config."""

    value = get_nested(config, "data.dataset.minimum_source_sketches", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "data.dataset.minimum_source_sketches must be a positive integer"
        )
    return value


def apply_minimum_source_override(
    config: dict[str, Any],
    value: int | None,
) -> None:
    """Apply a CLI-selected cleaned-source contract after positive validation."""

    if value is None:
        return
    if isinstance(value, bool) or value <= 0:
        raise ValueError("--minimum-source-sketches must be positive")
    config.setdefault("data", {}).setdefault("dataset", {})[
        "minimum_source_sketches"
    ] = int(value)


def _sync_token_dictionary_config(config: dict[str, Any]) -> None:
    """Keep tok-dict data/model vocabulary IDs from drifting apart."""

    if str(get_nested(config, "data.format.type", "stroke3")) not in {
        "tok_dict",
        "token",
        "tokens",
        "anchored_v3",
    }:
        return

    token_dictionary = get_nested(config, "data.format.token_dictionary", {})
    if not token_dictionary:
        return

    model = config.setdefault("model", {})
    model_input = model.setdefault("input", {})
    existing = model_input.get("token_dictionary", {})
    model_input["token_dictionary"] = deep_merge(existing, token_dictionary)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a user-requested device string."""

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_path(project_root: Path, path: str | Path) -> Path:
    """Resolve repo-relative paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def batch_limit(loader: Iterable[Any], limit: int | float | None) -> int:
    """Resolve a trainer limit value into an integer batch count."""

    loader_len = len(loader)  # type: ignore[arg-type]
    if limit is None:
        return loader_len
    if isinstance(limit, float):
        if limit <= 0:
            return 0
        if limit <= 1:
            return max(1, int(loader_len * limit))
        return min(loader_len, int(limit))
    return min(loader_len, int(limit))


def parse_batch_limit(value: str) -> int | float:
    """Parse CLI batch limits while preserving integer semantics."""

    if "." in value:
        return float(value)
    return int(value)


def limited(loader: Iterable[Any], limit: int | float | None) -> Iterator[Any]:
    """Yield at most ``limit`` batches from a loader."""

    max_batches = batch_limit(loader, limit)
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        yield batch


def format_logs(logs: dict[str, torch.Tensor | float], *, precision: int = 4) -> str:
    """Format scalar logs for terminal output."""

    parts = []
    for key, value in sorted(logs.items()):
        scalar = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        parts.append(f"{key}={scalar:.{precision}f}")
    return " ".join(parts)
