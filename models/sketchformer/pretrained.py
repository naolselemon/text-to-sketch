"""Pretrained Sketchformer asset inspection for the native model path."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PRETRAINED_ROOT = Path(
    "weights/pretrained/sketch-transformer-tf2-cvpr_tform_cont"
)


@dataclass(frozen=True)
class TensorFlowCheckpoint:
    """A TensorFlow v1/v2 checkpoint prefix plus its sidecar files."""

    prefix: Path
    index_file: Path
    data_files: tuple[Path, ...]

    @property
    def exists(self) -> bool:
        return self.index_file.exists() and bool(self.data_files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix": str(self.prefix),
            "index_file": str(self.index_file),
            "data_files": [str(path) for path in self.data_files],
            "exists": self.exists,
        }


@dataclass(frozen=True)
class ValidationIssue:
    """A validation warning or error for pretrained assets."""

    level: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "message": self.message}


@dataclass(frozen=True)
class PretrainedSketchformerManifest:
    """Resolved pretrained Sketchformer asset manifest."""

    root: Path
    config_path: Path
    weights_dir: Path
    plots_dir: Path
    config: dict[str, Any]
    checkpoints: tuple[TensorFlowCheckpoint, ...]
    recommended_checkpoint: TensorFlowCheckpoint
    plot_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "config_path": str(self.config_path),
            "weights_dir": str(self.weights_dir),
            "plots_dir": str(self.plots_dir),
            "config": self.config,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "recommended_checkpoint": self.recommended_checkpoint.to_dict(),
            "plot_count": self.plot_count,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def resolve_project_path(project_root: str | Path, path: str | Path) -> Path:
    """Resolve a path relative to the project root."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(project_root) / candidate


def inspect_tensorflow_checkpoint(prefix: str | Path) -> TensorFlowCheckpoint:
    """Inspect a TensorFlow checkpoint prefix without importing TensorFlow."""

    checkpoint_prefix = Path(prefix)
    if checkpoint_prefix.suffix == ".index":
        checkpoint_prefix = checkpoint_prefix.with_suffix("")

    index_file = checkpoint_prefix.with_suffix(".index")
    data_files = tuple(sorted(checkpoint_prefix.parent.glob(f"{checkpoint_prefix.name}.data-*")))
    return TensorFlowCheckpoint(
        prefix=checkpoint_prefix,
        index_file=index_file,
        data_files=data_files,
    )


def fingerprint_tensorflow_checkpoint(prefix: str | Path) -> str:
    """Hash the TensorFlow index and ordered data shards without loading the model."""
    checkpoint = inspect_tensorflow_checkpoint(prefix)
    if not checkpoint.exists:
        raise FileNotFoundError(f"Incomplete TensorFlow checkpoint: {checkpoint.prefix}")
    shard_indices: set[int] = set()
    shard_count: int | None = None
    for path in checkpoint.data_files:
        match = re.fullmatch(
            rf"{re.escape(checkpoint.prefix.name)}\.data-(\d+)-of-(\d+)",
            path.name,
        )
        if match is None:
            raise ValueError(f"Unexpected TensorFlow data shard: {path}")
        index, total = map(int, match.groups())
        if total < 1 or (shard_count is not None and total != shard_count):
            raise ValueError(f"Inconsistent TensorFlow checkpoint shards: {checkpoint.prefix}")
        shard_count = total
        shard_indices.add(index)
    if shard_count is None or shard_indices != set(range(shard_count)):
        raise FileNotFoundError(f"Incomplete TensorFlow checkpoint shards: {checkpoint.prefix}")
    digest = hashlib.sha256()
    for path in (checkpoint.index_file, *checkpoint.data_files):
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, issues: list[ValidationIssue]) -> dict[str, Any]:
    if not path.exists():
        issues.append(ValidationIssue("error", f"Missing config file: {path}"))
        return {}

    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        issues.append(ValidationIssue("error", f"Invalid JSON in {path}: {exc}"))
        return {}

    if not isinstance(value, dict):
        issues.append(ValidationIssue("error", f"Expected JSON object in {path}"))
        return {}
    return value


def _discover_checkpoints(weights_dir: Path) -> tuple[TensorFlowCheckpoint, ...]:
    if not weights_dir.exists():
        return ()
    return tuple(
        inspect_tensorflow_checkpoint(index_file)
        for index_file in sorted(weights_dir.glob("*.index"))
    )


def load_pretrained_manifest(
    root: str | Path = DEFAULT_PRETRAINED_ROOT,
    *,
    project_root: str | Path = ".",
    recommended_checkpoint: str = "weights/ckpt-12",
) -> PretrainedSketchformerManifest:
    """Load and validate the pretrained Sketchformer asset manifest."""

    resolved_root = resolve_project_path(project_root, root)
    config_path = resolved_root / "config.json"
    weights_dir = resolved_root / "weights"
    plots_dir = resolved_root / "plots"
    issues: list[ValidationIssue] = []

    if not resolved_root.exists():
        issues.append(ValidationIssue("error", f"Missing pretrained root: {resolved_root}"))
    if not weights_dir.exists():
        issues.append(ValidationIssue("error", f"Missing weights directory: {weights_dir}"))
    if not plots_dir.exists():
        issues.append(ValidationIssue("warning", f"Missing plots directory: {plots_dir}"))

    config = _load_json(config_path, issues)
    checkpoints = _discover_checkpoints(weights_dir)
    if not checkpoints:
        issues.append(ValidationIssue("error", f"No TensorFlow checkpoint indexes in {weights_dir}"))

    recommended = inspect_tensorflow_checkpoint(resolved_root / recommended_checkpoint)
    if not recommended.exists:
        issues.append(
            ValidationIssue(
                "error",
                f"Recommended checkpoint is incomplete: {recommended.prefix}",
            )
        )

    for checkpoint in checkpoints:
        if not checkpoint.exists:
            issues.append(
                ValidationIssue(
                    "error",
                    f"Incomplete TensorFlow checkpoint: {checkpoint.prefix}",
                )
            )

    plot_count = len(list(plots_dir.glob("*"))) if plots_dir.exists() else 0
    return PretrainedSketchformerManifest(
        root=resolved_root,
        config_path=config_path,
        weights_dir=weights_dir,
        plots_dir=plots_dir,
        config=config,
        checkpoints=checkpoints,
        recommended_checkpoint=recommended,
        plot_count=plot_count,
        issues=tuple(issues),
    )


def validate_pretrained_assets(
    root: str | Path = DEFAULT_PRETRAINED_ROOT,
    *,
    project_root: str | Path = ".",
    recommended_checkpoint: str = "weights/ckpt-12",
) -> PretrainedSketchformerManifest:
    """Alias for loading the manifest with validation issues populated."""

    return load_pretrained_manifest(
        root,
        project_root=project_root,
        recommended_checkpoint=recommended_checkpoint,
    )


def format_validation_report(manifest: PretrainedSketchformerManifest) -> str:
    """Format a manifest as a concise terminal report."""

    lines = [
        f"root={manifest.root}",
        f"config={manifest.config_path}",
        f"weights_dir={manifest.weights_dir}",
        f"plots_dir={manifest.plots_dir}",
        f"checkpoints={len(manifest.checkpoints)}",
        f"recommended_checkpoint={manifest.recommended_checkpoint.prefix}",
        f"recommended_checkpoint_complete={manifest.recommended_checkpoint.exists}",
        f"plot_count={manifest.plot_count}",
        f"ok={manifest.ok}",
    ]

    if manifest.config:
        lines.extend(
            [
                f"model_layers={manifest.config.get('num_layers')}",
                f"d_model={manifest.config.get('d_model')}",
                f"max_seq_len={manifest.config.get('max_seq_len')}",
                f"use_continuous_data={manifest.config.get('use_continuous_data')}",
            ]
        )

    if manifest.issues:
        lines.append("issues:")
        lines.extend(f"- {issue.level}: {issue.message}" for issue in manifest.issues)
    return "\n".join(lines)
