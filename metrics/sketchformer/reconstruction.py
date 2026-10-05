"""Reconstruction reporting helpers for native Sketchformer evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metrics.sketchformer.free_running import decode_token_sequence
from pipeline.stroke5 import Stroke5Transform
from utils.tokenizer import decode_tokens


@dataclass(frozen=True)
class ReconstructionExample:
    """One target/prediction pair prepared for qualitative inspection."""

    target: np.ndarray
    prediction: np.ndarray
    length: int
    source_file: str
    source_index: int
    label: int | None = None
    sample_id: str | None = None
    source_image_path: str | None = None
    canvas_transform: Stroke5Transform | None = None
    prediction_length: int | None = None
    decode_mode: str = "teacher-forced"
    coordinate_mode: str = "target-normalized"
    statistics: Mapping[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconstructionRenderMetadata:
    """Source-image context used to render a decoded reconstruction faithfully."""

    source_image_path: str | None
    canvas_transform: Stroke5Transform


def prediction_to_stroke3(output: Any) -> torch.Tensor:
    """Convert a model output object into predicted ``(dx, dy, pen)`` strokes."""

    if output.reconstruction is None:
        raise ValueError("Model output does not include reconstruction predictions")
    if getattr(output.reconstruction, "token_logits", None) is not None:
        raise ValueError("Token predictions require a codebook; use collect_reconstruction_examples")

    xy = output.reconstruction.xy
    pen = torch.argmax(output.reconstruction.pen_logits, dim=-1).to(dtype=xy.dtype)
    return torch.cat([xy, pen.unsqueeze(-1)], dim=-1)


def _batch_lengths(batch: Mapping[str, Any], fallback_mask: torch.Tensor | None) -> list[int]:
    lengths = batch.get("lengths")
    if torch.is_tensor(lengths):
        return [int(value) for value in lengths.detach().cpu().tolist()]
    if lengths is not None:
        return [int(value) for value in lengths]
    if fallback_mask is not None:
        return [int(value) for value in fallback_mask.detach().cpu().sum(dim=1).tolist()]
    return []


def collect_reconstruction_examples(
    output: Any,
    batch: Mapping[str, Any],
    *,
    max_examples: int,
    codebook: np.ndarray | None = None,
    token_layout: Mapping[str, Any] | None = None,
    coordinate_mode: str | None = None,
) -> list[ReconstructionExample]:
    """Collect a small CPU copy of target and predicted stroke3 sequences."""

    if max_examples <= 0:
        return []

    token_logits = (
        None
        if output.reconstruction is None
        else getattr(output.reconstruction, "token_logits", None)
    )
    if token_logits is not None and codebook is None:
        raise ValueError("A tok-dict codebook is required to plot token reconstructions")

    predictions = (
        torch.argmax(token_logits, dim=-1).detach().cpu().numpy()
        if token_logits is not None
        else prediction_to_stroke3(output).detach().cpu().numpy()
    )
    # The qualitative target is always the complete stored ground-truth
    # sequence. Autoregressive loss targets omit SOS, but decode_tokens already
    # ignores SOS and the review renderer represents the complete batch target.
    target_tensor = batch["targets"]
    targets = target_tensor.detach().cpu().numpy()
    lengths = _batch_lengths(batch, batch.get("valid_mask"))
    if not lengths:
        lengths = [targets.shape[1]] * targets.shape[0]
    prediction_lengths = _batch_lengths(
        {},
        getattr(output, "loss_valid_mask", None),
    )
    predictions_are_shifted = getattr(output, "loss_targets", None) is not None

    source_files = batch.get("source_files") or [""] * targets.shape[0]
    source_indices = batch.get("source_indices")
    labels = batch.get("labels")
    sample_ids = batch.get("sample_ids")
    if torch.is_tensor(source_indices):
        source_indices = source_indices.detach().cpu().tolist()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().tolist()

    examples: list[ReconstructionExample] = []
    for row in range(min(max_examples, targets.shape[0])):
        length = min(int(lengths[row]), targets.shape[1])
        label = int(labels[row]) if labels is not None else None
        source_index = int(source_indices[row]) if source_indices is not None else row
        if token_logits is not None:
            assert codebook is not None
            assert codebook is not None
            prediction_length = min(
                (
                    int(prediction_lengths[row])
                    if prediction_lengths
                    else max(0, length - 1 if predictions_are_shifted else length)
                ),
                predictions.shape[1],
            )
            target = decode_token_sequence(
                np.asarray(targets[row, :length], dtype=np.int64),
                codebook,
                token_layout=token_layout,
            )
            prediction = decode_token_sequence(
                np.asarray(predictions[row, :prediction_length], dtype=np.int64),
                codebook,
                token_layout=token_layout,
            )
        else:
            target = np.asarray(targets[row, :length], dtype=np.float32)
            prediction = np.asarray(predictions[row, :length], dtype=np.float32)
        examples.append(
            ReconstructionExample(
                target=target,
                prediction=prediction,
                length=length,
                source_file=str(source_files[row]),
                source_index=source_index,
                label=label,
                sample_id=str(sample_ids[row]) if sample_ids is not None else None,
                prediction_length=length,
                decode_mode="teacher-forced",
                coordinate_mode=(
                    coordinate_mode
                    or ("canvas" if _is_v3_layout(token_layout) else "target-normalized")
                ),
            )
        )
    return examples


def load_reconstruction_render_metadata(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
    source_images_root: str | Path | None = None,
) -> dict[str, ReconstructionRenderMetadata]:
    """Load source paths and canvas transforms from a preprocessing JSONL manifest."""

    manifest = Path(manifest_path)
    root = Path(project_root) if project_root is not None else Path.cwd()
    image_root = (
        _resolve_path(Path(source_images_root), root)
        if source_images_root is not None
        else None
    )
    metadata: dict[str, ReconstructionRenderMetadata] = {}

    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in preprocessing manifest {manifest}:{line_number}"
                ) from exc
            if record.get("status") not in {None, "accepted"}:
                continue
            transform_payload = record.get("transform")
            if not isinstance(transform_payload, Mapping):
                continue
            try:
                transform = Stroke5Transform(**dict(transform_payload))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid transform in preprocessing manifest {manifest}:{line_number}"
                ) from exc

            source_image_path = _resolve_source_image_path(
                record,
                manifest=manifest,
                project_root=root,
                source_images_root=image_root,
            )
            render_metadata = ReconstructionRenderMetadata(
                source_image_path=(
                    str(source_image_path) if source_image_path is not None else None
                ),
                canvas_transform=transform,
            )
            for sample_key in _record_sample_keys(record):
                existing = metadata.get(sample_key)
                if existing is not None and existing != render_metadata:
                    raise ValueError(
                        "Conflicting reconstruction metadata for "
                        f"sample {sample_key!r} in {manifest}"
                    )
                metadata[sample_key] = render_metadata
    return metadata


def attach_reconstruction_render_metadata(
    examples: list[ReconstructionExample],
    metadata: Mapping[str, ReconstructionRenderMetadata],
) -> list[ReconstructionExample]:
    """Attach manifest context to examples with matching sample identifiers."""

    enriched: list[ReconstructionExample] = []
    for example in examples:
        sample_key = _normalize_sample_key(
            example.sample_id
            if example.sample_id is not None
            else str(example.source_index)
        )
        render_metadata = metadata.get(sample_key)
        if render_metadata is None:
            enriched.append(example)
            continue
        enriched.append(
            replace(
                example,
                source_image_path=render_metadata.source_image_path,
                canvas_transform=render_metadata.canvas_transform,
            )
        )
    return enriched


def _record_sample_keys(record: Mapping[str, Any]) -> set[str]:
    values = [
        record.get("sample_id"),
        record.get("source_relative_path"),
        record.get("source_path"),
    ]
    return {
        key
        for value in values
        if value is not None and (key := _normalize_sample_key(str(value)))
    }


def _normalize_sample_key(value: str) -> str:
    key = value.replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    suffix = Path(key).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".npz"}:
        key = key[: -len(suffix)]
    return key


def _resolve_source_image_path(
    record: Mapping[str, Any],
    *,
    manifest: Path,
    project_root: Path,
    source_images_root: Path | None,
) -> Path | None:
    relative = record.get("source_relative_path")
    if source_images_root is not None:
        if relative:
            return source_images_root / Path(str(relative))
        sample_id = record.get("sample_id")
        if sample_id:
            return source_images_root / f"{sample_id}.png"

    source = record.get("source_path")
    if source is None:
        return None
    path = Path(str(source))
    if path.is_absolute():
        return path

    project_candidate = project_root / path
    if project_candidate.exists():
        return project_candidate
    manifest_candidate = manifest.parent / path
    if manifest_candidate.exists():
        return manifest_candidate
    return project_candidate


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def collect_generated_reconstruction_examples(
    generation: Any,
    batch: Mapping[str, Any],
    *,
    max_examples: int,
    codebook: np.ndarray,
    token_layout: Mapping[str, Any] | None = None,
    records: list[Mapping[str, float | int | str]] | None = None,
    coordinate_mode: str | None = None,
) -> list[ReconstructionExample]:
    """Collect target/free-running prediction pairs for qualitative plots."""

    if max_examples <= 0:
        return []
    predictions = generation.tokens.detach().cpu().numpy()
    prediction_lengths = generation.lengths.detach().cpu().tolist()
    targets = batch["targets"].detach().cpu().numpy()
    lengths = _batch_lengths(batch, batch.get("valid_mask"))
    source_files = batch.get("source_files") or [""] * targets.shape[0]
    source_indices = batch.get("source_indices")
    labels = batch.get("labels")
    sample_ids = batch.get("sample_ids")
    if torch.is_tensor(source_indices):
        source_indices = source_indices.detach().cpu().tolist()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().tolist()

    examples: list[ReconstructionExample] = []
    for row in range(min(max_examples, targets.shape[0])):
        target_length = min(int(lengths[row]), targets.shape[1])
        prediction_length = min(int(prediction_lengths[row]), predictions.shape[1])
        target = decode_token_sequence(
            np.asarray(targets[row, :target_length], dtype=np.int64),
            codebook,
            token_layout=token_layout,
        )
        prediction = decode_token_sequence(
            np.asarray(predictions[row, :prediction_length], dtype=np.int64),
            codebook,
            token_layout=token_layout,
        )
        examples.append(
            ReconstructionExample(
                target=target,
                prediction=prediction,
                length=target_length,
                source_file=str(source_files[row]),
                source_index=(
                    int(source_indices[row]) if source_indices is not None else row
                ),
                label=int(labels[row]) if labels is not None else None,
                sample_id=str(sample_ids[row]) if sample_ids is not None else None,
                prediction_length=prediction_length,
                decode_mode="free-running",
                coordinate_mode=(
                    coordinate_mode
                    or ("canvas" if _is_v3_layout(token_layout) else "target-normalized")
                ),
                statistics=(dict(records[row]) if records is not None else {}),
            )
        )
    return examples


def tensor_logs_to_floats(logs: Mapping[str, torch.Tensor | float | int]) -> dict[str, float]:
    """Convert scalar tensor logs into JSON-safe floats."""

    result: dict[str, float] = {}
    for key, value in logs.items():
        if torch.is_tensor(value):
            result[key] = float(value.detach().cpu())
        else:
            result[key] = float(value)
    return result


def write_metrics_report(
    output_path: str | Path,
    logs: Mapping[str, torch.Tensor | float | int],
    *,
    metadata: Mapping[str, Any] | None = None,
    records: list[Mapping[str, Any]] | None = None,
) -> Path:
    """Write a JSON report with aggregates and optional per-sample diagnostics."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": tensor_logs_to_floats(logs),
        "metadata": dict(metadata or {}),
    }
    if records is not None:
        report["records"] = [_json_safe_mapping(record) for record in records]
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return path


def _json_safe_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            result[str(key)] = value.detach().cpu().item()
        elif isinstance(value, np.generic):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def _is_v3_layout(token_layout: Mapping[str, Any] | None) -> bool:
    layout = token_layout or {}
    return str(layout.get("version", layout.get("token_layout_version", ""))).lower() in {
        "3",
        "v3",
        "anchored_v3",
    }
