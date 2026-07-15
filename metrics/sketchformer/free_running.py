"""Dataset-level diagnostics for unassisted autoregressive reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from utils.tokenizer import decode_tokens

LENGTH_BUCKETS = (
    (1, 512),
    (513, 1024),
    (1025, 2048),
    (2049, 4096),
)
CANVAS_SIZE = 256
CANVAS_MARGIN = 8


@torch.no_grad()
def free_running_reconstruction_metrics(
    generated_tokens: torch.Tensor,
    generated_lengths: torch.Tensor,
    batch: Mapping[str, object],
    codebook: np.ndarray,
    *,
    eos_token_id: int,
    token_layout: Mapping[str, Any] | None = None,
    coordinate_mode: str | None = None,
) -> dict[str, torch.Tensor]:
    """Compare free-running output to complete targets and decoded geometry."""

    records = free_running_reconstruction_records(
        generated_tokens,
        generated_lengths,
        batch,
        codebook,
        eos_token_id=eos_token_id,
        token_layout=token_layout,
        coordinate_mode=coordinate_mode,
    )
    return aggregate_free_running_records(records, device=generated_tokens.device)


@torch.no_grad()
def free_running_reconstruction_records(
    generated_tokens: torch.Tensor,
    generated_lengths: torch.Tensor,
    batch: Mapping[str, object],
    codebook: np.ndarray,
    *,
    eos_token_id: int,
    token_layout: Mapping[str, Any] | None = None,
    coordinate_mode: str | None = None,
) -> list[dict[str, float | int | str]]:
    """Return per-sketch metrics so variable batch sizes cannot bias totals."""

    targets = torch.as_tensor(batch["targets"]).detach().cpu()
    target_lengths = torch.as_tensor(batch["lengths"]).detach().cpu().long()
    generated = generated_tokens.detach().cpu().long()
    generated_lengths = generated_lengths.detach().cpu().long()
    layout = dict(token_layout or {})
    motion_start, motion_end = _motion_token_range(layout, len(codebook))
    structure_ids = _structure_token_ids(layout, len(codebook))
    stroke_counter_ids = _stroke_counter_token_ids(layout, len(codebook))
    geometry_mode = coordinate_mode or (
        "canvas" if _is_anchored_v3(layout) else "target-normalized"
    )
    records: list[dict[str, float | int | str]] = []
    sample_ids = batch.get("sample_ids")
    source_indices = batch.get("source_indices")
    if torch.is_tensor(source_indices):
        source_indices = source_indices.detach().cpu().tolist()

    for row in range(targets.shape[0]):
        target_length = int(target_lengths[row])
        generated_length = min(int(generated_lengths[row]), generated.shape[1])
        target = targets[row, :target_length].numpy().astype(np.int64)
        prediction = generated[row, :generated_length].numpy().astype(np.int64)

        overlap = min(target_length, generated_length)
        matches = int(np.count_nonzero(prediction[:overlap] == target[:overlap]))
        token_accuracy = matches / max(target_length, 1)
        exact_match = float(
            generated_length == target_length and np.array_equal(prediction, target)
        )
        target_eos_position = _first_position(target, eos_token_id)
        eos_position = _first_position(prediction, eos_token_id)
        eos_rate = float(eos_position >= 0)
        generation_limit = target_length
        max_length_hit = float(eos_position < 0 and generated_length >= generation_limit)
        premature_eos = float(
            eos_position >= 0
            and target_eos_position >= 0
            and eos_position < target_eos_position
        )

        target_stroke_count = _count_ids(target, stroke_counter_ids)
        generated_stroke_count = _count_ids(prediction, stroke_counter_ids)
        target_structure_count = _count_ids(target, structure_ids)
        generated_structure_count = _count_ids(prediction, structure_ids)
        motion = prediction[
            (prediction >= motion_start) & (prediction < motion_end)
        ]
        unique_motion_ratio = (
            float(len(np.unique(motion)) / len(motion)) if len(motion) else 0.0
        )

        target_stroke = decode_token_sequence(target, codebook, token_layout=layout)
        prediction_stroke = decode_token_sequence(
            prediction,
            codebook,
            token_layout=layout,
        )
        geometry = stroke_geometry_metrics(
            target_stroke,
            prediction_stroke,
            coordinate_mode=geometry_mode,
        )
        records.append(
            {
                "sample_id": (
                    str(sample_ids[row])
                    if sample_ids is not None
                    else str(source_indices[row] if source_indices is not None else row)
                ),
                "target_length": target_length,
                "generated_length": generated_length,
                "length_ratio": generated_length / max(target_length, 1),
                "absolute_length_error": abs(generated_length - target_length),
                "length_bucket": _length_bucket(target_length),
                "token_accuracy": token_accuracy,
                "exact_match": exact_match,
                "target_eos_position": target_eos_position,
                "eos_position": eos_position,
                "eos_rate": eos_rate,
                "premature_eos": premature_eos,
                "max_length_hit": max_length_hit,
                "target_stroke_count": target_stroke_count,
                "generated_stroke_count": generated_stroke_count,
                "stroke_count_error": abs(generated_stroke_count - target_stroke_count),
                "target_structure_count": target_structure_count,
                "generated_structure_count": generated_structure_count,
                "structure_count_error": abs(
                    generated_structure_count - target_structure_count
                ),
                "unique_motion_ratio": unique_motion_ratio,
                "longest_repeated_token_run": _longest_run(prediction),
                "first_divergence_position": _first_divergence(target, prediction),
                **geometry,
            }
        )
    return records


def aggregate_free_running_records(
    records: list[Mapping[str, float | int | str]],
    *,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Aggregate exact dataset-level means, quantiles, and length buckets."""

    def values(key: str, selected: list[Mapping[str, float | int | str]]) -> list[float]:
        return [float(record[key]) for record in selected]

    def mean_metric(key: str) -> torch.Tensor:
        return _tensor_mean(values(key, records), device)

    geometry_1px = values("geometry_f1_1px", records)
    geometry_2px = values("geometry_f1_2px", records)
    chamfer = values("symmetric_chamfer_px", records)
    present_eos = [
        float(record["eos_position"])
        for record in records
        if float(record["eos_position"]) >= 0
    ]
    present_divergence = [
        float(record["first_divergence_position"])
        for record in records
        if float(record["first_divergence_position"]) >= 0
    ]

    metrics = {
        "free_running/count": torch.tensor(float(len(records)), device=device),
        "free_running/token_accuracy": mean_metric("token_accuracy"),
        "free_running/exact_match": mean_metric("exact_match"),
        "free_running/target_length_mean": mean_metric("target_length"),
        "free_running/generated_length_mean": mean_metric("generated_length"),
        "free_running/length_ratio_mean": mean_metric("length_ratio"),
        "free_running/absolute_length_error_mean": mean_metric("absolute_length_error"),
        "free_running/eos_rate": mean_metric("eos_rate"),
        "free_running/eos_position_mean": _tensor_mean(present_eos, device),
        "free_running/premature_eos_rate": mean_metric("premature_eos"),
        "free_running/max_length_hit_rate": mean_metric("max_length_hit"),
        "free_running/target_stroke_count_mean": mean_metric("target_stroke_count"),
        "free_running/generated_stroke_count_mean": mean_metric("generated_stroke_count"),
        "free_running/stroke_count_error_mean": mean_metric("stroke_count_error"),
        "free_running/target_structure_count_mean": mean_metric(
            "target_structure_count"
        ),
        "free_running/generated_structure_count_mean": mean_metric(
            "generated_structure_count"
        ),
        "free_running/structure_count_error_mean": mean_metric(
            "structure_count_error"
        ),
        "free_running/unique_motion_ratio_mean": mean_metric("unique_motion_ratio"),
        "free_running/longest_repeated_token_run_mean": mean_metric(
            "longest_repeated_token_run"
        ),
        "free_running/first_divergence_position_mean": _tensor_mean(
            present_divergence,
            device,
        ),
        "free_running/geometry_f1_1px": _tensor_mean(geometry_1px, device),
        "free_running/geometry_f1_1px_median": _tensor_quantile(
            geometry_1px, 0.5, device
        ),
        "free_running/geometry_f1_2px": _tensor_mean(geometry_2px, device),
        "free_running/geometry_f1_2px_median": _tensor_quantile(
            geometry_2px, 0.5, device
        ),
        "free_running/geometry_f1_2px_p10": _tensor_quantile(
            geometry_2px, 0.1, device
        ),
        "free_running/symmetric_chamfer_px": _tensor_mean(chamfer, device),
        "free_running/symmetric_chamfer_px_p95": _tensor_quantile(
            chamfer, 0.95, device
        ),
    }
    bucket_medians: list[float] = []
    for low, high in LENGTH_BUCKETS:
        bucket = _bucket_name(low, high)
        selected = [record for record in records if record["length_bucket"] == bucket]
        bucket_f1_1px = values("geometry_f1_1px", selected)
        bucket_f1_2px = values("geometry_f1_2px", selected)
        bucket_chamfer = values("symmetric_chamfer_px", selected)
        # Empty buckets contribute zero. Silently dropping them would make a
        # one-bucket model look like a four-bucket macro score.
        bucket_medians.append(
            float(np.median(bucket_f1_2px)) if bucket_f1_2px else 0.0
        )
        metrics[f"free_running/geometry_f1_1px_median_length_{bucket}"] = (
            _tensor_quantile(bucket_f1_1px, 0.5, device)
        )
        metrics[f"free_running/geometry_f1_2px_length_{bucket}"] = _tensor_mean(
            bucket_f1_2px,
            device,
        )
        metrics[f"free_running/geometry_f1_2px_median_length_{bucket}"] = (
            _tensor_quantile(bucket_f1_2px, 0.5, device)
        )
        metrics[f"free_running/symmetric_chamfer_px_p95_length_{bucket}"] = (
            _tensor_quantile(bucket_chamfer, 0.95, device)
        )
        metrics[f"free_running/count_length_{bucket}"] = torch.tensor(
            float(len(bucket_f1_2px)),
            device=device,
        )
    metrics["free_running/macro_geometry_f1_2px_median"] = _tensor_mean(
        bucket_medians,
        device,
    )
    return metrics


def decode_token_sequence(
    tokens: np.ndarray,
    codebook: np.ndarray,
    *,
    token_layout: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Decode either the legacy relative grammar or anchored V3 grammar."""

    layout = dict(token_layout or {})
    if _is_anchored_v3(layout):
        return _decode_anchored_v3(tokens, codebook, layout)
    return decode_tokens(
        tokens,
        codebook,
        motion_token_offset=int(layout.get("motion_token_offset", 1)),
        pad_token_id=int(layout.get("pad_token_id", 0)),
        sep_token_id=_optional_int(layout.get("sep_token_id")),
        sos_token_id=_optional_int(layout.get("sos_token_id")),
        eos_token_id=_optional_int(layout.get("eos_token_id")),
    )


def stroke_geometry_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    coordinate_mode: str = "target-normalized",
) -> dict[str, float]:
    """Measure fixed-canvas raster F1 and symmetric Chamfer distance."""

    target_lines = _stroke_lines(target)
    prediction_lines = _stroke_lines(prediction)
    target_canvas, prediction_canvas = shared_raster_canvases(
        target_lines,
        prediction_lines,
        coordinate_mode=coordinate_mode,
    )
    if not target_canvas.any() and not prediction_canvas.any():
        return {
            "geometry_f1_1px": 1.0,
            "geometry_f1_2px": 1.0,
            "symmetric_chamfer_px": 0.0,
        }
    if not target_canvas.any() or not prediction_canvas.any():
        return {
            "geometry_f1_1px": 0.0,
            "geometry_f1_2px": 0.0,
            "symmetric_chamfer_px": float(np.hypot(CANVAS_SIZE, CANVAS_SIZE)),
        }

    target_distance = distance_transform_edt(~target_canvas)
    prediction_distance = distance_transform_edt(~prediction_canvas)
    return {
        "geometry_f1_1px": _canvas_f1(
            target_canvas,
            prediction_canvas,
            target_distance,
            prediction_distance,
            tolerance_px=1.0,
        ),
        "geometry_f1_2px": _canvas_f1(
            target_canvas,
            prediction_canvas,
            target_distance,
            prediction_distance,
            tolerance_px=2.0,
        ),
        "symmetric_chamfer_px": float(
            0.5
            * (
                np.mean(target_distance[prediction_canvas])
                + np.mean(prediction_distance[target_canvas])
            )
        ),
    }


def shared_raster_canvases(
    target_lines: list[np.ndarray],
    prediction_lines: list[np.ndarray],
    *,
    coordinate_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize target and prediction using one fixed 256-pixel transform."""

    if coordinate_mode not in {"canvas", "target-normalized"}:
        raise ValueError(
            "coordinate_mode must be 'canvas' or 'target-normalized', "
            f"got {coordinate_mode!r}"
        )
    if coordinate_mode == "canvas":
        scale = 1.0
        offset = np.zeros(2, dtype=np.float32)
    elif target_lines:
        points = np.concatenate(target_lines, axis=0)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        available = float(CANVAS_SIZE - 2 * CANVAS_MARGIN - 1)
        scale = available / max(float(np.max(maximum - minimum)), 1e-6)
        offset = np.full(2, CANVAS_MARGIN, dtype=np.float32) - minimum * scale
    else:
        scale = 1.0
        offset = np.zeros(2, dtype=np.float32)
    return (
        _rasterize_lines(target_lines, scale, offset),
        _rasterize_lines(prediction_lines, scale, offset),
    )


def _decode_anchored_v3(
    tokens: np.ndarray,
    codebook: np.ndarray,
    layout: Mapping[str, Any],
) -> np.ndarray:
    motion_start, motion_end = _motion_token_range(layout, len(codebook))
    x_start = int(layout.get("x_token_offset", layout.get("x_token_start", 2049)))
    y_start = int(layout.get("y_token_offset", layout.get("y_token_start", 2305)))
    coordinate_bins = int(layout.get("coordinate_bins", 256))
    stroke_start = int(layout.get("stroke_start_token_id", 2561))
    stroke_end = int(layout.get("stroke_end_token_id", 2562))
    eos = int(layout.get("eos_token_id", 2564))
    rows: list[list[float]] = []
    position = np.zeros(2, dtype=np.float32)
    pending_x: float | None = None
    expecting_x = False
    expecting_y = False
    in_stroke = False

    for raw_token in np.asarray(tokens).reshape(-1):
        token = int(raw_token)
        if token == eos:
            break
        if token == stroke_start:
            pending_x = None
            expecting_x = True
            expecting_y = False
            in_stroke = False
            continue
        if expecting_x and x_start <= token < x_start + coordinate_bins:
            pending_x = float(token - x_start)
            expecting_x = False
            expecting_y = True
            continue
        if expecting_y and y_start <= token < y_start + coordinate_bins:
            anchor = np.asarray([pending_x or 0.0, float(token - y_start)], dtype=np.float32)
            relocation = anchor - position
            rows.append([float(relocation[0]), float(relocation[1]), 0.0, 1.0, 0.0])
            rows.append([0.0, 0.0, 1.0, 0.0, 0.0])
            position = anchor
            expecting_y = False
            in_stroke = True
            continue
        if in_stroke and motion_start <= token < motion_end:
            delta = np.asarray(codebook[token - motion_start], dtype=np.float32)
            rows.append([float(delta[0]), float(delta[1]), 1.0, 0.0, 0.0])
            position += delta
            continue
        if token == stroke_end and in_stroke:
            if rows:
                rows[-1][2] = 0.0
                rows[-1][3] = 1.0
            in_stroke = False

    rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
    return np.asarray(rows, dtype=np.float32)


def _canvas_f1(
    target_canvas: np.ndarray,
    prediction_canvas: np.ndarray,
    target_distance: np.ndarray,
    prediction_distance: np.ndarray,
    *,
    tolerance_px: float,
) -> float:
    precision = float(np.mean(target_distance[prediction_canvas] <= tolerance_px))
    recall = float(np.mean(prediction_distance[target_canvas] <= tolerance_px))
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def _stroke_lines(stroke5: np.ndarray) -> list[np.ndarray]:
    array = np.asarray(stroke5, dtype=np.float32)
    if len(array) == 0:
        return []
    points = np.cumsum(array[:, :2], axis=0)
    lines: list[np.ndarray] = []
    start = 0
    for index, row in enumerate(array):
        if row[4] >= 0.5:
            if index - start >= 2:
                lines.append(points[start:index])
            break
        if row[3] >= 0.5:
            if index + 1 - start >= 2:
                lines.append(points[start : index + 1])
            start = index + 1
    return lines


def _rasterize_lines(
    lines: list[np.ndarray],
    scale: float,
    offset: np.ndarray,
) -> np.ndarray:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    for line in lines:
        points = np.rint(line * scale + offset).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, 1, thickness=1, lineType=cv2.LINE_8)
    return canvas.astype(bool)


def _motion_token_range(layout: Mapping[str, Any], codebook_size: int) -> tuple[int, int]:
    start = int(layout.get("motion_token_offset", layout.get("motion_token_start", 1)))
    # V3 contract ranges are inclusive; the internal comparison is half-open.
    end = (
        int(layout["motion_token_end"]) + 1
        if "motion_token_end" in layout
        else start + codebook_size
    )
    return start, end


def _structure_token_ids(layout: Mapping[str, Any], codebook_size: int) -> set[int]:
    if _is_anchored_v3(layout):
        return {
            int(layout.get("stroke_start_token_id", 2561)),
            int(layout.get("stroke_end_token_id", 2562)),
        }
    return {int(layout.get("sep_token_id", codebook_size + 1))}


def _stroke_counter_token_ids(layout: Mapping[str, Any], codebook_size: int) -> set[int]:
    if _is_anchored_v3(layout):
        return {int(layout.get("stroke_end_token_id", 2562))}
    return {int(layout.get("sep_token_id", codebook_size + 1))}


def _is_anchored_v3(layout: Mapping[str, Any]) -> bool:
    format_type = str(layout.get("type", layout.get("format_type", ""))).lower()
    version = str(layout.get("version", layout.get("token_layout_version", ""))).lower()
    return format_type == "anchored_v3" or version in {"3", "v3", "anchored_v3"}


def _first_position(tokens: np.ndarray, token_id: int) -> int:
    positions = np.flatnonzero(tokens == token_id)
    return int(positions[0]) if len(positions) else -1


def _count_ids(tokens: np.ndarray, token_ids: set[int]) -> int:
    return int(sum(np.count_nonzero(tokens == token_id) for token_id in token_ids))


def _longest_run(tokens: np.ndarray) -> int:
    if len(tokens) == 0:
        return 0
    changes = np.flatnonzero(np.diff(tokens) != 0) + 1
    boundaries = np.concatenate(([0], changes, [len(tokens)]))
    return int(np.max(np.diff(boundaries)))


def _first_divergence(target: np.ndarray, prediction: np.ndarray) -> int:
    overlap = min(len(target), len(prediction))
    differing = np.flatnonzero(target[:overlap] != prediction[:overlap])
    if len(differing):
        return int(differing[0])
    return -1 if len(target) == len(prediction) else overlap


def _length_bucket(length: int) -> str:
    for low, high in LENGTH_BUCKETS:
        if low <= length <= high:
            return _bucket_name(low, high)
    return "over_4096"


def _bucket_name(low: int, high: int) -> str:
    return f"{low}_{high}"


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _tensor_mean(values: list[float], device: torch.device | str) -> torch.Tensor:
    return torch.tensor(float(np.mean(values)) if values else 0.0, device=device)


def _tensor_quantile(
    values: list[float],
    quantile: float,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.tensor(
        float(np.quantile(values, quantile)) if values else 0.0,
        device=device,
    )
