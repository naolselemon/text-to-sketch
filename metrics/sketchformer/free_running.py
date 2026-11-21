"""Metrics for unassisted autoregressive token reconstruction."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from utils.tokenizer import decode_tokens
from metrics.sketchformer.drawing import score_drawing, target_canvas_transform, token_layout

LENGTH_BUCKETS = (
    (1, 512),
    (513, 1024),
    (1025, 2048),
    (2049, 4096),
)


@torch.no_grad()
def free_running_reconstruction_metrics(
    generated_tokens: torch.Tensor,
    generated_lengths: torch.Tensor,
    batch: Mapping[str, object],
    codebook: np.ndarray,
    *,
    eos_token_id: int,
) -> dict[str, torch.Tensor]:
    """Compare free-running output to complete token targets and decoded geometry."""

    records = free_running_reconstruction_records(
        generated_tokens,
        generated_lengths,
        batch,
        codebook,
        eos_token_id=eos_token_id,
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
    generation_limit: int | None = None,
    layout: dict | None = None,
) -> list[dict[str, float | int | str]]:
    """Return per-sketch metrics so variable batch sizes cannot bias totals."""

    targets = torch.as_tensor(batch["targets"]).detach().cpu()
    target_lengths = torch.as_tensor(batch["lengths"]).detach().cpu().long()
    generated = generated_tokens.detach().cpu().long()
    generated_lengths = generated_lengths.detach().cpu().long()
    records: list[dict[str, float | int | str]] = []

    for row in range(targets.shape[0]):
        target_length = int(target_lengths[row])
        generated_length = int(generated_lengths[row])
        target = targets[row, :target_length].numpy().astype(np.int64)
        prediction = generated[row, :generated_length].numpy().astype(np.int64)

        aligned = np.full(target_length, -1, dtype=np.int64)
        overlap = min(target_length, generated_length)
        aligned[:overlap] = prediction[:overlap]
        token_accuracy = float(np.mean(aligned == target))
        exact_match = float(
            generated_length == target_length and np.array_equal(prediction, target)
        )
        eos_rate = float(len(prediction) > 0 and prediction[-1] == eos_token_id)
        max_length_hit = float(
            generated_length >= generated.shape[1] and prediction[-1] != eos_token_id
        )

        target_stroke = decode_tokens(target, codebook)
        prediction_stroke = decode_tokens(prediction, codebook)
        drawing = score_drawing(
            generated[row].numpy(), target, codebook,
            generation_limit=generation_limit or generated.shape[1],
            layout=layout or {**token_layout(codebook), "eos_token_id": eos_token_id},
        )
        records.append(
            {
                "target_length": target_length,
                "length_bucket": _length_bucket(target_length),
                "token_accuracy": token_accuracy,
                "exact_match": exact_match,
                "eos_rate": eos_rate,
                "max_length_hit": max_length_hit,
                **drawing,
            }
        )
    return records


def aggregate_free_running_records(
    records: list[Mapping[str, float | int | str]],
    *,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Aggregate exact dataset-level means and medians."""

    def values(key: str, selected: list[Mapping[str, float | int | str]]) -> list[float]:
        return [float(record[key]) for record in selected]

    geometry_scores = values("geometry_f1_2px", records)

    metrics = {
        "free_running/token_accuracy": _tensor_mean(values("token_accuracy", records), device),
        "free_running/exact_match": _tensor_mean(values("exact_match", records), device),
        "free_running/eos_rate": _tensor_mean(values("eos_rate", records), device),
        "free_running/max_length_hit_rate": _tensor_mean(
            values("max_length_hit", records), device
        ),
        "free_running/geometry_f1_2px": _tensor_mean(geometry_scores, device),
        "free_running/geometry_f1_2px_median": _tensor_median(geometry_scores, device),
    }
    for low, high in LENGTH_BUCKETS:
        bucket = _bucket_name(low, high)
        selected = [record for record in records if record["length_bucket"] == bucket]
        bucket_values = values("geometry_f1_2px", selected)
        metrics[f"free_running/geometry_f1_2px_length_{bucket}"] = _tensor_mean(
            bucket_values,
            device,
        )
        metrics[f"free_running/geometry_f1_2px_median_length_{bucket}"] = _tensor_median(
            bucket_values,
            device,
        )
        metrics[f"free_running/count_length_{bucket}"] = torch.tensor(
            float(len(bucket_values)),
            device=device,
        )
    return metrics


def _stroke_f1(target: np.ndarray, prediction: np.ndarray, *, tolerance_px: float) -> float:
    target_lines = _stroke_lines(target)
    prediction_lines = _stroke_lines(prediction)
    if not target_lines or not prediction_lines:
        return float(not target_lines and not prediction_lines)

    scale, offset = target_canvas_transform(target_lines)
    target_canvas = _rasterize_lines(target_lines, scale, offset)
    prediction_canvas = _rasterize_lines(prediction_lines, scale, offset)
    if not target_canvas.any() or not prediction_canvas.any():
        return float(not target_canvas.any() and not prediction_canvas.any())

    target_distance = distance_transform_edt(~target_canvas)
    prediction_distance = distance_transform_edt(~prediction_canvas)
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
    canvas = np.zeros((256, 256), dtype=np.uint8)
    for line in lines:
        points = np.rint(line * scale + offset).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, 1, thickness=1, lineType=cv2.LINE_8)
    return canvas.astype(bool)


def _length_bucket(length: int) -> str:
    for low, high in LENGTH_BUCKETS:
        if low <= length <= high:
            return _bucket_name(low, high)
    return "over_4096"


def _bucket_name(low: int, high: int) -> str:
    return f"{low}_{high}"


def _tensor_mean(values: list[float], device: torch.device | str) -> torch.Tensor:
    return torch.tensor(float(np.mean(values)) if values else 0.0, device=device)


def _tensor_median(values: list[float], device: torch.device | str) -> torch.Tensor:
    return torch.tensor(float(np.median(values)) if values else 0.0, device=device)
