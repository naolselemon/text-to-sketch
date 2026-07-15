"""Qualitative reconstruction plots for native Sketchformer evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from metrics.sketchformer.reconstruction import ReconstructionExample

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
from pipeline.stroke5 import (
    Stroke5Transform,
    stroke5_to_canvas_strokes,
)
from pipeline.vectorization import rasterize_strokes, read_grayscale_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_FALLBACK_CANVAS_SHAPE = (512, 512)
_FALLBACK_MARGIN = 16


def _stroke_array_and_pen_lift(strokes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return stroke deltas and pen-lift mask for stroke3 or stroke5 arrays."""

    array = np.asarray(strokes, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] not in {3, 5}:
        raise ValueError(
            "Expected stroke array with shape (N, 3) or (N, 5), "
            f"got {array.shape}"
        )
    if array.shape[1] == 5:
        end_mask = array[:, 4] >= 0.5
        pen_lift = np.asarray((array[:, 3] >= 0.5) | end_mask, dtype=bool)
        return array[:, :2], pen_lift

    pen_lift = np.asarray(array[:, 2] >= 0.5, dtype=bool)
    return array[:, :2], pen_lift


def stroke3_to_points(stroke3: np.ndarray) -> np.ndarray:
    """Convert relative stroke3 or stroke5 deltas into absolute xy points."""

    deltas, _ = _stroke_array_and_pen_lift(stroke3)
    return np.cumsum(deltas, axis=0)


def _plot_stroke3(ax: plt.Axes, stroke3: np.ndarray, title: str) -> None:
    """Retained normalized plot helper; singleton pen-up moves stay invisible."""

    stroke5 = _as_stroke5(stroke3)
    transform = _fallback_transform(stroke5)
    image = rasterize_decoded_strokes(
        stroke5,
        transform,
        _FALLBACK_CANVAS_SHAPE,
    )
    _plot_raster(ax, image, title)


def rasterize_decoded_strokes(
    strokes: np.ndarray,
    transform: Stroke5Transform,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Render decoded strokes with the same rasterizer as preprocessing review."""

    stroke5 = _as_stroke5(strokes)
    canvas_strokes = stroke5_to_canvas_strokes(stroke5, transform)
    return np.where(
        rasterize_strokes(canvas_strokes, image_shape),
        0,
        255,
    ).astype(np.uint8)


def _as_stroke5(strokes: np.ndarray) -> np.ndarray:
    array = np.asarray(strokes, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] not in {3, 5}:
        raise ValueError(
            "Expected stroke array with shape (N, 3) or (N, 5), "
            f"got {array.shape}"
        )
    if array.shape[1] == 5:
        return array

    pen_lift = array[:, 2] >= 0.5
    rows = np.zeros((len(array) + 1, 5), dtype=np.float32)
    rows[:-1, :2] = array[:, :2]
    rows[:-1, 2] = (~pen_lift).astype(np.float32)
    rows[:-1, 3] = pen_lift.astype(np.float32)
    rows[-1, 4] = 1.0
    return rows


def _visible_lines(stroke5: np.ndarray) -> list[np.ndarray]:
    points = np.cumsum(stroke5[:, :2], axis=0)
    lines: list[np.ndarray] = []
    start = 0
    for index, row in enumerate(stroke5):
        if row[4] >= 0.5:
            if index - start >= 2:
                lines.append(points[start:index])
            break
        if row[3] >= 0.5:
            if index + 1 - start >= 2:
                lines.append(points[start : index + 1])
            start = index + 1
    else:
        if len(points) - start >= 2:
            lines.append(points[start:])
    return lines


def _fallback_transform(stroke5: np.ndarray) -> Stroke5Transform:
    lines = _visible_lines(stroke5)
    if lines:
        visible_points = np.concatenate(lines, axis=0)
    else:
        rows = stroke5[stroke5[:, 4] < 0.5]
        visible_points = (
            np.cumsum(rows[:, :2], axis=0)
            if len(rows)
            else np.zeros((1, 2), dtype=np.float32)
        )

    minimum = visible_points.min(axis=0)
    maximum = visible_points.max(axis=0)
    height, width = _FALLBACK_CANVAS_SHAPE
    drawable_width = max(1, width - 2 * _FALLBACK_MARGIN)
    drawable_height = max(1, height - 2 * _FALLBACK_MARGIN)
    extent = maximum - minimum
    scale = min(
        drawable_width / max(float(extent[0]), 1e-6),
        drawable_height / max(float(extent[1]), 1e-6),
    )
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    start_x = float(_FALLBACK_MARGIN - minimum[0] * scale)
    start_y = float(_FALLBACK_MARGIN - minimum[1] * scale)
    return Stroke5Transform(
        canvas_height=height,
        canvas_width=width,
        start_x=start_x,
        start_y=start_y,
        bbox_x=float(minimum[0]),
        bbox_y=float(minimum[1]),
        bbox_width=float(extent[0]),
        bbox_height=float(extent[1]),
        scale=float(scale),
        normalization_extent=1.0,
    )


def _load_source_image(example: ReconstructionExample) -> np.ndarray | None:
    if not example.source_image_path:
        return None
    path = Path(example.source_image_path)
    if not path.is_file():
        return None
    return read_grayscale_image(path)


def _render_context(
    example: ReconstructionExample,
) -> tuple[np.ndarray | None, Stroke5Transform, tuple[int, int], bool]:
    source = _load_source_image(example)
    if example.canvas_transform is not None:
        transform = example.canvas_transform
        image_shape = (transform.canvas_height, transform.canvas_width)
        return source, transform, image_shape, True

    target = _as_stroke5(example.target)
    transform = _fallback_transform(target)
    return source, transform, _FALLBACK_CANVAS_SHAPE, False


def _plot_raster(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


def save_reconstruction_pair(
    example: ReconstructionExample,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Save source, decoded target, and prediction on one shared canvas."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    source, transform, image_shape, uses_source_transform = _render_context(example)
    target = rasterize_decoded_strokes(example.target, transform, image_shape)
    prediction = rasterize_decoded_strokes(
        example.prediction,
        transform,
        image_shape,
    )

    panel_count = 3 if source is not None else 2
    fig, axes = plt.subplots(1, panel_count, figsize=(4 * panel_count, 4), dpi=150)
    axes_array = np.atleast_1d(axes)
    if title:
        fig.suptitle(title, fontsize=10)
    panel = 0
    if source is not None:
        _plot_raster(axes_array[panel], source, "original source")
        panel += 1
    target_title = (
        "decoded target"
        if uses_source_transform
        else f"decoded target | tokens={example.length}"
    )
    _plot_raster(axes_array[panel], target, target_title)
    stats = getattr(example, "statistics", {}) or {}
    if stats:
        pred_title = (
            f"model prediction | tokens={example.prediction_length or 0} "
            f"eos={stats.get('eos_position', 'n/a')} "
            f"strokes={stats.get('generated_stroke_count', 'n/a')}"
        )
    else:
        pred_title = "model prediction"
    _plot_raster(axes_array[panel + 1], prediction, pred_title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_reconstruction_examples(
    examples: list[ReconstructionExample],
    output_dir: str | Path,
    *,
    prefix: str = "reconstruction",
) -> list[Path]:
    """Save a numbered plot for each reconstruction example."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for index, example in enumerate(examples, start=1):
        title = (
            f"sample={example.sample_id or example.source_index} "
            f"length={example.length} label={example.label}"
        )
        output_path = directory / f"{prefix}_{index:03d}.png"
        saved.append(save_reconstruction_pair(example, output_path, title=title))
    return saved
