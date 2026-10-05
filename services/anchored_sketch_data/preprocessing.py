"""Deterministic fixed-canvas geometry preparation for anchored V3."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.vectorization import centerline_metrics

from .contract import CANVAS_MARGIN, CANVAS_SIZE

Point = tuple[float, float]
Stroke = list[Point]


@dataclass(frozen=True)
class CanvasTransform:
    """Invertible aspect-preserving mapping onto the fixed V3 canvas."""

    minimum: tuple[float, float]
    scale: float
    offset: tuple[float, float]


def normalize_to_canvas(
    strokes: Sequence[Sequence[Sequence[float]]],
    *,
    canvas_size: int = CANVAS_SIZE,
    margin: int = CANVAS_MARGIN,
) -> list[Stroke]:
    """Fit geometry into a square canvas while preserving aspect ratio."""

    if not any(len(stroke) >= 2 for stroke in strokes):
        return []
    normalized, _ = normalize_to_canvas_with_transform(
        strokes,
        canvas_size=canvas_size,
        margin=margin,
    )
    return normalized


def normalize_to_canvas_with_transform(
    strokes: Sequence[Sequence[Sequence[float]]],
    *,
    canvas_size: int = CANVAS_SIZE,
    margin: int = CANVAS_MARGIN,
) -> tuple[list[Stroke], CanvasTransform]:
    """Normalize strokes and retain the transform used for raster comparison."""

    clean = [_coerce_points(stroke) for stroke in strokes if len(stroke) >= 2]
    if not clean:
        raise ValueError("Cannot normalize an empty sketch")
    if canvas_size <= 2 * margin:
        raise ValueError("Canvas margin leaves no drawable area")
    all_points = np.concatenate(clean, axis=0)
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    extent = maximum - minimum
    longest = max(float(extent.max()), 1.0)
    drawable = float(canvas_size - 1 - 2 * margin)
    scale = drawable / longest
    used = extent * scale
    offset = np.asarray(
        [margin + (drawable - used[0]) / 2.0, margin + (drawable - used[1]) / 2.0],
        dtype=np.float32,
    )
    normalized = [
        [(float(x), float(y)) for x, y in ((points - minimum) * scale + offset)]
        for points in clean
    ]
    transform = CanvasTransform(
        minimum=(float(minimum[0]), float(minimum[1])),
        scale=float(scale),
        offset=(float(offset[0]), float(offset[1])),
    )
    return normalized, transform


def denormalize_from_canvas(
    strokes: Sequence[Sequence[Sequence[float]]],
    transform: CanvasTransform,
) -> list[Stroke]:
    """Map fixed-canvas strokes back onto their source raster coordinates."""

    if transform.scale <= 0:
        raise ValueError("Canvas transform scale must be positive")
    minimum = np.asarray(transform.minimum, dtype=np.float32)
    offset = np.asarray(transform.offset, dtype=np.float32)
    output: list[Stroke] = []
    for stroke in strokes:
        points = _coerce_points(stroke)
        restored = (points - offset) / float(transform.scale) + minimum
        output.append([(float(x), float(y)) for x, y in restored])
    return output


def preprocess_strokes(
    strokes: Sequence[Sequence[Sequence[float]]],
    *,
    source_shape: tuple[int, int],
    min_component_pixels: int = 8,
    min_arc_length: float = 3.0,
    border_margin: float = 2.0,
    merge_distance: float = 1.5,
) -> list[Stroke]:
    """Remove artifacts, merge compatible paths, and return stable ordering."""

    coerced = [[(float(x), float(y)) for x, y in _coerce_points(stroke)] for stroke in strokes]
    components = remove_small_components(
        coerced,
        source_shape,
        min_pixels=min_component_pixels,
    )
    no_frames = remove_border_frames(components, source_shape, margin=border_margin)
    long_enough = [stroke for stroke in no_frames if arc_length(stroke) >= min_arc_length]
    merged = merge_compatible_strokes(long_enough, max_gap=merge_distance)
    return deterministic_order(merged)


def remove_small_components(
    strokes: Sequence[Sequence[Sequence[float]]],
    image_shape: tuple[int, int],
    *,
    min_pixels: int = 8,
) -> list[Stroke]:
    if min_pixels <= 1:
        return [[(float(x), float(y)) for x, y in stroke] for stroke in strokes if len(stroke) >= 2]
    canvas = rasterize_strokes(strokes, image_shape)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        canvas.astype(np.uint8), connectivity=8
    )
    valid = {
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_pixels)
    }
    output: list[Stroke] = []
    height, width = image_shape
    for stroke in strokes:
        current: Stroke = []
        for point in stroke:
            x = int(np.clip(round(float(point[0])), 0, width - 1))
            y = int(np.clip(round(float(point[1])), 0, height - 1))
            if int(labels[y, x]) in valid:
                current.append((float(point[0]), float(point[1])))
            else:
                if len(current) >= 2:
                    output.append(current)
                current = []
        if len(current) >= 2:
            output.append(current)
    return output


def remove_border_frames(
    strokes: Sequence[Sequence[Sequence[float]]],
    image_shape: tuple[int, int],
    *,
    margin: float = 2.0,
    minimum_span_ratio: float = 0.75,
) -> list[Stroke]:
    height, width = image_shape
    output: list[Stroke] = []
    for stroke in strokes:
        points = _coerce_points(stroke)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        horizontal_frame = (
            (y_min <= margin or y_max >= height - 1 - margin)
            and x_max - x_min >= minimum_span_ratio * width
            and y_max - y_min <= 2 * margin
        )
        vertical_frame = (
            (x_min <= margin or x_max >= width - 1 - margin)
            and y_max - y_min >= minimum_span_ratio * height
            and x_max - x_min <= 2 * margin
        )
        touches = sum(
            (
                x_min <= margin,
                x_max >= width - 1 - margin,
                y_min <= margin,
                y_max >= height - 1 - margin,
            )
        )
        enclosing_frame = touches >= 3 and arc_length(stroke) >= 1.5 * (height + width)
        if not (horizontal_frame or vertical_frame or enclosing_frame):
            output.append([(float(x), float(y)) for x, y in points])
    return output


def merge_compatible_strokes(
    strokes: Sequence[Sequence[Sequence[float]]],
    *,
    max_gap: float = 1.5,
    minimum_cosine: float = 0.5,
) -> list[Stroke]:
    ordered = [list(stroke) for stroke in deterministic_order(strokes) if len(stroke) >= 2]
    if len(ordered) < 2 or max_gap < 0:
        return ordered

    # The original implementation rebuilt every possible path pair after every
    # merge. Detailed skeletons can contain hundreds of branch paths, making
    # that loop cubic. Stable path IDs preserve the original tie-breaking while
    # this heap keeps unchanged candidates and only recomputes candidates for a
    # newly merged path.
    paths = {index: path for index, path in enumerate(ordered)}
    versions = {index: 0 for index in paths}
    endpoint_index = _EndpointIndex(max_gap=max_gap)
    for path_id, path in paths.items():
        endpoint_index.add(path_id, path)

    candidates: list[tuple[float, int, int, bool, bool, int, int]] = []

    def add_pair(first_id: int, second_id: int) -> None:
        if first_id == second_id:
            return
        first_id, second_id = sorted((first_id, second_id))
        first_path = paths.get(first_id)
        second_path = paths.get(second_id)
        if first_path is None or second_path is None:
            return
        for reverse_first in (False, True):
            first_endpoint, first_direction = _connection_endpoint_and_direction(
                first_path,
                reverse=reverse_first,
                as_first=True,
            )
            for reverse_second in (False, True):
                second_endpoint, second_direction = _connection_endpoint_and_direction(
                    second_path,
                    reverse=reverse_second,
                    as_first=False,
                )
                gap = float(np.linalg.norm(second_endpoint - first_endpoint))
                if gap > max_gap:
                    continue
                denominator = float(
                    np.linalg.norm(first_direction) * np.linalg.norm(second_direction)
                )
                cosine = 1.0 if denominator == 0.0 else float(
                    np.dot(first_direction, second_direction) / denominator
                )
                if cosine >= minimum_cosine:
                    heapq.heappush(
                        candidates,
                        (
                            gap,
                            first_id,
                            second_id,
                            reverse_first,
                            reverse_second,
                            versions[first_id],
                            versions[second_id],
                        ),
                    )

    initial_pairs: set[tuple[int, int]] = set()
    for path_id, path in paths.items():
        for neighbor_id in endpoint_index.neighbors(path):
            if neighbor_id != path_id:
                initial_pairs.add(tuple(sorted((path_id, neighbor_id))))
    for first_id, second_id in sorted(initial_pairs):
        add_pair(first_id, second_id)

    while candidates:
        (
            _,
            first_id,
            second_id,
            reverse_first,
            reverse_second,
            first_version,
            second_version,
        ) = heapq.heappop(candidates)
        if (
            first_id not in paths
            or second_id not in paths
            or versions[first_id] != first_version
            or versions[second_id] != second_version
        ):
            continue

        first_path = paths[first_id]
        second_path = paths[second_id]
        endpoint_index.remove(first_id, first_path)
        endpoint_index.remove(second_id, second_path)
        first = list(reversed(first_path)) if reverse_first else first_path
        second = list(reversed(second_path)) if reverse_second else second_path
        merged = first + second[1:] if np.allclose(first[-1], second[0]) else first + second
        paths[first_id] = merged
        versions[first_id] += 1
        del paths[second_id]
        del versions[second_id]
        endpoint_index.add(first_id, merged)

        for neighbor_id in sorted(endpoint_index.neighbors(merged)):
            if neighbor_id != first_id:
                add_pair(first_id, neighbor_id)

    return deterministic_order(paths.values())


def _connection_endpoint_and_direction(
    path: Stroke,
    *,
    reverse: bool,
    as_first: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the joining endpoint/direction without copying or reversing a path."""

    if as_first:
        endpoint_index = 0 if reverse else -1
        adjacent_index = 1 if reverse else -2
        endpoint = np.asarray(path[endpoint_index])
        direction = endpoint - np.asarray(path[adjacent_index])
    else:
        endpoint_index = -1 if reverse else 0
        adjacent_index = -2 if reverse else 1
        endpoint = np.asarray(path[endpoint_index])
        direction = np.asarray(path[adjacent_index]) - endpoint
    return endpoint, direction


class _EndpointIndex:
    """Small spatial hash for paths whose endpoints can be within ``max_gap``."""

    def __init__(self, *, max_gap: float) -> None:
        self.max_gap = float(max_gap)
        self.cells: dict[tuple[int, int] | tuple[float, float], set[int]] = {}

    def add(self, path_id: int, path: Stroke) -> None:
        for key in self._path_keys(path):
            self.cells.setdefault(key, set()).add(path_id)

    def remove(self, path_id: int, path: Stroke) -> None:
        for key in self._path_keys(path):
            values = self.cells.get(key)
            if values is None:
                continue
            values.discard(path_id)
            if not values:
                del self.cells[key]

    def neighbors(self, path: Stroke) -> set[int]:
        output: set[int] = set()
        for point in (path[0], path[-1]):
            if self.max_gap == 0:
                output.update(self.cells.get((float(point[0]), float(point[1])), ()))
                continue
            center_x, center_y = self._cell(point)
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    output.update(
                        self.cells.get((center_x + offset_x, center_y + offset_y), ())
                    )
        return output

    def _path_keys(
        self, path: Stroke
    ) -> set[tuple[int, int] | tuple[float, float]]:
        if self.max_gap == 0:
            return {
                (float(path[0][0]), float(path[0][1])),
                (float(path[-1][0]), float(path[-1][1])),
            }
        return {self._cell(path[0]), self._cell(path[-1])}

    def _cell(self, point: Sequence[float]) -> tuple[int, int]:
        return (
            math.floor(float(point[0]) / self.max_gap),
            math.floor(float(point[1]) / self.max_gap),
        )


def deterministic_order(strokes: Sequence[Sequence[Sequence[float]]]) -> list[Stroke]:
    oriented: list[Stroke] = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        candidate = _canonical_orientation(stroke)
        # Canonicalization removes consecutive duplicate coordinates. RDP can
        # reduce a tiny closed loop to two identical endpoints, which therefore
        # becomes a singleton here and is not drawable by the V3 grammar.
        if len(candidate) >= 2:
            oriented.append(candidate)

    def key(stroke: Stroke) -> tuple[object, ...]:
        points = np.asarray(stroke, dtype=np.float64)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        rounded = tuple((round(x, 6), round(y, 6)) for x, y in stroke)
        return (
            round(float(minimum[1]), 6),
            round(float(minimum[0]), 6),
            round(float(maximum[1]), 6),
            round(float(maximum[0]), 6),
            len(stroke),
            rounded,
        )

    return sorted(oriented, key=key)


def simplify_strokes(
    strokes: Sequence[Sequence[Sequence[float]]], epsilon: float
) -> list[Stroke]:
    if epsilon < 0:
        raise ValueError("RDP epsilon must be non-negative")
    output: list[Stroke] = []
    for stroke in strokes:
        points = _coerce_points(stroke)
        if len(points) < 2:
            continue
        closed = len(points) > 2 and np.allclose(points[0], points[-1])
        if epsilon == 0:
            simplified = points
        else:
            simplified = _rdp(points, float(epsilon))
        if closed and not np.allclose(simplified[0], simplified[-1]):
            simplified = np.vstack((simplified, simplified[0]))
        if len(simplified) >= 2:
            output.append([(float(x), float(y)) for x, y in simplified])
    return deterministic_order(output)


def augment_strokes(
    strokes: Sequence[Sequence[Sequence[float]]],
    *,
    seed: int,
    horizontal_flip_probability: float = 0.5,
    max_rotation_degrees: float = 5.0,
    scale_range: tuple[float, float] = (0.9, 1.1),
    max_translation: float = 4.0,
    point_jitter_sigma: float = 0.25,
) -> list[Stroke]:
    rng = np.random.default_rng(int(seed))
    center = np.asarray([(CANVAS_SIZE - 1) / 2.0, (CANVAS_SIZE - 1) / 2.0])
    angle = math.radians(float(rng.uniform(-max_rotation_degrees, max_rotation_degrees)))
    scale = float(rng.uniform(*scale_range))
    cosine, sine = math.cos(angle), math.sin(angle)
    transform = scale * np.asarray([[cosine, -sine], [sine, cosine]])
    if float(rng.random()) < horizontal_flip_probability:
        transform = np.asarray([[-1.0, 0.0], [0.0, 1.0]]) @ transform
    translation = rng.uniform(-max_translation, max_translation, size=2)
    output: list[Stroke] = []
    for stroke in strokes:
        points = _coerce_points(stroke)
        jitter = rng.normal(0.0, point_jitter_sigma, size=points.shape)
        changed = (points - center) @ transform.T + center + translation + jitter
        changed = np.clip(changed, 0.0, CANVAS_SIZE - 1.0)
        output.append([(float(x), float(y)) for x, y in changed])
    return deterministic_order(output)


def rasterize_strokes(
    strokes: Sequence[Sequence[Sequence[float]]],
    image_shape: tuple[int, int] = (CANVAS_SIZE, CANVAS_SIZE),
) -> np.ndarray:
    canvas = np.zeros(image_shape, dtype=np.uint8)
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        points = np.rint(np.asarray(stroke, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, 1, thickness=1, lineType=cv2.LINE_8)
    return canvas.astype(bool)


def geometry_f1(
    reference: Sequence[Sequence[Sequence[float]]],
    candidate: Sequence[Sequence[Sequence[float]]],
    *,
    tolerance_px: float = 2.0,
) -> float:
    reference_raster = rasterize_strokes(reference)
    candidate_raster = rasterize_strokes(candidate)
    return centerline_metrics(
        reference_raster, candidate_raster, tolerance_px=tolerance_px
    ).f1


def perceptual_hash(image: np.ndarray) -> int:
    """Return a deterministic 64-bit difference hash for duplicate grouping."""

    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.ndim != 2:
        raise ValueError(f"Expected a grayscale or color image, got {gray.shape}")
    resized = cv2.resize(gray.astype(np.uint8), (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def stable_seed(namespace: str, seed: int) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def arc_length(stroke: Sequence[Sequence[float]]) -> float:
    points = _coerce_points(stroke)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _canonical_orientation(stroke: Sequence[Sequence[float]]) -> Stroke:
    points = [(float(x), float(y)) for x, y in _coerce_points(stroke)]
    if len(points) > 2 and np.allclose(points[0], points[-1]):
        ring = points[:-1]
        minimum = min((round(y, 6), round(x, 6)) for x, y in ring)
        candidates: list[list[Point]] = []
        for index, (x, y) in enumerate(ring):
            if (round(y, 6), round(x, 6)) != minimum:
                continue
            forward = ring[index:] + ring[:index]
            reversed_ring = list(reversed(ring))
            reverse_index = reversed_ring.index(ring[index])
            backward = reversed_ring[reverse_index:] + reversed_ring[:reverse_index]
            candidates.extend((forward, backward))
        chosen = min(candidates, key=lambda value: tuple((round(y, 6), round(x, 6)) for x, y in value))
        return chosen + [chosen[0]]
    forward_key = tuple((round(y, 6), round(x, 6)) for x, y in points)
    reverse = list(reversed(points))
    reverse_key = tuple((round(y, 6), round(x, 6)) for x, y in reverse)
    return points if forward_key <= reverse_key else reverse


def _coerce_points(stroke: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(stroke, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected stroke shape (N, 2), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Stroke coordinates must be finite")
    if len(points) > 1:
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.any(points[1:] != points[:-1], axis=1)
        points = points[keep]
    return points


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) <= 2:
        return points.copy()
    start, end = points[0], points[-1]
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator == 0.0:
        distances = np.linalg.norm(points[1:-1] - start, axis=1)
    else:
        positions = ((points[1:-1] - start) @ segment) / denominator
        projections = start + np.clip(positions, 0.0, 1.0)[:, None] * segment
        distances = np.linalg.norm(points[1:-1] - projections, axis=1)
    if len(distances) == 0 or float(distances.max()) <= epsilon:
        return np.vstack((start, end))
    split = int(np.argmax(distances)) + 1
    left = _rdp(points[: split + 1], epsilon)
    right = _rdp(points[split:], epsilon)
    return np.vstack((left[:-1], right))
