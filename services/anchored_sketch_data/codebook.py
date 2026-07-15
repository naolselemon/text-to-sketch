"""Training-split-only motion dictionary fitting."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .contract import CODEBOOK_SIZE


@dataclass(frozen=True)
class TrainingStrokeSample:
    sample_id: str
    split: str
    strokes: Sequence[Sequence[Sequence[float]]]


def within_stroke_deltas(
    strokes: Sequence[Sequence[Sequence[float]]],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for stroke in strokes:
        points = np.asarray(stroke, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected stroke shape (N, 2), got {points.shape}")
        if len(points) >= 2:
            chunks.append(np.diff(points, axis=0))
    if not chunks:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def fit_training_codebook(
    samples: Iterable[TrainingStrokeSample],
    *,
    n_clusters: int = CODEBOOK_SIZE,
    seed: int = 42,
    batch_size: int = 8192,
    max_iter: int = 200,
    max_deltas: int = 2_000_000,
) -> np.ndarray:
    """Fit MiniBatchKMeans and reject accidental validation/test inputs."""

    values = list(samples)
    non_training = [sample.sample_id for sample in values if sample.split != "train"]
    if non_training:
        preview = ", ".join(non_training[:3])
        raise ValueError(f"Codebook inputs must all be training samples; got {preview}")
    if max_deltas < n_clusters:
        raise ValueError("max_deltas must be at least n_clusters")
    deltas = [within_stroke_deltas(sample.strokes) for sample in values]
    deltas = [chunk for chunk in deltas if len(chunk)]
    if not deltas:
        raise ValueError("Training samples contain no within-stroke deltas")
    total_deltas = sum(len(chunk) for chunk in deltas)
    if total_deltas < int(n_clusters):
        raise ValueError(
            f"Need at least {n_clusters} training deltas, found {total_deltas}"
        )
    motions = _bounded_motion_sample(deltas, maximum=int(max_deltas), seed=int(seed))
    estimator = MiniBatchKMeans(
        n_clusters=int(n_clusters),
        random_state=int(seed),
        batch_size=max(int(batch_size), int(n_clusters)),
        max_iter=int(max_iter),
        n_init=3,
        reassignment_ratio=0.01,
    )
    estimator.fit(motions)
    centers = np.asarray(estimator.cluster_centers_, dtype=np.float32)
    # Cluster numbering is otherwise an implementation detail. Sorting makes
    # equal input and seed yield a stable token dictionary across environments.
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return centers[order]


def _bounded_motion_sample(
    chunks: list[np.ndarray],
    *,
    maximum: int,
    seed: int,
) -> np.ndarray:
    total = sum(len(chunk) for chunk in chunks)
    if total <= maximum:
        return np.concatenate(chunks, axis=0)
    exact_allocations = [maximum * len(chunk) / total for chunk in chunks]
    allocations = [int(value) for value in exact_allocations]
    remaining = maximum - sum(allocations)
    remainder_order = sorted(
        range(len(chunks)),
        key=lambda index: (-(exact_allocations[index] - allocations[index]), index),
    )
    for index in remainder_order[:remaining]:
        allocations[index] += 1

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for chunk, allocation in zip(chunks, allocations):
        if allocation == 0:
            continue
        if allocation == len(chunk):
            selected.append(chunk)
        else:
            indices = np.sort(rng.choice(len(chunk), size=allocation, replace=False))
            selected.append(chunk[indices])
    return np.concatenate(selected, axis=0)
