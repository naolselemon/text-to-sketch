"""
Tok-Dict Builder — K-means Codebook.

Builds a discrete motion vocabulary by clustering the (Δx, Δy) displacement
pairs from a corpus of stroke-5 arrays using MiniBatchKMeans.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


# Core builder

def build_codebook(
    stroke5_arrays: list[np.ndarray],
    K: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster (Δx, Δy) pairs from stroke-5 arrays into K centroids."""
    drawing_deltas: list[np.ndarray] = []

    for s5 in stroke5_arrays:
        mask = s5[:, 2] == 1.0   # p1 == 1  →  pen drawing
        subset = s5[mask, :2]
        if len(subset) > 0:
            drawing_deltas.append(subset)

    if not drawing_deltas:
        raise ValueError(
            "No drawing-stroke samples found in the provided stroke-5 arrays. "
            "Cannot build a codebook from empty data."
        )

    deltas = np.concatenate(drawing_deltas, axis=0)

    # Reduce K if we have fewer samples than requested clusters.
    K_actual = min(K, len(deltas))
    if K_actual < K:
        print(
            f"[tokdict] Warning: only {len(deltas)} samples available — "
            f"reducing K from {K} to {K_actual}."
        )

    kmeans = MiniBatchKMeans(
        n_clusters=K_actual,
        random_state=random_state,
        n_init=3,
        batch_size=max(1024, K_actual),
        verbose=0,
    )
    kmeans.fit(deltas)

    return kmeans.cluster_centers_.astype(np.float32)


# Persistence helpers

def save_codebook(
    codebook: np.ndarray,
    output_dir: Path,
    K: int,
    n_samples: int,
) -> tuple[Path, Path]:
    """Persist the codebook array and a companion metadata JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path  = output_dir / "codebook.npy"
    meta_path = output_dir / "metadata.json"

    np.save(npy_path, codebook)

    metadata = {
        "K": K,
        "n_samples": n_samples,
        "codebook_shape": list(codebook.shape),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    return npy_path, meta_path


def load_codebook_from_dir(tokdict_dir: Path) -> tuple[np.ndarray, dict]:
    """Load a saved codebook and its metadata from *tokdict_dir*.

    Returns
    -------
    (codebook, metadata) tuple.
    """
    tokdict_dir = Path(tokdict_dir)
    codebook = np.load(tokdict_dir / "codebook.npy")
    with open(tokdict_dir / "metadata.json") as fh:
        metadata = json.load(fh)
    return codebook, metadata
