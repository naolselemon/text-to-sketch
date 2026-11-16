"""Sketch token dictionary builder.

Builds a discrete motion vocabulary by clustering the (Δx, Δy) displacement
pairs from a corpus of stroke-5 arrays using MiniBatchKMeans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from utils.paths import DEFAULT_SKETCH_TOKEN_DIR, DEFAULT_STROKE5_DIR


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
            f"[sketch_token] Warning: only {len(deltas)} samples available — "
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

def compute_codebook_artifact_checksum(codebook: np.ndarray | Path | str) -> str:
    """Compute deterministic SHA256 checksum over normalized float32 cluster centers.

    Ensures identical checksum whether supplied as a file path (.npy) or an in-memory array.
    """
    if isinstance(codebook, (str, Path)):
        p = Path(codebook)
        if not p.is_file():
            raise FileNotFoundError(f"Codebook file not found: {p}")
        if p.suffix == ".npy":
            arr = np.load(p, allow_pickle=False)
            return compute_codebook_artifact_checksum(arr)
        return hashlib.sha256(p.read_bytes()).hexdigest()

    array = np.ascontiguousarray(np.asarray(codebook, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def compute_codebook_fingerprint(codebook: np.ndarray | Path | str) -> str:
    """Backward-compatible alias for compute_codebook_artifact_checksum."""
    return compute_codebook_artifact_checksum(codebook)


def compute_token_dictionary_fingerprint(
    codebook: np.ndarray | Path | str,
    *,
    codebook_size: int | None = None,
    motion_token_offset: int = 1,
    pad_token_id: int = 0,
    sep_token_id: int | None = None,
    sos_token_id: int | None = None,
    eos_token_id: int | None = None,
    vocab_size: int | None = None,
) -> str:
    """Compute semantic token dictionary fingerprint including codebook and token layout.

    Distinguishes semantic vocabulary identity from file format differences. Does not
    hash non-semantic metadata such as creation timestamps or file paths.
    """
    codebook_sha256 = compute_codebook_artifact_checksum(codebook)
    if isinstance(codebook, (str, Path)):
        p = Path(codebook)
        if p.suffix == ".npy":
            arr = np.load(p, allow_pickle=False)
            k = len(arr)
        else:
            k = codebook_size if codebook_size is not None else 1000
    else:
        k = len(codebook)

    k_actual = codebook_size if codebook_size is not None else k
    sep_id = sep_token_id if sep_token_id is not None else (k_actual + motion_token_offset)
    sos_id = sos_token_id if sos_token_id is not None else (sep_id + 1)
    eos_id = eos_token_id if eos_token_id is not None else (sos_id + 1)
    v_size = vocab_size if vocab_size is not None else (eos_id + 1)

    payload = {
        "codebook_sha256": codebook_sha256,
        "codebook_size": int(k_actual),
        "motion_token_offset": int(motion_token_offset),
        "pad_token_id": int(pad_token_id),
        "sep_token_id": int(sep_id),
        "sos_token_id": int(sos_id),
        "eos_token_id": int(eos_id),
        "vocab_size": int(v_size),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_token_dict_fingerprint(
    token_dict_dir: Path | str,
    *,
    allow_unverified: bool = False,
) -> str:
    """Return canonical semantic token dictionary fingerprint of token_dict_dir."""
    codebook, metadata = load_codebook_from_dir(token_dict_dir, allow_unverified=allow_unverified)
    if not metadata.get("is_verified", False):
        if not allow_unverified:
            raise ValueError(
                f"Cannot compute verified token dictionary fingerprint for unverified token dictionary at {token_dict_dir}"
            )
        return ""
    token_layout = metadata.get("token_layout", {})
    return compute_token_dictionary_fingerprint(
        codebook,
        codebook_size=metadata.get("K", len(codebook)),
        motion_token_offset=int(token_layout.get("motion_token_offset", 1)),
        pad_token_id=int(token_layout.get("pad_token_id", 0)),
        sep_token_id=int(token_layout["sep_token_id"]) if "sep_token_id" in token_layout else None,
        sos_token_id=int(token_layout["sos_token_id"]) if "sos_token_id" in token_layout else None,
        eos_token_id=int(token_layout["eos_token_id"]) if "eos_token_id" in token_layout else None,
        vocab_size=int(token_layout["vocab_size"]) if "vocab_size" in token_layout else None,
    )


def save_codebook(
    codebook: np.ndarray,
    output_dir: Path,
    K: int,
    n_samples: int,
    *,
    source: str | None = None,
) -> tuple[Path, Path]:
    """Persist the codebook array and a companion metadata JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / "codebook.npy"
    meta_path = output_dir / "metadata.json"

    np.save(npy_path, codebook)
    codebook_sha256 = compute_codebook_artifact_checksum(codebook)

    token_layout = {
        "pad_token_id": 0,
        "motion_token_offset": 1,
        "sep_token_id": K + 1,
        "sos_token_id": K + 2,
        "eos_token_id": K + 3,
        "vocab_size": K + 4,
    }
    semantic_fingerprint = compute_token_dictionary_fingerprint(
        codebook,
        codebook_size=K,
        **token_layout,
    )

    metadata = {
        "K": K,
        "n_samples": n_samples,
        "codebook_shape": list(codebook.shape),
        "codebook_sha256": codebook_sha256,
        "dictionary_fingerprint": semantic_fingerprint,
        "token_layout": token_layout,
        "is_verified": True,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    if source is not None:
        metadata["source"] = source
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return npy_path, meta_path


def load_codebook_from_dir(
    sketch_token_dir: Path | str,
    *,
    allow_unverified: bool = False,
) -> tuple[np.ndarray, dict]:
    """Load a saved codebook and its metadata from *sketch_token_dir*, validating checksum.

    Parameters
    ----------
    sketch_token_dir : Path | str
        Directory containing codebook.npy and metadata.json.
    allow_unverified : bool
        If False (default), strictly requires metadata.json to exist with a valid
        recorded 'codebook_sha256' that matches the codebook array.
        If True, permits loading legacy codebooks lacking recorded checksums,
        marking metadata['is_verified'] = False and metadata['codebook_sha256'] = None.
        Corrupted codebooks with mismatching declared checksums are ALWAYS rejected.
    """
    sketch_token_dir = Path(sketch_token_dir)
    npy_path = sketch_token_dir / "codebook.npy"
    if not npy_path.is_file():
        raise FileNotFoundError(f"Codebook file not found in {sketch_token_dir}: {npy_path}")

    codebook = np.load(npy_path, allow_pickle=False)
    actual_sha256 = compute_codebook_artifact_checksum(codebook)

    meta_path = sketch_token_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as fh:
            metadata = json.load(fh)

    declared_sha256 = metadata.get("codebook_sha256")
    if declared_sha256 is not None and str(declared_sha256).strip():
        declared_str = str(declared_sha256).strip()
        if declared_str != actual_sha256:
            raise ValueError(
                f"Codebook artifact checksum mismatch in {sketch_token_dir}: "
                f"metadata records {declared_str}, actual is {actual_sha256}. "
                f"Codebook file has been modified or corrupted!"
            )
        metadata["codebook_sha256"] = declared_str
        metadata["is_verified"] = True
    else:
        if not allow_unverified:
            raise ValueError(
                f"Token dictionary at {sketch_token_dir} lacks verified 'codebook_sha256' metadata. "
                "Codebook artifact integrity cannot be verified. Pass allow_unverified=True for legacy dictionaries."
            )
        warnings.warn(
            f"Loading codebook from {sketch_token_dir} in unverified legacy mode without recorded checksum.",
            UserWarning,
            stacklevel=2,
        )
        metadata["codebook_sha256"] = None
        metadata["is_verified"] = False
        metadata.pop("dictionary_fingerprint", None)

    return codebook, metadata


def load_original_token_dict_centers(token_dict_path: Path) -> np.ndarray:
    """Load cluster centers from the original Sketchformer sklearn pickle."""

    try:
        import sklearn.cluster._kmeans as kmeans_module

        sys.modules.setdefault("sklearn.cluster.k_means_", kmeans_module)
    except Exception:
        pass

    with Path(token_dict_path).open("rb") as handle:
        token_dict = pickle.load(handle, encoding="latin1")
    centers = getattr(token_dict, "cluster_centers_", None)
    if centers is None:
        raise ValueError(f"Original token dictionary has no cluster_centers_: {token_dict_path}")
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"Expected original token centers with shape (K, 2), got {centers.shape}")
    return centers


def load_stroke5_arrays(source_dir: Path) -> list[np.ndarray]:
    """Load all stroke-5 arrays from a directory of one-array ``.npz`` files."""

    arrays: list[np.ndarray] = []
    for path in sorted(Path(source_dir).glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        if len(data.files) != 1:
            raise ValueError(f"Expected one array in {path}, found {data.files}")
        array = np.asarray(data[data.files[0]], dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 5:
            raise ValueError(f"Expected stroke5 shape (N, 5) in {path}, got {array.shape}")
        arrays.append(array)
    if not arrays:
        raise FileNotFoundError(f"No .npz stroke5 files found in {source_dir}")
    return arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sketch tok-dict codebook.")
    parser.add_argument("--source-dir", default=DEFAULT_STROKE5_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_SKETCH_TOKEN_DIR)
    parser.add_argument("--K", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-token-dict-pkl",
        default=None,
        help="Export centers from the released Sketchformer token_dict.pkl instead of fitting K-means.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source_token_dict_pkl:
        codebook = load_original_token_dict_centers(Path(args.source_token_dict_pkl))
        npy_path, meta_path = save_codebook(
            codebook,
            Path(args.output_dir),
            K=len(codebook),
            n_samples=len(codebook),
            source=str(args.source_token_dict_pkl),
        )
        print(f"Saved released Sketchformer codebook: {npy_path}")
        print(f"Saved metadata: {meta_path}")
        return 0

    stroke5_arrays = load_stroke5_arrays(Path(args.source_dir))
    codebook = build_codebook(stroke5_arrays, K=args.K, random_state=args.seed)
    npy_path, meta_path = save_codebook(
        codebook,
        Path(args.output_dir),
        K=len(codebook),
        n_samples=len(stroke5_arrays),
    )
    print(f"Saved codebook: {npy_path}")
    print(f"Saved metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
