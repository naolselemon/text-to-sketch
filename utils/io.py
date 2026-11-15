"""
Shared I/O utilities for the  Pipeline.

Provides thin wrappers around numpy / json persistence so that all pipeline
scripts use a consistent interface for reading and writing pipeline artefacts.

"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Generator

import numpy as np


def _atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed NumPy archive without exposing partial output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, _replacement_mode(path))
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_path, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def _replacement_mode(path: Path) -> int:
    """Match an existing target or the process's normal new-file mode."""

    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        current_umask = os.umask(0)
        os.umask(current_umask)
        return 0o666 & ~current_umask


# Stroke-5

def save_stroke5(stroke5: np.ndarray, path: Path | str) -> None:
    """Save a stroke-5 array to a compressed .npz file.

    Parameters
    ----------
    stroke5 : np.ndarray, shape (N, 5)
    path    : destination path (parent dirs are created automatically).
    """
    path = Path(path)
    _atomic_savez_compressed(path, stroke5=stroke5)


def load_stroke5(path: Path | str) -> np.ndarray:
    """Load a stroke-5 array previously saved with :func:`save_stroke5`.

    Returns
    -------
    np.ndarray, shape (N, 5)
    """
    with np.load(str(path)) as data:
        return np.array(data["stroke5"], copy=True)


def load_all_stroke5(
    stroke5_dir: Path | str,
) -> Generator[tuple[Path, np.ndarray], None, None]:
    """Yield (path, stroke5_array) for every .npz file in *stroke5_dir*.

    Parameters
    ----------
    stroke5_dir : directory to scan recursively.

    Yields
    ------
    (path, stroke5) tuples.
    """
    stroke5_dir = Path(stroke5_dir)
    for npz_path in sorted(stroke5_dir.rglob("*.npz")):
        try:
            yield npz_path, load_stroke5(npz_path)
        except Exception as exc:
            print(f"[io] Warning: could not load {npz_path}: {exc}")


# Token sequences

def save_token_sequence(
    tokens: np.ndarray,
    path: Path | str,
    *,
    build_id: str | None = None,
    token_dict_sha256: str | None = None,
    sample_id: str | None = None,
    **metadata: Any,
) -> None:
    """Save a token-index sequence and optional provenance metadata to a compressed .npz file."""
    path = Path(path)
    payload: dict[str, Any] = {"tokens": np.asarray(tokens, dtype=np.int64)}
    if build_id is not None:
        payload["build_id"] = np.asarray(str(build_id))
    if token_dict_sha256 is not None:
        payload["token_dict_sha256"] = np.asarray(str(token_dict_sha256))
    if sample_id is not None:
        payload["sample_id"] = np.asarray(str(sample_id))
    for key, value in metadata.items():
        if value is not None:
            payload[key] = np.asarray(value)
    _atomic_savez_compressed(path, **payload)


def load_token_sequence(path: Path | str) -> np.ndarray:
    """Load a token-index sequence saved with :func:`save_token_sequence`."""
    with np.load(str(path)) as data:
        return np.array(data["tokens"], copy=True)


def load_token_sequence_with_metadata(
    path: Path | str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a token sequence alongside any stored provenance metadata."""
    with np.load(str(path), allow_pickle=True) as data:
        if "tokens" in data:
            token_key = "tokens"
        elif data.files == ["arr_0"]:
            token_key = "arr_0"
        else:
            raise ValueError(f"Token archive has no tokens array: {path}")
        tokens = np.asarray(data[token_key])
        if tokens.ndim != 1:
            raise ValueError(f"Token sequence must be one-dimensional: {path}")
        meta: dict[str, Any] = {}
        for key in data.files:
            if key != token_key:
                val = data[key]
                meta[key] = val.item() if val.shape == () else val
        return tokens, meta


# Codebook

def load_codebook(codebook_path: Path | str) -> np.ndarray:
    """Load a sketch-token codebook array from a .npy file.

    Returns
    -------
    np.ndarray, shape (K, 2)
    """
    return np.load(str(codebook_path))
