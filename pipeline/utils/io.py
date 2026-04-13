"""
Shared I/O utilities for the  Pipeline.

Provides thin wrappers around numpy / json persistence so that all pipeline
scripts use a consistent interface for reading and writing pipeline artefacts.

"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np



# Stroke-5

def save_stroke5(stroke5: np.ndarray, path: Path | str) -> None:
    """Save a stroke-5 array to a compressed .npz file.

    Parameters
    ----------
    stroke5 : np.ndarray, shape (N, 5)
    path    : destination path (parent dirs are created automatically).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, stroke5=stroke5)


def load_stroke5(path: Path | str) -> np.ndarray:
    """Load a stroke-5 array previously saved with :func:`save_stroke5`.

    Returns
    -------
    np.ndarray, shape (N, 5)
    """
    data = np.load(str(path))
    return data["stroke5"]


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

def save_token_sequence(tokens: np.ndarray, path: Path | str) -> None:
    """Save a token-index sequence to a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, tokens=tokens)


def load_token_sequence(path: Path | str) -> np.ndarray:
    """Load a token-index sequence saved with :func:`save_token_sequence`."""
    data = np.load(str(path))
    return data["tokens"]


# Codebook

def load_codebook(codebook_path: Path | str) -> np.ndarray:
    """Load a Tok-Dict codebook array from a .npy file.

    Returns
    -------
    np.ndarray, shape (K, 2)
    """
    return np.load(str(codebook_path))
