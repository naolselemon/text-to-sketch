"""
Unified interactive runner for the Hand Simulation Pipeline.

Runs Stages 1 → 5 on a configurable number of extracted sketches, then
builds the Tok-Dict codebook from the resulting stroke-5 corpus.

Prerequisites
-------------
    Run `python scripts/extract_sketches.py` first to populate
    data/processed/sketches/ with binary line-art images.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.steps.vectorizer import vectorize_image
from pipeline.steps.ordering_algorithms import (
    order_directional_bias,
    order_greedy_nearest_neighbor,
    order_tsp,
)
from pipeline.steps.kinematics import generate_kinematics
from pipeline.steps.stroke5_formatter import to_stroke5
from pipeline.tokdict.builder import build_codebook, save_codebook
from pipeline.utils.io import save_stroke5


# Interactive prompts

_BANNER = """
╔════════════════════════════════════════════════════════════╗
║          Hand Simulation Pipeline — Text-to-Sketch         ║
║ Stages: Vectorize → Order → Kinematics → Stroke5 → Tok-Dict║
╚════════════════════════════════════════════════════════════╝"""

_ORDERING_METHODS: dict[str, str] = {
    "1": "directional",
    "2": "greedy",
    "3": "tsp",
}
_DEFAULT_ORDERING = "1"


def _prompt_int(prompt: str, default: int, lo: int = 1, hi: int = 10_000) -> int:
    """Prompt the user for an integer, returning *default* on empty/invalid input."""
    while True:
        try:
            raw = input(prompt).strip()
            if not raw:
                return default
            value = int(raw)
            if lo <= value <= hi:
                return value
            print(f"  Please enter a number between {lo} and {hi}.")
        except (ValueError, EOFError):
            return default


def _prompt_ordering() -> str:
    """Prompt the user to select a stroke-ordering method."""
    print("\nStroke-ordering method:")
    print("  1) Directional bias [default]  — top-left → bottom-right")
    print("  2) Greedy nearest-neighbor     — minimise pen travel locally")
    print("  3) TSP approximation           — globally minimise pen travel")
    while True:
        try:
            raw = input("Choose [1/2/3, default: 1]: ").strip() or _DEFAULT_ORDERING
        except EOFError:
            raw = _DEFAULT_ORDERING
        if raw in _ORDERING_METHODS:
            return _ORDERING_METHODS[raw]
        print("  Invalid choice — please enter 1, 2, or 3.")


# Pipeline

_ORDER_FN_MAP = {
    "directional": order_directional_bias,
    "greedy":      order_greedy_nearest_neighbor,
    "tsp":         order_tsp,
}


def run_pipeline(
    sketches_dir: Path,
    stroke5_dir: Path,
    tokdict_dir: Path,
    n_sketches: int,
    ordering: str,
    codebook_K: int = 1000,
    seed: int = 42,
) -> None:
    """Core pipeline logic (importable for programmatic use)."""

    all_sketches = sorted(sketches_dir.rglob("*.png"))
    if not all_sketches:
        print(f"[error] No sketches found in {sketches_dir}")
        print("        Run 'python scripts/extract_sketches.py' first.")
        sys.exit(1)

    n = min(n_sketches, len(all_sketches))
    random.seed(seed)
    samples = random.sample(all_sketches, n)

    order_fn = _ORDER_FN_MAP[ordering]

    print(f"\n[pipeline] {n} sketches  ·  ordering: {ordering}  ·  K={codebook_K}")
    print(f"[pipeline] stroke-5 output → {stroke5_dir}")
    print()

    stroke5_arrays: list[np.ndarray] = []
    ok = skipped = 0

    for img_path in tqdm(samples, desc="Steps B-E", unit="sketch"):
        try:
            # B: Vectorize
            strokes = vectorize_image(img_path)
            if not strokes:
                skipped += 1
                continue

            # C: Order
            ordered = order_fn(strokes)

            # D: Kinematics
            timed = generate_kinematics(ordered)
            if not timed:
                skipped += 1
                continue

            # E: stroke-5
            s5 = to_stroke5(timed)
            stroke5_arrays.append(s5)

            out_path = stroke5_dir / (img_path.stem + ".npz")
            save_stroke5(s5, out_path)
            ok += 1

        except Exception as exc:
            tqdm.write(f"  [skip] {img_path.name}: {exc}")
            skipped += 1

    # Summary
    print()
    print("─" * 58)
    print(f"  Stroke-5 conversion  : {ok:>5} OK   {skipped:>5} skipped")
    print(f"  Output directory     : {stroke5_dir}")
    print("─" * 58)

    if not stroke5_arrays:
        print("[tokdict] No stroke-5 data — skipping codebook build.")
        return

    # Tok-Dict
    n_drawing_pts = int(sum(
        int((s5[:, 2] == 1.0).sum()) for s5 in stroke5_arrays
    ))
    print(f"\n[tokdict] Building K-means codebook …")
    print(f"[tokdict]   sequences    : {len(stroke5_arrays)}")
    print(f"[tokdict]   drawing pts  : {n_drawing_pts:,}")
    print(f"[tokdict]   requested K  : {codebook_K}")

    codebook = build_codebook(stroke5_arrays, K=codebook_K)
    npy_path, meta_path = save_codebook(
        codebook, tokdict_dir,
        K=len(codebook),
        n_samples=n_drawing_pts,
    )

    print(f"[tokdict]   actual K     : {len(codebook)}")
    print(f"[tokdict]   codebook     : {npy_path}")
    print(f"[tokdict]   metadata     : {meta_path}")



# Entry-point


def main() -> None:
    sketches_dir = PROJECT_ROOT / "data" / "processed" / "sketches"
    stroke5_dir  = PROJECT_ROOT / "data" / "processed" / "stroke5"
    tokdict_dir  = PROJECT_ROOT / "data" / "processed" / "tokdict"

    # Count available sketches before prompting.
    available = list(sketches_dir.rglob("*.png"))

    print(_BANNER)
    print(f"\n  Available sketches : {len(available)}")

    if not available:
        print(f"\n[error] No sketches found in {sketches_dir}")
        print("        Run 'python scripts/extract_sketches.py' first.")
        sys.exit(1)

    default_n = min(50, len(available))
    n = _prompt_int(
        f"\nHow many sketches to process? [default: {default_n}]: ",
        default=default_n,
        lo=1,
        hi=len(available), 
    )
    ordering = _prompt_ordering()

    print(f"\n  Will process {n} sketches with '{ordering}' ordering.")

    run_pipeline(
        sketches_dir=sketches_dir,
        stroke5_dir=stroke5_dir,
        tokdict_dir=tokdict_dir,
        n_sketches=n,
        ordering=ordering,
    )

    print(f"\n{'═' * 58}")
    print("  Pipeline complete!")
    print(f"{'═' * 58}\n")


if __name__ == "__main__":
    main()
