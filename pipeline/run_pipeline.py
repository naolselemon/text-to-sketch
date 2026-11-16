"""Interactive CLI for the main text-to-sketch pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.paths import (
    DEFAULT_FILTERED_SKETCHES_DIR,
    DEFAULT_STROKE5_DIR,
    DEFAULT_SKETCH_TOKEN_DIR,
)
from pipeline.vectorization import (
    DEFAULT_MAX_GEOMETRY_ERROR,
    DEFAULT_RDP_EPSILON,
    THRESHOLD_PROFILES,
    VECTORIZERS,
)
from pipeline.workflow import run_pipeline

_BANNER = """
╔════════════════════════════════════════════════════════════╗
║          Centerline Sketch Pipeline                        ║
║ Grayscale -> Centerline -> Stroke5 -> Released Tok-Dict    ║
╚════════════════════════════════════════════════════════════╝"""

_ORDERING_METHODS: dict[str, str] = {
    "1": "continuity",
    "2": "directional",
    "3": "greedy",
    "4": "tsp",
    "5": "continuity-topology",
}
_DEFAULT_ORDERING = "1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run centerline or legacy contour sketch preprocessing."
    )
    parser.add_argument(
        "--sketches-dir",
        type=Path,
        default=DEFAULT_FILTERED_SKETCHES_DIR,
        help=f"Filtered sketch image directory (default: {DEFAULT_FILTERED_SKETCHES_DIR}).",
    )
    parser.add_argument(
        "--stroke5-dir",
        type=Path,
        default=DEFAULT_STROKE5_DIR,
        help=f"Output directory for stroke5 .npz files (default: {DEFAULT_STROKE5_DIR}).",
    )
    parser.add_argument(
        "--sketch-token-dir",
        type=Path,
        default=DEFAULT_SKETCH_TOKEN_DIR,
        help=f"Output directory for sketch-token codebook files (default: {DEFAULT_SKETCH_TOKEN_DIR}).",
    )
    parser.add_argument(
        "--token-dict-dir",
        type=Path,
        default=None,
        help="Released token dictionary directory used by centerline preprocessing.",
    )
    parser.add_argument(
        "--n-sketches",
        type=int,
        default=None,
        help="Number of sketches to process. If omitted, prompt interactively.",
    )
    parser.add_argument(
        "--ordering",
        choices=sorted(set(_ORDERING_METHODS.values())),
        default=None,
        help="Stroke ordering method. If omitted, prompt interactively.",
    )
    parser.add_argument(
        "--vectorizer",
        choices=VECTORIZERS,
        default="centerline",
        help="centerline is the default path; contour retains legacy behavior.",
    )
    parser.add_argument(
        "--threshold-profile",
        choices=THRESHOLD_PROFILES,
        default="hysteresis",
        help="Grayscale threshold profile used before centerline thinning.",
    )
    parser.add_argument(
        "--extractor-name",
        default=None,
        help="Extractor profile recorded in the preprocessing manifest.",
    )
    parser.add_argument(
        "--rdp-epsilon",
        type=float,
        default=DEFAULT_RDP_EPSILON,
        help=f"RDP simplification epsilon used during vectorization (default: {DEFAULT_RDP_EPSILON}).",
    )
    parser.add_argument(
        "--codebook-k",
        type=int,
        default=1000,
        help="Requested K-means sketch-token codebook size (default: 1000).",
    )
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=4096,
        help="Reject sketches that cannot fit this limit without excess geometry loss.",
    )
    parser.add_argument(
        "--max-geometry-error",
        type=float,
        default=DEFAULT_MAX_GEOMETRY_ERROR,
        help="Maximum adaptive RDP tolerance in source-image pixels.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSONL audit manifest path.",
    )
    parser.add_argument(
        "--fail-on-overlength",
        action="store_true",
        help="Exit unsuccessfully after writing the manifest if any sketch is overlength.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of centerline sketches to process concurrently. "
            "Use 1 for the existing serial behavior (default: 1)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sketch sampling (default: 42).",
    )
    return parser.parse_args(argv)


def _prompt_int(prompt: str, default: int, lo: int = 1, hi: int = 10_000) -> int:
    """Prompt for an integer, returning *default* on empty/invalid input."""
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
    print("  1) Centerline continuity [default]")
    print("  2) Directional bias")
    print("  3) Greedy nearest-neighbor")
    print("  4) TSP approximation")
    print("  5) Continuity topology")
    while True:
        try:
            raw = input("Choose [1/2/3/4/5, default: 1]: ").strip() or _DEFAULT_ORDERING
        except EOFError:
            raw = _DEFAULT_ORDERING
        if raw in _ORDERING_METHODS:
            return _ORDERING_METHODS[raw]
        print("  Invalid choice; please enter 1, 2, 3, 4, or 5.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    sketches_dir = args.sketches_dir
    stroke5_dir = args.stroke5_dir
    sketch_token_dir = args.sketch_token_dir

    available = list(sketches_dir.rglob("*.png"))

    print(_BANNER)
    print(f"\n  Available sketches : {len(available)}")

    if not available:
        print(f"\n[error] No sketches found in {sketches_dir}")
        print("        Run 'python scripts/prepare_data/filter_sketches_by_points.py' first.")
        sys.exit(1)

    default_n = min(50, len(available))
    if args.n_sketches is None:
        n = _prompt_int(
            f"\nHow many sketches to process? [default: {default_n}]: ",
            default=default_n,
            lo=1,
            hi=len(available),
        )
    else:
        if args.n_sketches < 1:
            print("\n[error] --n-sketches must be at least 1.")
            sys.exit(1)
        n = min(args.n_sketches, len(available))

    if args.num_workers < 1:
        print("\n[error] --num-workers must be at least 1.")
        sys.exit(1)

    ordering = args.ordering or _prompt_ordering()

    print(
        f"\n  Will process {n} sketches with '{ordering}' ordering "
        f"using {args.num_workers} worker(s)."
    )

    run_pipeline(
        sketches_dir=sketches_dir,
        stroke5_dir=stroke5_dir,
        sketch_token_dir=sketch_token_dir,
        n_sketches=n,
        ordering=ordering,
        rdp_epsilon=args.rdp_epsilon,
        codebook_K=args.codebook_k,
        seed=args.seed,
        vectorizer=args.vectorizer,
        threshold_profile=args.threshold_profile,
        max_token_length=args.max_token_length,
        max_geometry_error=args.max_geometry_error,
        token_dict_dir=args.token_dict_dir,
        manifest_path=args.manifest,
        fail_on_overlength=args.fail_on_overlength,
        extractor_name=args.extractor_name,
        num_workers=args.num_workers,
    )

    print(f"\n{'═' * 58}")
    print("  Pipeline complete!")
    print(f"{'═' * 58}\n")


if __name__ == "__main__":
    main()
