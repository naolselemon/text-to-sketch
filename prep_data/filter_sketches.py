"""
Filter sketch images by original image point count.

By default this copies sketches whose original foreground pixel count is at or
below --max-points into a *new* output directory. Choose a different --output-dir
and --report-path for every run; existing output is never merged or replaced.
This happens before any vectorization/RDP simplification:

    .venv/bin/python scripts/prepare_data/filter_sketches_by_points.py --output-dir data/processed/builds/RUN/sketches_filtered --report-path data/processed/builds/RUN/filter_report.csv
Use --action move-rejected if you want to clean data/processed/sketches in
place by moving high-point sketches out of it.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(iterable, **_: object):
        return iterable


from utils.paths import (
    DEFAULT_FILTERED_SKETCHES_DIR,
    DEFAULT_SKETCHES_DIR,
    PROCESSED_DATA_DIR,
)

DEFAULT_RDP_EPSILON = 0.5
DEFAULT_INPUT_DIR = DEFAULT_SKETCHES_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_FILTERED_SKETCHES_DIR
DEFAULT_REJECTED_DIR = PROCESSED_DATA_DIR / "sketches_rejected"
DEFAULT_REPORT_PATH = PROCESSED_DATA_DIR / "sketch_point_filter_report.csv"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass(frozen=True)
class FilterResult:
    path: Path
    relative_path: Path
    point_count: int | None
    raw_point_count: int | None
    simplified_point_count: int | None
    kept: bool
    action: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter sketch images whose point count is above a threshold. "
            "Sketches with exactly max-points are kept."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Sketch directory to scan recursively (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Destination for kept sketches when --action copy-kept is used; "
            "must not already exist (use a unique path for each run; "
            f"default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--rejected-dir",
        type=Path,
        default=DEFAULT_REJECTED_DIR,
        help=(
            "Destination for rejected sketches when --action move-rejected is "
            "used; must not already exist "
            f"(default: {DEFAULT_REJECTED_DIR})."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"New CSV report path; must not already exist (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20000,
        help="Reject sketches with point count above this value (default: 20000).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_RDP_EPSILON,
        help=f"RDP epsilon used for vectorization (default: {DEFAULT_RDP_EPSILON}).",
    )
    parser.add_argument(
        "--count",
        choices=("original", "raw-contour", "simplified"),
        default="original",
        help=(
            "Which point count to filter on. original counts foreground pixels "
            "directly from the sketch image before vectorization; raw-contour "
            "counts OpenCV contour points before RDP; simplified counts points "
            "after RDP (default: original)."
        ),
    )
    parser.add_argument(
        "--foreground-threshold",
        type=int,
        default=40,
        help=(
            "Minimum grayscale contrast from the background for a pixel to "
            "count as an original sketch point (default: 40). For dark "
            "background sketches, pixels brighter than this value are counted; "
            "for light background sketches, pixels darker than 255 minus this "
            "value are counted."
        ),
    )
    parser.add_argument(
        "--background",
        choices=("auto", "dark", "light"),
        default="auto",
        help=(
            "Sketch background polarity for original point counting. auto "
            "infers it from the image border (default: auto)."
        ),
    )
    parser.add_argument(
        "--action",
        choices=("copy-kept", "move-rejected", "delete-rejected", "report-only"),
        default="copy-kept",
        help=(
            "copy-kept copies valid sketches to output-dir; move-rejected moves "
            "high-point sketches to rejected-dir; delete-rejected removes "
            "high-point sketches; report-only only writes the CSV report "
            "(default: copy-kept)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only scan the first N sketches. Useful for a quick test run.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required with --action delete-rejected.",
    )
    return parser.parse_args()


def collect_sketches(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def infer_background(img) -> str:
    import numpy as np

    border = [img[0, :], img[-1, :], img[:, 0], img[:, -1]]
    border_median = float(np.median(np.concatenate(border)))
    return "dark" if border_median < 128 else "light"


def count_original_points(
    path: Path,
    foreground_threshold: int,
    background: str,
) -> int:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {path}")

    resolved_background = infer_background(img) if background == "auto" else background
    if resolved_background == "dark":
        return int((img > foreground_threshold).sum())
    return int((img < 255 - foreground_threshold).sum())


def point_count_for(
    path: Path,
    epsilon: float,
    count_kind: str,
    foreground_threshold: int,
    background: str,
) -> tuple[int, int | None, int | None]:
    if count_kind == "original":
        return count_original_points(path, foreground_threshold, background), None, None

    from pipeline.vectorization import vectorize_image_with_stats

    _, stats = vectorize_image_with_stats(path, epsilon=epsilon)
    point_count = (
        stats.raw_point_count
        if count_kind == "raw-contour"
        else stats.simplified_point_count
    )
    return point_count, stats.raw_point_count, stats.simplified_point_count


def copy_kept(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def move_rejected(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def write_report(report_path: Path, results: list[FilterResult]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "relative_path",
                "point_count",
                "raw_point_count",
                "simplified_point_count",
                "kept",
                "action",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "path": str(result.path),
                    "relative_path": str(result.relative_path),
                    "point_count": result.point_count,
                    "raw_point_count": result.raw_point_count,
                    "simplified_point_count": result.simplified_point_count,
                    "kept": result.kept,
                    "action": result.action,
                    "error": result.error,
                }
            )


def filter_sketches(args: argparse.Namespace) -> list[FilterResult]:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    rejected_dir = args.rejected_dir.resolve()
    report_path = args.report_path.resolve()
    if report_path.exists():
        raise FileExistsError(f"Filter report already exists: {report_path}. Choose a new --report-path.")
    destination = (
        output_dir if args.action == "copy-kept"
        else rejected_dir if args.action == "move-rejected"
        else None
    )
    if destination is not None:
        if destination == input_dir or destination in input_dir.parents or input_dir in destination.parents:
            raise ValueError("Filter destination must not overlap the input directory")
        if report_path == destination or report_path in destination.parents:
            raise ValueError("Filter report path cannot be the output directory or its parent")
        destination.mkdir(parents=True, exist_ok=False)


    sketches = collect_sketches(input_dir)
    if args.limit is not None:
        sketches = sketches[: args.limit]

    results: list[FilterResult] = []
    for path in tqdm(sketches, desc="Filtering sketches", unit="sketch"):
        relative_path = path.relative_to(input_dir)
        try:
            point_count, raw_point_count, simplified_point_count = point_count_for(
                path,
                epsilon=args.epsilon,
                count_kind=args.count,
                foreground_threshold=args.foreground_threshold,
                background=args.background,
            )
            kept = point_count <= args.max_points
            action = "kept" if kept else "rejected"

            if args.action == "copy-kept" and kept:
                copy_kept(path, output_dir / relative_path)
                action = "copied"
            elif args.action == "move-rejected" and not kept:
                move_rejected(path, rejected_dir / relative_path)
                action = "moved"
            elif args.action == "delete-rejected" and not kept:
                path.unlink()
                action = "deleted"
            elif args.action == "report-only":
                action = "reported"

            results.append(
                FilterResult(
                    path=path,
                    relative_path=relative_path,
                    point_count=point_count,
                    raw_point_count=raw_point_count,
                    simplified_point_count=simplified_point_count,
                    kept=kept,
                    action=action,
                    error="",
                )
            )
        except Exception as exc:
            results.append(
                FilterResult(
                    path=path,
                    relative_path=relative_path,
                    point_count=None,
                    raw_point_count=None,
                    simplified_point_count=None,
                    kept=False,
                    action="error",
                    error=str(exc),
                )
            )

    return results


def print_summary(args: argparse.Namespace, results: list[FilterResult]) -> None:
    total = len(results)
    errors = sum(1 for result in results if result.error)
    kept = sum(1 for result in results if result.kept and not result.error)
    rejected = sum(1 for result in results if not result.kept and not result.error)

    print()
    print("-" * 64)
    print(f"Scanned          : {total}")
    print(f"Kept             : {kept} (<= {args.max_points:,} {args.count} points)")
    print(f"Rejected         : {rejected} (> {args.max_points:,} {args.count} points)")
    print(f"Errors           : {errors}")
    print(f"Report           : {args.report_path.resolve()}")
    if args.action == "copy-kept":
        print(f"Filtered sketches: {args.output_dir.resolve()}")
    elif args.action == "move-rejected":
        print(f"Rejected sketches: {args.rejected_dir.resolve()}")
    print("-" * 64)


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        print(f"[error] Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.max_points < 1:
        print("[error] --max-points must be at least 1.", file=sys.stderr)
        sys.exit(1)

    if args.action == "delete-rejected" and not args.yes:
        print(
            "[error] --action delete-rejected permanently removes files. "
            "Re-run with --yes if that is intended.",
            file=sys.stderr,
        )
        sys.exit(1)

    results = filter_sketches(args)
    write_report(args.report_path, results)
    print_summary(args, results)


if __name__ == "__main__":
    main()
