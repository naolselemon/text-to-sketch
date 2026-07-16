"""Measure anchored V3 preprocessing throughput without changing its output."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root")


_add_project_to_path()

from services.anchored_sketch_data.builder import _preprocess_sources


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare serial and multiprocessing throughput for deterministic "
            "anchored V3 image preprocessing."
        )
    )
    parser.add_argument("--images", type=int, default=256)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers. 0 uses all CPUs available to the process.",
    )
    parser.add_argument("--minimum-speedup", type=float, default=1.25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.images < 32:
        raise ValueError("--images must be at least 32 to amortize process startup")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.minimum_speedup <= 0:
        raise ValueError("--minimum-speedup must be positive")

    available_cpus = _available_cpu_count()
    worker_count = min(args.images, args.workers or available_cpus)
    if worker_count < 2:
        raise ValueError("parallel preprocessing eval requires at least two workers")

    with tempfile.TemporaryDirectory() as tmp:
        source_root = Path(tmp)
        paths = _write_workload(source_root, count=args.images)

        serial_started = time.perf_counter()
        serial = _preprocess_sources(
            paths,
            source_root=source_root,
            threshold_profile="hysteresis",
            workers=1,
            show_progress=False,
        )
        serial_seconds = time.perf_counter() - serial_started

        parallel_started = time.perf_counter()
        parallel = _preprocess_sources(
            paths,
            source_root=source_root,
            threshold_profile="hysteresis",
            workers=worker_count,
            show_progress=False,
        )
        parallel_seconds = time.perf_counter() - parallel_started

    if serial != parallel:
        raise AssertionError("parallel preprocessing changed source geometry or rejection data")
    if len(serial[0]) != args.images or serial[1]:
        raise AssertionError(
            f"synthetic workload produced {len(serial[0])} accepted and "
            f"{len(serial[1])} rejected images"
        )

    speedup = serial_seconds / max(parallel_seconds, 1e-9)
    report = {
        "eval": "anchored_v3_parallel_preprocessing",
        "status": "pass" if speedup >= args.minimum_speedup else "fail",
        "images": args.images,
        "workers": worker_count,
        "serial_seconds": serial_seconds,
        "parallel_seconds": parallel_seconds,
        "serial_images_per_second": args.images / serial_seconds,
        "parallel_images_per_second": args.images / parallel_seconds,
        "speedup": speedup,
        "minimum_speedup": args.minimum_speedup,
        "deterministic_equivalence": True,
    }
    print(json.dumps(report, sort_keys=True))
    if speedup < args.minimum_speedup:
        raise AssertionError(
            f"parallel speedup {speedup:.2f}x is below {args.minimum_speedup:.2f}x"
        )
    return 0


def _available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _write_workload(root: Path, *, count: int) -> list[Path]:
    paths: list[Path] = []
    width = height = 512
    x_values = np.arange(16, width - 16, dtype=np.int32)
    for image_index in range(count):
        image = np.full((height, width), 255, dtype=np.uint8)
        for curve_index in range(8):
            phase = image_index * 0.17 + curve_index * 0.63
            baseline = 40 + curve_index * 57
            y_values = baseline + np.rint(
                16.0 * np.sin(x_values / (17.0 + curve_index) + phase)
            ).astype(np.int32)
            points = np.column_stack((x_values, y_values)).reshape(-1, 1, 2)
            cv2.polylines(image, [points], False, 0, thickness=1, lineType=cv2.LINE_8)
        center = (
            256 + int(round(35 * math.sin(image_index * 0.11))),
            256 + int(round(35 * math.cos(image_index * 0.13))),
        )
        cv2.circle(image, center, 28 + image_index % 13, 0, thickness=1)
        path = root / f"workload-{image_index:05d}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write eval fixture {path}")
        paths.append(path)
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
