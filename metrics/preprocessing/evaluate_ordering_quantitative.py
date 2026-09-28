"""Compare stroke-ordering methods quantitatively under fixed controls."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from statistics import fmean, median
from typing import Callable

import numpy as np

from pipeline.ordering import (
    order_continuity_greedy,
    order_outer_to_inner,
    order_directional_bias,
    order_greedy_nearest_neighbor,
    order_tsp,
)
from pipeline.stroke5 import strokes_to_stroke5
from pipeline.vectorization import (
    THRESHOLD_PROFILES,
    CenterlineBranch,
    read_grayscale_image,
    vectorize_image,
    vectorize_image_with_stats,
)
from utils.tokenizer import ErrorFeedbackQuantizer

Point = tuple[int, int]
Stroke = list[Point]
Orderer = Callable[[list[Stroke]], list[Stroke]]

METHODS = ("continuity_greedy", "outer_to_inner", "directional_bias", "nn_greedy", "tsp")
METHOD_LABELS = {
    "continuity_greedy": "Continuity-greedy",
    "outer_to_inner": "Outer-to-inner",
    "directional_bias": "Directional bias",
    "nn_greedy": "NN-greedy",
    "tsp": "TSP",
}
ORDERERS: dict[str, Orderer] = {
    "directional_bias": order_directional_bias,
    "nn_greedy": order_greedy_nearest_neighbor,
    "tsp": order_tsp,
}

METRICS = {
    "output_stroke_count": {
        "label": "Output strokes",
        "direction": "lower",
    },
    "branch_join_count": {
        "label": "Branch joins",
        "direction": "higher",
    },
    "branch_join_rate": {
        "label": "Branch join rate",
        "direction": "higher",
    },
    "pen_lift_count": {
        "label": "Pen lifts",
        "direction": "lower",
    },
    "joinable_pen_lift_count": {
        "label": "Joinable pen lifts",
        "direction": "lower",
    },
    "mean_continuous_trajectory_length_px": {
        "label": "Mean continuous trajectory length (px)",
        "direction": "higher",
    },
    "median_continuous_trajectory_length_px": {
        "label": "Median continuous trajectory length (px)",
        "direction": "higher",
    },
    "max_continuous_trajectory_length_px": {
        "label": "Maximum continuous trajectory length (px)",
        "direction": "higher",
    },
    "output_point_count": {
        "label": "Output points",
        "direction": "lower",
    },
    "token_length": {
        "label": "Encoded tokens",
        "direction": "lower",
    },
    "separator_token_count": {
        "label": "Separator tokens",
        "direction": "lower",
    },
    "modeled_drawing_duration_seconds": {
        "label": "Modeled drawing duration (s)",
        "direction": "lower",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare continuity-greedy, directional bias, NN-greedy, and TSP "
            "using identical vectorized input and fixed settings."
        )
    )
    parser.add_argument(
        "--sketches-dir",
        type=Path,
        required=True,
        help="Directory containing source PNG sketches; searched recursively.",
    )
    parser.add_argument(
        "--codebook",
        type=Path,
        required=True,
        help="Path to the shared two-dimensional codebook.npy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the complete JSON evidence report.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help=(
            "Destination for the presentation-ready Markdown summary. "
            "Default: the JSON output path with a .md suffix."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Optional deterministic sample size; default evaluates every PNG.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed used only when --samples is supplied.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.5,
        help="Fixed RDP epsilon in source-image pixels.",
    )
    parser.add_argument(
        "--threshold-profile",
        choices=THRESHOLD_PROFILES,
        default="hysteresis",
        help="Fixed centerline threshold profile.",
    )
    parser.add_argument(
        "--junction-tolerance",
        type=float,
        default=2.0,
        help="Continuity and joinable-lift endpoint tolerance in pixels.",
    )
    parser.add_argument(
        "--normalization-extent",
        type=float,
        default=1.0,
        help="Fixed Stroke-5 normalization extent used for every method.",
    )
    parser.add_argument(
        "--min-stroke-duration",
        type=float,
        default=0.10,
        help="Minimum modeled duration of one trajectory in seconds.",
    )
    parser.add_argument(
        "--delay-between-strokes",
        type=float,
        default=0.15,
        help="Modeled delay for each pen lift in seconds.",
    )
    args = parser.parse_args()

    if args.samples is not None and args.samples <= 0:
        parser.error("--samples must be positive")
    if args.epsilon < 0:
        parser.error("--epsilon must be non-negative")
    if args.junction_tolerance < 0:
        parser.error("--junction-tolerance must be non-negative")
    if args.normalization_extent <= 0:
        parser.error("--normalization-extent must be positive")
    if args.min_stroke_duration < 0:
        parser.error("--min-stroke-duration must be non-negative")
    if args.delay_between_strokes < 0:
        parser.error("--delay-between-strokes must be non-negative")
    return args


def euclidean_distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def trajectory_length(stroke: Stroke) -> float:
    return math.fsum(
        euclidean_distance(first, second)
        for first, second in zip(stroke, stroke[1:])
    )


def select_images(
    sketches_dir: Path,
    *,
    samples: int | None,
    seed: int,
) -> list[Path]:
    if not sketches_dir.is_dir():
        raise NotADirectoryError(f"Sketch directory not found: {sketches_dir}")

    paths = sorted(
        path
        for path in sketches_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if not paths:
        raise FileNotFoundError(f"No PNG sketches found under {sketches_dir}")
    if samples is None:
        return paths
    if samples > len(paths):
        raise ValueError(
            f"Requested {samples} samples, but only {len(paths)} PNG files exist"
        )
    return sorted(random.Random(seed).sample(paths, samples))


def apply_orderer(
    method: str,
    source_strokes: list[Stroke],
    *,
    junction_tolerance: float,
    branches: list[CenterlineBranch] | None = None,
) -> list[Stroke]:
    # Each method receives an independent copy because orderers may reverse paths.
    strokes = copy.deepcopy(source_strokes)
    if method == "outer_to_inner":
        if branches is not None:
            return order_outer_to_inner(branches)
        synthetic = [
            CenterlineBranch(
                branch_id=i,
                points=list(s),
                start_node_id=i * 2,
                end_node_id=i * 2 + 1,
                component_id=i,
                is_loop=(len(s) > 2 and s[0] == s[-1]),
            )
            for i, s in enumerate(strokes)
        ]
        return order_outer_to_inner(synthetic)
    if method == "continuity_greedy":
        return order_continuity_greedy(
            strokes,
            junction_tolerance=junction_tolerance,
        )
    return ORDERERS[method](strokes)


def modeled_drawing_duration(
    trajectory_lengths: list[float],
    *,
    min_stroke_duration: float,
    delay_between_strokes: float,
) -> float:
    stroke_time = math.fsum(
        max(min_stroke_duration, 0.04 * math.sqrt(length))
        for length in trajectory_lengths
    )
    lift_time = max(0, len(trajectory_lengths) - 1) * delay_between_strokes
    return stroke_time + lift_time


def evaluate_method(
    *,
    source_image: Path,
    image_shape: tuple[int, int],
    source_strokes: list[Stroke],
    method: str,
    quantizer: ErrorFeedbackQuantizer,
    separator_token_id: int,
    junction_tolerance: float,
    normalization_extent: float,
    min_stroke_duration: float,
    delay_between_strokes: float,
    branches: list[CenterlineBranch] | None = None,
) -> dict[str, object]:
    ordered = apply_orderer(
        method,
        source_strokes,
        junction_tolerance=junction_tolerance,
        branches=branches,
    )
    if not ordered:
        raise ValueError(f"{method} produced no output for {source_image}")
    if any(len(stroke) < 2 for stroke in ordered):
        raise ValueError(f"{method} produced a stroke with fewer than two points")

    input_stroke_count = len(source_strokes)
    output_stroke_count = len(ordered)
    branch_join_count = input_stroke_count - output_stroke_count
    if branch_join_count < 0:
        raise ValueError(
            f"{method} increased stroke count from "
            f"{input_stroke_count} to {output_stroke_count}"
        )

    lengths = [trajectory_length(stroke) for stroke in ordered]
    pen_lift_distances = [
        euclidean_distance(current[-1], following[0])
        for current, following in zip(ordered, ordered[1:])
    ]

    stroke5, _ = strokes_to_stroke5(
        ordered,
        canvas_shape=image_shape,
        normalization_extent=normalization_extent,
    )
    tokens = quantizer.encode(stroke5)

    return {
        "source_image": str(source_image),
        "method": method,
        "input_stroke_count": input_stroke_count,
        "output_stroke_count": output_stroke_count,
        "branch_join_count": branch_join_count,
        "branch_join_rate": branch_join_count / input_stroke_count,
        "pen_lift_count": max(0, output_stroke_count - 1),
        "joinable_pen_lift_count": sum(
            distance <= junction_tolerance for distance in pen_lift_distances
        ),
        "mean_continuous_trajectory_length_px": float(fmean(lengths)),
        "median_continuous_trajectory_length_px": float(median(lengths)),
        "max_continuous_trajectory_length_px": float(max(lengths)),
        "output_point_count": sum(len(stroke) for stroke in ordered),
        "token_length": int(len(tokens)),
        "separator_token_count": int(
            np.count_nonzero(tokens == separator_token_id)
        ),
        "modeled_drawing_duration_seconds": modeled_drawing_duration(
            lengths,
            min_stroke_duration=min_stroke_duration,
            delay_between_strokes=delay_between_strokes,
        ),
    }


def aggregate_records(
    records: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method in METHODS:
        method_records = [record for record in records if record["method"] == method]
        if not method_records:
            raise ValueError(f"No records were produced for {method}")

        method_summary: dict[str, dict[str, float]] = {}
        for metric in METRICS:
            values = [float(record[metric]) for record in method_records]
            method_summary[metric] = {
                "mean": float(fmean(values)),
                "median": float(median(values)),
                "minimum": float(min(values)),
                "maximum": float(max(values)),
            }
        summary[method] = method_summary
    return summary


def continuity_head_to_head(
    records: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, int]]]:
    images = sorted({str(record["source_image"]) for record in records})
    indexed = {
        (str(record["source_image"]), str(record["method"])): record
        for record in records
    }
    comparisons: dict[str, dict[str, dict[str, int]]] = {}

    for metric, specification in METRICS.items():
        direction = specification["direction"]
        metric_comparisons: dict[str, dict[str, int]] = {}
        for baseline in METHODS[1:]:
            wins = ties = losses = 0
            for image in images:
                continuity_value = float(indexed[(image, "continuity_greedy")][metric])
                baseline_value = float(indexed[(image, baseline)][metric])
                if math.isclose(
                    continuity_value,
                    baseline_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    ties += 1
                elif (
                    direction == "lower" and continuity_value < baseline_value
                ) or (
                    direction == "higher" and continuity_value > baseline_value
                ):
                    wins += 1
                else:
                    losses += 1
            metric_comparisons[baseline] = {
                "continuity_wins": wins,
                "ties": ties,
                "continuity_losses": losses,
            }
        comparisons[metric] = metric_comparisons
    return comparisons


def format_metric_value(metric: str, value: float) -> str:
    if metric == "branch_join_rate":
        return f"{value:.2%}"
    if "count" in metric or metric == "token_length":
        return f"{value:,.2f}"
    return f"{value:,.2f}"


def print_macro_means(
    summary: dict[str, dict[str, dict[str, float]]],
) -> None:
    print("\nMacro mean across evaluated sketches")
    print(
        f"{'Metric':<45}"
        f"{'Continuity':>15}"
        f"{'Directional':>15}"
        f"{'NN-greedy':>15}"
        f"{'TSP':>15}"
    )
    print("-" * 105)
    for metric, specification in METRICS.items():
        values = [summary[method][metric]["mean"] for method in METHODS]
        print(
            f"{specification['label']:<45}"
            f"{values[0]:>15.4f}"
            f"{values[1]:>15.4f}"
            f"{values[2]:>15.4f}"
            f"{values[3]:>15.4f}"
        )


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown_summary(
    path: Path,
    report: dict[str, object],
) -> None:
    configuration = report["configuration"]
    summary = report["summary"]
    head_to_head = report["continuity_head_to_head"]

    lines = [
        "# Stroke-Ordering Quantitative Evaluation",
        "",
        "## Evaluation configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    configuration_rows = (
        ("Images evaluated", report["images_evaluated"]),
        ("Sketch directory", configuration["sketches_dir"]),
        ("Codebook", configuration["codebook"]),
        ("RDP epsilon", configuration["epsilon"]),
        ("Vectorizer", configuration["vectorizer"]),
        ("Threshold profile", configuration["threshold_profile"]),
        ("Junction tolerance", f"{configuration['junction_tolerance_px']} px"),
        ("Normalization extent", configuration["normalization_extent"]),
        ("Sampling seed", configuration["seed"]),
    )
    for label, value in configuration_rows:
        lines.append(f"| {label} | {markdown_escape(value)} |")

    lines.extend(
        [
            "",
            "## Macro means",
            "",
            (
                "Bold values are best according to the declared metric direction. "
                "All methods used the same vectorized input."
            ),
            "",
            f"| Metric | Better | {' | '.join(METHOD_LABELS[m] for m in METHODS)} |",
            "|---|---:|" + "---:|" * len(METHODS),
        ]
    )

    for metric, specification in METRICS.items():
        values = {
            method: float(summary[method][metric]["mean"])
            for method in METHODS
        }
        best = (
            min(values.values())
            if specification["direction"] == "lower"
            else max(values.values())
        )
        cells = []
        for method in METHODS:
            value = values[method]
            rendered = format_metric_value(metric, value)
            if math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12):
                rendered = f"**{rendered}**"
            cells.append(rendered)
        lines.append(
            f"| {specification['label']} | {specification['direction'].title()} | "
            + " | ".join(cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Continuity-greedy head-to-head",
            "",
            "Wins and losses are counted per sketch from continuity-greedy's perspective.",
            "",
            "| Metric | Baseline | Wins | Ties | Losses |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for metric, specification in METRICS.items():
        for baseline in METHODS[1:]:
            comparison = head_to_head[metric][baseline]
            lines.append(
                f"| {specification['label']} | {METHOD_LABELS[baseline]} | "
                f"{comparison['continuity_wins']} | "
                f"{comparison['ties']} | "
                f"{comparison['continuity_losses']} |"
            )

    skipped = report["skipped_images"]
    if skipped:
        lines.extend(
            [
                "",
                "## Skipped images",
                "",
                "| Image | Reason |",
                "|---|---|",
            ]
        )
        for item in skipped:
            lines.append(
                f"| {markdown_escape(item['source_image'])} | "
                f"{markdown_escape(item['reason'])} |"
            )

    lines.extend(
        [
            "",
            "## Evidence files",
            "",
            f"- Complete JSON report: `{configuration['json_output']}`",
            f"- This Markdown summary: `{configuration['markdown_output']}`",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    markdown_output = args.markdown_output or args.output.with_suffix(".md")
    if args.output.resolve() == markdown_output.resolve():
        raise ValueError("JSON and Markdown output paths must be different")

    image_paths = select_images(
        args.sketches_dir,
        samples=args.samples,
        seed=args.seed,
    )

    if not args.codebook.is_file():
        raise FileNotFoundError(f"Codebook not found: {args.codebook}")
    codebook = np.load(args.codebook, allow_pickle=False)
    quantizer = ErrorFeedbackQuantizer(codebook)
    separator_token_id = len(codebook) + 1

    records: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for image_index, image_path in enumerate(image_paths, start=1):
        print(
            f"[{image_index:>3}/{len(image_paths)}] {image_path.name}",
            flush=True,
        )
        image = read_grayscale_image(image_path)
        branches, _ = vectorize_image_with_stats(
            image_path,
            epsilon=args.epsilon,
            method="centerline",
            threshold_profile=args.threshold_profile,
            structured=True,
        )
        source_strokes = [list(b.points) for b in branches if len(b.points) >= 2]
        source_strokes = [list(stroke) for stroke in source_strokes if len(stroke) >= 2]
        if not source_strokes:
            skipped.append(
                {
                    "source_image": str(image_path),
                    "reason": "centerline vectorization produced no valid strokes",
                }
            )
            continue

        for method in METHODS:
            records.append(
                evaluate_method(
                    source_image=image_path,
                    image_shape=image.shape,
                    source_strokes=source_strokes,
                    method=method,
                    quantizer=quantizer,
                    separator_token_id=separator_token_id,
                    junction_tolerance=args.junction_tolerance,
                    normalization_extent=args.normalization_extent,
                    min_stroke_duration=args.min_stroke_duration,
                    delay_between_strokes=args.delay_between_strokes,
                    branches=branches,
                )
            )

    if not records:
        raise RuntimeError("No non-empty sketches were available for evaluation")

    summary = aggregate_records(records)
    head_to_head = continuity_head_to_head(records)
    configuration = {
        "sketches_dir": str(args.sketches_dir),
        "codebook": str(args.codebook),
        "json_output": str(args.output),
        "markdown_output": str(markdown_output),
        "requested_samples": args.samples,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "vectorizer": "centerline",
        "threshold_profile": args.threshold_profile,
        "junction_tolerance_px": args.junction_tolerance,
        "normalization_extent": args.normalization_extent,
        "min_stroke_duration_seconds": args.min_stroke_duration,
        "delay_between_strokes_seconds": args.delay_between_strokes,
        "methods": list(METHODS),
    }
    report = {
        "configuration": configuration,
        "metric_definitions": METRICS,
        "images_evaluated": len(records) // len(METHODS),
        "skipped_images": skipped,
        "summary": summary,
        "continuity_head_to_head": head_to_head,
        "per_sketch": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(markdown_output, report)

    print_macro_means(summary)
    print(f"\nJSON evidence:     {args.output}")
    print(f"Markdown summary: {markdown_output}")


if __name__ == "__main__":
    main()