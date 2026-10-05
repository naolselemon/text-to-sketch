"""Render publication-ready figures from an ordering evaluation JSON report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, PercentFormatter

METHODS = ("continuity_greedy", "outer_to_inner", "directional_bias", "nn_greedy", "tsp")
BASELINES = METHODS[1:]
METHOD_LABELS = {
    "continuity_greedy": "Continuity-greedy",
    "outer_to_inner": "Outer-to-inner",
    "directional_bias": "Directional bias",
    "nn_greedy": "NN-greedy",
    "tsp": "TSP",
}
METHOD_COLORS = {
    "continuity_greedy": "#0072B2",
    "outer_to_inner": "#D55E00",
    "directional_bias": "#E69F00",
    "nn_greedy": "#009E73",
    "tsp": "#CC79A7",
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
        "label": "Mean continuous trajectory length",
        "direction": "higher",
    },
    "median_continuous_trajectory_length_px": {
        "label": "Median continuous trajectory length",
        "direction": "higher",
    },
    "max_continuous_trajectory_length_px": {
        "label": "Maximum continuous trajectory length",
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
        "label": "Modeled drawing duration",
        "direction": "lower",
    },
}
CORE_METRICS = (
    "output_stroke_count",
    "pen_lift_count",
    "mean_continuous_trajectory_length_px",
    "output_point_count",
    "token_length",
    "modeled_drawing_duration_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-ready charts from an ordering JSON report."
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="JSON report produced by evaluate_ordering_quantitative.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will receive figures and visualization_manifest.json.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
        help="One or more output formats. Default: png pdf.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Raster resolution for PNG output. Default: 240.",
    )
    parser.add_argument(
        "--title",
        default="Stroke-Ordering Quantitative Evaluation",
        help="Main title used across the figure suite.",
    )
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    return args


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {path}")
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)

    required = {
        "configuration",
        "images_evaluated",
        "summary",
        "continuity_head_to_head",
        "per_sketch",
    }
    missing = required.difference(report)
    if missing:
        raise ValueError(f"Evaluation report is missing keys: {sorted(missing)}")

    missing_methods = [m for m in METHODS if m not in report["summary"]]
    if missing_methods:
        raise ValueError(f"Evaluation summary is missing methods: {missing_methods}")
    for metric in METRICS:
        for method in METHODS:
            if metric not in report["summary"][method]:
                raise ValueError(f"Summary for {method} is missing metric: {metric}")
        if metric not in report["continuity_head_to_head"]:
            raise ValueError(f"Head-to-head results are missing metric: {metric}")
    if not report["per_sketch"]:
        raise ValueError("Evaluation report contains no per-sketch records")
    return report


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#F8FAFC",
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0F172A",
            "legend.frameon": False,
            "savefig.facecolor": "#F8FAFC",
            "savefig.edgecolor": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def metric_mean(report: dict[str, Any], method: str, metric: str) -> float:
    return float(report["summary"][method][metric]["mean"])


def format_value(metric: str, value: float) -> str:
    if metric == "branch_join_rate":
        return f"{value:.1%}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def axis_number_formatter(value: float, _position: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def is_best(value: float, values: list[float], direction: str) -> bool:
    best = min(values) if direction == "lower" else max(values)
    return math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12)


def subtitle(report: dict[str, Any]) -> str:
    config = report["configuration"]
    return (
        f"n={report['images_evaluated']} sketches  |  "
        f"centerline  |  epsilon={config.get('epsilon', 'n/a')}  |  "
        f"threshold={config.get('threshold_profile', 'n/a')}  |  "
        f"junction tolerance={config.get('junction_tolerance_px', 'n/a')} px"
    )


def save_figure(
    figure: plt.Figure,
    *,
    output_dir: Path,
    stem: str,
    title: str,
    formats: tuple[str, ...],
    dpi: int,
) -> dict[str, Any]:
    files: list[str] = []
    for extension in formats:
        destination = output_dir / f"{stem}.{extension}"
        save_options: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.22,
        }
        if extension == "png":
            save_options["dpi"] = dpi
        figure.savefig(destination, **save_options)
        files.append(str(destination))
    plt.close(figure)
    return {"id": stem, "title": title, "files": files}


def build_executive_dashboard(
    report: dict[str, Any],
    *,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
    main_title: str,
) -> dict[str, Any]:
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes_flat = axes.ravel()

    for axis, metric in zip(axes_flat, CORE_METRICS):
        specification = METRICS[metric]
        values = [metric_mean(report, method, metric) for method in METHODS]
        positions = np.arange(len(METHODS))
        bars = axis.barh(
            positions,
            values,
            color=[METHOD_COLORS[method] for method in METHODS],
            height=0.62,
            alpha=0.94,
        )
        axis.set_yticks(positions, [METHOD_LABELS[method] for method in METHODS])
        axis.invert_yaxis()
        axis.set_title(
            f"{specification['label']}\n{specification['direction'].title()} is better",
            loc="left",
            pad=10,
        )
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_formatter(FuncFormatter(axis_number_formatter))
        maximum = max(values)
        axis.set_xlim(0, maximum * 1.22 if maximum > 0 else 1.0)

        for bar, value in zip(bars, values):
            if is_best(value, values, specification["direction"]):
                bar.set_edgecolor("#0F172A")
                bar.set_linewidth(1.8)
            axis.text(
                bar.get_width() + maximum * 0.025,
                bar.get_y() + bar.get_height() / 2,
                format_value(metric, value),
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold" if is_best(
                    value, values, specification["direction"]
                ) else "normal",
            )

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.suptitle(main_title, fontsize=22, fontweight="bold", x=0.055, ha="left")
    figure.text(0.055, 0.935, subtitle(report), fontsize=10.5, color="#475569")
    figure.text(
        0.055,
        0.02,
        "Outlined bars are best for their metric. Raw macro means; no cross-metric normalization.",
        fontsize=9,
        color="#64748B",
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.98,
        top=0.87,
        bottom=0.09,
        hspace=0.45,
        wspace=0.42,
    )
    return save_figure(
        figure,
        output_dir=output_dir,
        stem="01_executive_dashboard",
        title="Executive metric dashboard",
        formats=formats,
        dpi=dpi,
    )


def relative_improvement(continuity: float, baseline: float, direction: str) -> float:
    if math.isclose(baseline, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return float("nan")
    if direction == "lower":
        return (baseline - continuity) / abs(baseline) * 100.0
    return (continuity - baseline) / abs(baseline) * 100.0


def build_relative_improvement_chart(
    report: dict[str, Any],
    *,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
    main_title: str,
) -> dict[str, Any]:
    figure, axis = plt.subplots(figsize=(14, 8.5))
    positions = np.arange(len(CORE_METRICS))
    bar_height = 0.8 / max(1, len(BASELINES))

    all_values: list[float] = []
    for baseline_index, baseline in enumerate(BASELINES):
        improvements = []
        for metric in CORE_METRICS:
            improvements.append(
                relative_improvement(
                    metric_mean(report, "continuity_greedy", metric),
                    metric_mean(report, baseline, metric),
                    METRICS[metric]["direction"],
                )
            )
        all_values.extend(value for value in improvements if math.isfinite(value))
        offsets = positions + (baseline_index - (len(BASELINES) - 1) / 2.0) * bar_height
        bars = axis.barh(
            offsets,
            improvements,
            height=bar_height * 0.86,
            color=METHOD_COLORS[baseline],
            label=f"vs {METHOD_LABELS[baseline]}",
            alpha=0.94,
        )
        for bar, value in zip(bars, improvements):
            if not math.isfinite(value):
                continue
            axis.text(
                value + (1.2 if value >= 0 else -1.2),
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.1f}%",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
                fontweight="semibold",
            )

    axis.axvline(0, color="#0F172A", linewidth=1.2)
    axis.set_yticks(
        positions,
        [METRICS[metric]["label"] for metric in CORE_METRICS],
    )
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    axis.set_xlabel("Relative improvement; positive values favor continuity-greedy")
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    minimum = min([0.0, *all_values])
    maximum = max([0.0, *all_values])
    span = max(maximum - minimum, 10.0)
    axis.set_xlim(minimum - span * 0.08, maximum + span * 0.18)
    axis.legend(loc="lower right", ncol=3)

    figure.suptitle(
        f"{main_title}: Relative Advantage",
        fontsize=20,
        fontweight="bold",
        x=0.08,
        ha="left",
    )
    figure.text(0.08, 0.925, subtitle(report), fontsize=10.5, color="#475569")
    figure.subplots_adjust(left=0.28, right=0.97, top=0.86, bottom=0.13)
    return save_figure(
        figure,
        output_dir=output_dir,
        stem="02_relative_improvement",
        title="Relative improvement against each baseline",
        formats=formats,
        dpi=dpi,
    )


def build_head_to_head_matrix(
    report: dict[str, Any],
    *,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
    main_title: str,
) -> dict[str, Any]:
    metric_names = list(METRICS)
    matrix = np.zeros((len(metric_names), len(BASELINES)), dtype=float)
    annotations: list[list[str]] = []

    for row_index, metric in enumerate(metric_names):
        row_annotations = []
        for column_index, baseline in enumerate(BASELINES):
            comparison = report["continuity_head_to_head"][metric][baseline]
            wins = int(comparison["continuity_wins"])
            ties = int(comparison["ties"])
            losses = int(comparison["continuity_losses"])
            total = wins + ties + losses
            matrix[row_index, column_index] = wins / total if total else 0.0
            row_annotations.append(f"{wins}W / {ties}T / {losses}L")
        annotations.append(row_annotations)

    color_map = LinearSegmentedColormap.from_list(
        "professional_win_rate",
        ("#B91C1C", "#FEF3C7", "#15803D"),
    )
    figure, axis = plt.subplots(figsize=(12.5, 10))
    image = axis.imshow(matrix, cmap=color_map, vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(
        np.arange(len(BASELINES)),
        [METHOD_LABELS[baseline] for baseline in BASELINES],
    )
    axis.set_yticks(
        np.arange(len(metric_names)),
        [METRICS[metric]["label"] for metric in metric_names],
    )
    axis.tick_params(axis="x", labelrotation=0)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                annotations[row_index][column_index],
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if value <= 0.20 or value >= 0.75 else "#0F172A",
            )

    axis.set_xticks(np.arange(-0.5, len(BASELINES), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(metric_names), 1), minor=True)
    axis.grid(which="minor", color="#FFFFFF", linewidth=2.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    color_bar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    color_bar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    color_bar.set_label("Continuity-greedy per-sketch win rate")

    figure.suptitle(
        f"{main_title}: Head-to-Head Consistency",
        fontsize=20,
        fontweight="bold",
        x=0.12,
        ha="left",
    )
    figure.text(
        0.12,
        0.925,
        "Each cell reports continuity-greedy wins / ties / losses against one baseline.",
        fontsize=10.5,
        color="#475569",
    )
    figure.subplots_adjust(left=0.34, right=0.91, top=0.88, bottom=0.09)
    return save_figure(
        figure,
        output_dir=output_dir,
        stem="03_head_to_head_matrix",
        title="Per-sketch head-to-head consistency matrix",
        formats=formats,
        dpi=dpi,
    )


def build_distribution_panels(
    report: dict[str, Any],
    *,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
    main_title: str,
) -> dict[str, Any]:
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes_flat = axes.ravel()
    records = report["per_sketch"]
    random_generator = np.random.default_rng(42)

    for axis, metric in zip(axes_flat, CORE_METRICS):
        values_by_method = [
            [
                float(record[metric])
                for record in records
                if record["method"] == method
            ]
            for method in METHODS
        ]
        positions = np.arange(len(METHODS))
        boxplot = axis.boxplot(
            values_by_method,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            medianprops={"color": "#0F172A", "linewidth": 1.6},
            meanprops={
                "marker": "D",
                "markerfacecolor": "#FFFFFF",
                "markeredgecolor": "#0F172A",
                "markersize": 4.5,
            },
            whiskerprops={"color": "#64748B", "linewidth": 1.1},
            capprops={"color": "#64748B", "linewidth": 1.1},
        )
        for patch, method in zip(boxplot["boxes"], METHODS):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.60)
            patch.set_edgecolor(METHOD_COLORS[method])
            patch.set_linewidth(1.4)

        for method_index, (method, values) in enumerate(
            zip(METHODS, values_by_method)
        ):
            jitter = random_generator.normal(0.0, 0.035, size=len(values))
            axis.scatter(
                np.full(len(values), method_index) + jitter,
                values,
                s=22,
                color=METHOD_COLORS[method],
                alpha=0.72,
                edgecolors="#FFFFFF",
                linewidths=0.45,
                zorder=3,
            )

        specification = METRICS[metric]
        axis.set_title(
            f"{specification['label']}\n{specification['direction'].title()} is better",
            loc="left",
            pad=10,
        )
        axis.set_xticks(
            positions,
            [METHOD_LABELS[method] for method in METHODS],
            rotation=18,
            ha="right",
        )
        axis.yaxis.set_major_formatter(FuncFormatter(axis_number_formatter))
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.suptitle(
        f"{main_title}: Per-Sketch Distributions",
        fontsize=20,
        fontweight="bold",
        x=0.055,
        ha="left",
    )
    figure.text(0.055, 0.935, subtitle(report), fontsize=10.5, color="#475569")
    figure.text(
        0.055,
        0.02,
        "Boxes show interquartile range; center lines are medians; diamonds are means; dots are sketches.",
        fontsize=9,
        color="#64748B",
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.87,
        bottom=0.12,
        hspace=0.50,
        wspace=0.30,
    )
    return save_figure(
        figure,
        output_dir=output_dir,
        stem="04_per_sketch_distributions",
        title="Per-sketch metric distributions",
        formats=formats,
        dpi=dpi,
    )


def main() -> None:
    args = parse_args()
    formats = tuple(dict.fromkeys(args.formats))
    report = load_report(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_publication_style()

    figure_builders = (
        build_executive_dashboard,
        build_relative_improvement_chart,
        build_head_to_head_matrix,
        build_distribution_panels,
    )
    figures = []
    for builder in figure_builders:
        figure_record = builder(
            report,
            output_dir=args.output_dir,
            formats=formats,
            dpi=args.dpi,
            main_title=args.title,
        )
        figures.append(figure_record)
        print(f"Created {figure_record['title']}")
        for path in figure_record["files"]:
            print(f"  {path}")

    manifest = {
        "source_report": str(args.report),
        "output_directory": str(args.output_dir),
        "images_evaluated": int(report["images_evaluated"]),
        "formats": list(formats),
        "png_dpi": args.dpi,
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "figures": figures,
    }
    manifest_path = args.output_dir / "visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Visualization manifest: {manifest_path}")


if __name__ == "__main__":
    main()