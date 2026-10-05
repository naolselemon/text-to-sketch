"""Validate the fixed 100-sketch anchored-V3 human review release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from services.anchored_sketch_data.contract import (
    TOKEN_LAYOUT,
    validate_anchored_v3_runtime_config,
)


REVIEW_SCHEMA_VERSION = 1
REVIEW_CRITERIA = ("face_shape", "eyes", "hair", "major_accessories")
EXPECTED_SAMPLE_COUNT = 100
MINIMUM_PASSES = 95
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build_review_template(
    sample_ids: Sequence[str],
    *,
    evaluation_report_sha256: str,
    plot_records: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Create the stable manual-review input format in plot order."""

    values = [str(sample_id) for sample_id in sample_ids]
    if not values or len(values) != len(set(values)):
        raise ValueError("Human-review sample IDs must be non-empty and unique")
    if _SHA256.fullmatch(str(evaluation_report_sha256)) is None:
        raise ValueError("evaluation_report_sha256 must be a SHA-256 digest")
    if len(plot_records) != len(values):
        raise ValueError("Every human-review sample must have one plot record")
    plots_by_id = {str(record.get("sample_id")): record for record in plot_records}
    if set(plots_by_id) != set(values) or len(plots_by_id) != len(values):
        raise ValueError("Plot records must match human-review sample IDs exactly")
    for sample_id, record in plots_by_id.items():
        if not str(record.get("plot_path", "")):
            raise ValueError(f"Plot record {sample_id!r} has no path")
        if _SHA256.fullmatch(str(record.get("plot_sha256", ""))) is None:
            raise ValueError(f"Plot record {sample_id!r} has no valid SHA-256")
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "evaluation_report_sha256": str(evaluation_report_sha256),
        "instructions": (
            "Set every criterion to true only when prediction preserves the target. "
            "Use true for major_accessories when the target has none."
        ),
        "reviews": [
            {
                "sample_id": sample_id,
                "plot_path": str(plots_by_id[sample_id]["plot_path"]),
                "plot_sha256": str(plots_by_id[sample_id]["plot_sha256"]),
                **{criterion: None for criterion in REVIEW_CRITERIA},
            }
            for sample_id in values
        ],
    }


def evaluate_human_reviews(
    reviews_payload: Mapping[str, Any],
    expected_sample_ids: Sequence[str],
    *,
    minimum_passes: int = 95,
) -> dict[str, Any]:
    """Validate identities and criteria, then count all-criterion passes."""

    expected = [str(sample_id) for sample_id in expected_sample_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("Evaluation report contains duplicate review sample IDs")
    if int(reviews_payload.get("schema_version", -1)) != REVIEW_SCHEMA_VERSION:
        raise ValueError("Unsupported human-review schema version")
    reviews = reviews_payload.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("Human-review payload must contain a reviews list")
    review_ids = [
        str(review.get("sample_id"))
        for review in reviews
        if isinstance(review, Mapping)
    ]
    if len(review_ids) != len(reviews) or len(review_ids) != len(set(review_ids)):
        raise ValueError("Human-review entries must have unique sample IDs")
    if set(review_ids) != set(expected):
        missing = sorted(set(expected) - set(review_ids))
        unexpected = sorted(set(review_ids) - set(expected))
        raise ValueError(
            f"Human-review samples do not match evaluation plots: "
            f"missing={missing} unexpected={unexpected}"
        )
    if minimum_passes < 0 or minimum_passes > len(expected):
        raise ValueError("minimum_passes must be within the reviewed sample count")

    by_id = {str(review["sample_id"]): review for review in reviews}
    criterion_passes = {criterion: 0 for criterion in REVIEW_CRITERIA}
    passed_ids: list[str] = []
    failed_ids: list[str] = []
    for sample_id in expected:
        review = by_id[sample_id]
        for criterion in REVIEW_CRITERIA:
            if not isinstance(review.get(criterion), bool):
                raise ValueError(
                    f"Review {sample_id!r} criterion {criterion!r} must be true or false"
                )
            criterion_passes[criterion] += int(review[criterion])
        if all(bool(review[criterion]) for criterion in REVIEW_CRITERIA):
            passed_ids.append(sample_id)
        else:
            failed_ids.append(sample_id)

    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "sample_count": len(expected),
        "minimum_passes": int(minimum_passes),
        "pass_count": len(passed_ids),
        "passed": len(passed_ids) >= minimum_passes,
        "criterion_pass_counts": criterion_passes,
        "passed_sample_ids": passed_ids,
        "failed_sample_ids": failed_ids,
    }


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_release_evaluation_report(
    evaluation: Mapping[str, Any],
) -> list[str]:
    """Require the exact held-out V3 report that produced the review plots."""

    metadata = evaluation.get("metadata")
    records = evaluation.get("records")
    metrics = evaluation.get("metrics")
    if not isinstance(metadata, Mapping) or not isinstance(records, list):
        raise ValueError("Human review requires a metrics report with metadata and records")
    if not isinstance(metrics, Mapping):
        raise ValueError("Human review evaluation report has no metrics")
    expected_metadata = {
        "format_type": "anchored_v3",
        "format_version": 3,
        "split": "test",
        "decode_mode": "free-running",
        "limit_batches": 1.0,
        "max_generation_length": None,
        "enforce_v3_gates": True,
        "allow_legacy_checkpoint": False,
    }
    mismatches = [
        f"{name}={metadata.get(name)!r} expected {expected!r}"
        for name, expected in expected_metadata.items()
        if metadata.get(name) != expected
    ]
    checkpoint_contract = metadata.get("checkpoint_contract")
    if not isinstance(checkpoint_contract, Mapping):
        mismatches.append("checkpoint_contract is missing")
    elif int(checkpoint_contract.get("token_layout_version", -1)) != 3:
        mismatches.append("checkpoint token layout is not V3")
    else:
        compatibility_config = checkpoint_contract.get("compatibility_config")
        if not isinstance(compatibility_config, Mapping):
            mismatches.append("checkpoint compatibility_config is missing")
        else:
            try:
                validate_anchored_v3_runtime_config(compatibility_config)
            except ValueError as exc:
                mismatches.append(f"checkpoint is not canonical anchored V3: {exc}")
            checkpoint_data = compatibility_config.get("data")
            checkpoint_dataset = (
                checkpoint_data.get("dataset")
                if isinstance(checkpoint_data, Mapping)
                else None
            )
            configured_minimum = (
                checkpoint_dataset.get("minimum_source_sketches")
                if isinstance(checkpoint_dataset, Mapping)
                else None
            )
            report_minimum = metadata.get("minimum_source_sketches")
            if (
                isinstance(report_minimum, bool)
                or not isinstance(report_minimum, int)
                or report_minimum <= 0
            ):
                mismatches.append("evaluation minimum_source_sketches is not positive")
            elif configured_minimum != report_minimum:
                mismatches.append(
                    "evaluation minimum_source_sketches does not match checkpoint"
                )
    if metadata.get("token_layout") != TOKEN_LAYOUT.to_dict():
        mismatches.append("evaluation token layout is not canonical anchored V3")
    if metadata.get("decoder_memory_source") != "encoder":
        mismatches.append("evaluation decoder memory source is not encoder")
    if len(records) < EXPECTED_SAMPLE_COUNT:
        mismatches.append(
            f"records={len(records)} expected at least {EXPECTED_SAMPLE_COUNT}"
        )
    if records:
        try:
            f1_values = [float(record["geometry_f1_2px"]) for record in records]
            chamfer_values = [
                float(record["symmetric_chamfer_px"]) for record in records
            ]
            premature_values = [float(record["premature_eos"]) for record in records]
            max_hit_values = [float(record["max_length_hit"]) for record in records]
            long_f1 = [
                float(record["geometry_f1_2px"])
                for record in records
                if 2049 <= int(record["target_length"]) <= 4096
            ]
        except (KeyError, TypeError, ValueError) as exc:
            mismatches.append(f"records lack release metrics: {exc}")
        else:
            record_checks = {
                "median_f1_2px": float(np.quantile(f1_values, 0.5)) >= 0.95,
                "long_median_f1_2px": bool(long_f1)
                and float(np.quantile(long_f1, 0.5)) >= 0.90,
                "p10_f1_2px": float(np.quantile(f1_values, 0.1)) >= 0.85,
                "p95_chamfer_px": float(np.quantile(chamfer_values, 0.95)) <= 3.0,
                "premature_eos_rate": float(np.mean(premature_values)) <= 0.02,
                "max_length_hit_rate": float(np.mean(max_hit_values)) <= 0.02,
            }
            failed = [name for name, passed in record_checks.items() if not passed]
            if failed:
                mismatches.append("record-level V3 gates failed: " + ", ".join(failed))
    if mismatches:
        raise ValueError("Invalid human-review evaluation report: " + "; ".join(mismatches))
    return [str(record["sample_id"]) for record in records[:EXPECTED_SAMPLE_COUNT]]


def validate_review_binding(
    reviews_payload: Mapping[str, Any],
    *,
    evaluation_report_sha256: str,
) -> None:
    """Bind every review decision to exact evaluation and plot bytes."""

    if reviews_payload.get("evaluation_report_sha256") != evaluation_report_sha256:
        raise ValueError("Human-review template does not match the evaluation report hash")
    reviews = reviews_payload.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("Human-review payload must contain a reviews list")
    for review in reviews:
        if not isinstance(review, Mapping):
            raise ValueError("Human-review entries must be mappings")
        plot_path = Path(str(review.get("plot_path", "")))
        expected_hash = str(review.get("plot_sha256", ""))
        if not plot_path.is_file():
            raise ValueError(f"Human-review plot does not exist: {plot_path}")
        if _SHA256.fullmatch(expected_hash) is None or sha256_file(plot_path) != expected_hash:
            raise ValueError(f"Human-review plot hash mismatch: {plot_path}")


def main() -> int:
    args = parse_args()
    evaluation_path = Path(args.evaluation_report)
    reviews_path = Path(args.reviews)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    expected_ids = validate_release_evaluation_report(evaluation)
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    evaluation_hash = sha256_file(evaluation_path)
    validate_review_binding(
        reviews,
        evaluation_report_sha256=evaluation_hash,
    )
    result = evaluate_human_reviews(
        reviews,
        expected_ids,
        minimum_passes=MINIMUM_PASSES,
    )
    report = {
        **result,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_report": str(evaluation_path),
        "evaluation_report_sha256": evaluation_hash,
        "reviews": str(reviews_path),
        "reviews_sha256": sha256_file(reviews_path),
    }
    output = write_json_atomic(args.output, report)
    print(
        f"human_review={'PASS' if result['passed'] else 'FAIL'} "
        f"passes={result['pass_count']}/{result['sample_count']} report={output}"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
