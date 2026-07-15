"""Build a deterministic anchored-V3 train-source scaling-curve report.

Input reports must be complete held-out test evaluations produced by
``scripts.sketchformer.evaluate``.  This utility deliberately consumes the
per-sample records as well as the aggregate metrics so a stale or manually
edited summary cannot enter the scaling curve unnoticed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCALING_CURVE_SCHEMA_VERSION = 1
LEGACY_REPORT_SPECS: tuple[tuple[str, int | None], ...] = (
    ("1400", 1_400),
    ("5000", 5_000),
    ("10000", 10_000),
    ("full", None),
)
# Public compatibility alias retained for callers that imported the original
# fixed study definition.  New studies derive their points from report labels.
REPORT_SPECS = LEGACY_REPORT_SPECS
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SUMMARY_TOLERANCE = 1.0e-6
_METRIC_KEYS: tuple[tuple[str, str], ...] = (
    ("median_f1_2px", "geometry_f1_2px_median"),
    ("p10_f1_2px", "geometry_f1_2px_p10"),
    ("p95_chamfer_px", "symmetric_chamfer_px_p95"),
    ("premature_eos_rate", "premature_eos_rate"),
    ("max_length_hit_rate", "max_length_hit_rate"),
)


class ScalingCurveValidationError(ValueError):
    """Raised when an evaluation report cannot belong to the scaling study."""


@dataclass(frozen=True)
class ValidatedEvaluation:
    """The small, verified projection needed by the aggregate report."""

    label: str
    train_source_limit: int | None
    report_sha256: str
    dataset_manifest_sha256: str
    codebook_sha256: str
    sample_ids: tuple[str, ...]
    metrics: dict[str, float]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        nargs=2,
        default=None,
        metavar=("LIMIT", "PATH"),
        help=(
            "Add a scaling point. LIMIT is a positive train-source limit or "
            "'full'. Repeat for every point."
        ),
    )
    parser.add_argument(
        "--report-1400",
        default=None,
        help="Legacy alias for --report 1400 PATH.",
    )
    parser.add_argument(
        "--report-5000",
        default=None,
        help="Legacy alias for --report 5000 PATH.",
    )
    parser.add_argument(
        "--report-10000",
        default=None,
        help="Legacy alias for --report 10000 PATH.",
    )
    parser.add_argument(
        "--report-full",
        default=None,
        help="Legacy alias for --report full PATH.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def build_scaling_curve(
    report_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Validate an arbitrary nested study and return a stable payload."""

    reports = [
        load_evaluation_report(
            path,
            label=label,
            expected_train_source_limit=expected_limit,
        )
        for label, expected_limit, path in _normalize_report_specs(report_paths)
    ]
    reference = reports[0]
    for report in reports[1:]:
        if report.dataset_manifest_sha256 != reference.dataset_manifest_sha256:
            raise ScalingCurveValidationError(
                f"Report {report.label!r} dataset_manifest_sha256 does not match "
                f"report {reference.label!r}"
            )
        if report.codebook_sha256 != reference.codebook_sha256:
            raise ScalingCurveValidationError(
                f"Report {report.label!r} codebook_sha256 does not match "
                f"report {reference.label!r}"
            )
        if report.sample_ids != reference.sample_ids:
            mismatch = _first_sequence_mismatch(reference.sample_ids, report.sample_ids)
            raise ScalingCurveValidationError(
                f"Report {report.label!r} ordered test sample IDs do not match "
                f"report {reference.label!r}: {mismatch}"
            )

    sample_ids_json = json.dumps(
        reference.sample_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": SCALING_CURVE_SCHEMA_VERSION,
        "study": "anchored_v3_train_source_scaling",
        "format": {"type": "anchored_v3", "version": 3},
        "dataset_manifest_sha256": reference.dataset_manifest_sha256,
        "codebook_sha256": reference.codebook_sha256,
        "test_sample_count": len(reference.sample_ids),
        "ordered_test_sample_ids_sha256": hashlib.sha256(sample_ids_json).hexdigest(),
        "points": [
            {
                "train_source_label": report.label,
                "train_source_limit": report.train_source_limit,
                "evaluation_report_sha256": report.report_sha256,
                **report.metrics,
            }
            for report in reports
        ],
    }


def _normalize_report_specs(
    report_paths: Mapping[str, str | Path],
) -> tuple[tuple[str, int | None, str | Path], ...]:
    """Canonicalize, validate, and order arbitrary scaling points."""

    if not isinstance(report_paths, Mapping):
        raise ScalingCurveValidationError("Scaling reports must be a mapping")
    if len(report_paths) < 2:
        raise ScalingCurveValidationError(
            "Scaling study requires at least two reports, including 'full'"
        )

    normalized: dict[str, tuple[int | None, str | Path]] = {}
    for raw_label, path in report_paths.items():
        label, limit = _parse_report_label(raw_label)
        if label in normalized:
            raise ScalingCurveValidationError(
                f"Duplicate scaling report limit {label!r}"
            )
        normalized[label] = (limit, path)

    if "full" not in normalized:
        raise ScalingCurveValidationError(
            "Scaling study requires exactly one 'full' report"
        )

    ordered = sorted(
        (
            (label, limit, path)
            for label, (limit, path) in normalized.items()
        ),
        key=lambda item: (item[1] is None, item[1] if item[1] is not None else 0),
    )
    return tuple(ordered)


def _parse_report_label(raw_label: Any) -> tuple[str, int | None]:
    if not isinstance(raw_label, str):
        raise ScalingCurveValidationError(
            f"Scaling report limit must be a string; got {raw_label!r}"
        )
    label = raw_label.strip().lower()
    if label == "full":
        return "full", None
    if not label.isdecimal():
        raise ScalingCurveValidationError(
            f"Scaling report limit {raw_label!r} must be a positive integer or 'full'"
        )
    limit = int(label)
    if limit <= 0:
        raise ScalingCurveValidationError(
            f"Scaling report limit {raw_label!r} must be positive"
        )
    return str(limit), limit


def load_evaluation_report(
    path: str | Path,
    *,
    label: str,
    expected_train_source_limit: int | None,
) -> ValidatedEvaluation:
    """Read one JSON report, reject ambiguous JSON, and validate its contract."""

    report_path = Path(path)
    try:
        raw = report_path.read_bytes()
    except OSError as exc:
        raise ScalingCurveValidationError(
            f"Could not read report {label!r}: {report_path}: {exc}"
        ) from exc
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_mapping_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ScalingCurveValidationError) as exc:
        raise ScalingCurveValidationError(
            f"Report {label!r} is not strict UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ScalingCurveValidationError(f"Report {label!r} root must be an object")

    metadata = _required_mapping(payload, "metadata", label)
    metrics = _required_mapping(payload, "metrics", label)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ScalingCurveValidationError(
            f"Report {label!r} must contain non-empty per-sample records"
        )
    _validate_evaluation_metadata(metadata, label=label)
    contract = _required_mapping(metadata, "checkpoint_contract", label)
    manifest_hash, codebook_hash = _validate_checkpoint_contract(
        contract,
        label=label,
        expected_train_source_limit=expected_train_source_limit,
    )
    sample_ids, recomputed = _validate_records(records, label=label)
    official_metrics = _validate_summary_metrics(
        metrics,
        recomputed=recomputed,
        sample_count=len(sample_ids),
        label=label,
    )
    return ValidatedEvaluation(
        label=label,
        train_source_limit=expected_train_source_limit,
        report_sha256=hashlib.sha256(raw).hexdigest(),
        dataset_manifest_sha256=manifest_hash,
        codebook_sha256=codebook_hash,
        sample_ids=sample_ids,
        metrics=official_metrics,
    )


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace ``path`` with canonical, finite JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
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


def _validate_evaluation_metadata(metadata: Mapping[str, Any], *, label: str) -> None:
    expected: tuple[tuple[str, Any, type], ...] = (
        ("format_type", "anchored_v3", str),
        ("format_version", 3, int),
        ("split", "test", str),
        ("decode_mode", "free-running", str),
        ("limit_batches", 1.0, float),
        ("max_generation_length", None, type(None)),
        ("allow_legacy_checkpoint", False, bool),
    )
    mismatches = [
        f"{name}={metadata.get(name)!r} expected {value!r}"
        for name, value, expected_type in expected
        if type(metadata.get(name)) is not expected_type or metadata.get(name) != value
    ]
    if not isinstance(metadata.get("enforce_v3_gates"), bool):
        mismatches.append("enforce_v3_gates must be a boolean")
    checkpoint = metadata.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        mismatches.append("checkpoint must be a non-empty path")
    if mismatches:
        raise ScalingCurveValidationError(
            f"Report {label!r} is not a complete anchored-V3 test/free-running "
            "evaluation: " + "; ".join(mismatches)
        )


def _validate_checkpoint_contract(
    contract: Mapping[str, Any],
    *,
    label: str,
    expected_train_source_limit: int | None,
) -> tuple[str, str]:
    if type(contract.get("schema_version")) is not int or contract.get(
        "schema_version"
    ) != 1:
        raise ScalingCurveValidationError(
            f"Report {label!r} checkpoint schema_version must be 1"
        )
    if type(contract.get("token_layout_version")) is not int or contract.get(
        "token_layout_version"
    ) != 3:
        raise ScalingCurveValidationError(
            f"Report {label!r} checkpoint token_layout_version must be 3"
        )
    config = contract.get("config")
    if not isinstance(config, Mapping) or not config:
        raise ScalingCurveValidationError(
            f"Report {label!r} checkpoint config must be a non-empty object"
        )
    if not isinstance(contract.get("monitored_metrics"), Mapping):
        raise ScalingCurveValidationError(
            f"Report {label!r} checkpoint monitored_metrics must be an object"
        )
    git_commit = contract.get("git_commit")
    if not isinstance(git_commit, str) or not git_commit.strip():
        raise ScalingCurveValidationError(
            f"Report {label!r} checkpoint git_commit must be non-empty"
        )
    manifest_hash = _required_sha256(
        contract,
        "dataset_manifest_sha256",
        label=label,
    )
    codebook_hash = _required_sha256(contract, "codebook_sha256", label=label)
    compatibility = _required_mapping(contract, "compatibility_config", label)
    format_type = _nested_value(
        compatibility,
        ("data", "format", "type"),
        label=label,
    )
    format_version = _nested_value(
        compatibility,
        ("data", "format", "version"),
        label=label,
    )
    memory_source = _nested_value(
        compatibility,
        ("model", "decoder", "memory_source"),
        label=label,
    )
    if format_type != "anchored_v3" or type(format_version) is not int or format_version != 3:
        raise ScalingCurveValidationError(
            f"Report {label!r} checkpoint compatibility is not anchored V3"
        )
    if memory_source != "encoder":
        raise ScalingCurveValidationError(
            f"Report {label!r} checkpoint decoder.memory_source must be 'encoder'"
        )
    actual_limit = _nested_value(
        compatibility,
        ("data", "dataset", "train_source_limit"),
        label=label,
    )
    if expected_train_source_limit is None:
        limit_matches = actual_limit is None
    else:
        limit_matches = (
            type(actual_limit) is int and actual_limit == expected_train_source_limit
        )
    if not limit_matches:
        raise ScalingCurveValidationError(
            f"Report {label!r} checkpoint compatibility train_source_limit="
            f"{actual_limit!r}; expected {expected_train_source_limit!r}"
        )
    return manifest_hash, codebook_hash


def _validate_records(
    records: list[Any],
    *,
    label: str,
) -> tuple[tuple[str, ...], dict[str, float]]:
    sample_ids: list[str] = []
    f1_values: list[float] = []
    chamfer_values: list[float] = []
    premature_values: list[float] = []
    max_hit_values: list[float] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ScalingCurveValidationError(
                f"Report {label!r} record {index} must be an object"
            )
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ScalingCurveValidationError(
                f"Report {label!r} record {index} has no non-empty string sample_id"
            )
        sample_ids.append(sample_id)
        f1_values.append(
            _finite_number(record, "geometry_f1_2px", label=label, record=index)
        )
        chamfer_values.append(
            _finite_number(
                record,
                "symmetric_chamfer_px",
                label=label,
                record=index,
            )
        )
        premature_values.append(
            _binary_number(record, "premature_eos", label=label, record=index)
        )
        max_hit_values.append(
            _binary_number(record, "max_length_hit", label=label, record=index)
        )
    if len(sample_ids) != len(set(sample_ids)):
        raise ScalingCurveValidationError(
            f"Report {label!r} contains duplicate test sample IDs"
        )
    for index, value in enumerate(f1_values):
        if not 0.0 <= value <= 1.0:
            raise ScalingCurveValidationError(
                f"Report {label!r} record {index} geometry_f1_2px must be in [0, 1]"
            )
    for index, value in enumerate(chamfer_values):
        if value < 0.0:
            raise ScalingCurveValidationError(
                f"Report {label!r} record {index} symmetric_chamfer_px must be >= 0"
            )
    recomputed = {
        "median_f1_2px": float(np.quantile(f1_values, 0.5)),
        "p10_f1_2px": float(np.quantile(f1_values, 0.1)),
        "p95_chamfer_px": float(np.quantile(chamfer_values, 0.95)),
        "premature_eos_rate": float(np.mean(premature_values)),
        "max_length_hit_rate": float(np.mean(max_hit_values)),
    }
    return tuple(sample_ids), recomputed


def _validate_summary_metrics(
    metrics: Mapping[str, Any],
    *,
    recomputed: Mapping[str, float],
    sample_count: int,
    label: str,
) -> dict[str, float]:
    prefix = "test/free_running"
    reported_count = _finite_number(metrics, f"{prefix}/count", label=label)
    if reported_count != float(sample_count):
        raise ScalingCurveValidationError(
            f"Report {label!r} summary count={reported_count!r}; "
            f"records={sample_count}"
        )
    result: dict[str, float] = {}
    for output_name, report_name in _METRIC_KEYS:
        value = _finite_number(metrics, f"{prefix}/{report_name}", label=label)
        expected = float(recomputed[output_name])
        if not math.isclose(
            value,
            expected,
            rel_tol=_SUMMARY_TOLERANCE,
            abs_tol=_SUMMARY_TOLERANCE,
        ):
            raise ScalingCurveValidationError(
                f"Report {label!r} metric {report_name}={value:.12g} does not "
                f"match per-sample value {expected:.12g}"
            )
        result[output_name] = value
    for name in ("median_f1_2px", "p10_f1_2px", "premature_eos_rate", "max_length_hit_rate"):
        if not 0.0 <= result[name] <= 1.0:
            raise ScalingCurveValidationError(
                f"Report {label!r} metric {name} must be in [0, 1]"
            )
    if result["p95_chamfer_px"] < 0.0:
        raise ScalingCurveValidationError(
            f"Report {label!r} metric p95_chamfer_px must be >= 0"
        )
    return result


def _required_mapping(
    parent: Mapping[str, Any],
    key: str,
    label: str,
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ScalingCurveValidationError(
            f"Report {label!r} field {key!r} must be an object"
        )
    return value


def _required_sha256(
    parent: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ScalingCurveValidationError(
            f"Report {label!r} checkpoint {key} must be a lowercase SHA-256 digest"
        )
    return value


def _nested_value(
    parent: Mapping[str, Any],
    path: Sequence[str],
    *,
    label: str,
) -> Any:
    value: Any = parent
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ScalingCurveValidationError(
                f"Report {label!r} checkpoint compatibility is missing "
                + ".".join(path)
            )
        value = value[key]
    return value


def _finite_number(
    parent: Mapping[str, Any],
    key: str,
    *,
    label: str,
    record: int | None = None,
) -> float:
    value = parent.get(key)
    location = f"record {record}" if record is not None else "summary"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScalingCurveValidationError(
            f"Report {label!r} {location} field {key!r} must be numeric"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ScalingCurveValidationError(
            f"Report {label!r} {location} field {key!r} must be finite"
        )
    return result


def _binary_number(
    parent: Mapping[str, Any],
    key: str,
    *,
    label: str,
    record: int,
) -> float:
    value = _finite_number(parent, key, label=label, record=record)
    if value not in (0.0, 1.0):
        raise ScalingCurveValidationError(
            f"Report {label!r} record {record} field {key!r} must be 0 or 1"
        )
    return value


def _first_sequence_mismatch(reference: Sequence[str], candidate: Sequence[str]) -> str:
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        if expected != actual:
            return f"index {index}: {actual!r} expected {expected!r}"
    return f"sample counts differ: {len(candidate)} expected {len(reference)}"


def _mapping_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScalingCurveValidationError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ScalingCurveValidationError(f"Non-finite JSON constant {value!r}")


def _report_paths_from_args(args: argparse.Namespace) -> dict[str, str]:
    generic_reports = list(args.report or [])
    legacy_values = {
        "1400": args.report_1400,
        "5000": args.report_5000,
        "10000": args.report_10000,
        "full": args.report_full,
    }
    supplied_legacy = {label: path for label, path in legacy_values.items() if path}
    if generic_reports and supplied_legacy:
        raise ScalingCurveValidationError(
            "Do not mix repeatable --report arguments with legacy --report-* arguments"
        )
    if generic_reports:
        paths: dict[str, str] = {}
        for raw_label, path in generic_reports:
            label, _ = _parse_report_label(raw_label)
            if label in paths:
                raise ScalingCurveValidationError(
                    f"Duplicate scaling report limit {label!r}"
                )
            paths[label] = path
        return paths
    if supplied_legacy:
        missing = [label for label, path in legacy_values.items() if not path]
        if missing:
            raise ScalingCurveValidationError(
                "Legacy scaling mode requires all of --report-1400, --report-5000, "
                "--report-10000, and --report-full; missing " + ", ".join(missing)
            )
        return {label: str(path) for label, path in legacy_values.items()}
    raise ScalingCurveValidationError(
        "Provide repeatable --report LIMIT PATH arguments or all four legacy reports"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = _report_paths_from_args(args)
        payload = build_scaling_curve(paths)
        output = write_json_atomic(args.output, payload)
    except (ScalingCurveValidationError, OSError, TypeError, ValueError) as exc:
        print(f"scaling-curve validation failed: {exc}", file=sys.stderr)
        return 2
    print(f"[scaling-curve] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
