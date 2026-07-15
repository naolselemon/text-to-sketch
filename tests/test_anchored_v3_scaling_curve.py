from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.sketchformer.scaling_curve import (
    ScalingCurveValidationError,
    build_scaling_curve,
    main,
    write_json_atomic,
)


REPORT_LIMITS = {
    "1400": 1_400,
    "5000": 5_000,
    "10000": 10_000,
    "full": None,
}


def evaluation_report(
    train_source_limit: int | None,
    *,
    sample_ids: tuple[str, ...] = ("anime-001", "anime-002"),
    manifest_hash: str = "a" * 64,
    codebook_hash: str = "b" * 64,
) -> dict:
    f1_values = [0.8, 1.0]
    chamfer_values = [1.0, 3.0]
    premature_values = [0.0, 0.0]
    max_hit_values = [0.0, 1.0]
    records = [
        {
            "sample_id": sample_id,
            "geometry_f1_2px": f1_values[index],
            "symmetric_chamfer_px": chamfer_values[index],
            "premature_eos": premature_values[index],
            "max_length_hit": max_hit_values[index],
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    prefix = "test/free_running"
    return {
        "created_at": "2026-01-01T00:00:00+00:00",
        "metadata": {
            "format_type": "anchored_v3",
            "format_version": 3,
            "split": "test",
            "decode_mode": "free-running",
            "limit_batches": 1.0,
            "max_generation_length": None,
            "enforce_v3_gates": False,
            "allow_legacy_checkpoint": False,
            "checkpoint": "weights/best.pt",
            "checkpoint_contract": {
                "schema_version": 1,
                "token_layout_version": 3,
                "dataset_manifest_sha256": manifest_hash,
                "codebook_sha256": codebook_hash,
                "git_commit": "c" * 40,
                "monitored_metrics": {},
                "config": {"experiment": {"name": "anchored-v3-scaling"}},
                "compatibility_config": {
                    "data": {
                        "dataset": {"train_source_limit": train_source_limit},
                        "format": {"type": "anchored_v3", "version": 3},
                    },
                    "model": {"decoder": {"memory_source": "encoder"}},
                },
            },
        },
        "metrics": {
            f"{prefix}/count": float(len(records)),
            f"{prefix}/geometry_f1_2px_median": float(
                np.quantile(f1_values, 0.5)
            ),
            f"{prefix}/geometry_f1_2px_p10": float(np.quantile(f1_values, 0.1)),
            f"{prefix}/symmetric_chamfer_px_p95": float(
                np.quantile(chamfer_values, 0.95)
            ),
            f"{prefix}/premature_eos_rate": float(np.mean(premature_values)),
            f"{prefix}/max_length_hit_rate": float(np.mean(max_hit_values)),
        },
        "records": records,
    }


class AnchoredV3ScalingCurveTest(unittest.TestCase):
    def _write_reports(self, directory: Path) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for label, limit in REPORT_LIMITS.items():
            path = directory / f"evaluation-{label}.json"
            path.write_text(
                json.dumps(evaluation_report(limit), sort_keys=True),
                encoding="utf-8",
            )
            paths[label] = path
        return paths

    def test_builds_ordered_deterministic_curve_with_requested_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._write_reports(root)

            first = build_scaling_curve(paths)
            second = build_scaling_curve(dict(reversed(list(paths.items()))))
            first_path = write_json_atomic(root / "first.json", first)
            second_path = write_json_atomic(root / "second.json", second)

            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(
                [point["train_source_label"] for point in first["points"]],
                ["1400", "5000", "10000", "full"],
            )
            self.assertEqual(
                [point["train_source_limit"] for point in first["points"]],
                [1_400, 5_000, 10_000, None],
            )
            self.assertEqual(first["test_sample_count"], 2)
            self.assertEqual(first["dataset_manifest_sha256"], "a" * 64)
            self.assertEqual(first["codebook_sha256"], "b" * 64)
            self.assertEqual(first["points"][0]["median_f1_2px"], 0.9)
            self.assertEqual(first["points"][0]["p10_f1_2px"], 0.8200000000000001)
            self.assertEqual(first["points"][0]["p95_chamfer_px"], 2.9)
            self.assertEqual(first["points"][0]["premature_eos_rate"], 0.0)
            self.assertEqual(first["points"][0]["max_length_hit_rate"], 0.5)

    def test_rejects_hash_sample_order_and_train_limit_mismatches(self) -> None:
        mutations = {
            "dataset_manifest_sha256": lambda payload: payload["metadata"][
                "checkpoint_contract"
            ].__setitem__("dataset_manifest_sha256", "c" * 64),
            "codebook_sha256": lambda payload: payload["metadata"][
                "checkpoint_contract"
            ].__setitem__("codebook_sha256", "c" * 64),
            "ordered test sample IDs": lambda payload: payload["records"].reverse(),
            "train_source_limit": lambda payload: payload["metadata"][
                "checkpoint_contract"
            ]["compatibility_config"]["data"]["dataset"].__setitem__(
                "train_source_limit", 9_999
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for expected_error, mutate in mutations.items():
                with self.subTest(expected_error=expected_error):
                    paths = self._write_reports(root)
                    changed = evaluation_report(5_000)
                    mutate(changed)
                    paths["5000"].write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ScalingCurveValidationError,
                        expected_error,
                    ):
                        build_scaling_curve(paths)

    def test_rejects_non_v3_partial_or_inconsistent_metric_report(self) -> None:
        mutations = {
            "anchored-V3": lambda payload: payload["metadata"].__setitem__(
                "format_type", "tok_dict"
            ),
            "test/free-running": lambda payload: payload["metadata"].__setitem__(
                "split", "valid"
            ),
            "does not match per-sample": lambda payload: payload["metrics"].__setitem__(
                "test/free_running/geometry_f1_2px_median", 0.1
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for expected_error, mutate in mutations.items():
                with self.subTest(expected_error=expected_error):
                    paths = self._write_reports(root)
                    changed = evaluation_report(1_400)
                    mutate(changed)
                    paths["1400"].write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ScalingCurveValidationError,
                        expected_error,
                    ):
                        build_scaling_curve(paths)

    def test_cli_returns_nonzero_and_preserves_output_on_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._write_reports(root)
            bad = evaluation_report(10_000, codebook_hash="c" * 64)
            paths["10000"].write_text(json.dumps(bad), encoding="utf-8")
            output = root / "curve.json"
            output.write_text("existing\n", encoding="utf-8")

            exit_code = main(
                [
                    "--report-1400",
                    str(paths["1400"]),
                    "--report-5000",
                    str(paths["5000"]),
                    "--report-10000",
                    str(paths["10000"]),
                    "--report-full",
                    str(paths["full"]),
                    "--output",
                    str(output),
                ]
            )

            self.assertNotEqual(exit_code, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing\n")


if __name__ == "__main__":
    unittest.main()
