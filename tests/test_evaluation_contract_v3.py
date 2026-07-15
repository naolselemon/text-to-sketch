from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from metrics.sketchformer.free_running import (
    aggregate_free_running_records,
    decode_token_sequence,
    free_running_reconstruction_records,
    stroke_geometry_metrics,
)
from metrics.sketchformer.reconstruction import ReconstructionExample, write_metrics_report
from metrics.sketchformer.visualisation import save_reconstruction_examples
from scripts.sketchformer.evaluate import _validate_evaluation_request
from scripts.sketchformer.config import compose_training_config


V3_LAYOUT = {
    "type": "anchored_v3",
    "version": 3,
    "pad_token_id": 0,
    "motion_token_start": 1,
    "motion_token_end": 2048,
    "x_token_start": 2049,
    "x_token_end": 2304,
    "y_token_start": 2305,
    "y_token_end": 2560,
    "stroke_start_token_id": 2561,
    "stroke_end_token_id": 2562,
    "sos_token_id": 2563,
    "eos_token_id": 2564,
    "mask_token_id": 2565,
}


class EvaluationContractV3Test(unittest.TestCase):
    def setUp(self) -> None:
        self.codebook = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.target = torch.tensor(
            [[2563, 2561, 2059, 2325, 1, 1, 2, 2562, 2564]],
            dtype=torch.long,
        )

    def test_release_request_rejects_legacy_partial_or_non_test_shortcuts(self) -> None:
        config = compose_training_config(
            "configs/train.yaml",
            experiment="anime_anchored_v3_direct",
        )
        valid = dict(
            allow_legacy_checkpoint=False,
            enforce_v3_gates=True,
            split="test",
            decode_mode="free-running",
            checkpoint="best.pt",
            limit_batches=1.0,
            max_generation_length=None,
            metrics_output="report.json",
            human_review_template=None,
            plots_output_dir=None,
            num_plots=8,
        )
        _validate_evaluation_request(SimpleNamespace(**valid), config)
        for override in (
            {"allow_legacy_checkpoint": True},
            {"split": "valid"},
            {"limit_batches": 1},
            {"limit_batches": 0.5},
            {"max_generation_length": 512},
            {"checkpoint": None},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                _validate_evaluation_request(
                    SimpleNamespace(**{**valid, **override}),
                    config,
                )

        wrong_layout = copy.deepcopy(config)
        wrong_layout["data"]["format"]["token_dictionary"]["eos_token_id"] = 7
        with self.assertRaisesRegex(ValueError, "runtime contract mismatch"):
            _validate_evaluation_request(SimpleNamespace(**valid), wrong_layout)

        latent_memory = copy.deepcopy(config)
        latent_memory["model"]["decoder"]["memory_source"] = "latent_expander"
        with self.assertRaisesRegex(ValueError, "memory_source must be encoder"):
            _validate_evaluation_request(SimpleNamespace(**valid), latent_memory)

    def test_v3_decoder_uses_absolute_anchor_and_within_stroke_motion(self) -> None:
        decoded = decode_token_sequence(
            self.target[0].numpy(),
            self.codebook,
            token_layout=V3_LAYOUT,
        )
        points = np.cumsum(decoded[:, :2], axis=0)

        np.testing.assert_allclose(points[0], [10.0, 20.0])
        np.testing.assert_allclose(points[-2], [12.0, 21.0])
        self.assertEqual(decoded[-1, 4], 1.0)

    def test_free_running_records_include_all_collapse_diagnostics(self) -> None:
        prediction = torch.tensor(
            [[2563, 2561, 2059, 2325, 1, 1, 1, 2562, 2564]],
            dtype=torch.long,
        )
        records = free_running_reconstruction_records(
            prediction,
            torch.tensor([9]),
            {"targets": self.target, "lengths": torch.tensor([9])},
            self.codebook,
            eos_token_id=2564,
            token_layout=V3_LAYOUT,
        )
        record = records[0]

        expected_keys = {
            "generated_length",
            "target_length",
            "eos_position",
            "stroke_count_error",
            "target_structure_count",
            "generated_structure_count",
            "unique_motion_ratio",
            "longest_repeated_token_run",
            "first_divergence_position",
            "geometry_f1_1px",
            "geometry_f1_2px",
            "symmetric_chamfer_px",
        }
        self.assertTrue(expected_keys.issubset(record))
        self.assertEqual(record["longest_repeated_token_run"], 3)
        self.assertEqual(record["first_divergence_position"], 6)
        self.assertEqual(record["eos_position"], 8)

    def test_identity_geometry_and_dataset_quantiles(self) -> None:
        records = free_running_reconstruction_records(
            self.target,
            torch.tensor([9]),
            {"targets": self.target.clone(), "lengths": torch.tensor([9])},
            self.codebook,
            eos_token_id=2564,
            token_layout=V3_LAYOUT,
        )
        summary = aggregate_free_running_records(records, device="cpu")

        self.assertEqual(records[0]["geometry_f1_1px"], 1.0)
        self.assertEqual(records[0]["geometry_f1_2px"], 1.0)
        self.assertEqual(records[0]["symmetric_chamfer_px"], 0.0)
        self.assertEqual(summary["free_running/geometry_f1_2px_p10"].item(), 1.0)
        self.assertEqual(
            summary["free_running/macro_geometry_f1_2px_median"].item(),
            0.25,
        )
        self.assertEqual(summary["free_running/symmetric_chamfer_px_p95"].item(), 0.0)

    def test_empty_prediction_has_bounded_failure_metrics(self) -> None:
        target = decode_token_sequence(
            self.target[0].numpy(), self.codebook, token_layout=V3_LAYOUT
        )
        empty = np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        metrics = stroke_geometry_metrics(target, empty, coordinate_mode="canvas")

        self.assertEqual(metrics["geometry_f1_2px"], 0.0)
        self.assertGreater(metrics["symmetric_chamfer_px"], 3.0)

    def test_report_contains_per_sample_records_and_plot_is_mode_qualified(self) -> None:
        stroke = decode_token_sequence(
            self.target[0].numpy(), self.codebook, token_layout=V3_LAYOUT
        )
        record = {
            "sample_id": "anime-1",
            "geometry_f1_2px": 1.0,
            "symmetric_chamfer_px": 0.0,
            "eos_position": 8,
            "generated_stroke_count": 1,
            "longest_repeated_token_run": 2,
        }
        example = ReconstructionExample(
            target=stroke,
            prediction=stroke,
            length=9,
            prediction_length=9,
            source_file="source.npz",
            source_index=0,
            sample_id="anime-1",
            decode_mode="free-running",
            coordinate_mode="canvas",
            statistics=record,
        )
        with tempfile.TemporaryDirectory() as directory:
            report_path = write_metrics_report(
                Path(directory) / "report.json",
                {"test/free_running/geometry_f1_2px": torch.tensor(1.0)},
                records=[record],
            )
            plots = save_reconstruction_examples([example], directory)
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(report["records"][0]["sample_id"], "anime-1")
            self.assertIn("free-running", plots[0].name)
            self.assertTrue(plots[0].is_file())


if __name__ == "__main__":
    unittest.main()
