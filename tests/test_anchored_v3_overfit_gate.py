from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.sketchformer.overfit_gate import (
    _validate_v3_config,
    build_gate_report,
    evaluate_overfit_model,
    write_gate_report_atomic,
)
from scripts.sketchformer.config import compose_training_config
from scripts.sketchformer.train import _validate_overfit_gate


V3_LAYOUT = {
    "type": "anchored_v3",
    "version": 3,
    "pad_token_id": 0,
    "motion_token_offset": 1,
    "x_token_offset": 2049,
    "y_token_offset": 2305,
    "coordinate_bins": 256,
    "stroke_start_token_id": 2561,
    "stroke_end_token_id": 2562,
    "sos_token_id": 2563,
    "eos_token_id": 2564,
}


class _IdentityModel(torch.nn.Module):
    def __init__(self, *, break_uncached: bool = False) -> None:
        super().__init__()
        self.break_uncached = break_uncached

    def forward(self, batch):
        targets = batch["targets"][:, 1:]
        logits = torch.full(
            (*targets.shape, 2566),
            -100.0,
            dtype=torch.float32,
            device=targets.device,
        )
        logits.scatter_(2, targets.unsqueeze(-1), 100.0)
        return SimpleNamespace(
            reconstruction=SimpleNamespace(token_logits=logits),
            loss_targets=targets,
            loss_valid_mask=targets != 0,
        )

    def generate(self, batch, *, use_cache: bool):
        tokens = batch["targets"].clone()
        if self.break_uncached and not use_cache:
            tokens[:, 4] = 2
        return SimpleNamespace(tokens=tokens, lengths=batch["lengths"].clone())


def _identity_batch():
    tokens = torch.tensor(
        [[2563, 2561, 2059, 2325, 1, 1, 2, 2562, 2564]],
        dtype=torch.long,
    )
    return {
        "tokens": tokens,
        "targets": tokens.clone(),
        "lengths": torch.tensor([tokens.shape[1]]),
        "valid_mask": torch.ones_like(tokens, dtype=torch.bool),
        "sample_ids": ["overfit-00"],
    }


class AnchoredV3OverfitGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.codebook = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def test_identity_model_passes_all_three_gates(self) -> None:
        metrics = evaluate_overfit_model(
            _IdentityModel(),
            [_identity_batch()],
            codebook=self.codebook,
            token_layout=V3_LAYOUT,
            device="cpu",
            expected_samples=1,
        )
        report = build_gate_report(metrics)

        self.assertEqual(metrics["teacher_forced_token_accuracy"], 1.0)
        self.assertEqual(metrics["free_running_geometry_f1_2px_median"], 1.0)
        self.assertTrue(metrics["cached_uncached_exact_match"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["failures"], [])

    def test_cache_divergence_fails_and_names_the_sample(self) -> None:
        metrics = evaluate_overfit_model(
            _IdentityModel(break_uncached=True),
            [_identity_batch()],
            codebook=self.codebook,
            token_layout=V3_LAYOUT,
            device="cpu",
            expected_samples=1,
        )
        report = build_gate_report(metrics)

        self.assertFalse(report["passed"])
        self.assertEqual(report["failures"], ["cached_uncached_exact_match"])
        self.assertEqual(
            metrics["cached_uncached_mismatch_sample_ids"], ["overfit-00"]
        )

    def test_thresholds_are_inclusive_and_subthreshold_values_fail(self) -> None:
        at_threshold = build_gate_report(
            {
                "teacher_forced_token_accuracy": 0.995,
                "free_running_geometry_f1_2px_median": 0.99,
                "cached_uncached_exact_match": True,
            }
        )
        below = build_gate_report(
            {
                "teacher_forced_token_accuracy": 0.9949,
                "free_running_geometry_f1_2px_median": 0.9899,
                "cached_uncached_exact_match": True,
            }
        )

        self.assertTrue(at_threshold["passed"])
        self.assertFalse(below["passed"])
        self.assertEqual(
            below["failures"],
            [
                "teacher_forced_token_accuracy",
                "free_running_geometry_f1_2px_median",
            ],
        )

    def test_report_write_is_atomic_and_preserves_training_interface(self) -> None:
        report = build_gate_report(
            {
                "teacher_forced_token_accuracy": 1.0,
                "free_running_geometry_f1_2px_median": 1.0,
                "cached_uncached_exact_match": True,
            },
            metadata={"checkpoint": "best.pt"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_gate_report_atomic(Path(directory) / "gate.json", report)
            loaded = json.loads(path.read_text(encoding="utf-8"))

            self.assertTrue(loaded["passed"])
            self.assertEqual(loaded["teacher_forced_token_accuracy"], 1.0)
            self.assertEqual(
                loaded["free_running_geometry_f1_2px_median"], 1.0
            )
            self.assertTrue(loaded["cached_uncached_exact_match"])
            self.assertEqual(loaded["metadata"]["checkpoint"], "best.pt")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_gate_rejects_wrong_sample_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 32 sketches"):
            evaluate_overfit_model(
                _IdentityModel(),
                [_identity_batch()],
                codebook=self.codebook,
                token_layout=V3_LAYOUT,
                device="cpu",
            )

    def test_overfit_config_is_v3_direct_memory_and_fp32(self) -> None:
        config = compose_training_config(
            "configs/train.yaml",
            experiment="anime_anchored_v3_overfit",
        )
        _validate_v3_config(config)

        config["trainer"]["runtime"]["precision"] = "bf16-mixed"
        with self.assertRaisesRegex(ValueError, "must run in FP32"):
            _validate_v3_config(config)

    def test_full_training_rejects_gate_report_from_different_artifacts(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codebook = root / "codebook.npy"
            manifest = root / "manifest.jsonl"
            np.save(codebook, np.zeros((2048, 2), dtype=np.float32))
            manifest.write_text("{}\n", encoding="utf-8")
            report_path = root / "gate.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "passed": True,
                        "teacher_forced_token_accuracy": 1.0,
                        "free_running_geometry_f1_2px_median": 1.0,
                        "cached_uncached_exact_match": True,
                        "metrics": {"sample_count": 32},
                        "metadata": {
                            "codebook_sha256": hashlib.sha256(
                                codebook.read_bytes()
                            ).hexdigest(),
                            "manifest_sha256": "f" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "data": {
                    "dataset": {
                        "root": str(root),
                        "manifest_file": "manifest.jsonl",
                    },
                    "format": {
                        "type": "anchored_v3",
                        "token_dictionary": {"codebook_path": str(codebook)},
                    },
                },
                "trainer": {
                    "gates": {
                        "require_overfit_report": True,
                        "overfit_report": str(report_path),
                    }
                },
            }

            with self.assertRaisesRegex(RuntimeError, "manifest_sha256"):
                _validate_overfit_gate(config)


if __name__ == "__main__":
    unittest.main()
