from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from core.callbacks import CheckpointCallback
from core.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointContract,
    CheckpointContractError,
    load_checkpoint,
    save_checkpoint,
)


class CheckpointContractV3Test(unittest.TestCase):
    def setUp(self) -> None:
        self.model = torch.nn.Linear(2, 2)
        self.contract = CheckpointContract(
            config={"model": {"name": "tiny"}, "data": {"format": {"version": 3}}},
            token_layout_version=3,
            codebook_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            git_commit="c" * 40,
        )

    def test_current_checkpoint_round_trip_validates_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                path,
                self.model,
                epoch=4,
                step=19,
                metrics={"valid/free_running/macro": 0.91},
                contract=self.contract,
                require_contract=True,
            )
            restored = torch.nn.Linear(2, 2)
            result = load_checkpoint(
                path,
                restored,
                strict=True,
                expected_contract={
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "config": self.contract.config,
                    "token_layout_version": 3,
                    "codebook_sha256": "a" * 64,
                    "dataset_manifest_sha256": "b" * 64,
                },
                require_contract=True,
            )

            self.assertFalse(result.legacy)
            self.assertEqual(result.epoch, 4)
            self.assertEqual(result.step, 19)
            self.assertEqual(
                result.contract["monitored_metrics"]["valid/free_running/macro"],
                0.91,
            )
            for expected, actual in zip(self.model.parameters(), restored.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_hash_mismatch_is_rejected_before_loading_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, self.model, contract=self.contract, require_contract=True)
            restored = torch.nn.Linear(2, 2)
            before = [parameter.detach().clone() for parameter in restored.parameters()]

            with self.assertRaisesRegex(CheckpointContractError, "codebook_sha256"):
                load_checkpoint(
                    path,
                    restored,
                    expected_contract={"codebook_sha256": "d" * 64},
                    require_contract=True,
                )

            for expected, actual in zip(before, restored.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_contract_object_ignores_run_paths_but_rejects_model_changes(self) -> None:
        base = CheckpointContract(
            config={
                "model": {"name": "tiny", "width": 2},
                "data": {"format": {"version": 3}},
                "experiment": {
                    "run": {"output_dir": "first", "resume_from_checkpoint": None},
                    "pretrained": {"path": "initial.pt"},
                },
                "trainer": {"gates": {"overfit_report": "first.json"}},
            },
            token_layout_version=3,
            codebook_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            git_commit="c" * 40,
        )
        moved = CheckpointContract(
            config={
                **base.config,
                "experiment": {
                    "run": {
                        "output_dir": "second",
                        "resume_from_checkpoint": "last.pt",
                    },
                    "pretrained": {"path": "other.pt"},
                },
                "trainer": {"gates": {"overfit_report": "second.json"}},
            },
            token_layout_version=3,
            codebook_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            git_commit="c" * 40,
        )
        changed_model = CheckpointContract(
            config={**moved.config, "model": {"name": "tiny", "width": 4}},
            token_layout_version=3,
            codebook_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            git_commit="c" * 40,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, self.model, contract=base, require_contract=True)

            load_checkpoint(
                path,
                torch.nn.Linear(2, 2),
                expected_contract=moved,
                require_contract=True,
            )
            with self.assertRaisesRegex(
                CheckpointContractError,
                "compatibility_config",
            ):
                load_checkpoint(
                    path,
                    torch.nn.Linear(2, 2),
                    expected_contract=changed_model,
                    require_contract=True,
                )

    def test_legacy_checkpoint_needs_explicit_compatibility_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"model": self.model.state_dict()}, path)
            with self.assertRaisesRegex(CheckpointContractError, "no artifact contract"):
                load_checkpoint(path, torch.nn.Linear(2, 2), require_contract=True)

            result = load_checkpoint(
                path,
                torch.nn.Linear(2, 2),
                strict=True,
                require_contract=False,
            )
            self.assertTrue(result.legacy)

    def test_resume_rejects_missing_optimizer_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, self.model, contract=self.contract, require_contract=True)
            optimizer = torch.optim.AdamW(self.model.parameters())

            with self.assertRaisesRegex(CheckpointContractError, "optimizer state"):
                load_checkpoint(
                    path,
                    self.model,
                    optimizer=optimizer,
                    expected_contract=self.contract,
                    require_contract=True,
                )

    def test_resume_training_state_round_trips_through_callback(self) -> None:
        training_state = {
            "schema_version": 1,
            "curriculum_stage": {
                "index": 3,
                "name": "long_reconstruction",
                "completed_epochs": 7,
            },
            "full_validation_early_stopping": {
                "monitor": "val_full/free_running/macro_geometry_f1_2px_median",
                "mode": "max",
                "best": 0.93,
                "non_improving": 4,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            optimizer = torch.optim.AdamW(self.model.parameters())
            callback = CheckpointCallback(
                directory,
                monitor="val/free_running/macro_geometry_f1_2px_median",
                mode="max",
                contract=self.contract,
            )
            callback.on_validation_end(
                self.model,
                optimizer=optimizer,
                scheduler=None,
                epoch=13,
                step=29,
                metrics={"val/free_running/macro_geometry_f1_2px_median": 0.93},
                training_state=training_state,
            )

            result = load_checkpoint(
                Path(directory) / "last.pt",
                torch.nn.Linear(2, 2),
                strict=True,
                expected_contract=self.contract,
                require_contract=True,
                require_training_state=True,
            )

        self.assertEqual(result.training_state, training_state)

    def test_strict_resume_rejects_missing_training_state_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, self.model, contract=self.contract, require_contract=True)
            restored = torch.nn.Linear(2, 2)
            before = [parameter.detach().clone() for parameter in restored.parameters()]

            with self.assertRaisesRegex(CheckpointContractError, "training_state"):
                load_checkpoint(
                    path,
                    restored,
                    strict=True,
                    expected_contract=self.contract,
                    require_contract=True,
                    require_training_state=True,
                )

            for expected, actual in zip(before, restored.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_strict_tensor_key_failure_does_not_mutate_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            save_checkpoint(path, self.model, contract=self.contract, require_contract=True)
            checkpoint = torch.load(path, map_location="cpu")
            checkpoint["model"].pop("bias")
            torch.save(checkpoint, path)
            restored = torch.nn.Linear(2, 2)
            before = [parameter.detach().clone() for parameter in restored.parameters()]

            with self.assertRaisesRegex(CheckpointContractError, "missing=.*bias"):
                load_checkpoint(
                    path,
                    restored,
                    strict=True,
                    expected_contract=self.contract,
                    require_contract=True,
                )

            for expected, actual in zip(before, restored.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_callback_refuses_to_write_incomplete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckpointCallback(directory)
            with self.assertRaisesRegex(CheckpointContractError, "requires an artifact"):
                callback.on_validation_end(
                    self.model,
                    optimizer=None,
                    scheduler=None,
                    epoch=1,
                    step=1,
                    metrics={"val/token_loss": 1.0},
                )


if __name__ == "__main__":
    unittest.main()
