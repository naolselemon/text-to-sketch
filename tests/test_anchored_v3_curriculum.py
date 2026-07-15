from __future__ import annotations

import unittest

import torch
from torch import nn

from builders.optimizer_builder import build_optimizer
from builders.scheduler_builder import build_scheduler
from scripts.sketchformer.curriculum import (
    build_stage_parameter_groups,
    parse_curriculum,
)
from scripts.sketchformer.train import (
    _full_validation_training_state,
    _restore_full_validation_early_stopping,
)
from core import CheckpointContractError


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_embedding = nn.Embedding(8, 2)
        self.target_embedding = self.input_embedding
        self.encoder = _Encoder()
        self.pool = nn.Linear(2, 2)
        self.latent_expander = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)
        self.reconstruction_head = nn.Linear(2, 8)
        self.classification_head = None


class AnchoredV3CurriculumTest(unittest.TestCase):
    def test_final_stage_resume_restores_full_validation_patience_history(self) -> None:
        state = _full_validation_training_state(
            stage_index=3,
            stage_name="long_reconstruction",
            completed_stage_epochs=9,
            monitor="val_full/free_running/macro_geometry_f1_2px_median",
            mode="max",
            best=0.94,
            non_improving=4,
        )

        best, non_improving = _restore_full_validation_early_stopping(
            state,
            stage_index=3,
            stage_name="long_reconstruction",
            completed_stage_epochs=9,
            monitor="val_full/free_running/macro_geometry_f1_2px_median",
            mode="max",
        )

        self.assertEqual(best, 0.94)
        self.assertEqual(non_improving, 4)
        self.assertEqual(non_improving + 1, 5)

    def test_final_stage_resume_rejects_inconsistent_epoch_history(self) -> None:
        state = _full_validation_training_state(
            stage_index=3,
            stage_name="long_reconstruction",
            completed_stage_epochs=8,
            monitor="val_full/free_running/macro_geometry_f1_2px_median",
            mode="max",
            best=0.94,
            non_improving=4,
        )

        with self.assertRaisesRegex(CheckpointContractError, "stage mismatch"):
            _restore_full_validation_early_stopping(
                state,
                stage_index=3,
                stage_name="long_reconstruction",
                completed_stage_epochs=9,
                monitor="val_full/free_running/macro_geometry_f1_2px_median",
                mode="max",
            )

    def test_parses_group_learning_rates(self) -> None:
        stages = parse_curriculum(
            {
                "curriculum": {
                    "enabled": True,
                    "stages": [
                        {
                            "name": "v3",
                            "max_length": 4096,
                            "epochs": 2,
                            "learning_rate": 1e-4,
                            "trainable": "decoder_top_encoder",
                            "learning_rates": {
                                "new_modules": 2e-4,
                                "decoder": 5e-5,
                                "encoder": 1e-5,
                            },
                        }
                    ],
                }
            },
            default_max_length=4096,
        )
        self.assertEqual(stages[0].learning_rates["new_modules"], 2e-4)

    def test_parameter_groups_deduplicate_tied_embeddings(self) -> None:
        model = _TinyModel()
        stage = parse_curriculum(
            {
                "curriculum": {
                    "enabled": True,
                    "stages": [
                        {
                            "max_length": 4096,
                            "epochs": 1,
                            "learning_rate": 1e-4,
                            "trainable": "all",
                            "learning_rates": {
                                "new_modules": 3e-4,
                                "decoder": 1e-4,
                                "encoder": 1e-5,
                            },
                        }
                    ],
                }
            },
            default_max_length=4096,
        )[0]
        groups = build_stage_parameter_groups(model, stage)
        identities = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(identities), len(set(identities)))
        rates = {group["name"]: group["lr"] for group in groups}
        self.assertEqual(rates["new_modules"], 3e-4)
        self.assertEqual(rates["decoder"], 1e-4)
        self.assertEqual(rates["encoder"], 1e-5)

    def test_new_grammar_stage_trains_decoder_but_keeps_encoder_frozen(self) -> None:
        model = _TinyModel()
        stage = parse_curriculum(
            {
                "curriculum": {
                    "enabled": True,
                    "stages": [
                        {
                            "max_length": 4096,
                            "epochs": 1,
                            "learning_rate": 3e-4,
                            "trainable": "new_modules",
                            "learning_rates": {
                                "new_modules": 3e-4,
                                "decoder": 1e-4,
                                "encoder": 1e-5,
                            },
                        }
                    ],
                }
            },
            default_max_length=4096,
        )[0]

        groups = build_stage_parameter_groups(model, stage)

        rates = {group["name"]: group["lr"] for group in groups}
        self.assertEqual(rates["decoder"], 1e-4)
        self.assertTrue(all(parameter.requires_grad for parameter in model.decoder.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.encoder.parameters()))

    def test_decoder_top_encoder_unfreezes_last_two_layers_only(self) -> None:
        model = _TinyModel()
        stage = parse_curriculum(
            {
                "curriculum": {
                    "enabled": True,
                    "stages": [
                        {
                            "max_length": 4096,
                            "epochs": 1,
                            "learning_rate": 1e-4,
                            "trainable": "decoder_top_encoder",
                        }
                    ],
                }
            },
            default_max_length=4096,
        )[0]
        build_stage_parameter_groups(model, stage)
        self.assertFalse(any(p.requires_grad for p in model.encoder.layers[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.encoder.layers[-1].parameters()))

    def test_optimizer_preserves_group_learning_rates(self) -> None:
        model = _TinyModel()
        stage = parse_curriculum(
            {
                "curriculum": {
                    "enabled": True,
                    "stages": [
                        {
                            "max_length": 4096,
                            "epochs": 1,
                            "learning_rate": 1e-4,
                            "trainable": "all",
                            "learning_rates": {"new_modules": 3e-4, "encoder": 1e-5},
                        }
                    ],
                }
            },
            default_max_length=4096,
        )[0]
        groups = build_stage_parameter_groups(model, stage)
        optimizer = build_optimizer(
            groups,
            {"optimizer": {"type": "adamw", "lr": 1e-4}},
        )
        self.assertEqual(optimizer.param_groups[0]["lr"], 3e-4)

    def test_scheduler_derives_capped_fractional_warmup(self) -> None:
        parameter = nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        bundle = build_scheduler(
            optimizer,
            {
                "scheduler": {
                    "type": "cosine_with_warmup",
                    "warmup_fraction": 0.05,
                    "warmup_cap_steps": 500,
                }
            },
            total_steps=20_000,
        )
        assert bundle.scheduler is not None
        self.assertAlmostEqual(bundle.scheduler.lr_lambdas[0](249), 0.5)
        self.assertAlmostEqual(bundle.scheduler.lr_lambdas[0](499), 1.0)


if __name__ == "__main__":
    unittest.main()
