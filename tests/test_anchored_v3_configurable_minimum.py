from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.sketchformer import evaluate as evaluate_cli
from scripts.sketchformer import overfit_gate as overfit_cli
from scripts.sketchformer import train as train_cli


def _base_config() -> dict:
    return {
        "data": {
            "dataset": {
                "minimum_source_sketches": 25_000,
                "train_source_limit": None,
            },
            "format": {
                "type": "anchored_v3",
                "token_dictionary": {
                    "codebook_path": "dataset/current/codebook.npy",
                },
            },
        },
        "experiment": {
            "run": {},
            "pretrained": {},
        },
        "trainer": {
            "runtime": {},
            "training": {},
            "gates": {},
        },
    }


def _parse(module, *arguments: str):
    with patch.object(sys, "argv", [module.__name__, *arguments]):
        return module.parse_args()


class AnchoredV3ConfigurableMinimumCliTest(unittest.TestCase):
    def test_train_cli_overrides_the_configured_source_minimum(self) -> None:
        config = _base_config()
        args = _parse(train_cli, "--minimum-source-sketches", "7942")

        train_cli._apply_cli_overrides(config, args)

        self.assertEqual(config["data"]["dataset"]["minimum_source_sketches"], 7_942)

    def test_evaluate_cli_overrides_the_configured_source_minimum(self) -> None:
        config = _base_config()
        args = _parse(evaluate_cli, "--minimum-source-sketches", "7942")

        evaluate_cli._apply_cli_overrides(config, args)

        self.assertEqual(config["data"]["dataset"]["minimum_source_sketches"], 7_942)

    def test_overfit_cli_overrides_the_configured_source_minimum(self) -> None:
        config = _base_config()
        args = _parse(
            overfit_cli,
            "--checkpoint",
            "checkpoint.pt",
            "--minimum-source-sketches",
            "7942",
        )

        overfit_cli._apply_cli_overrides(config, args)

        self.assertEqual(config["data"]["dataset"]["minimum_source_sketches"], 7_942)

    def test_all_clis_reject_non_positive_source_minimums(self) -> None:
        cases = (
            (train_cli, ()),
            (evaluate_cli, ()),
            (overfit_cli, ("--checkpoint", "checkpoint.pt")),
        )
        for module, required_arguments in cases:
            for invalid in ("0", "-1"):
                with self.subTest(module=module.__name__, minimum=invalid):
                    config = _base_config()
                    args = _parse(
                        module,
                        *required_arguments,
                        "--minimum-source-sketches",
                        invalid,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "--minimum-source-sketches must be positive",
                    ):
                        module._apply_cli_overrides(config, args)

    def test_training_enforcement_uses_the_configured_source_minimum(self) -> None:
        config = _base_config()
        config["data"]["dataset"]["minimum_source_sketches"] = 7_942
        datamodule = SimpleNamespace(
            train_dataset=SimpleNamespace(
                metadata={
                    "preparation": {
                        "accepted_source_sketches": 7_942,
                    },
                },
            ),
        )

        train_cli._require_production_v3_dataset(config, datamodule)

        config["data"]["dataset"]["minimum_source_sketches"] = 7_943
        with self.assertRaisesRegex(ValueError, "at least 7943"):
            train_cli._require_production_v3_dataset(config, datamodule)


if __name__ == "__main__":
    unittest.main()
