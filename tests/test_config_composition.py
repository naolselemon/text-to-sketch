from __future__ import annotations

import unittest

from builders import build_model
from scripts.sketchformer.config import compose_training_config


class ConfigCompositionTest(unittest.TestCase):
    def test_default_training_config_targets_tok_dict(self) -> None:
        config = compose_training_config("configs/train.yaml", experiment="smoke_test")

        self.assertEqual(config["data"]["format"]["type"], "tok_dict")
        self.assertEqual(config["model"]["name"], "sketchformer_tok_dict")
        model_tokens = config["model"]["input"]["token_dictionary"]
        data_tokens = config["data"]["format"]["token_dictionary"]
        for key in (
            "codebook_size",
            "motion_token_offset",
            "pad_token_id",
            "sep_token_id",
            "sos_token_id",
            "eos_token_id",
            "vocab_size",
        ):
            self.assertEqual(model_tokens[key], data_tokens[key])
        self.assertEqual(model_tokens["pad_token_id"], 0)
        self.assertEqual(model_tokens["sep_token_id"], 1001)
        self.assertEqual(model_tokens["sos_token_id"], 1002)
        self.assertEqual(model_tokens["eos_token_id"], 1003)
        self.assertEqual(model_tokens["vocab_size"], 1004)
        self.assertEqual(config["trainer"]["checkpointing"]["monitor"], "val/token_loss")

    def test_anchored_v3_config_is_synced_and_reconstruction_first(self) -> None:
        config = compose_training_config(
            "configs/train.yaml",
            experiment="anime_anchored_v3_direct",
        )
        tokens = config["data"]["format"]["token_dictionary"]
        self.assertEqual(config["data"]["format"]["type"], "anchored_v3")
        self.assertEqual(config["model"]["input"]["token_dictionary"], tokens)
        self.assertIsNone(tokens["codebook_dir"])
        self.assertTrue(tokens["codebook_path"].endswith("current/codebook.npy"))
        self.assertEqual(tokens["vocab_size"], 2566)
        self.assertEqual(config["model"]["decoder"]["memory_source"], "encoder")
        self.assertEqual(config["experiment"]["pretrained"]["mode"], "strict")
        self.assertTrue(
            config["experiment"]["pretrained"]["path"].endswith(
                "sketchformer_anchored_v3_transformer_init.pt"
            )
        )
        self.assertIsNone(config["data"]["dataset"]["train_source_limit"])
        self.assertEqual(config["data"]["dataset"]["minimum_source_sketches"], 1)
        self.assertEqual(
            config["trainer"]["checkpointing"]["monitor"],
            "val/free_running/macro_geometry_f1_2px_median",
        )

    def test_overfit_experiment_inherits_v3_contract(self) -> None:
        config = compose_training_config(
            "configs/train.yaml",
            experiment="anime_anchored_v3_overfit",
        )
        self.assertEqual(config["model"]["decoder"]["memory_source"], "encoder")
        self.assertEqual(config["trainer"]["runtime"]["precision"], "32-true")
        self.assertFalse(config["trainer"]["gates"]["require_overfit_report"])
        self.assertEqual(config["data"]["dataset"]["train_source_limit"], 32)
        self.assertFalse(config["data"]["dataset"]["include_augmentations"])
        self.assertEqual(
            config["data"]["dataset"]["source_subset_strategy"],
            "length_stratified",
        )
        self.assertTrue(config["trainer"]["gates"]["overfit_mode"])

    def test_wide_fallback_changes_only_v3_capacity(self) -> None:
        narrow = compose_training_config(
            "configs/train.yaml",
            experiment="anime_anchored_v3_direct",
        )
        wide = compose_training_config(
            "configs/train.yaml",
            experiment="anime_anchored_v3_wide",
        )

        self.assertEqual(wide["data"], narrow["data"])
        self.assertEqual(wide["model"]["architecture"]["d_model"], 256)
        self.assertEqual(wide["model"]["architecture"]["num_encoder_layers"], 6)
        self.assertEqual(wide["model"]["architecture"]["num_decoder_layers"], 6)
        self.assertEqual(wide["model"]["architecture"]["dim_feedforward"], 1024)
        self.assertEqual(wide["model"]["decoder"], narrow["model"]["decoder"])
        self.assertTrue(
            wide["experiment"]["pretrained"]["path"].endswith(
                "sketchformer_anchored_v3_wide_transformer_init.pt"
            )
        )
        model = build_model(wide["model"])
        self.assertEqual(model.config.d_model, 256)


if __name__ == "__main__":
    unittest.main()
