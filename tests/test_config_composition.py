from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
        self.assertFalse(config["trainer"]["checkpointing"]["retain_candidates"])

    def test_long_experiment_retains_candidates(self):
        config = compose_training_config("configs/train.yaml", experiment="anime_tok_dict_long")
        self.assertTrue(config["trainer"]["checkpointing"]["retain_candidates"])

    def test_frozen_json_config_preserves_scientific_notation_and_settings(self):
        config = compose_training_config("configs/train.yaml", experiment="anime_tok_dict_long")
        config['experiment'].pop('overrides', None)
        config['model']['architecture']['layer_norm_eps'] = 1e-5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'inference_config.json'
            path.write_text(json.dumps(config))
            restored = compose_training_config(path)
        self.assertEqual(restored, config)
        self.assertIsInstance(restored['model']['architecture']['layer_norm_eps'], float)


if __name__ == "__main__":
    unittest.main()
