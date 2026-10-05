from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.trainer import average_logs
from dataloaders.datamodule import TokenBudgetBatchSampler
from dataloaders.masks import build_sequence_masks
from dataloaders.transforms import TokenSequenceTransform
from metrics.sketchformer.free_running import free_running_reconstruction_metrics
from models.sketchformer.config import (
    PositionalEncodingConfig,
    ReconstructionHeadConfig,
    SketchformerConfig,
    TokenDictionaryConfig,
)
from models.sketchformer.decoder import LatentExpander
from models.sketchformer.model import SketchformerModel
from scripts.sketchformer.config import compose_training_config
from scripts.sketchformer.curriculum import (
    parse_curriculum,
    resume_epoch_for_stage,
    set_trainable_scope,
)
from scripts.sketchformer.parity_check import parity_metrics


def _tiny_token_model() -> SketchformerModel:
    config = SketchformerConfig(
        name="sketchformer_tok_dict",
        input_mode="tok_dict",
        max_seq_len=16,
        token_dictionary=TokenDictionaryConfig(
            codebook_size=4,
            motion_token_offset=1,
            pad_token_id=0,
            sep_token_id=5,
            sos_token_id=6,
            eos_token_id=7,
            vocab_size=8,
        ),
        d_model=16,
        latent_dim=16,
        pool_hidden_dim=8,
        pooling_mode="tf_self_attn_v1",
        latent_expander_mode="tf_dense",
        latent_expander_base_length=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dim_feedforward=32,
        dropout=0.0,
        activation="relu",
        norm_first=False,
        use_final_norm=False,
        gradient_checkpointing=False,
        positional_encoding=PositionalEncodingConfig(type="sinusoidal", max_length=16),
        decoder_autoregressive=True,
        reconstruction=ReconstructionHeadConfig(enabled=True, target="tok_dict"),
    )
    return SketchformerModel(config)


class LongSequenceTest(unittest.TestCase):
    def test_variable_batches_use_valid_token_weighted_metrics(self) -> None:
        logs = [
            {"val/token_loss": torch.tensor(1.0), "val/valid_tokens": torch.tensor(1.0)},
            {"val/token_loss": torch.tensor(3.0), "val/valid_tokens": torch.tensor(3.0)},
        ]

        averaged = average_logs(logs, weight_key="val/valid_tokens")

        self.assertEqual(averaged["val/token_loss"].item(), 2.5)

    def test_adaptive_expander_is_exact_at_checkpoint_base_length(self) -> None:
        expander = LatentExpander(8, 8, 16, mode="tf_dense", base_length=4)
        latent = torch.randn(2, 8)

        expected = expander.expand_layer(latent.unsqueeze(2)).transpose(1, 2)
        actual = expander(latent, 4)
        extended = expander(latent, 16)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(extended.shape, (2, 16, 8))
        self.assertEqual(expander.long_weight.count_nonzero().item(), 0)

    def test_forward_builds_memory_for_current_batch_length(self) -> None:
        model = _tiny_token_model().eval()
        tokens = torch.tensor([[6, 1, 5, 7, 0, 0]], dtype=torch.long)
        batch = {"tokens": tokens, "targets": tokens.clone(), **build_sequence_masks([4], 6)}

        with patch.object(
            model.latent_expander,
            "forward",
            wraps=model.latent_expander.forward,
        ) as expand_forward, torch.no_grad():
            model(batch)

        self.assertEqual(expand_forward.call_args.args[1], 6)

    def test_cached_generation_matches_uncached_greedy_generation(self) -> None:
        torch.manual_seed(4)
        model = _tiny_token_model().eval()
        tokens = torch.tensor([[6, 1, 5, 7, 0, 0]], dtype=torch.long)
        batch = {"tokens": tokens, "targets": tokens.clone(), **build_sequence_masks([4], 6)}

        cached = model.generate(batch, max_length=8, use_cache=True)
        uncached = model.generate(batch, max_length=8, use_cache=False)

        self.assertTrue(torch.equal(cached.tokens, uncached.tokens))
        self.assertTrue(torch.equal(cached.lengths, uncached.lengths))

    def test_extended_model_is_checkpoint_exact_at_base_length(self) -> None:
        torch.manual_seed(8)
        extended = _tiny_token_model().eval()
        reference_config = extended.config.__class__(
            **{
                **extended.config.__dict__,
                "max_seq_len": 4,
                "positional_encoding": PositionalEncodingConfig(
                    type="sinusoidal",
                    max_length=4,
                ),
            }
        )
        reference = SketchformerModel(reference_config).eval()
        state = extended.state_dict()
        reference.load_state_dict(
            {key: value for key, value in state.items() if key in reference.state_dict()},
            strict=True,
        )
        tokens = torch.tensor([[6, 1, 5, 7]], dtype=torch.long)
        batch = {"tokens": tokens, "targets": tokens.clone(), **build_sequence_masks([4], 4)}

        with torch.no_grad():
            extended_logits = extended(batch).reconstruction.token_logits
            reference_logits = reference(batch).reconstruction.token_logits
        cosine, agreement = parity_metrics(extended_logits, reference_logits)

        self.assertGreaterEqual(cosine, 0.999)
        self.assertGreaterEqual(agreement, 0.99)

    def test_token_budget_sampler_never_exceeds_padded_budget(self) -> None:
        lengths = [500, 512, 900, 1024, 1800, 2048, 4096]
        sampler = TokenBudgetBatchSampler(
            lengths,
            max_tokens=4096,
            max_batch_size=8,
            shuffle=False,
            pad_to_multiple_of=8,
        )

        batches = list(sampler)

        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(len(lengths))))
        for batch in batches:
            padded = ((max(lengths[index] for index in batch) + 7) // 8) * 8
            self.assertLessEqual(padded * len(batch), 4096)

        regression = TokenBudgetBatchSampler(
            [585] * 7,
            max_tokens=4096,
            max_batch_size=8,
            shuffle=False,
            pad_to_multiple_of=8,
        )
        self.assertEqual([len(batch) for batch in regression], [6, 1])

    def test_shuffled_token_budget_sampler_length_is_stable(self) -> None:
        sampler = TokenBudgetBatchSampler(
            [128, 200, 500, 512, 900, 1024, 1800, 2048, 4096],
            max_tokens=4096,
            max_batch_size=8,
            shuffle=True,
        )

        self.assertEqual(len(list(iter(sampler))), len(sampler))
        self.assertEqual(len(list(iter(sampler))), len(sampler))

    def test_curriculum_and_scope_are_decision_complete(self) -> None:
        config = compose_training_config(
            "configs/train.yaml",
            experiment="anime_tok_dict_long",
        )
        stages = parse_curriculum(config["trainer"], default_max_length=4096)
        model = _tiny_token_model()

        count = set_trainable_scope(model, "expander")

        self.assertEqual([stage.max_length for stage in stages], [512, 1024, 2048, 4096])
        self.assertEqual(sum(stage.epochs for stage in stages), 21)
        self.assertEqual(config["model"]["input"]["max_seq_len"], 4096)
        self.assertFalse(config["data"]["sequence"]["truncate_long_sequences"])
        self.assertGreater(count, 0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.latent_expander.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.encoder.parameters()))

    def test_curriculum_resume_skips_completed_epochs(self) -> None:
        config = compose_training_config(
            "configs/train.yaml",
            experiment="anime_tok_dict_long",
        )
        stages = parse_curriculum(config["trainer"], default_max_length=4096)

        offsets = [
            resume_epoch_for_stage(stages, index, completed_epochs=2)
            for index in range(len(stages))
        ]

        self.assertEqual(offsets, [1, 1, 0, 0])

    def test_transform_refuses_to_truncate(self) -> None:
        transform = TokenSequenceTransform(
            split="train",
            max_length=4,
            truncate_long_sequences=False,
        )

        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            transform({"tokens": np.arange(5), "length": 5})

    def test_identity_free_running_geometry_scores_one(self) -> None:
        codebook = np.asarray(
            [[0.1, 0.0], [0.0, 0.1], [-0.1, 0.0], [0.0, -0.1]],
            dtype=np.float32,
        )
        tokens = torch.tensor([[6, 1, 2, 5, 7]], dtype=torch.long)
        batch = {
            "targets": tokens.clone(),
            "lengths": torch.tensor([5]),
        }

        metrics = free_running_reconstruction_metrics(
            tokens,
            torch.tensor([5]),
            batch,
            codebook,
            eos_token_id=7,
        )

        self.assertEqual(metrics["free_running/token_accuracy"].item(), 1.0)
        self.assertEqual(metrics["free_running/geometry_f1_2px"].item(), 1.0)
        self.assertEqual(metrics["free_running/geometry_f1_2px_median"].item(), 1.0)
        self.assertEqual(
            metrics["free_running/geometry_f1_2px_median_length_1_512"].item(),
            1.0,
        )
        self.assertEqual(metrics["free_running/eos_rate"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
