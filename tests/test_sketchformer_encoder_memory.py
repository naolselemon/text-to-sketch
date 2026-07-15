from __future__ import annotations

import unittest

import torch

from dataloaders.masks import build_sequence_masks
from models.sketchformer.config import (
    PositionalEncodingConfig,
    ReconstructionHeadConfig,
    SketchformerConfig,
    TokenDictionaryConfig,
)
from models.sketchformer.heads import TokenReconstructionHead
from models.sketchformer.model import SketchformerModel


def _anchored_token_config() -> TokenDictionaryConfig:
    return TokenDictionaryConfig(
        codebook_size=4,
        motion_token_offset=1,
        pad_token_id=0,
        sep_token_id=None,
        sos_token_id=15,
        eos_token_id=16,
        vocab_size=18,
        x_token_offset=5,
        y_token_offset=9,
        coordinate_bins=4,
        stroke_start_token_id=13,
        stroke_end_token_id=14,
        mask_token_id=17,
    )


def _encoder_memory_model(*, tie_weights: bool = True) -> SketchformerModel:
    config = SketchformerConfig(
        name="sketchformer_tok_dict_anchored_v3",
        input_mode="tok_dict",
        max_seq_len=16,
        token_dictionary=_anchored_token_config(),
        d_model=16,
        latent_dim=16,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        dim_feedforward=32,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
        use_final_norm=True,
        gradient_checkpointing=False,
        positional_encoding=PositionalEncodingConfig(
            type="sinusoidal",
            max_length=16,
        ),
        decoder_autoregressive=True,
        decoder_memory_source="encoder",
        tie_token_weights=tie_weights,
        generation_grammar="anchored_v3",
        reconstruction=ReconstructionHeadConfig(
            enabled=True,
            target="tok_dict",
        ),
    )
    return SketchformerModel(config).eval()


def _padded_batch(sequences: list[list[int]]) -> dict[str, torch.Tensor]:
    max_length = max(len(sequence) for sequence in sequences)
    tokens = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        tokens[row, : len(sequence)] = torch.tensor(sequence)
    return {
        "tokens": tokens,
        "targets": tokens.clone(),
        **build_sequence_masks([len(sequence) for sequence in sequences], max_length),
    }


class SketchformerEncoderMemoryTest(unittest.TestCase):
    def test_mapping_parses_v3_decoder_contract(self) -> None:
        config = SketchformerConfig.from_mapping(
            {
                "name": "sketchformer_tok_dict_anchored_v3",
                "input": {
                    "type": "tok_dict",
                    "max_seq_len": 4096,
                    "token_dictionary": {
                        "codebook_size": 2048,
                        "motion_token_offset": 1,
                        "pad_token_id": 0,
                        "x_token_offset": 2049,
                        "y_token_offset": 2305,
                        "coordinate_bins": 256,
                        "stroke_start_token_id": 2561,
                        "stroke_end_token_id": 2562,
                        "sos_token_id": 2563,
                        "eos_token_id": 2564,
                        "mask_token_id": 2565,
                        "vocab_size": 2566,
                    },
                },
                "embedding": {
                    "positional_encoding": {"max_length": 4096},
                },
                "decoder": {
                    "autoregressive": True,
                    "memory_source": "encoder",
                    "tie_token_weights": True,
                    "generation_grammar": "anchored_v3",
                },
                "heads": {
                    "reconstruction": {"target": "tok_dict"},
                },
            }
        )

        config.validate()

        self.assertEqual(config.decoder_memory_source, "encoder")
        self.assertTrue(config.tie_token_weights)
        self.assertEqual(config.resolved_generation_grammar, "anchored_v3")
        self.assertIsNone(config.token_dictionary.sep_token_id)

    def test_encoder_memory_omits_expander_and_ties_token_weights(self) -> None:
        model = _encoder_memory_model()
        assert isinstance(model.reconstruction_head, TokenReconstructionHead)

        input_weight = model.input_embedding.token_embedding.weight
        target_weight = model.target_embedding.token_embedding.weight
        output_weight = model.reconstruction_head.projection.weight

        self.assertIsNone(model.latent_expander)
        self.assertEqual(input_weight.data_ptr(), target_weight.data_ptr())
        self.assertEqual(input_weight.data_ptr(), output_weight.data_ptr())

    def test_legacy_configuration_keeps_latent_expander_default(self) -> None:
        config = SketchformerConfig(
            max_seq_len=8,
            d_model=8,
            latent_dim=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            dim_feedforward=16,
            dropout=0.0,
            gradient_checkpointing=False,
            positional_encoding=PositionalEncodingConfig(max_length=8),
            reconstruction=ReconstructionHeadConfig(
                enabled=True,
                xy_distribution="deterministic",
            ),
        )

        model = SketchformerModel(config)

        self.assertEqual(config.decoder_memory_source, "latent_expander")
        self.assertIsNotNone(model.latent_expander)

    def test_teacher_first_step_matches_cached_decoder_first_step(self) -> None:
        torch.manual_seed(11)
        model = _encoder_memory_model()
        sequence = [15, 13, 5, 9, 1, 14, 16]
        batch = _padded_batch([sequence])

        with torch.no_grad():
            teacher_logits = model(batch).reconstruction.token_logits[:, 0]
            encoded = model.encode(
                batch["tokens"],
                attention_mask=batch["sdpa_mask"],
            )
            decoder_input = model.target_embedding(
                batch["tokens"][:, :1],
                position_offset=0,
            )
            cross_mask = model._cross_attention_mask(
                batch["valid_mask"],
                1,
                encoded.shape[1],
            )
            decoded, _ = model.decoder.forward_step(
                decoder_input,
                encoded,
                cross_attention_mask=cross_mask,
            )
            cached_logits = model.reconstruction_head(decoded).token_logits[:, 0]

        torch.testing.assert_close(teacher_logits, cached_logits, atol=1e-6, rtol=1e-6)

    def test_corrupted_decoder_inputs_do_not_replace_loss_targets(self) -> None:
        model = _encoder_memory_model()
        sequence = [15, 13, 5, 9, 1, 14, 16]
        batch = _padded_batch([sequence])
        decoder_inputs = batch["targets"].clone()
        decoder_inputs[:, 4] = 17

        output = model(
            batch,
            targets=batch["targets"],
            decoder_inputs=decoder_inputs,
        )

        self.assertTrue(torch.equal(output.loss_targets, batch["targets"][:, 1:]))

    def test_forward_is_invariant_to_padding_and_batch_companions(self) -> None:
        torch.manual_seed(17)
        model = _encoder_memory_model(tie_weights=False)
        short = [15, 13, 5, 9, 1, 14, 16]
        long = [15, 13, 6, 10, 2, 3, 14, 13, 7, 11, 4, 14, 16]
        single = _padded_batch([short])
        combined = _padded_batch([short, long])

        with torch.no_grad():
            single_logits = model(single).reconstruction.token_logits
            combined_logits = model(combined).reconstruction.token_logits[:1]
            assert isinstance(model.reconstruction_head, TokenReconstructionHead)
            bias = model.reconstruction_head.projection.bias
            assert bias is not None
            bias.zero_()
            bias[13] = 100.0
            bias[5:9] = torch.arange(20.0, 24.0)
            bias[9:13] = torch.arange(30.0, 34.0)
            bias[1:5] = torch.arange(40.0, 44.0)
            bias[14] = 39.0
            single_generation = model.generate(single, use_cache=True)
            combined_generation = model.generate(combined, use_cache=True)

        torch.testing.assert_close(
            single_logits,
            combined_logits[:, : single_logits.shape[1]],
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertTrue(
            torch.equal(
                single_generation.tokens[0],
                combined_generation.tokens[0, : single_generation.tokens.shape[1]],
            )
        )
        self.assertEqual(
            single_generation.lengths[0].item(),
            combined_generation.lengths[0].item(),
        )

    def test_cached_and_uncached_generation_match_and_use_per_sample_caps(self) -> None:
        torch.manual_seed(23)
        model = _encoder_memory_model()
        short = [15, 13, 5, 9, 1, 14, 16]
        long = [15, 13, 6, 10, 2, 3, 14, 13, 7, 11, 4, 14, 16]
        batch = _padded_batch([short, long])
        assert isinstance(model.reconstruction_head, TokenReconstructionHead)
        with torch.no_grad():
            bias = model.reconstruction_head.projection.bias
            assert bias is not None
            bias.zero_()
            bias[13] = 100.0
            bias[1] = 90.0
            bias[14] = 80.0
            bias[16] = 70.0

            cached = model.generate(batch, max_length=10, use_cache=True)
            uncached = model.generate(batch, max_length=10, use_cache=False)

        self.assertTrue(torch.equal(cached.tokens, uncached.tokens))
        self.assertTrue(torch.equal(cached.lengths, uncached.lengths))
        self.assertEqual(cached.tokens.shape, (2, 10))
        self.assertEqual(cached.lengths.tolist(), [7, 10])
        self.assertTrue(torch.equal(cached.tokens[0, 7:], torch.zeros(3, dtype=torch.long)))

    def test_anchored_generation_constraints_encode_the_complete_grammar(self) -> None:
        model = _encoder_memory_model()
        previous = torch.tensor([15, 13, 5, 9, 1, 14, 16])

        allowed = model._anchored_v3_allowed_tokens(previous)

        self.assertEqual(allowed.sum(dim=1).tolist(), [2, 4, 4, 4, 5, 2, 1])
        self.assertTrue(allowed[0, 13])
        self.assertTrue(allowed[0, 16])
        self.assertTrue(allowed[1, 5:9].all())
        self.assertTrue(allowed[2, 9:13].all())
        self.assertTrue(allowed[3, 1:5].all())
        self.assertTrue(allowed[4, 1:5].all())
        self.assertTrue(allowed[4, 14])


if __name__ == "__main__":
    unittest.main()
