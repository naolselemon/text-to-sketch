from __future__ import annotations

import unittest

import numpy as np

from services.anchored_sketch_data.contract import TOKEN_LAYOUT, artifact_contract
from services.anchored_sketch_data.grammar import AnchoredGrammar, GrammarState, validate_tokens
from services.anchored_sketch_data.preprocessing import geometry_f1, rasterize_strokes
from services.anchored_sketch_data.tokenizer import AnchoredTokenizer


def exact_test_codebook() -> np.ndarray:
    centers = np.column_stack(
        (
            np.arange(2048, dtype=np.float32) + 1000.0,
            np.arange(2048, dtype=np.float32) + 2000.0,
        )
    )
    centers[0] = [1.0, 0.0]
    centers[1] = [0.0, 1.0]
    centers[2] = [-1.0, 0.0]
    centers[3] = [0.0, -1.0]
    return centers


class AnchoredV3TokenizerTest(unittest.TestCase):
    def test_exact_public_token_layout(self) -> None:
        self.assertEqual(
            TOKEN_LAYOUT.to_dict(),
            {
                "version": 3,
                "vocab_size": 2566,
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
            },
        )
        self.assertEqual(artifact_contract()["format_type"], "anchored_v3")
        self.assertEqual(artifact_contract()["canvas_size"], 256)

    def test_encode_uses_anchor_and_one_motion_per_later_point(self) -> None:
        tokenizer = AnchoredTokenizer(exact_test_codebook())
        strokes = [
            [(10.0, 20.0), (11.0, 20.0), (12.0, 20.0)],
            [(100.0, 110.0), (100.0, 111.0)],
        ]

        tokens = tokenizer.encode(strokes)
        summary = validate_tokens(tokens)

        self.assertEqual(
            tokens.tolist(),
            [
                2563,
                2561,
                2059,
                2325,
                1,
                1,
                2562,
                2561,
                2149,
                2415,
                2,
                2562,
                2564,
            ],
        )
        self.assertEqual(summary.stroke_count, 2)
        self.assertEqual(summary.motion_count, 3)
        self.assertEqual(summary.unpadded_length, len(tokens))

    def test_decode_and_raster_round_trip_are_exact_for_codebook_motions(self) -> None:
        tokenizer = AnchoredTokenizer(exact_test_codebook())
        source = [
            [(8.0, 8.0), (9.0, 8.0), (10.0, 8.0), (10.0, 9.0)],
            [(200.0, 200.0), (199.0, 200.0), (199.0, 199.0)],
        ]

        restored = tokenizer.decode(tokenizer.encode(source))

        self.assertEqual(restored, source)
        self.assertEqual(geometry_f1(source, restored), 1.0)
        np.testing.assert_array_equal(rasterize_strokes(source), rasterize_strokes(restored))

    def test_quantization_residual_restarts_at_every_stroke_anchor(self) -> None:
        codebook = exact_test_codebook()
        codebook[0] = [0.75, 0.0]
        codebook[1] = [1.3, 0.0]
        tokenizer = AnchoredTokenizer(codebook)
        tokens = tokenizer.encode(
            [
                [(10.0, 10.0), (11.0, 10.0)],
                [(100.0, 100.0), (101.0, 100.0)],
            ]
        )

        motion_tokens = [token for token in tokens.tolist() if 1 <= token <= 2048]

        self.assertEqual(motion_tokens, [1, 1])
        decoded = tokenizer.decode(tokens)
        self.assertAlmostEqual(decoded[0][-1][0], 10.75)
        self.assertAlmostEqual(decoded[1][-1][0], 100.75)

    def test_grammar_rejects_empty_strokes_missing_motion_and_post_eos_content(self) -> None:
        grammar = AnchoredGrammar()
        invalid = (
            [2563, 2564],
            [2563, 2561, 2049, 2305, 2562, 2564],
            [2563, 2561, 2049, 2305, 1, 2562, 2564, 1],
            [2563, 2561, 2049, 2305, 2565, 2562, 2564],
        )

        for tokens in invalid:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                grammar.validate(tokens)

    def test_grammar_masks_only_tokens_valid_in_current_state(self) -> None:
        grammar = AnchoredGrammar()
        initial = grammar.allowed_token_mask(GrammarState.EXPECT_STROKE_OR_EOS)
        after_stroke = grammar.allowed_token_mask(
            GrammarState.EXPECT_STROKE_OR_EOS, stroke_count=1
        )

        self.assertEqual(np.flatnonzero(initial).tolist(), [2561])
        self.assertEqual(np.flatnonzero(after_stroke).tolist(), [2561, 2564])


if __name__ == "__main__":
    unittest.main()

