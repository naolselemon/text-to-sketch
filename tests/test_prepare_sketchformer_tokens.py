from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from prep_data.prepare_sketchformer_tokens import load_token_file, truncate_tokens
from utils.tokenizer import ErrorFeedbackQuantizer, decode_tokens, encode_stroke5


class PrepareSketchformerTokensTest(unittest.TestCase):
    def test_truncate_tokens_preserves_eos_at_max_length(self) -> None:
        tokens = np.asarray([1002, 1, 2, 3, 4, 1003], dtype=np.int64)

        truncated = truncate_tokens(
            tokens,
            max_length=4,
            sep_token_id=1001,
            eos_token_id=1003,
        )

        self.assertEqual(truncated.tolist(), [1002, 1, 1001, 1003])

    def test_tokenizer_uses_original_tensorflow_token_layout(self) -> None:
        codebook = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        stroke5 = np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        tokens = encode_stroke5(stroke5, codebook)
        decoded = decode_tokens(tokens, codebook)

        self.assertEqual(tokens.tolist(), [4, 1, 2, 3, 5])
        self.assertEqual(decoded[-1].tolist(), [0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(decoded[1].tolist(), [1.0, 0.0, 0.0, 1.0, 0.0])

    def test_preencoded_tokens_load_without_requantization(self) -> None:
        expected = np.asarray([1002, 17, 1001, 1003], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            np.savez_compressed(path, tokens=expected)

            actual = load_token_file(path)

        np.testing.assert_array_equal(actual, expected)

    def test_pair_search_quantizer_corrects_a_missing_small_motion(self) -> None:
        codebook = np.asarray(
            [[0.12, 0.0], [0.08, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        stroke5 = np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        tokens = ErrorFeedbackQuantizer(codebook).encode(stroke5)
        decoded = decode_tokens(tokens, codebook)

        self.assertAlmostEqual(float(decoded[:-1, 0].sum()), 0.2, places=6)

    def test_parse_args_defaults_and_flags(self) -> None:
        from prep_data.prepare_sketchformer_tokens import parse_args

        args = parse_args([])
        self.assertTrue(args.reject_unexpected_files)
        self.assertIsNone(args.manifest)
        self.assertIsNone(args.build_id)

        args = parse_args(["--no-reject-unexpected-files", "--build-id", "b1", "--manifest", "m.jsonl"])
        self.assertFalse(args.reject_unexpected_files)
        self.assertEqual(args.build_id, "b1")
        self.assertEqual(args.manifest, "m.jsonl")

    def test_discover_manifest_finds_file_in_dir_or_parent(self) -> None:
        from prep_data.prepare_sketchformer_tokens import discover_manifest

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tokens_dir = tmp_path / "tokens"
            tokens_dir.mkdir()
            manifest = tmp_path / "manifest.jsonl"
            manifest.write_text("{}", encoding="utf-8")

            found = discover_manifest(tokens_dir)
            self.assertEqual(found, manifest)

    def test_discover_manifest_finds_preprocessing_manifest(self) -> None:
        from prep_data.prepare_sketchformer_tokens import discover_manifest

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tokens_dir = tmp_path / "tokens"
            tokens_dir.mkdir()
            manifest = tmp_path / "preprocessing_manifest.jsonl"
            manifest.write_text("{}", encoding="utf-8")

            found = discover_manifest(tokens_dir)
            self.assertEqual(found, manifest)

    def test_load_manifest_records_parses_jsonl_and_detects_duplicates(self) -> None:
        import json
        from prep_data.prepare_sketchformer_tokens import load_manifest_records

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            records = [{"sample_id": "s1", "status": "accepted"}, {"sample_id": "s2", "status": "rejected"}]
            manifest.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

            loaded = load_manifest_records(manifest)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["sample_id"], "s1")

            # Duplicate check
            dup_manifest = Path(tmp) / "dup.jsonl"
            dup_manifest.write_text('{"sample_id": "s1"}\n{"sample_id": "s1"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate drawing IDs detected"):
                load_manifest_records(dup_manifest)


if __name__ == "__main__":
    unittest.main()
