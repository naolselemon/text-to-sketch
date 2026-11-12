from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dataloaders.stroke_sequence_dataset import StrokeSequenceDataset
from prep_data import prepare_sketchformer, prepare_sketchformer_tokens
from prep_data.sketch_token.create_token_dict import save_codebook


def _create_mock_codebook(target_dir: Path, K: int = 10) -> tuple[Path, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    codebook = rng.standard_normal((K, 2)).astype(np.float32)
    save_codebook(codebook, target_dir, K=K, n_samples=100)
    codebook_path = target_dir / "codebook.npy"
    sha256 = hashlib.sha256(codebook_path.read_bytes()).hexdigest()
    return codebook_path, sha256


def _write_mock_token_file(path: Path, sample_id: str, K: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sos = K + 2
    eos = K + 3
    sep = K + 1
    # Valid tokens within bounds [0, K + 4)
    tokens = np.array([sos, 1, 2, sep, 3, eos], dtype=np.int64)
    np.savez_compressed(path, tokens=tokens)


def _write_mock_stroke5_file(path: Path, sample_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stroke5 = np.array(
        [
            [0.1, 0.2, 1.0, 0.0, 0.0],
            [0.3, 0.4, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],  # sentinel row
        ],
        dtype=np.float32,
    )
    np.savez(path, stroke5=stroke5)


class DatasetIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dict_dir = self.root / "sketch_token"
        self.codebook_path, self.codebook_hash = _create_mock_codebook(self.dict_dir, K=10)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_accepted_then_rejected_smaller_rebuild(self) -> None:
        """Build with 10 items, rebuild with 5 (marking 5 rejected).

        Assert strictly only 5 can load and zero remnants of the 5 rejected drawings leak through.
        """
        tokens_dir = self.root / "tokens"
        manifest_1 = self.root / "manifest_1.jsonl"
        manifest_2 = self.root / "manifest_2.jsonl"
        build_v1 = self.root / "build_v1"
        build_v2 = self.root / "build_v2"

        # Create 10 mock token files
        all_ids = [f"sketch_{i:03d}" for i in range(10)]
        for sid in all_ids:
            _write_mock_token_file(tokens_dir / f"{sid}.npz", sid, K=10)

        # Manifest 1: all 10 accepted
        with open(manifest_1, "w", encoding="utf-8") as fh:
            for sid in all_ids:
                record = {
                    "sample_id": sid,
                    "status": "accepted",
                    "tokens_path": str(tokens_dir / f"{sid}.npz"),
                    "token_dictionary_sha256": self.codebook_hash,
                    "token_dictionary_size": 10,
                }
                fh.write(json.dumps(record) + "\n")

        with patch.object(
            sys,
            "argv",
            [
                "prepare_tokens",
                "--tokens-dir",
                str(tokens_dir),
                "--token-dict-dir",
                str(self.dict_dir),
                "--target-dir",
                str(build_v1),
                "--manifest",
                str(manifest_1),
                "--n-chunks",
                "2",
            ],
        ):
            ret = prepare_sketchformer_tokens.main()
            self.assertEqual(ret, 0)

        # Verify build_v1 loads all 10 items
        ds_train = StrokeSequenceDataset(build_v1, split="train", format_type="tok_dict")
        ds_valid = StrokeSequenceDataset(build_v1, split="valid", format_type="tok_dict")
        ds_test = StrokeSequenceDataset(build_v1, split="test", format_type="tok_dict")
        self.assertEqual(len(ds_train) + len(ds_valid) + len(ds_test), 10)

        # Manifest 2: 5 accepted, 5 rejected
        accepted_ids = all_ids[:5]
        rejected_ids = all_ids[5:]
        with open(manifest_2, "w", encoding="utf-8") as fh:
            for sid in accepted_ids:
                record = {
                    "sample_id": sid,
                    "status": "accepted",
                    "tokens_path": str(tokens_dir / f"{sid}.npz"),
                    "token_dictionary_sha256": self.codebook_hash,
                    "token_dictionary_size": 10,
                }
                fh.write(json.dumps(record) + "\n")
            for sid in rejected_ids:
                record = {
                    "sample_id": sid,
                    "status": "rejected",
                    "rejection_reason": "overlength",
                    "tokens_path": str(tokens_dir / f"{sid}.npz"),
                    "token_dictionary_sha256": self.codebook_hash,
                    "token_dictionary_size": 10,
                }
                fh.write(json.dumps(record) + "\n")

        # Build v2 from manifest 2
        with patch.object(
            sys,
            "argv",
            [
                "prepare_tokens",
                "--tokens-dir",
                str(tokens_dir),
                "--token-dict-dir",
                str(self.dict_dir),
                "--target-dir",
                str(build_v2),
                "--manifest",
                str(manifest_2),
                "--n-chunks",
                "2",
            ],
        ):
            ret = prepare_sketchformer_tokens.main()
            self.assertEqual(ret, 0)

        # Verify build_v2 loads strictly only 5 items
        v2_train = StrokeSequenceDataset(build_v2, split="train", format_type="tok_dict")
        v2_valid = StrokeSequenceDataset(build_v2, split="valid", format_type="tok_dict")
        v2_test = StrokeSequenceDataset(build_v2, split="test", format_type="tok_dict")
        total_v2 = len(v2_train) + len(v2_valid) + len(v2_test)
        self.assertEqual(total_v2, 5)

        # Verify absolutely zero remnants of the 5 rejected items can be loaded
        loaded_ids: set[str] = set()
        for ds in (v2_train, v2_valid, v2_test):
            for i in range(len(ds)):
                loaded_ids.add(ds[i]["sample_id"])

        self.assertEqual(loaded_ids, set(accepted_ids))
        for rejected_id in rejected_ids:
            self.assertNotIn(rejected_id, loaded_ids)

        # Also inspect chunk files directly on disk in build_v2
        for chunk_file in build_v2.glob("*.npz"):
            if chunk_file.name == "meta.npz":
                continue
            with np.load(chunk_file, allow_pickle=True) as data:
                if "sample_ids" in data:
                    chunk_sids = set(data["sample_ids"])
                    for rejected_id in rejected_ids:
                        self.assertNotIn(rejected_id, chunk_sids)

    def test_duplicate_ids_fails(self) -> None:
        """Assert build failure (ValueError) on input containing duplicate identifiers."""
        tokens_dir = self.root / "tokens"
        manifest = self.root / "manifest_dup.jsonl"
        target_dir = self.root / "build_dup"

        _write_mock_token_file(tokens_dir / "sketch_001.npz", "sketch_001", K=10)

        with open(manifest, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"sample_id": "sketch_001", "status": "accepted"}) + "\n")
            fh.write(json.dumps({"sample_id": "sketch_001", "status": "accepted"}) + "\n")

        with patch.object(
            sys,
            "argv",
            [
                "prepare_tokens",
                "--tokens-dir",
                str(tokens_dir),
                "--token-dict-dir",
                str(self.dict_dir),
                "--target-dir",
                str(target_dir),
                "--manifest",
                str(manifest),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "Duplicate sample ID"):
                prepare_sketchformer_tokens.main()

        # Also test duplicate in --accepted-ids
        with patch.object(
            sys,
            "argv",
            [
                "prepare_tokens",
                "--tokens-dir",
                str(tokens_dir),
                "--token-dict-dir",
                str(self.dict_dir),
                "--target-dir",
                str(target_dir),
                "--accepted-ids",
                "sketch_001,sketch_001",
            ],
        ):
            with self.assertRaisesRegex(ValueError, "Duplicate sample ID"):
                prepare_sketchformer_tokens.main()

    def test_missing_file_fails(self) -> None:
        """Confirm build failure (FileNotFoundError) if manifest-listed files are missing."""
        tokens_dir = self.root / "tokens"
        manifest = self.root / "manifest_missing.jsonl"
        target_dir = self.root / "build_missing"

        _write_mock_token_file(tokens_dir / "sketch_exists.npz", "sketch_exists", K=10)

        with open(manifest, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"sample_id": "sketch_exists", "status": "accepted"}) + "\n")
            fh.write(json.dumps({"sample_id": "sketch_ghost", "status": "accepted"}) + "\n")

        with patch.object(
            sys,
            "argv",
            [
                "prepare_tokens",
                "--tokens-dir",
                str(tokens_dir),
                "--token-dict-dir",
                str(self.dict_dir),
                "--target-dir",
                str(target_dir),
                "--manifest",
                str(manifest),
            ],
        ):
            with self.assertRaises(FileNotFoundError):
                prepare_sketchformer_tokens.main()

    def test_changed_dictionary_fails(self) -> None:
        """Assert builder fails with ValueError on dictionary hash mismatch or out-of-bounds tokens."""
        tokens_dir = self.root / "tokens"
        target_dir = self.root / "build_changed_dict"

        _write_mock_token_file(tokens_dir / "sketch_001.npz", "sketch_001", K=10)

        # Hash mismatch in manifest
        manifest_hash_mismatch = self.root / "manifest_hash_mismatch.jsonl"
        with open(manifest_hash_mismatch, "w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "sample_id": "sketch_001",
                        "status": "accepted",
                        "token_dictionary_sha256": "wrong_hash_1234567890",
                        "token_dictionary_size": 10,
                    }
                )
                + "\n"
            )

        with patch.object(
            sys,
            "argv",
            [
                "prepare_tokens",
                "--tokens-dir",
                str(tokens_dir),
                "--token-dict-dir",
                str(self.dict_dir),
                "--target-dir",
                str(target_dir),
                "--manifest",
                str(manifest_hash_mismatch),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "Token dictionary hash mismatch"):
                prepare_sketchformer_tokens.main()

        # Out-of-bounds token in file
        bad_token_file = tokens_dir / "sketch_oob.npz"
        # codebook K=10, vocab_size=14; 9999 is out of bounds
        np.savez_compressed(bad_token_file, tokens=np.array([12, 9999, 13], dtype=np.int64))

        manifest_oob = self.root / "manifest_oob.jsonl"
        with open(manifest_oob, "w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "sample_id": "sketch_oob",
                        "status": "accepted",
                        "token_dictionary_sha256": self.codebook_hash,
                        "token_dictionary_size": 10,
                    }
                )
                + "\n"
            )

        with patch.object(
            sys,
            "argv",
            [
                "prepare_tokens",
                "--tokens-dir",
                str(tokens_dir),
                "--token-dict-dir",
                str(self.dict_dir),
                "--target-dir",
                str(target_dir),
                "--manifest",
                str(manifest_oob),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "outside valid dictionary bounds"):
                prepare_sketchformer_tokens.main()

        with patch.object(
            sys,
            "argv",
            [
                "prepare_tokens",
                "--tokens-dir",
                str(tokens_dir),
                "--token-dict-dir",
                str(self.dict_dir),
                "--target-dir",
                str(target_dir),
                "--accepted-ids",
                "sketch_001",
                "--dictionary-hash",
                "non_matching_cli_hash",
            ],
        ):
            with self.assertRaisesRegex(ValueError, "does not match expected --dictionary-hash"):
                prepare_sketchformer_tokens.main()

    def test_interrupted_build_fails(self) -> None:
        """Assert archive without valid post-completion meta publication mark is rejected on load."""
        target_dir = self.root / "build_interrupted"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Chunks exist on disk
        chunk_file = target_dir / "train_000.npz"
        tokens = np.array([np.array([12, 1, 13], dtype=np.int64)], dtype=object)
        labels = np.array([0], dtype=np.int32)
        np.savez_compressed(chunk_file, x=tokens, y=labels)

        # Case 1: meta.npz completely missing
        with self.assertRaises(FileNotFoundError):
            StrokeSequenceDataset(target_dir, split="train", format_type="tok_dict")

        # Case 2: meta.npz exists but lacks publication signature
        np.savez(target_dir / "meta.npz", status="incomplete", published=False, example_count=1)
        with self.assertRaisesRegex(RuntimeError, "missing a valid publication record"):
            StrokeSequenceDataset(target_dir, split="train", format_type="tok_dict")

        # Case 3: meta.npz claims published, but example count does not match loaded chunks
        np.savez(
            target_dir / "meta.npz",
            status="complete",
            published=True,
            n_samples_train=999,  # actual chunk has only 1 item
            example_count=999,
        )
        with self.assertRaises(AssertionError):
            StrokeSequenceDataset(target_dir, split="train", format_type="tok_dict")

    def test_untouched_baseline_files(self) -> None:
        """Guarantee run-isolated builds write strictly to target dir leaving baseline files untouched."""
        tokens_dir = self.root / "tokens"
        baseline_dir = self.root / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "isolated_target"

        # Create baseline files in baseline_dir
        (baseline_dir / "prior_data.txt").write_text("critical prior data", encoding="utf-8")
        (baseline_dir / "model.pt").write_bytes(b"\x00\x01\x02\x03\x04")

        # Record mtimes and sha256 hashes of baseline files
        baseline_state: dict[Path, tuple[int, str]] = {}
        for p in baseline_dir.iterdir():
            baseline_state[p] = (
                p.stat().st_mtime_ns,
                hashlib.sha256(p.read_bytes()).hexdigest(),
            )

        # Also create a mock token and manifest
        sid = "sketch_001"
        _write_mock_token_file(tokens_dir / f"{sid}.npz", sid, K=10)
        manifest = self.root / "manifest_isolated.jsonl"
        with open(manifest, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"sample_id": sid, "status": "accepted"}) + "\n")

        with patch.object(
            sys,
            "argv",
            [
                "prepare_tokens",
                "--tokens-dir",
                str(tokens_dir),
                "--token-dict-dir",
                str(self.dict_dir),
                "--target-dir",
                str(target_dir),
                "--manifest",
                str(manifest),
                "--train-frac",
                "1.0",
                "--valid-frac",
                "0.0",
                "--test-frac",
                "0.0",
                "--n-chunks",
                "1",
            ],
        ):
            ret = prepare_sketchformer_tokens.main()
            self.assertEqual(ret, 0)

        # Confirm target directory was populated
        self.assertTrue((target_dir / "meta.npz").exists())
        self.assertTrue((target_dir / "train_000.npz").exists())

        # Verify baseline files were completely untouched
        for p, (expected_mtime, expected_hash) in baseline_state.items():
            self.assertTrue(p.exists(), f"Baseline file {p} was deleted!")
            current_mtime = p.stat().st_mtime_ns
            current_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(
                current_hash, expected_hash, f"Baseline file {p} content was modified!"
            )
            self.assertEqual(
                current_mtime, expected_mtime, f"Baseline file {p} was touched/written!"
            )


class ContinuousStroke3IsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def test_stroke3_requires_explicit_manifest_or_ids(self) -> None:
        """Ensure prepare_sketchformer raises ValueError if neither manifest nor accepted-ids is passed."""
        with patch.object(
            sys,
            "argv",
            [
                "prepare_sketchformer",
                "--source-dir",
                str(self.root / "stroke5"),
                "--target-dir",
                str(self.root / "target"),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "An explicit input list is required"):
                prepare_sketchformer.main()

    def test_stroke3_builds_and_publishes_atomically(self) -> None:
        """Test continuous stroke3 builder isolation with manifest and atomic meta."""
        stroke5_dir = self.root / "stroke5"
        manifest = self.root / "manifest_s3.jsonl"
        target_dir = self.root / "target_s3"

        sids = ["sk_0", "sk_1", "sk_2"]
        for sid in sids:
            _write_mock_stroke5_file(stroke5_dir / f"{sid}.npz", sid)

        with open(manifest, "w", encoding="utf-8") as fh:
            for sid in sids:
                fh.write(json.dumps({"sample_id": sid, "status": "accepted"}) + "\n")

        with patch.object(
            sys,
            "argv",
            [
                "prepare_sketchformer",
                "--source-dir",
                str(stroke5_dir),
                "--target-dir",
                str(target_dir),
                "--manifest",
                str(manifest),
                "--n-chunks",
                "1",
            ],
        ):
            ret = prepare_sketchformer.main()
            self.assertEqual(ret, 0)

        # Verify loader verifies publication and counts
        ds_train = StrokeSequenceDataset(target_dir, split="train", format_type="stroke3")
        self.assertGreater(len(ds_train), 0)


if __name__ == "__main__":
    unittest.main()
