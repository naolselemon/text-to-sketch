"""Tests for Issue #13: Data leakage prevention across runs and rebuilds."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataloaders.stroke_sequence_dataset import StrokeSequenceDataset
from prep_data.prepare_sketchformer_tokens import main as prepare_tokens_main
from prep_data.sketch_token.create_token_dict import compute_codebook_fingerprint
from utils.io import save_token_sequence


class DataLeakageIssue13Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.token_dict_dir = self.root / "token_dict"
        self.token_dict_dir.mkdir(parents=True, exist_ok=True)

        self.codebook_size = 16
        self.codebook = np.zeros((self.codebook_size, 2), dtype=np.float32)
        for i in range(self.codebook_size):
            self.codebook[i] = [float(i) * 0.1, float(i) * 0.1]
        np.save(self.token_dict_dir / "codebook.npy", self.codebook)
        self.codebook_sha256 = compute_codebook_fingerprint(self.token_dict_dir / "codebook.npy")
        (self.token_dict_dir / "metadata.json").write_text(
            json.dumps({"codebook_size": self.codebook_size, "codebook_sha256": self.codebook_sha256}),
            encoding="utf-8",
        )

        self.sos = self.codebook_size + 2  # 18
        self.eos = self.codebook_size + 3  # 19

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _make_valid_tokens(self, length: int = 10) -> np.ndarray:
        seq = [self.sos] + [((i % 16) + 1) for i in range(length - 2)] + [self.eos]
        return np.asarray(seq, dtype=np.int64)

    def test_accepted_then_rejected_across_runs_does_not_leak(self) -> None:
        """Scenario 1: Sample B was accepted in Run 1, but rejected in Run 2.
        Run 2 manifest marks B as rejected. B's old .npz file remains on disk.
        Verify new dataset build excludes B, contains only Picture A, and leaves B.npz untouched.
        """
        tokens_dir = self.root / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "chunks"

        tokens_a = self._make_valid_tokens(8)
        tokens_b = self._make_valid_tokens(12)

        file_a = tokens_dir / "picture_a.npz"
        file_b = tokens_dir / "picture_b.npz"

        # Run 1: saved both picture_a and picture_b
        save_token_sequence(tokens_a, file_a, build_id="run_1", token_dict_sha256=self.codebook_sha256, sample_id="picture_a")
        save_token_sequence(tokens_b, file_b, build_id="run_1", token_dict_sha256=self.codebook_sha256, sample_id="picture_b")

        # Run 2: picture_a is accepted and saved with run_2; picture_b is rejected so file_b is untouched
        save_token_sequence(tokens_a, file_a, build_id="run_2", token_dict_sha256=self.codebook_sha256, sample_id="picture_a")
        b_stat_before = file_b.stat()

        manifest_path = tokens_dir / "manifest.jsonl"
        records = [
            {
                "sample_id": "picture_a",
                "status": "accepted",
                "build_id": "run_2",
                "token_dict_sha256": self.codebook_sha256,
                "tokens_path": str(file_a),
                "tokens_sha256": hashlib.sha256(file_a.read_bytes()).hexdigest(),
            },
            {
                "sample_id": "picture_b",
                "status": "rejected",
                "build_id": "run_2",
                "rejection_reason": "overlength",
            },
        ]
        with manifest_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ret = prepare_tokens_main([
            "--tokens-dir", str(tokens_dir),
            "--token-dict-dir", str(self.token_dict_dir),
            "--target-dir", str(target_dir),
            "--n-chunks", "1",
            "--train-frac", "1.0",
            "--valid-frac", "0.0",
            "--test-frac", "0.0",
            "--build-id", "run_2",
        ])
        self.assertEqual(ret, 0)

        self.assertTrue(file_b.exists())
        self.assertEqual(file_b.stat().st_size, b_stat_before.st_size)

        dataset = StrokeSequenceDataset(target_dir, split="train")
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["sample_id"], "picture_a")

    def test_smaller_rebuild_ignores_stale_chunks_and_preserves_previous(self) -> None:
        """Scenario 2: Initial build creates 4 chunks. Rebuild creates only 2 chunks.
        Verify StrokeSequenceDataset loads only the 2 new chunks, ignores stale chunks,
        and the previous build is snapshotted to versions/<prev_build_id>.
        """
        tokens_dir = self.root / "tokens_rebuild"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "chunks_rebuild"

        records_1 = []
        for i in range(12):
            sid = f"sample_{i:02d}"
            fpath = tokens_dir / f"{sid}.npz"
            tok = self._make_valid_tokens(10)
            save_token_sequence(tok, fpath, build_id="build_1", token_dict_sha256=self.codebook_sha256, sample_id=sid)
            records_1.append({
                "sample_id": sid,
                "status": "accepted",
                "build_id": "build_1",
                "tokens_sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
                "token_dict_sha256": self.codebook_sha256,
            })
        manifest_path = tokens_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for r in records_1:
                f.write(json.dumps(r) + "\n")

        prepare_tokens_main([
            "--tokens-dir", str(tokens_dir),
            "--token-dict-dir", str(self.token_dict_dir),
            "--target-dir", str(target_dir),
            "--n-chunks", "4",
            "--train-frac", "1.0",
            "--valid-frac", "0.0",
            "--test-frac", "0.0",
            "--build-id", "build_1",
        ])
        ds1 = StrokeSequenceDataset(target_dir, split="train")
        self.assertEqual(len(ds1), 12)
        self.assertEqual(len(ds1.files), 4)

        records_2 = records_1[:4]
        for r in records_2:
            r["build_id"] = "build_2"
            fpath = tokens_dir / f"{r['sample_id']}.npz"
            save_token_sequence(self._make_valid_tokens(10), fpath, build_id="build_2", token_dict_sha256=self.codebook_sha256, sample_id=r["sample_id"])
            r["tokens_sha256"] = hashlib.sha256(fpath.read_bytes()).hexdigest()
        for r in records_1[4:]:
            records_2.append({"sample_id": r["sample_id"], "status": "rejected", "build_id": "build_2"})

        with manifest_path.open("w", encoding="utf-8") as f:
            for r in records_2:
                f.write(json.dumps(r) + "\n")

        prepare_tokens_main([
            "--tokens-dir", str(tokens_dir),
            "--token-dict-dir", str(self.token_dict_dir),
            "--target-dir", str(target_dir),
            "--n-chunks", "2",
            "--train-frac", "1.0",
            "--valid-frac", "0.0",
            "--test-frac", "0.0",
            "--build-id", "build_2",
        ])

        self.assertTrue((target_dir / "train_000_build_1.npz").exists())
        self.assertTrue((target_dir / "train_002_build_1.npz").exists())
        self.assertTrue((target_dir / "train_003_build_1.npz").exists())

        ds2 = StrokeSequenceDataset(target_dir, split="train")
        self.assertEqual(len(ds2), 4)
        self.assertEqual(len(ds2.files), 2)
        self.assertEqual([f.name for f in ds2.files], ["train_000_build_2.npz", "train_001_build_2.npz"])

        prev_snapshot = target_dir / "versions" / "build_1"
        self.assertTrue(prev_snapshot.is_dir())
        self.assertTrue((prev_snapshot / "meta.npz").is_file())
        self.assertTrue((prev_snapshot / "train_003_build_1.npz").is_file())

    def test_duplicate_drawing_ids_in_manifest_raises_value_error(self) -> None:
        """Scenario 3: Duplicate IDs in manifest must fail fast."""
        tokens_dir = self.root / "tokens_dup"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "chunks_dup"

        manifest_path = tokens_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"sample_id": "dup_1", "status": "accepted"}) + "\n")
            f.write(json.dumps({"sample_id": "dup_1", "status": "accepted"}) + "\n")

        with self.assertRaisesRegex(ValueError, "Duplicate drawing IDs detected in manifest"):
            prepare_tokens_main([
                "--tokens-dir", str(tokens_dir),
                "--token-dict-dir", str(self.token_dict_dir),
                "--target-dir", str(target_dir),
            ])

    def test_missing_token_file_for_accepted_sample_raises_file_not_found(self) -> None:
        """Scenario 4: Missing token file for an accepted manifest record raises FileNotFoundError."""
        tokens_dir = self.root / "tokens_missing"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "chunks_missing"

        manifest_path = tokens_dir / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps({"sample_id": "ghost_sample", "status": "accepted"}) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(FileNotFoundError, "Missing token file for accepted sample"):
            prepare_tokens_main([
                "--tokens-dir", str(tokens_dir),
                "--token-dict-dir", str(self.token_dict_dir),
                "--target-dir", str(target_dir),
            ])

    def test_changed_token_dictionary_raises_value_error(self) -> None:
        """Scenario 5: Changed token dictionary fingerprint raises ValueError."""
        tokens_dir = self.root / "tokens_dict_mismatch"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "chunks_dict_mismatch"

        fpath = tokens_dir / "s1.npz"
        save_token_sequence(self._make_valid_tokens(8), fpath, token_dict_sha256="wrong_hash_12345", sample_id="s1")

        manifest_path = tokens_dir / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps({
                "sample_id": "s1",
                "status": "accepted",
                "token_dict_sha256": "wrong_hash_12345",
                "tokens_sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
            }) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Token dictionary fingerprint mismatch"):
            prepare_tokens_main([
                "--tokens-dir", str(tokens_dir),
                "--token-dict-dir", str(self.token_dict_dir),
                "--target-dir", str(target_dir),
            ])

    def test_tokens_sha256_mismatch_raises_value_error(self) -> None:
        """Refinement 2: tokens_sha256 mismatch raises ValueError."""
        tokens_dir = self.root / "tokens_hash_mismatch"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "chunks_hash_mismatch"

        fpath = tokens_dir / "s1.npz"
        save_token_sequence(self._make_valid_tokens(8), fpath, token_dict_sha256=self.codebook_sha256, sample_id="s1")

        manifest_path = tokens_dir / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps({
                "sample_id": "s1",
                "status": "accepted",
                "tokens_sha256": "tampered_sha256_hash",
                "token_dict_sha256": self.codebook_sha256,
            }) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "SHA-256 hash mismatch"):
            prepare_tokens_main([
                "--tokens-dir", str(tokens_dir),
                "--token-dict-dir", str(self.token_dict_dir),
                "--target-dir", str(target_dir),
            ])

    def test_untracked_unexpected_file_rejected_by_default(self) -> None:
        """Refinement 1: Untracked file in tokens_dir not in manifest raises ValueError by default."""
        tokens_dir = self.root / "tokens_untracked"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "chunks_untracked"

        tracked_path = tokens_dir / "tracked.npz"
        save_token_sequence(self._make_valid_tokens(8), tracked_path, token_dict_sha256=self.codebook_sha256, sample_id="tracked")

        untracked_path = tokens_dir / "surprise_leak.npz"
        save_token_sequence(self._make_valid_tokens(8), untracked_path, token_dict_sha256=self.codebook_sha256, sample_id="surprise_leak")

        manifest_path = tokens_dir / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps({
                "sample_id": "tracked",
                "status": "accepted",
                "tokens_sha256": hashlib.sha256(tracked_path.read_bytes()).hexdigest(),
                "token_dict_sha256": self.codebook_sha256,
            }) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Unexpected untracked token file found on disk"):
            prepare_tokens_main([
                "--tokens-dir", str(tokens_dir),
                "--token-dict-dir", str(self.token_dict_dir),
                "--target-dir", str(target_dir),
            ])

    def test_interrupted_build_without_publication_marker_rejected(self) -> None:
        """Scenario 6: Interrupted or unpublished build raises RuntimeError when loaded."""
        target_dir = self.root / "chunks_interrupted"
        target_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(target_dir / "train_000.npz", x=np.asarray([[1, 2]], dtype=object), y=np.asarray([0]))
        # Case A: no meta.npz at all
        with self.assertRaisesRegex(RuntimeError, "not published or build was interrupted"):
            StrokeSequenceDataset(target_dir, split="train", format_type="tok_dict")

        # Case B: meta.npz exists but is_published=False
        np.savez(target_dir / "meta.npz", format="tok_dict", is_published=False)
        with self.assertRaisesRegex(RuntimeError, "not published or build was interrupted"):
            StrokeSequenceDataset(target_dir, split="train", format_type="tok_dict")
        with self.assertRaisesRegex(RuntimeError, "not published or build was interrupted"):
            StrokeSequenceDataset(target_dir, split="train")

    def test_old_user_files_in_target_dir_are_preserved(self) -> None:
        """Scenario 7: User data, notes, and previous versions in target_dir are not destroyed."""
        tokens_dir = self.root / "tokens_preserve"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "chunks_preserve"
        target_dir.mkdir(parents=True, exist_ok=True)

        user_readme = target_dir / "README.txt"
        user_readme.write_text("My custom notes", encoding="utf-8")
        user_code = target_dir / "my_custom_script.py"
        user_code.write_text("print('hello')", encoding="utf-8")

        fpath = tokens_dir / "s1.npz"
        save_token_sequence(self._make_valid_tokens(8), fpath, token_dict_sha256=self.codebook_sha256, sample_id="s1")
        manifest_path = tokens_dir / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps({
                "sample_id": "s1",
                "status": "accepted",
                "tokens_sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
                "token_dict_sha256": self.codebook_sha256,
            }) + "\n",
            encoding="utf-8",
        )

        prepare_tokens_main([
            "--tokens-dir", str(tokens_dir),
            "--token-dict-dir", str(self.token_dict_dir),
            "--target-dir", str(target_dir),
            "--n-chunks", "1",
            "--train-frac", "1.0",
            "--valid-frac", "0.0",
            "--test-frac", "0.0",
        ])

        self.assertTrue(user_readme.is_file())
        self.assertEqual(user_readme.read_text(encoding="utf-8"), "My custom notes")
        self.assertTrue(user_code.is_file())
        self.assertEqual(user_code.read_text(encoding="utf-8"), "print('hello')")

    def test_preprocessing_manifest_discovered_automatically(self) -> None:
        """Finding 1: preprocessing_manifest.jsonl discovered automatically in tokens parent dir."""
        pipeline_output = self.root / "pipeline_run"
        tokens_dir = pipeline_output / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "chunks_auto_manifest"

        fpath = tokens_dir / "sample_auto.npz"
        save_token_sequence(self._make_valid_tokens(8), fpath, build_id="auto_build_1", token_dict_sha256=self.codebook_sha256, sample_id="sample_auto")

        # preprocessing_manifest.jsonl written in pipeline_output (parent of tokens_dir)
        manifest_path = pipeline_output / "preprocessing_manifest.jsonl"
        manifest_path.write_text(
            json.dumps({
                "sample_id": "sample_auto",
                "status": "accepted",
                "build_id": "auto_build_1",
                "tokens_sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
                "token_dict_sha256": self.codebook_sha256,
            }) + "\n",
            encoding="utf-8",
        )

        # Run without --manifest argument
        ret = prepare_tokens_main([
            "--tokens-dir", str(tokens_dir),
            "--token-dict-dir", str(self.token_dict_dir),
            "--target-dir", str(target_dir),
            "--n-chunks", "1",
            "--train-frac", "1.0",
            "--valid-frac", "0.0",
            "--test-frac", "0.0",
        ])
        self.assertEqual(ret, 0)
        ds = StrokeSequenceDataset(target_dir, split="train")
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]["sample_id"], "sample_auto")

    def test_rejects_manifest_with_multiple_distinct_build_ids(self) -> None:
        """Finding 2: Manifest containing multiple distinct build IDs among accepted samples is rejected."""
        tokens_dir = self.root / "tokens_multi_build"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "chunks_multi_build"

        fpath1 = tokens_dir / "s1.npz"
        fpath2 = tokens_dir / "s2.npz"
        save_token_sequence(self._make_valid_tokens(8), fpath1, build_id="build_A", token_dict_sha256=self.codebook_sha256, sample_id="s1")
        save_token_sequence(self._make_valid_tokens(8), fpath2, build_id="build_B", token_dict_sha256=self.codebook_sha256, sample_id="s2")

        manifest_path = tokens_dir / "manifest.jsonl"
        records = [
            {
                "sample_id": "s1",
                "status": "accepted",
                "build_id": "build_A",
                "tokens_sha256": hashlib.sha256(fpath1.read_bytes()).hexdigest(),
                "token_dict_sha256": self.codebook_sha256,
            },
            {
                "sample_id": "s2",
                "status": "accepted",
                "build_id": "build_B",
                "tokens_sha256": hashlib.sha256(fpath2.read_bytes()).hexdigest(),
                "token_dict_sha256": self.codebook_sha256,
            },
        ]
        with manifest_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        with self.assertRaisesRegex(ValueError, "Multiple distinct build IDs detected among accepted records in manifest"):
            prepare_tokens_main([
                "--tokens-dir", str(tokens_dir),
                "--token-dict-dir", str(self.token_dict_dir),
                "--target-dir", str(target_dir),
            ])

    def test_rejects_manifest_with_build_id_mismatching_cli(self) -> None:
        """Finding 2: Manifest record build ID mismatch with explicit --build-id CLI argument is rejected."""
        tokens_dir = self.root / "tokens_cli_mismatch"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "chunks_cli_mismatch"

        fpath = tokens_dir / "s1.npz"
        save_token_sequence(self._make_valid_tokens(8), fpath, build_id="build_A", token_dict_sha256=self.codebook_sha256, sample_id="s1")

        manifest_path = tokens_dir / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps({
                "sample_id": "s1",
                "status": "accepted",
                "build_id": "build_A",
                "tokens_sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
                "token_dict_sha256": self.codebook_sha256,
            }) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Build ID mismatch for accepted sample 's1'"):
            prepare_tokens_main([
                "--tokens-dir", str(tokens_dir),
                "--token-dict-dir", str(self.token_dict_dir),
                "--target-dir", str(target_dir),
                "--build-id", "build_EXPECTED",
            ])

    def test_rebuild_keeps_active_chunks_immutable_until_meta_swap(self) -> None:
        """Finding 3: Active published chunks are never overwritten in-place before meta.npz is swapped."""
        tokens_dir = self.root / "tokens_immutable"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / "chunks_immutable"

        fpath1 = tokens_dir / "s1.npz"
        save_token_sequence(self._make_valid_tokens(8), fpath1, build_id="build_1", token_dict_sha256=self.codebook_sha256, sample_id="s1")

        manifest_path = tokens_dir / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps({
                "sample_id": "s1",
                "status": "accepted",
                "build_id": "build_1",
                "tokens_sha256": hashlib.sha256(fpath1.read_bytes()).hexdigest(),
                "token_dict_sha256": self.codebook_sha256,
            }) + "\n",
            encoding="utf-8",
        )

        # Build 1
        prepare_tokens_main([
            "--tokens-dir", str(tokens_dir),
            "--token-dict-dir", str(self.token_dict_dir),
            "--target-dir", str(target_dir),
            "--n-chunks", "1",
            "--train-frac", "1.0",
            "--valid-frac", "0.0",
            "--test-frac", "0.0",
            "--build-id", "build_1",
        ])

        chunk_b1 = target_dir / "train_000_build_1.npz"
        self.assertTrue(chunk_b1.is_file())
        b1_content = chunk_b1.read_bytes()

        # Build 2 with different sample
        fpath2 = tokens_dir / "s2.npz"
        save_token_sequence(self._make_valid_tokens(14), fpath2, build_id="build_2", token_dict_sha256=self.codebook_sha256, sample_id="s2")
        manifest_path.write_text(
            json.dumps({
                "sample_id": "s2",
                "status": "accepted",
                "build_id": "build_2",
                "tokens_sha256": hashlib.sha256(fpath2.read_bytes()).hexdigest(),
                "token_dict_sha256": self.codebook_sha256,
            }) + "\n" +
            json.dumps({
                "sample_id": "s1",
                "status": "rejected",
                "build_id": "build_2",
            }) + "\n",
            encoding="utf-8",
        )

        prepare_tokens_main([
            "--tokens-dir", str(tokens_dir),
            "--token-dict-dir", str(self.token_dict_dir),
            "--target-dir", str(target_dir),
            "--n-chunks", "1",
            "--train-frac", "1.0",
            "--valid-frac", "0.0",
            "--test-frac", "0.0",
            "--build-id", "build_2",
        ])

        # Verify build_1 chunk was never overwritten in-place
        self.assertTrue(chunk_b1.is_file())
        self.assertEqual(chunk_b1.read_bytes(), b1_content)

        # Verify build_2 chunk is present under build-specific filename
        chunk_b2 = target_dir / "train_000_build_2.npz"
        self.assertTrue(chunk_b2.is_file())

        # Verify active dataset now references build_2
        ds = StrokeSequenceDataset(target_dir, split="train")
        self.assertEqual([f.name for f in ds.files], ["train_000_build_2.npz"])
        self.assertEqual(ds[0]["sample_id"], "s2")


if __name__ == "__main__":
    unittest.main()
