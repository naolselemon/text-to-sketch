"""Evaluation must read the active published build, not stale chunk aliases."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evals.benchmark_manifest import build_manifest
from evals.failure_analysis import analyze_codebook


class ActiveDatasetEvalsTest(unittest.TestCase):
    def test_benchmark_and_codebook_ignore_old_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = np.array([[1002, 1, 1, 1003]], dtype=np.int64)
            new = np.array([[1002, 2, 1003]], dtype=np.int64)
            np.savez(
                root / "train_000_old.npz", x=old, y=[0], sample_ids=["old"]
            )
            np.savez(
                root / "train_000_new.npz", x=new, y=[0], sample_ids=["new"]
            )
            np.savez(
                root / "meta.npz", is_published=True, format="tok_dict",
                train_chunks=["train_000_new.npz"], train_sample_ids=["new"],
                valid_sample_ids=[], test_sample_ids=[],
                n_samples_train=1, schema_version=2,
            )
            codebook = root / "codebook.npy"
            np.save(codebook, np.zeros((2, 2), dtype=np.float32))

            records = build_manifest(root, "train", sep_token_id=1001,
                                     sos_token_id=1002, eos_token_id=1003)
            self.assertEqual(
                [record["sample_id"] for record in records], ["new"]
            )
            self.assertEqual(records[0]["source_file"], "train_000_new.npz")
            usage = analyze_codebook(
                codebook, root, "train", sos_token_id=1002, eos_token_id=1003
            )
            self.assertEqual(usage["total_motion_tokens"], 1)
            self.assertEqual(usage["used_entries"], 1)

    def test_evaluators_fail_on_corrupt_active_sample_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "train_000_new.npz", x=[[1002, 2, 1003]], y=[0],
                     sample_ids=["unexpected"])
            np.savez(
                root / "meta.npz", is_published=True, format="tok_dict",
                train_chunks=["train_000_new.npz"],
                train_sample_ids=["expected"], valid_sample_ids=[],
                test_sample_ids=[], n_samples_train=1, schema_version=2,
            )
            codebook = root / "codebook.npy"
            np.save(codebook, np.zeros((2, 2), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "Sample IDs mismatch"):
                build_manifest(root, "train")
            with self.assertRaisesRegex(ValueError, "Sample IDs mismatch"):
                analyze_codebook(codebook, root, "train")

    def test_manifest_discovery_follows_active_report(self) -> None:
        import json
        from scripts.sketchformer.evaluate import _discover_preprocessing_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            old_manifest = root / "old" / "preprocessing_manifest.jsonl"
            new_manifest = root / "new" / "preprocessing_manifest.jsonl"
            old_manifest.parent.mkdir()
            new_manifest.parent.mkdir()
            old_manifest.write_text("{}\n", encoding="utf-8")
            new_manifest.write_text("{}\n", encoding="utf-8")
            (dataset / "preparation_report.json").write_text(
                json.dumps({"build_id": "old", "manifest_path": str(old_manifest)}),
                encoding="utf-8",
            )
            active_report = dataset / "preparation_report_new.json"
            active_report.write_text(
                json.dumps({"build_id": "new", "manifest_path": str(new_manifest)}),
                encoding="utf-8",
            )
            np.savez(dataset / "meta.npz", build_id="new",
                     report_file=active_report.name, is_published=True)
            config = {"data": {"dataset": {"root": str(dataset)}}}
            self.assertEqual(
                _discover_preprocessing_manifest(config, None), new_manifest
            )

            active_report.write_text(
                json.dumps({"build_id": "old", "manifest_path": str(old_manifest)}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "differs from active dataset"):
                _discover_preprocessing_manifest(config, None)


if __name__ == "__main__":
    unittest.main()
