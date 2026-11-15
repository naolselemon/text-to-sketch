"""Fixture-only CPU checks for auditing saved benchmark evidence (not model scores)."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from evals.long_sequence_eval import LENGTH_DEFINITION, main


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data)) + kind + data
        + struct.pack(">I", zlib.crc32(kind + data))
    )


PNG = (
    b"\x89PNG\r\n\x1a\n"
    + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    + png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
    + png_chunk(b"IEND", b"")
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LongSequenceEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "weights.safetensors"
        self.codebook = self.root / "codebook.npy"
        self.codebook.write_bytes(b"fixture codebook only")
        size = 4
        layout = {
            "codebook_size": size,
            "motion_token_offset": 1,
            "pad_token_id": 0,
            "sep_token_id": size + 1,
            "sos_token_id": size + 2,
            "eos_token_id": size + 3,
            "vocab_size": size + 4,
        }
        metadata = {
            "source_tf_checkpoint_sha256": sha(b"original TF fixture"),
            "original_token_dict_sha256": sha(b"original dictionary fixture"),
            "codebook_sha256": sha(self.codebook.read_bytes()),
            "token_layout_json": json.dumps(layout, sort_keys=True, separators=(",", ":")),
        }
        header = json.dumps({
            "__metadata__": metadata,
            "tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }).encode()
        self.checkpoint_bytes = struct.pack("<Q", len(header)) + header + bytes(4)
        self.checkpoint.write_bytes(self.checkpoint_bytes)
        self.data = self.root / "data"
        self.data.mkdir()
        dataset = b"fixture dataset only"
        (self.data / "valid.npz").write_bytes(dataset)
        files = {"valid.npz": sha(dataset)}
        self.provenance = {
            "length_definition": LENGTH_DEFINITION,
            "split": "valid",
            "random_initialization": False,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha(self.checkpoint.read_bytes()),
            "checkpoint_load": {"strict": True, "missing_keys": [], "unexpected_keys": []},
            "codebook_sha256": sha(self.codebook.read_bytes()),
            "data_root": str(self.data),
            "dataset_files": files,
            "dataset_sha256": sha(json.dumps(files, sort_keys=True).encode()),
            "seed": 7,
        }
        self.report_path = self.root / "heldout_report.json"
        self.output = self.root / "length_bins.json"
        self.teacher = []
        self.running = []
        self.selection = []
        self.plot_dir = self.root / "heldout_plots"
        self.plot_dir.mkdir()

    def add_sample(self, length: int, *, eos: int = 1, hit: int = 0) -> None:
        sample_id = f"sample-{length}"
        shared = {
            "sample_id": sample_id,
            "source_file": str(self.data / "valid.npz"),
            "source_index": len(self.teacher),
            "target_length": length,
            "length_bucket": (
                "short" if length <= 512 else "medium" if length <= 1024
                else "long" if length <= 2048 else "extended"
            ),
            "target_sha256": sha(sample_id.encode()),
            "stroke_count": 1,
        }
        self.teacher.append({**shared, "tf_scored_tokens": length, "tf_token_accuracy": 0.6})
        self.running.append({
            **shared,
            "eos_rate": float(eos),
            "max_length_hit": float(hit),
            "exact_match": 0.0,
            "token_accuracy": 0.5,
            "geometry_f1_2px": 0.4,
        })
        self.selection.append({
            "sample_id": sample_id,
            "target_length": length,
            "length_bucket": self.running[-1]["length_bucket"],
        })
        (self.plot_dir / f"reconstruction_{len(self.selection):03d}.png").write_bytes(PNG)

    def save(self) -> None:
        (self.root / "heldout_teacher_forced_per_sample.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.teacher)
        )
        (self.root / "heldout_per_sample.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.running)
        )
        (self.root / "heldout_comparison_manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "seed": 7,
            "length_definition": LENGTH_DEFINITION,
            "dataset_sha256": self.provenance["dataset_sha256"],
            "samples": self.selection,
        }))
        self.report_path.write_text(json.dumps({
            "schema_version": 2,
            "provenance": self.provenance,
            "teacher_forced": {
                "mode": "teacher-forced",
                "total_samples": len(self.teacher),
                "per_sample_path": "heldout_teacher_forced_per_sample.jsonl",
            },
            "free_running": {
                "mode": "free-running",
                "total_samples": len(self.running),
                "per_sample_path": "heldout_per_sample.jsonl",
                "comparison_manifest_path": "heldout_comparison_manifest.json",
            },
        }))

    def run_cli(self) -> int:
        return main([
            "--benchmark-report", str(self.report_path),
            "--codebook-path", str(self.codebook),
            "--output", str(self.output),
        ])

    def test_all_bin_boundaries_completion_geometry_and_saved_plot(self) -> None:
        for length in (1, 200, 201, 512, 513, 1024, 1025, 2048, 2049, 4096):
            self.add_sample(length, eos=int(length % 2 == 0), hit=int(length % 2 != 0))
        self.save()
        self.assertEqual(self.run_cli(), 0)
        report = json.loads(self.output.read_text())
        self.assertEqual(report["status"], "review_required")
        self.assertEqual(report["total_samples"], 10)
        self.assertEqual([item["sample_count"] for item in report["bins"].values()], [2] * 5)
        for length in (200, 512, 1024, 2048, 4096):
            label = next(key for key, value in report["bins"].items() if value["range"][1] == length)
            bucket = report["bins"][label]
            self.assertEqual(bucket["eos_count"], 1)
            self.assertEqual(bucket["max_length_hit_count"], 1)
            self.assertEqual(bucket["eos_rate"], 0.5)
            self.assertEqual(bucket["geometry_f1_2px_mean"], 0.4)
            self.assertEqual(bucket["qualitative_review"], "pending")
            representative = bucket["representative"]
            self.assertTrue(Path(representative["plot"]).is_file())
            self.assertIn(representative, bucket["comparison_plots"])
            self.assertIn(representative["sample_id"], {r["sample_id"] for r in self.running})
        self.assertEqual(self.run_cli(), 2)  # Never overwrite existing analysis.

    def test_prepares_five_bin_comparison_manifest_from_teacher_only_report(self) -> None:
        for length in (200, 201, 512, 513, 1025, 2049):
            self.add_sample(length)
        self.save()
        teacher_only = json.loads(self.report_path.read_text())
        del teacher_only["free_running"]
        del teacher_only["provenance"]["codebook_sha256"]  # TF-only benchmark does not load codebook.
        self.report_path.write_text(json.dumps(teacher_only))
        manifest_path = self.root / "five_bins_manifest.json"
        self.assertEqual(main([
            "--benchmark-report", str(self.report_path),
            "--codebook-path", str(self.codebook),
            "--output", str(manifest_path),
            "--prepare-comparison-manifest",
        ]), 0)
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["seed"], 7)
        self.assertEqual(manifest["length_definition"], LENGTH_DEFINITION)
        self.assertEqual(manifest["dataset_sha256"], self.provenance["dataset_sha256"])
        self.assertEqual(
            [row["target_length"] <= 200 for row in manifest["samples"][:2]],
            [True, False],
        )
        self.assertEqual(len(manifest["samples"]), 5)
        self.assertEqual(
            {row["sample_id"] for row in manifest["samples"]} & {"sample-200", "sample-201", "sample-512"},
            {"sample-200", manifest["samples"][1]["sample_id"]},
        )
        another = self.root / "five_bins_again.json"
        self.assertEqual(main([
            "--benchmark-report", str(self.report_path),
            "--codebook-path", str(self.codebook),
            "--output", str(another),
            "--prepare-comparison-manifest",
        ]), 0)
        self.assertEqual(manifest_path.read_bytes(), another.read_bytes())

    def test_checkpoint_label_must_bind_codebook_not_just_benchmark_hash(self) -> None:
        self.add_sample(200)
        self.save()
        replacement = self.checkpoint_bytes.replace(
            sha(self.codebook.read_bytes()).encode(), sha(b"other dictionary").encode()
        )
        self.assertNotEqual(replacement, self.checkpoint_bytes)
        self.checkpoint.write_bytes(replacement)
        self.provenance["checkpoint_sha256"] = sha(replacement)
        self.save()
        self.assertEqual(self.run_cli(), 2)
        self.assertFalse(self.output.exists())

    def test_empty_bins_write_insufficient_report_and_exit_nonzero(self) -> None:
        self.add_sample(200)
        self.save()
        self.assertEqual(self.run_cli(), 1)
        report = json.loads(self.output.read_text())
        self.assertEqual(report["bins"]["1-200"]["sample_count"], 1)
        for label in ("201-512", "513-1024", "1025-2048", "2049-4096"):
            self.assertEqual(report["bins"][label]["status"], "insufficient")
            self.assertIsNone(report["bins"][label]["eos_rate"])
            self.assertIsNone(report["bins"][label]["representative"])

    def test_bad_hashes_and_missing_artifacts_fail_without_report(self) -> None:
        self.add_sample(200)
        self.save()
        self.codebook.write_bytes(b"wrong dictionary")
        self.assertEqual(self.run_cli(), 2)
        self.assertFalse(self.output.exists())
        self.codebook.write_bytes(b"fixture codebook only")
        self.checkpoint.write_bytes(b"wrong checkpoint")
        self.assertEqual(self.run_cli(), 2)
        self.checkpoint.write_bytes(self.checkpoint_bytes)
        (self.data / "valid.npz").write_bytes(b"changed dataset")
        self.assertEqual(self.run_cli(), 2)
        (self.data / "valid.npz").write_bytes(b"fixture dataset only")
        (self.plot_dir / "reconstruction_001.png").unlink()
        self.assertEqual(self.run_cli(), 2)
        self.assertFalse(self.output.exists())

    def test_rejects_incomplete_joins_duplicates_and_missing_bin_plot(self) -> None:
        for length in (200, 201, 513, 1025, 2049):
            self.add_sample(length)
        self.save()
        self.running[0]["target_sha256"] = sha(b"different")
        self.save()
        self.assertEqual(self.run_cli(), 2)
        self.running[0]["target_sha256"] = self.teacher[0]["target_sha256"]
        self.teacher.append(self.teacher[0].copy())
        self.running.append(self.running[0].copy())
        self.save()
        self.assertEqual(self.run_cli(), 2)
        self.running.pop()
        self.teacher.pop()
        self.selection = self.selection[:-1]
        self.save()
        self.assertEqual(self.run_cli(), 2)
        self.assertFalse(self.output.exists())

    def test_rejects_unsupported_mode_random_init_and_bad_completion(self) -> None:
        self.add_sample(200)
        self.save()
        self.running[0]["eos_rate"] = 1.0
        self.running[0]["max_length_hit"] = 1.0
        self.save()
        self.assertEqual(self.run_cli(), 2)
        self.running[0]["max_length_hit"] = 0.0
        self.provenance["random_initialization"] = True
        self.save()
        self.assertEqual(self.run_cli(), 2)
        self.provenance["random_initialization"] = False
        report = json.loads(self.report_path.read_text())
        del report["teacher_forced"]
        self.report_path.write_text(json.dumps(report))
        self.assertEqual(self.run_cli(), 2)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
