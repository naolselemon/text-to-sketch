from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from services.anchored_sketch_data.artifacts import (
    EncodedSample,
    RejectedSample,
    require_minimum_source_sketches,
    validate_dataset,
    write_dataset_atomic,
)
from services.anchored_sketch_data.tokenizer import AnchoredTokenizer


def artifact_test_codebook() -> np.ndarray:
    centers = np.column_stack(
        (
            np.arange(2048, dtype=np.float32) + 1000.0,
            np.arange(2048, dtype=np.float32) + 2000.0,
        )
    )
    centers[0] = [1.0, 0.0]
    return centers


def encoded_sample(
    sample_id: str,
    split: str,
    *,
    group_id: str | None = None,
    source_sha256: str | None = None,
) -> EncodedSample:
    tokens = AnchoredTokenizer(artifact_test_codebook()).encode(
        [[(8.0, 8.0), (9.0, 8.0), (10.0, 8.0)]]
    )
    return EncodedSample(
        sample_id=sample_id,
        source_path=f"source/{sample_id}.png",
        source_sha256=source_sha256 or hashlib.sha256(sample_id.encode()).hexdigest(),
        perceptual_hash=int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big"),
        group_id=group_id or hashlib.sha256(f"group:{sample_id}".encode()).hexdigest(),
        split=split,
        point_count=3,
        stroke_count=1,
        tokens=tokens,
        preprocessing={"rdp_epsilon": 0.5},
    )


class AnchoredV3ArtifactTest(unittest.TestCase):
    def test_production_source_floor_cannot_be_bypassed_by_builder_override(self) -> None:
        metadata = {"preparation": {"accepted_source_sketches": 24_999}}
        with self.assertRaisesRegex(ValueError, "at least 25000"):
            require_minimum_source_sketches(metadata)
        with self.assertRaises(TypeError):
            require_minimum_source_sketches(metadata, minimum=1)
        metadata["preparation"]["accepted_source_sketches"] = 25_000
        self.assertEqual(require_minimum_source_sketches(metadata), 25_000)

    def test_atomic_dataset_round_trip_records_contract_and_rejections(self) -> None:
        samples = [
            encoded_sample("train", "train"),
            encoded_sample("valid", "valid"),
            encoded_sample("test", "test"),
        ]
        rejected = RejectedSample(
            sample_id="bad",
            source_path="source/bad.png",
            source_sha256="f" * 64,
            perceptual_hash=None,
            rejection_reason="empty",
        )
        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_dataset_atomic(
                tmp,
                samples,
                artifact_test_codebook(),
                preparation={"test": True},
                rejected=[rejected],
                shard_size=1,
            )

            metadata = validate_dataset(dataset)
            manifest = [
                json.loads(line)
                for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertTrue(dataset.name.startswith("anchored_v3-"))
            self.assertEqual(metadata["format_version"], 3)
            self.assertEqual(metadata["token_layout_version"], 3)
            self.assertEqual(metadata["split_counts"], {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(metadata["rejected_count"], 1)
            self.assertEqual({entry["status"] for entry in manifest}, {"accepted", "rejected"})
            self.assertEqual(len(list((dataset / "shards").glob("*.npz"))), 3)
            current = Path(tmp) / "current"
            self.assertTrue(current.is_symlink())
            self.assertEqual(current.readlink(), Path(dataset.name))
            self.assertEqual(current.resolve(), dataset.resolve())

    def test_writer_refuses_split_overlap_and_existing_nonempty_version(self) -> None:
        shared_group = "a" * 64
        split_leak = [
            encoded_sample("one", "train", group_id=shared_group),
            encoded_sample("two", "valid", group_id=shared_group),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "group_id"):
                write_dataset_atomic(
                    tmp,
                    split_leak,
                    artifact_test_codebook(),
                    preparation={"test": True},
                )

            good = [encoded_sample("one", "train")]
            write_dataset_atomic(
                tmp,
                good,
                artifact_test_codebook(),
                preparation={"test": True},
            )
            with self.assertRaisesRegex(FileExistsError, "non-empty target"):
                write_dataset_atomic(
                    tmp,
                    good,
                    artifact_test_codebook(),
                    preparation={"test": True},
                )

    def test_writer_refuses_stale_temporary_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / ".anchored_v3-interrupted.tmp"
            stale.mkdir()
            (stale / "train-00000.npz").write_bytes(b"partial")

            with self.assertRaisesRegex(FileExistsError, "stale temporary target"):
                write_dataset_atomic(
                    tmp,
                    [encoded_sample("one", "train")],
                    artifact_test_codebook(),
                    preparation={"test": True},
                )

    def test_current_symlink_rotates_and_conflicting_real_directory_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_dataset_atomic(
                root,
                [encoded_sample("one", "train")],
                artifact_test_codebook(),
                preparation={"version": 1},
            )
            second = write_dataset_atomic(
                root,
                [encoded_sample("two", "train")],
                artifact_test_codebook(),
                preparation={"version": 2},
            )

            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())
            self.assertEqual((root / "current").readlink(), Path(second.name))
            self.assertFalse(any(root.glob(".current-*.tmp")))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "current").mkdir()

            with self.assertRaisesRegex(FileExistsError, "conflicting real path"):
                write_dataset_atomic(
                    root,
                    [encoded_sample("one", "train")],
                    artifact_test_codebook(),
                    preparation={"version": 1},
                )

            self.assertEqual(list(root.glob("anchored_v3-*")), [])

    def test_validation_detects_manifest_codebook_and_stale_shard_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_dataset_atomic(
                tmp,
                [encoded_sample("one", "train")],
                artifact_test_codebook(),
                preparation={"test": True},
            )
            (dataset / "shards" / "stale.npz").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "exactly match"):
                validate_dataset(dataset)

        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_dataset_atomic(
                tmp,
                [encoded_sample("one", "train")],
                artifact_test_codebook(),
                preparation={"test": True},
            )
            with (dataset / "manifest.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "Manifest SHA-256 mismatch"):
                validate_dataset(dataset)

        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_dataset_atomic(
                tmp,
                [encoded_sample("one", "train")],
                artifact_test_codebook(),
                preparation={"test": True},
            )
            codebook = np.load(dataset / "codebook.npy", allow_pickle=False)
            codebook[0, 0] += 1.0
            np.save(dataset / "codebook.npy", codebook, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "Codebook SHA-256 mismatch"):
                validate_dataset(dataset)

    def test_validation_detects_metadata_payload_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_dataset_atomic(
                tmp,
                [encoded_sample("one", "train")],
                artifact_test_codebook(),
                preparation={"test": True},
            )
            metadata_path = dataset / "meta.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["preparation"] = {"test": False}
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "Metadata payload SHA-256 mismatch"):
                validate_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
