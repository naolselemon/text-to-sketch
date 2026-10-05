from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataloaders.anchored_v3_dataset import AnchoredV3Dataset, AnchoredV3SampleIndex
from dataloaders.datamodule import StrokeSequenceDataModule
from services.anchored_sketch_data.artifacts import EncodedSample, write_dataset_atomic
from services.anchored_sketch_data.contract import CODEBOOK_SIZE, TOKEN_LAYOUT


class AnchoredV3DatasetLoaderTest(unittest.TestCase):
    def test_overfit_source_selection_is_length_stratified(self) -> None:
        selector = object.__new__(AnchoredV3Dataset)
        selector.subset_seed = 42
        selector.max_source_samples = 8
        selector.source_subset_strategy = "length_stratified"
        lengths = [100, 200, 600, 700, 1200, 1300, 3000, 3100]
        entries = [
            AnchoredV3SampleIndex(
                file_path=Path("fixture.npz"),
                shard_index=index,
                length=length,
                sample_id=f"sample-{index}",
                source_sample_id=f"sample-{index}",
                augmented=False,
            )
            for index, length in enumerate(lengths)
        ]

        selected = selector._ordered_source_ids(entries)

        selected_lengths = [lengths[int(sample_id.split("-")[1])] for sample_id in selected]
        counts = [
            sum(
                (length <= 512 if bucket == 0 else 513 <= length <= 1024 if bucket == 1 else 1025 <= length <= 2048 if bucket == 2 else length >= 2049)
                for length in selected_lengths
            )
            for bucket in range(4)
        ]
        self.assertEqual(counts, [2, 2, 2, 2])

    def test_current_symlink_is_pinned_for_the_lifetime_of_a_dataset(self) -> None:
        codebook = np.zeros((CODEBOOK_SIZE, 2), dtype=np.float32)
        tokens = np.asarray(
            [2563, 2561, 2049, 2305, 1, 2562, 2564], dtype=np.int32
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = write_dataset_atomic(
                output,
                [
                    EncodedSample(
                        sample_id="first",
                        source_path="first.png",
                        source_sha256="a" * 64,
                        perceptual_hash=1,
                        group_id="first-group",
                        split="train",
                        point_count=2,
                        stroke_count=1,
                        tokens=tokens,
                    )
                ],
                codebook,
                preparation={"generation": 1},
            )
            dataset = AnchoredV3Dataset(output / "current", split="train")
            second = write_dataset_atomic(
                output,
                [
                    EncodedSample(
                        sample_id="second",
                        source_path="second.png",
                        source_sha256="b" * 64,
                        perceptual_hash=2,
                        group_id="second-group",
                        split="train",
                        point_count=2,
                        stroke_count=1,
                        tokens=tokens,
                    )
                ],
                codebook,
                preparation={"generation": 2},
            )

            self.assertEqual(dataset.root, first.resolve())
            self.assertEqual((output / "current").resolve(), second.resolve())
            self.assertEqual(dataset[0]["sample_id"], "first")

    def test_reads_only_manifest_listed_split_from_flat_shards(self) -> None:
        codebook = np.zeros((CODEBOOK_SIZE, 2), dtype=np.float32)
        tokens = np.asarray(
            [
                TOKEN_LAYOUT.sos_token_id,
                TOKEN_LAYOUT.stroke_start_token_id,
                TOKEN_LAYOUT.x_token_start,
                TOKEN_LAYOUT.y_token_start,
                TOKEN_LAYOUT.motion_token_start,
                TOKEN_LAYOUT.stroke_end_token_id,
                TOKEN_LAYOUT.eos_token_id,
            ],
            dtype=np.int32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = write_dataset_atomic(
                Path(directory),
                [
                    EncodedSample(
                        sample_id=f"sample-{split}",
                        source_path=f"{split}.png",
                        source_sha256=("a" if split == "train" else "b" if split == "valid" else "c") * 64,
                        perceptual_hash=index,
                        group_id=f"group-{split}",
                        split=split,
                        point_count=2,
                        stroke_count=1,
                        tokens=tokens,
                    )
                    for index, split in enumerate(("train", "valid", "test"))
                ],
                codebook,
                preparation={"fixture": True},
                shard_size=1,
            )
            dataset = AnchoredV3Dataset(root, split="valid")
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["sample_id"], "sample-valid")
            np.testing.assert_array_equal(dataset[0]["tokens"], tokens)

    def test_datamodule_accepts_null_legacy_separator_without_truncating(self) -> None:
        codebook = np.zeros((CODEBOOK_SIZE, 2), dtype=np.float32)
        tokens = np.asarray(
            [
                TOKEN_LAYOUT.sos_token_id,
                TOKEN_LAYOUT.stroke_start_token_id,
                TOKEN_LAYOUT.x_token_start,
                TOKEN_LAYOUT.y_token_start,
                TOKEN_LAYOUT.motion_token_start,
                TOKEN_LAYOUT.stroke_end_token_id,
                TOKEN_LAYOUT.eos_token_id,
            ],
            dtype=np.int32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = write_dataset_atomic(
                Path(directory),
                [
                    EncodedSample(
                        sample_id=f"sample-{split}",
                        source_path=f"{split}.png",
                        source_sha256=("a" if split == "train" else "b") * 64,
                        perceptual_hash=index,
                        group_id=f"group-{split}",
                        split=split,
                        point_count=2,
                        stroke_count=1,
                        tokens=tokens,
                    )
                    for index, split in enumerate(("train", "valid"))
                ],
                codebook,
                preparation={"fixture": True},
                shard_size=1,
            )
            module = StrokeSequenceDataModule(
                {
                    "dataset": {"root": str(root)},
                    "format": {
                        "type": "anchored_v3",
                        "token_dictionary": {
                            "pad_token_id": TOKEN_LAYOUT.pad_token_id,
                            "sep_token_id": None,
                            "stroke_end_token_id": TOKEN_LAYOUT.stroke_end_token_id,
                            "sos_token_id": TOKEN_LAYOUT.sos_token_id,
                            "eos_token_id": TOKEN_LAYOUT.eos_token_id,
                        },
                    },
                    "sequence": {
                        "max_length": 7,
                        "truncate_long_sequences": False,
                        "add_start_token": False,
                        "add_end_token": False,
                        "pad_to_multiple_of": 1,
                    },
                    "batching": {
                        "batch_size": 1,
                        "eval_batch_size": 1,
                        "num_workers": 0,
                    },
                }
            )
            module.setup("fit")

            batch = next(iter(module.val_dataloader()))

            self.assertEqual(batch["lengths"].tolist(), [7])
            self.assertEqual(batch["tokens"][0, -1].item(), TOKEN_LAYOUT.eos_token_id)

    def test_train_source_limits_are_deterministic_and_nested(self) -> None:
        codebook = np.zeros((CODEBOOK_SIZE, 2), dtype=np.float32)
        tokens = np.asarray(
            [
                TOKEN_LAYOUT.sos_token_id,
                TOKEN_LAYOUT.stroke_start_token_id,
                TOKEN_LAYOUT.x_token_start,
                TOKEN_LAYOUT.y_token_start,
                TOKEN_LAYOUT.motion_token_start,
                TOKEN_LAYOUT.stroke_end_token_id,
                TOKEN_LAYOUT.eos_token_id,
            ],
            dtype=np.int32,
        )
        samples = []
        for index, source_id in enumerate(("source-a", "source-b", "source-c")):
            source_hash = f"{index + 1:064x}"
            group_id = f"group-{source_id}"
            for augmented in (False, True):
                samples.append(
                    EncodedSample(
                        sample_id=(source_id if not augmented else f"{source_id}#aug-01"),
                        source_path=f"{source_id}.png",
                        source_sha256=source_hash,
                        perceptual_hash=index,
                        group_id=group_id,
                        split="train",
                        point_count=2,
                        stroke_count=1,
                        tokens=tokens,
                        preprocessing={
                            "augmented": augmented,
                            "source_sample_id": source_id,
                        },
                    )
                )
        with tempfile.TemporaryDirectory() as directory:
            root = write_dataset_atomic(
                Path(directory),
                samples,
                codebook,
                preparation={"fixture": True},
            )

            one = AnchoredV3Dataset(
                root,
                split="train",
                max_source_samples=1,
                subset_seed=9,
            )
            two = AnchoredV3Dataset(
                root,
                split="train",
                max_source_samples=2,
                subset_seed=9,
            )
            originals_only = AnchoredV3Dataset(
                root,
                split="train",
                max_source_samples=2,
                subset_seed=9,
                include_augmentations=False,
            )

            one_ids = {entry.source_sample_id for entry in one.index}
            two_ids = {entry.source_sample_id for entry in two.index}
            self.assertEqual(len(one), 2)
            self.assertEqual(len(two), 4)
            self.assertEqual(len(originals_only), 2)
            self.assertEqual(len(one_ids), 1)
            self.assertEqual(len(two_ids), 2)
            self.assertTrue(one_ids < two_ids)


if __name__ == "__main__":
    unittest.main()
