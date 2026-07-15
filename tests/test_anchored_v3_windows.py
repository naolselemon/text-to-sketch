from __future__ import annotations

import unittest

import numpy as np
from torch.utils.data import Dataset

from dataloaders.collate import TokenSequenceCollator
from dataloaders.datamodule import (
    CompleteStrokeWindowDataset,
    LengthStratifiedSubset,
)
from scripts.sketchformer.train import _require_fixed_length_stratified_validation


class _Dataset(Dataset):
    def __init__(self) -> None:
        self.sample = {
            "tokens": np.asarray(
                [13, 11, 3, 7, 1, 2, 12, 11, 4, 8, 2, 1, 12, 14],
                dtype=np.int64,
            ),
            "sample_id": "face",
            "length": 14,
            "label": 0,
            "source_file": "fixture",
            "source_index": 0,
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, item: int):
        return dict(self.sample)


class AnchoredV3WindowTest(unittest.TestCase):
    def test_fixed_validation_contract_requires_exact_count_and_four_buckets(self) -> None:
        class Loader:
            class Data:
                lengths = [100, 600, 1200, 3000]

            dataset = Data()

        _require_fixed_length_stratified_validation(Loader(), 4)
        Loader.dataset.lengths = [100, 200, 1200, 3000]
        with self.assertRaisesRegex(ValueError, "four length buckets"):
            _require_fixed_length_stratified_validation(Loader(), 4)

    def test_fixed_validation_subset_balances_all_length_buckets(self) -> None:
        class LengthDataset(Dataset):
            lengths = [100 + index for index in range(20)] + [
                600 + index for index in range(20)
            ] + [1200 + index for index in range(20)] + [3000 + index for index in range(20)]

            def __len__(self) -> int:
                return len(self.lengths)

            def __getitem__(self, item: int):
                return item

        subset = LengthStratifiedSubset(
            LengthDataset(),
            LengthDataset.lengths,
            16,
        )
        counts = [0, 0, 0, 0]
        for length in subset.lengths:
            counts[0 if length <= 512 else 1 if length <= 1024 else 2 if length <= 2048 else 3] += 1

        self.assertEqual(counts, [4, 4, 4, 4])
        self.assertEqual(subset.indices, LengthStratifiedSubset(
            LengthDataset(), LengthDataset.lengths, 16
        ).indices)

    def test_long_drawing_is_split_only_between_complete_strokes(self) -> None:
        windows = CompleteStrokeWindowDataset(
            _Dataset(),
            8,
            sos_token_id=13,
            eos_token_id=14,
            stroke_start_token_id=11,
            stroke_end_token_id=12,
        )
        self.assertEqual(len(windows), 2)
        for sample in (windows[0], windows[1]):
            self.assertEqual(sample["tokens"][0], 13)
            self.assertEqual(sample["tokens"][-1], 14)
            self.assertEqual(np.count_nonzero(sample["tokens"] == 11), 1)
            self.assertEqual(np.count_nonzero(sample["tokens"] == 12), 1)
            self.assertLessEqual(sample["length"], 8)

    def test_v3_collator_refuses_overlength_sequences(self) -> None:
        collator = TokenSequenceCollator(
            max_length=6,
            pad_token_id=0,
            sos_token_id=13,
            eos_token_id=14,
            add_start_token=False,
            add_end_token=False,
            truncate_long_sequences=False,
        )
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            collator([_Dataset()[0]])


if __name__ == "__main__":
    unittest.main()
