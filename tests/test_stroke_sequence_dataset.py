from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataloaders import StrokeSequenceDataset


def _stroke(length: int) -> np.ndarray:
    rows = []
    for index in range(length):
        pen_state = 1.0 if index == length - 1 else 0.0
        rows.append([0.1 * index, 0.05 * index, pen_state])
    return np.asarray(rows, dtype=np.float32)


def _write_chunk(path: Path, lengths: list[int]) -> None:
    sketches = np.asarray([_stroke(length) for length in lengths], dtype=object)
    labels = np.zeros(len(lengths), dtype=np.int32)
    np.savez_compressed(path, x=sketches, y=labels)


def _write_token_chunk(path: Path, sequences: list[list[int]]) -> None:
    tokens = np.asarray([np.asarray(seq, dtype=np.int64) for seq in sequences], dtype=object)
    labels = np.zeros(len(sequences), dtype=np.int32)
    np.savez_compressed(path, x=tokens, y=labels)


class StrokeSequenceDatasetTest(unittest.TestCase):
    def test_loads_sketchformer_ready_train_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_chunk(root / "train_000.npz", [3, 5])
            _write_chunk(root / "valid.npz", [4])
            _write_chunk(root / "test.npz", [2])
            np.savez(root / "meta.npz", n_classes=1)

            dataset = StrokeSequenceDataset(root, split="train")

            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(sample["stroke3"].shape, (3, 3))
            self.assertEqual(sample["label"], 0)
            self.assertEqual(sample["length"], 3)
            self.assertEqual(dataset.metadata["n_classes"], 1)

    def test_loads_tok_dict_train_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_token_chunk(root / "train_000.npz", [[1002, 1, 1003], [1002, 2, 1001, 1003]])

            dataset = StrokeSequenceDataset(
                root,
                split="train",
                format_type="tok_dict",
            )

            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(sample["tokens"].tolist(), [1002, 1, 1003])
            self.assertNotIn("stroke3", sample)
            self.assertEqual(sample["length"], 3)

    def test_raises_for_missing_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                StrokeSequenceDataset(Path(tmp), split="train")


if __name__ == "__main__":
    unittest.main()
