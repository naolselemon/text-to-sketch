from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataloaders import (
    Stroke3Collator,
    StrokeSequenceDataModule,
    TokenSequenceCollator,
    build_sequence_masks,
    causal_mask,
    lengths_to_valid_mask,
    make_sdpa_self_attention_mask,
    valid_to_padding_mask,
)


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


class CollateAndMaskTest(unittest.TestCase):
    def test_length_masks_use_true_for_valid_tokens(self) -> None:
        valid_mask = lengths_to_valid_mask([2, 4], max_length=5)

        self.assertEqual(valid_mask.tolist(), [
            [True, True, False, False, False],
            [True, True, True, True, False],
        ])
        self.assertEqual(valid_to_padding_mask(valid_mask).tolist(), [
            [False, False, True, True, True],
            [False, False, False, False, True],
        ])

    def test_sdpa_mask_shape_and_causal_behavior(self) -> None:
        valid_mask = torch.tensor([[True, True, True, False]])
        sdpa = make_sdpa_self_attention_mask(valid_mask, causal=True)

        self.assertEqual(sdpa.shape, (1, 1, 1, 4))
        self.assertEqual(sdpa[0, 0, 0].tolist(), [True, True, True, False])
        self.assertEqual(causal_mask(3).tolist(), [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ])

    def test_sdpa_mask_has_pin_memory_safe_storage(self) -> None:
        valid_mask = torch.tensor([
            [True, True, False, False],
            [True, True, True, False],
        ])

        for causal in (False, True):
            sdpa = make_sdpa_self_attention_mask(valid_mask, causal=causal)

            self.assertTrue(sdpa.is_contiguous())
            if hasattr(torch, "_debug_has_internal_overlap"):
                self.assertEqual(torch._debug_has_internal_overlap(sdpa), 0)

    def test_collator_pads_and_builds_standard_masks(self) -> None:
        samples = [
            {
                "stroke3": _stroke(3),
                "label": 0,
                "length": 3,
                "source_file": "a.npz",
                "source_index": 0,
            },
            {
                "stroke3": _stroke(5),
                "label": 0,
                "length": 5,
                "source_file": "b.npz",
                "source_index": 1,
            },
        ]

        batch = Stroke3Collator(max_length=8, pad_to_multiple_of=4)(samples)

        self.assertEqual(batch["strokes"].shape, (2, 8, 3))
        self.assertEqual(batch["targets"].shape, (2, 8, 3))
        self.assertEqual(batch["valid_mask"].sum().item(), 8)
        self.assertEqual(batch["sdpa_mask"].shape, (2, 1, 1, 8))
        self.assertEqual(batch["lengths"].tolist(), [3, 5])

    def test_collator_can_skip_full_attention_mask_for_long_sequences(self) -> None:
        samples = [
            {
                "stroke3": _stroke(3),
                "label": 0,
                "length": 3,
                "source_file": "a.npz",
                "source_index": 0,
            },
        ]

        batch = Stroke3Collator(
            max_length=8,
            pad_to_multiple_of=4,
            build_attention_mask=False,
        )(samples)

        self.assertIsNone(batch["sdpa_mask"])
        self.assertEqual(batch["valid_mask"].shape, (1, 4))

    def test_token_collator_pads_with_configured_pad_token(self) -> None:
        samples = [
            {
                "tokens": np.asarray([6, 1, 7], dtype=np.int64),
                "label": 0,
                "length": 3,
                "source_file": "a.npz",
                "source_index": 0,
            },
            {
                "tokens": np.asarray([6, 2, 5, 7], dtype=np.int64),
                "label": 0,
                "length": 4,
                "source_file": "b.npz",
                "source_index": 1,
            },
        ]

        batch = TokenSequenceCollator(
            max_length=8,
            pad_token_id=0,
            pad_to_multiple_of=4,
        )(samples)

        self.assertEqual(batch["tokens"].shape, (2, 4))
        self.assertEqual(batch["tokens"][0].tolist(), [6, 1, 7, 0])
        self.assertEqual(batch["targets"][1].tolist(), [6, 2, 5, 7])
        self.assertEqual(batch["valid_mask"].sum().item(), 7)
        self.assertEqual(batch["pad_token_id"], 0)

    def test_token_collator_adds_sos_and_forces_truncated_tail(self) -> None:
        samples = [
            {
                "tokens": np.asarray([1, 2, 3, 4, 1003], dtype=np.int64),
                "label": 0,
                "length": 5,
                "source_file": "a.npz",
                "source_index": 0,
            }
        ]

        batch = TokenSequenceCollator(
            max_length=5,
            pad_token_id=0,
            sep_token_id=1001,
            sos_token_id=1002,
            eos_token_id=1003,
            pad_to_multiple_of=None,
        )(samples)

        self.assertEqual(batch["tokens"][0].tolist(), [1002, 1, 2, 1001, 1003])
        self.assertEqual(batch["lengths"].tolist(), [5])
        self.assertEqual(batch["raw_lengths"].tolist(), [6])

    def test_datamodule_returns_train_and_validation_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_chunk(root / "train_000.npz", [3, 5])
            _write_chunk(root / "valid.npz", [4])
            _write_chunk(root / "test.npz", [2])

            config = {
                "dataset": {"root": str(root)},
                "sequence": {
                    "max_length": 8,
                    "truncate_long_sequences": True,
                    "pad_value": 0.0,
                    "pad_to_multiple_of": 4,
                },
                "preprocessing": {
                    "normalize_by_bounds": False,
                    "delta_clip": 1000.0,
                    "max_cached_files": 1,
                },
                "augmentation": {"enabled": False},
                "batching": {
                    "batch_size": 2,
                    "eval_batch_size": 1,
                    "num_workers": 0,
                    "persistent_workers": False,
                    "drop_last": False,
                    "bucket_by_length": False,
                    "pin_memory": False,
                },
            }

            datamodule = StrokeSequenceDataModule(config)
            datamodule.setup("fit")
            train_batch = next(iter(datamodule.train_dataloader()))
            val_batch = next(iter(datamodule.val_dataloader()))

            self.assertEqual(train_batch["strokes"].shape[0], 2)
            self.assertEqual(val_batch["strokes"].shape[0], 1)
            self.assertIn("valid_mask", build_sequence_masks([1], max_length=2))

    def test_datamodule_returns_tok_dict_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_token_chunk(root / "train_000.npz", [[6, 1, 7], [6, 2, 5, 7]])
            _write_token_chunk(root / "valid.npz", [[6, 3, 7]])
            _write_token_chunk(root / "test.npz", [[6, 1, 7]])

            config = {
                "dataset": {"root": str(root), "metadata_file": "meta.npz"},
                "format": {
                    "type": "tok_dict",
                    "token_dictionary": {
                        "sep_token_id": 5,
                        "sos_token_id": 6,
                        "eos_token_id": 7,
                        "pad_token_id": 0,
                    },
                },
                "sequence": {
                    "max_length": 8,
                    "truncate_long_sequences": True,
                    "add_end_token": True,
                    "pad_to_multiple_of": 4,
                },
                "preprocessing": {"max_cached_files": 1},
                "augmentation": {"enabled": False},
                "batching": {
                    "batch_size": 2,
                    "eval_batch_size": 1,
                    "num_workers": 0,
                    "persistent_workers": False,
                    "drop_last": False,
                    "bucket_by_length": False,
                    "pin_memory": False,
                },
            }

            datamodule = StrokeSequenceDataModule(config)
            datamodule.setup("fit")
            train_batch = next(iter(datamodule.train_dataloader()))
            val_batch = next(iter(datamodule.val_dataloader()))

            self.assertEqual(train_batch["tokens"].shape[0], 2)
            self.assertEqual(val_batch["tokens"].shape[0], 1)
            self.assertEqual(train_batch["pad_token_id"], 0)


if __name__ == "__main__":
    unittest.main()
