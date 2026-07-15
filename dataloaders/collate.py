"""Batch collation for variable-length stroke3 and tok-dict samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dataloaders.masks import build_sequence_masks


def _round_up(value: int, multiple: int | None) -> int:
    if not multiple or multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


@dataclass
class Stroke3Collator:
    """Pad stroke3 samples and build masks for attention/losses."""

    max_length: int | None = None
    pad_value: float = 0.0
    pad_to_multiple_of: int | None = 8
    causal_attention: bool = False
    build_attention_mask: bool = True

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")

        raw_lengths = [int(sample["length"]) for sample in samples]
        lengths = [
            min(length, self.max_length) if self.max_length is not None else length
            for length in raw_lengths
        ]
        batch_max = max(lengths)
        if self.max_length is not None:
            batch_max = min(batch_max, self.max_length)
        padded_length = _round_up(batch_max, self.pad_to_multiple_of)

        strokes = torch.full(
            (len(samples), padded_length, 3),
            fill_value=float(self.pad_value),
            dtype=torch.float32,
        )
        labels = torch.empty(len(samples), dtype=torch.long)
        source_indices = torch.empty(len(samples), dtype=torch.long)
        source_files: list[str] = []
        sample_ids: list[str] = []

        for row, sample in enumerate(samples):
            sequence = np.asarray(sample["stroke3"], dtype=np.float32)
            length = lengths[row]
            strokes[row, :length] = torch.from_numpy(sequence[:length])
            labels[row] = int(sample["label"])
            source_indices[row] = int(sample["source_index"])
            source_files.append(str(sample["source_file"]))
            sample_ids.append(str(sample.get("sample_id", sample["source_index"])))

        lengths_tensor = torch.as_tensor(lengths, dtype=torch.long)
        masks = build_sequence_masks(
            lengths_tensor,
            max_length=padded_length,
            causal=self.causal_attention,
            device=strokes.device,
        )
        if not self.build_attention_mask:
            masks["sdpa_mask"] = None

        return {
            "strokes": strokes,
            "targets": strokes.clone(),
            "lengths": lengths_tensor,
            "raw_lengths": torch.as_tensor(raw_lengths, dtype=torch.long),
            "labels": labels,
            "source_indices": source_indices,
            "source_files": source_files,
            "sample_ids": sample_ids,
            **masks,
        }


@dataclass
class TokenSequenceCollator:
    """Pad tok-dict samples and build masks for attention/losses."""

    max_length: int | None = None
    pad_token_id: int = 0
    sep_token_id: int | None = None
    sos_token_id: int | None = None
    eos_token_id: int | None = None
    add_start_token: bool = True
    add_end_token: bool = True
    truncate_long_sequences: bool = True
    pad_to_multiple_of: int | None = 8
    causal_attention: bool = False
    build_attention_mask: bool = True

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")

        prepared = [self._prepare_sequence(sample["tokens"]) for sample in samples]
        raw_lengths = [raw_length for _, raw_length in prepared]
        lengths = [len(sequence) for sequence, _ in prepared]
        batch_max = max(lengths)
        if self.max_length is not None:
            batch_max = min(batch_max, self.max_length)
        padded_length = _round_up(batch_max, self.pad_to_multiple_of)

        tokens = torch.full(
            (len(samples), padded_length),
            fill_value=int(self.pad_token_id),
            dtype=torch.long,
        )
        labels = torch.empty(len(samples), dtype=torch.long)
        source_indices = torch.empty(len(samples), dtype=torch.long)
        source_files: list[str] = []
        sample_ids: list[str] = []

        for row, sample in enumerate(samples):
            sequence = prepared[row][0]
            length = lengths[row]
            tokens[row, :length] = torch.from_numpy(sequence).long()
            labels[row] = int(sample["label"])
            source_indices[row] = int(sample["source_index"])
            source_files.append(str(sample["source_file"]))
            sample_ids.append(str(sample.get("sample_id", sample["source_index"])))

        lengths_tensor = torch.as_tensor(lengths, dtype=torch.long)
        masks = build_sequence_masks(
            lengths_tensor,
            max_length=padded_length,
            causal=self.causal_attention,
            device=tokens.device,
        )
        if not self.build_attention_mask:
            masks["sdpa_mask"] = None

        return {
            "tokens": tokens,
            "targets": tokens.clone(),
            "lengths": lengths_tensor,
            "raw_lengths": torch.as_tensor(raw_lengths, dtype=torch.long),
            "labels": labels,
            "source_indices": source_indices,
            "source_files": source_files,
            "sample_ids": sample_ids,
            "pad_token_id": int(self.pad_token_id),
            **masks,
        }

    def _prepare_sequence(self, tokens: np.ndarray) -> tuple[np.ndarray, int]:
        sequence = np.asarray(tokens, dtype=np.int64)
        if sequence.ndim != 1:
            raise ValueError(f"Expected token array with shape (N,), got {sequence.shape}")
        if len(sequence) == 0:
            raise ValueError("Token sequence is empty")

        if (
            self.add_start_token
            and self.sos_token_id is not None
            and int(sequence[0]) != int(self.sos_token_id)
        ):
            sequence = np.concatenate(
                [np.asarray([self.sos_token_id], dtype=np.int64), sequence]
            )
        if (
            self.add_end_token
            and self.eos_token_id is not None
            and int(sequence[-1]) != int(self.eos_token_id)
        ):
            sequence = np.concatenate(
                [sequence, np.asarray([self.eos_token_id], dtype=np.int64)]
            )

        raw_length = int(len(sequence))
        if self.max_length is not None and raw_length > self.max_length:
            if not self.truncate_long_sequences:
                raise ValueError(
                    f"Sequence length {raw_length} exceeds max_length={self.max_length}"
                )
            sequence = np.array(sequence[: self.max_length], copy=True, dtype=np.int64)
            if self.add_end_token and self.eos_token_id is not None:
                if len(sequence) > 1 and self.sep_token_id is not None:
                    sequence[-2:] = [int(self.sep_token_id), int(self.eos_token_id)]
                else:
                    sequence[-1] = int(self.eos_token_id)
        return sequence.astype(np.int64, copy=False), raw_length
