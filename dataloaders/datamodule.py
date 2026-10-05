"""DataModule-style loaders for in-repo Sketchformer fine-tuning."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

from dataloaders.collate import Stroke3Collator, TokenSequenceCollator
from dataloaders.anchored_v3_dataset import AnchoredV3Dataset
from dataloaders.stroke_sequence_dataset import StrokeSequenceDataset
from dataloaders.transforms import Stroke3Transform, TokenSequenceTransform


def _to_plain_config(config: Any) -> Any:
    if isinstance(config, Mapping):
        return {key: _to_plain_config(value) for key, value in config.items()}
    if isinstance(config, list):
        return [_to_plain_config(value) for value in config]
    return config


def _get(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group sequence indices with similar lengths into batches."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        boundaries: Sequence[int],
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.boundaries = sorted(int(boundary) for boundary in boundaries)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        buckets = self._build_buckets()
        batches: list[list[int]] = []

        for bucket in buckets:
            if self.shuffle:
                rng.shuffle(bucket)
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        total = 0
        for bucket in self._build_buckets():
            if self.drop_last:
                total += len(bucket) // self.batch_size
            else:
                total += math.ceil(len(bucket) / self.batch_size)
        return total

    def _build_buckets(self) -> list[list[int]]:
        buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for index, length in enumerate(self.lengths):
            bucket_index = 0
            while bucket_index < len(self.boundaries) and length > self.boundaries[bucket_index]:
                bucket_index += 1
            buckets[bucket_index].append(index)
        return [bucket for bucket in buckets if bucket]


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Build variable-size batches bounded by padded sequence token count."""

    def __init__(
        self,
        lengths: Sequence[int],
        max_tokens: int,
        *,
        max_batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        pad_to_multiple_of: int = 1,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if any(length <= 0 for length in lengths):
            raise ValueError("sequence lengths must be positive")
        if pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        self.pad_to_multiple_of = int(pad_to_multiple_of)
        padded_lengths = [self._padded_length(int(length)) for length in lengths]
        if any(length > max_tokens for length in padded_lengths):
            longest = max(padded_lengths)
            raise ValueError(
                f"padded sequence length {longest} exceeds "
                f"max_tokens_per_batch={max_tokens}"
            )
        self.lengths = [int(length) for length in lengths]
        self.max_tokens = int(max_tokens)
        self.max_batch_size = int(max_batch_size) if max_batch_size else None
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def _padded_length(self, length: int) -> int:
        multiple = self.pad_to_multiple_of
        return ((int(length) + multiple - 1) // multiple) * multiple

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._build_batches(self.seed + self.epoch if self.shuffle else None)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._build_batches(None))

    def _build_batches(self, shuffle_seed: int | None) -> list[list[int]]:
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda index: self.lengths[index])

        batches: list[list[int]] = []
        batch: list[int] = []
        batch_max = 0
        for index in indices:
            length = self._padded_length(self.lengths[index])
            next_max = max(batch_max, length)
            exceeds_tokens = next_max * (len(batch) + 1) > self.max_tokens
            exceeds_size = (
                self.max_batch_size is not None
                and len(batch) + 1 > self.max_batch_size
            )
            if batch and (exceeds_tokens or exceeds_size):
                batches.append(batch)
                batch = []
                batch_max = 0
            batch.append(index)
            batch_max = max(batch_max, length)
        if batch and (not self.drop_last or self.max_batch_size is None or len(batch) == self.max_batch_size):
            batches.append(batch)
        if shuffle_seed is not None:
            rng = random.Random(shuffle_seed)
            for values in batches:
                rng.shuffle(values)
            rng.shuffle(batches)
        return batches


class SequenceLengthSubset(Dataset):
    """Read-only dataset view containing complete sequences up to a stage limit."""

    def __init__(self, dataset: Dataset, lengths: Sequence[int], max_length: int) -> None:
        self.dataset = dataset
        self.indices = [index for index, length in enumerate(lengths) if length <= max_length]
        self._lengths = [int(lengths[index]) for index in self.indices]
        if not self.indices:
            raise ValueError(f"No complete sequences fit curriculum max_length={max_length}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        return self.dataset[self.indices[item]]

    @property
    def lengths(self) -> list[int]:
        return list(self._lengths)


class LengthStratifiedSubset(Dataset):
    """Fixed deterministic subset balanced across V3 length buckets."""

    def __init__(
        self,
        dataset: Dataset,
        lengths: Sequence[int],
        maximum_samples: int,
        *,
        boundaries: Sequence[int] = (512, 1024, 2048, 4096),
    ) -> None:
        if maximum_samples <= 0:
            raise ValueError("maximum_samples must be positive")
        if len(lengths) != len(dataset):
            raise ValueError("lengths must align with the dataset")
        self.dataset = dataset
        self.indices = self._select_indices(lengths, maximum_samples, boundaries)
        self._lengths = [int(lengths[index]) for index in self.indices]
        if not self.indices:
            raise ValueError("length-stratified validation subset is empty")

    @staticmethod
    def _select_indices(
        lengths: Sequence[int],
        maximum_samples: int,
        boundaries: Sequence[int],
    ) -> list[int]:
        ordered_boundaries = sorted(int(value) for value in boundaries)
        buckets: list[list[int]] = [[] for _ in ordered_boundaries]
        for index, raw_length in enumerate(lengths):
            length = int(raw_length)
            for bucket_index, upper in enumerate(ordered_boundaries):
                if length <= upper:
                    buckets[bucket_index].append(index)
                    break
        nonempty = [bucket for bucket in buckets if bucket]
        if not nonempty:
            return []
        limit = min(int(maximum_samples), sum(len(bucket) for bucket in nonempty))
        base, remainder = divmod(limit, len(nonempty))
        selected: list[int] = []
        for bucket_index, bucket in enumerate(nonempty):
            quota = min(len(bucket), base + (bucket_index < remainder))
            selected.extend(LengthStratifiedSubset._evenly_spaced(bucket, quota))
        if len(selected) < limit:
            chosen = set(selected)
            remaining = [
                index
                for bucket in nonempty
                for index in bucket
                if index not in chosen
            ]
            selected.extend(
                LengthStratifiedSubset._evenly_spaced(
                    remaining,
                    limit - len(selected),
                )
            )
        return sorted(selected)

    @staticmethod
    def _evenly_spaced(indices: Sequence[int], count: int) -> list[int]:
        if count <= 0:
            return []
        if count >= len(indices):
            return list(indices)
        if count == 1:
            return [indices[len(indices) // 2]]
        return [
            indices[round(position * (len(indices) - 1) / (count - 1))]
            for position in range(count)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        return self.dataset[self.indices[item]]

    @property
    def lengths(self) -> list[int]:
        return list(self._lengths)


class PrefixSubset(Dataset):
    """First N examples of a deterministic evaluation dataset."""

    def __init__(self, dataset: Dataset, lengths: Sequence[int], maximum_samples: int) -> None:
        if maximum_samples <= 0:
            raise ValueError("maximum_samples must be positive")
        self.dataset = dataset
        count = min(len(dataset), int(maximum_samples))
        self.indices = list(range(count))
        self._lengths = [int(lengths[index]) for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        return self.dataset[self.indices[item]]

    @property
    def lengths(self) -> list[int]:
        return list(self._lengths)


class CompleteStrokeWindowDataset(Dataset):
    """Expose complete anchored-V3 stroke windows without dropping long drawings."""

    def __init__(
        self,
        dataset: Dataset,
        max_length: int,
        *,
        sos_token_id: int,
        eos_token_id: int,
        stroke_start_token_id: int,
        stroke_end_token_id: int,
    ) -> None:
        if max_length < 7:
            raise ValueError("anchored-V3 windows require max_length >= 7")
        self.dataset = dataset
        self.max_length = int(max_length)
        self.sos_token_id = int(sos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.stroke_start_token_id = int(stroke_start_token_id)
        self.stroke_end_token_id = int(stroke_end_token_id)
        self._windows: list[tuple[int, int, int]] = []
        self._lengths: list[int] = []
        self._build_windows()
        if not self._windows:
            raise ValueError("anchored-V3 curriculum produced no complete-stroke windows")

    def _build_windows(self) -> None:
        for dataset_index in range(len(self.dataset)):
            sample = self.dataset[dataset_index]
            tokens = sample["tokens"]
            spans = self._stroke_spans(tokens)
            window_start: int | None = None
            window_end: int | None = None
            for start, end in spans:
                proposed_start = start if window_start is None else window_start
                proposed_length = 2 + (end - proposed_start)
                if proposed_length > self.max_length:
                    if window_start is None or window_end is None:
                        raise ValueError(
                            f"single anchored-V3 stroke exceeds curriculum max_length={self.max_length}"
                        )
                    self._append_window(dataset_index, window_start, window_end)
                    window_start = start
                elif window_start is None:
                    window_start = start
                window_end = end
            if window_start is not None and window_end is not None:
                self._append_window(dataset_index, window_start, window_end)

    def _append_window(self, dataset_index: int, start: int, end: int) -> None:
        length = 2 + (end - start)
        if length > self.max_length:
            raise ValueError("complete stroke window exceeds configured maximum")
        self._windows.append((dataset_index, start, end))
        self._lengths.append(length)

    def _stroke_spans(self, tokens) -> list[tuple[int, int]]:
        starts = [index for index, token in enumerate(tokens) if int(token) == self.stroke_start_token_id]
        ends = [index + 1 for index, token in enumerate(tokens) if int(token) == self.stroke_end_token_id]
        if len(starts) != len(ends) or not starts:
            raise ValueError("invalid anchored-V3 sequence: unbalanced or missing strokes")
        spans = list(zip(starts, ends))
        if any(start >= end for start, end in spans):
            raise ValueError("invalid anchored-V3 stroke ordering")
        return spans

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, item: int):
        dataset_index, start, end = self._windows[item]
        sample = dict(self.dataset[dataset_index])
        body = sample["tokens"][start:end]
        tokens = np.concatenate(
            (
                np.asarray([self.sos_token_id], dtype=np.int64),
                np.asarray(body, dtype=np.int64),
                np.asarray([self.eos_token_id], dtype=np.int64),
            )
        )
        sample["tokens"] = tokens
        sample["length"] = len(tokens)
        sample["sample_id"] = f"{sample['sample_id']}:window{item}"
        return sample

    @property
    def lengths(self) -> list[int]:
        return list(self._lengths)


class StrokeSequenceDataModule:
    """Small Lightning-compatible DataModule without requiring Lightning import."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        project_root: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        self.config = _to_plain_config(config)
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.seed = int(seed)

        self.train_dataset: StrokeSequenceDataset | AnchoredV3Dataset | None = None
        self.valid_dataset: StrokeSequenceDataset | AnchoredV3Dataset | None = None
        self.test_dataset: StrokeSequenceDataset | AnchoredV3Dataset | None = None

    @property
    def format_type(self) -> str:
        return str(_get(self.config, "format.type", "stroke3"))

    @property
    def uses_token_format(self) -> bool:
        return self.format_type in {"tok_dict", "token", "tokens", "anchored_v3"}

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            self.train_dataset = self._make_dataset("train")
            self.valid_dataset = self._make_dataset("valid")
        if stage in {None, "test"}:
            self.test_dataset = self._make_dataset("test")

    def train_dataloader(self, max_sequence_length: int | None = None) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        dataset: Any = self.train_dataset
        if max_sequence_length is not None:
            if self.format_type == "anchored_v3":
                token_dictionary = _get(self.config, "format.token_dictionary", {})
                dataset = CompleteStrokeWindowDataset(
                    self.train_dataset,
                    int(max_sequence_length),
                    sos_token_id=int(token_dictionary["sos_token_id"]),
                    eos_token_id=int(token_dictionary["eos_token_id"]),
                    stroke_start_token_id=int(token_dictionary["stroke_start_token_id"]),
                    stroke_end_token_id=int(token_dictionary["stroke_end_token_id"]),
                )
            else:
                dataset = SequenceLengthSubset(
                    self.train_dataset,
                    self.train_dataset.lengths,
                    int(max_sequence_length),
                )
        return self._make_loader(dataset, split="train")

    def val_dataloader(
        self,
        max_sequence_length: int | None = None,
        *,
        max_samples: int | None = None,
        stratify_by_length: bool = False,
    ) -> DataLoader:
        if self.valid_dataset is None:
            self.setup("fit")
        assert self.valid_dataset is not None
        dataset: Any = self.valid_dataset
        if max_sequence_length is not None:
            dataset = SequenceLengthSubset(
                self.valid_dataset,
                self.valid_dataset.lengths,
                int(max_sequence_length),
            )
        if max_samples is not None:
            if stratify_by_length:
                dataset = LengthStratifiedSubset(
                    dataset,
                    dataset.lengths,
                    int(max_samples),
                    boundaries=_get(
                        self.config,
                        "batching.bucket_boundaries",
                        [512, 1024, 2048, 4096],
                    ),
                )
            else:
                dataset = PrefixSubset(
                    dataset,
                    dataset.lengths,
                    int(max_samples),
                )
        return self._make_loader(dataset, split="valid")

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        assert self.test_dataset is not None
        return self._make_loader(self.test_dataset, split="test")

    def _make_dataset(self, split: str) -> StrokeSequenceDataset | AnchoredV3Dataset:
        root = Path(_get(self.config, "dataset.root"))
        if not root.is_absolute():
            root = self.project_root / root

        if self.uses_token_format:
            separator_id = _get(
                self.config,
                "format.token_dictionary.sep_token_id",
            )
            if separator_id is None:
                separator_id = _get(
                    self.config,
                    "format.token_dictionary.stroke_end_token_id",
                    0,
                )
            transform = TokenSequenceTransform(
                split=split,
                max_length=int(_get(self.config, "sequence.max_length")),
                truncate_long_sequences=bool(
                    _get(self.config, "sequence.truncate_long_sequences", True)
                ),
                add_start_token=bool(_get(self.config, "sequence.add_start_token", True)),
                add_end_token=bool(_get(self.config, "sequence.add_end_token", True)),
                sos_token_id=int(
                    _get(self.config, "format.token_dictionary.sos_token_id")
                ),
                sep_token_id=int(separator_id),
                eos_token_id=int(
                    _get(self.config, "format.token_dictionary.eos_token_id")
                ),
            )
        else:
            transform = Stroke3Transform(
                split=split,
                max_length=int(_get(self.config, "sequence.max_length")),
                truncate_long_sequences=bool(
                    _get(self.config, "sequence.truncate_long_sequences", True)
                ),
                normalize=bool(_get(self.config, "preprocessing.normalize_by_bounds", True)),
                delta_clip=float(_get(self.config, "preprocessing.delta_clip", 1000.0)),
                augmentation=_get(self.config, "augmentation", {}),
                seed=self.seed,
            )

        if self.format_type == "anchored_v3":
            source_limit = (
                _get(self.config, "dataset.train_source_limit")
                if split == "train"
                else None
            )
            return AnchoredV3Dataset(
                root,
                split=split,
                transform=transform,
                max_cached_files=int(_get(self.config, "preprocessing.max_cached_files", 2)),
                validate_artifacts=bool(
                    _get(self.config, "dataset.validate_artifacts", True)
                ),
                max_source_samples=(
                    int(source_limit) if source_limit is not None else None
                ),
                subset_seed=int(_get(self.config, "dataset.subset_seed", self.seed)),
                include_augmentations=bool(
                    _get(self.config, "dataset.include_augmentations", True)
                ),
                source_subset_strategy=str(
                    _get(self.config, "dataset.source_subset_strategy", "nested_hash")
                ),
            )

        return StrokeSequenceDataset(
            root=root,
            split=split,
            train_pattern=str(_get(self.config, "dataset.train_pattern", "train_*.npz")),
            valid_file=str(_get(self.config, "dataset.valid_file", "valid.npz")),
            test_file=str(_get(self.config, "dataset.test_file", "test.npz")),
            metadata_file=str(_get(self.config, "dataset.metadata_file", "meta.npz")),
            format_type=self.format_type,
            transform=transform,
            max_cached_files=int(_get(self.config, "preprocessing.max_cached_files", 2)),
        )

    def _make_loader(self, dataset: Dataset, *, split: str) -> DataLoader:
        is_train = split == "train"
        batch_size = int(
            _get(
                self.config,
                "batching.batch_size" if is_train else "batching.eval_batch_size",
                2,
            )
        )
        num_workers = int(_get(self.config, "batching.num_workers", 0))
        persistent_workers = bool(_get(self.config, "batching.persistent_workers", False))
        if num_workers == 0:
            persistent_workers = False

        if self.uses_token_format:
            separator_id = _get(
                self.config,
                "format.token_dictionary.sep_token_id",
            )
            if separator_id is None:
                separator_id = _get(
                    self.config,
                    "format.token_dictionary.stroke_end_token_id",
                    0,
                )
            collate_fn = TokenSequenceCollator(
                max_length=int(_get(self.config, "sequence.max_length")),
                pad_token_id=int(
                    _get(self.config, "format.token_dictionary.pad_token_id")
                ),
                sep_token_id=int(separator_id),
                sos_token_id=int(
                    _get(self.config, "format.token_dictionary.sos_token_id")
                ),
                eos_token_id=int(
                    _get(self.config, "format.token_dictionary.eos_token_id")
                ),
                add_start_token=bool(_get(self.config, "sequence.add_start_token", True)),
                add_end_token=bool(_get(self.config, "sequence.add_end_token", True)),
                truncate_long_sequences=bool(
                    _get(self.config, "sequence.truncate_long_sequences", True)
                ),
                pad_to_multiple_of=int(_get(self.config, "sequence.pad_to_multiple_of", 8)),
                causal_attention=False,
                build_attention_mask=bool(
                    _get(self.config, "sequence.build_attention_mask", True)
                ),
            )
        else:
            collate_fn = Stroke3Collator(
                max_length=int(_get(self.config, "sequence.max_length")),
                pad_value=float(_get(self.config, "sequence.pad_value", 0.0)),
                pad_to_multiple_of=int(_get(self.config, "sequence.pad_to_multiple_of", 8)),
                causal_attention=False,
                build_attention_mask=bool(
                    _get(self.config, "sequence.build_attention_mask", True)
                ),
            )

        max_tokens_per_batch = int(
            _get(
                self.config,
                "batching.max_tokens_per_batch"
                if is_train
                else "batching.eval_max_tokens_per_batch",
                _get(self.config, "batching.max_tokens_per_batch", 0),
            )
            or 0
        )
        if max_tokens_per_batch > 0:
            batch_sampler = TokenBudgetBatchSampler(
                dataset.lengths,
                max_tokens=max_tokens_per_batch,
                max_batch_size=batch_size,
                shuffle=is_train,
                drop_last=bool(_get(self.config, "batching.drop_last", False)) if is_train else False,
                seed=self.seed,
                pad_to_multiple_of=int(
                    _get(self.config, "sequence.pad_to_multiple_of", 1)
                ),
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=bool(_get(self.config, "batching.pin_memory", False)),
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )

        if is_train and bool(_get(self.config, "batching.bucket_by_length", False)):
            batch_sampler = LengthBucketBatchSampler(
                dataset.lengths,
                batch_size=batch_size,
                boundaries=_get(self.config, "batching.bucket_boundaries", [256, 512, 1024, 2048]),
                shuffle=True,
                drop_last=bool(_get(self.config, "batching.drop_last", False)),
                seed=self.seed,
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=bool(_get(self.config, "batching.pin_memory", False)),
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=bool(_get(self.config, "batching.drop_last", False)) if is_train else False,
            num_workers=num_workers,
            pin_memory=bool(_get(self.config, "batching.pin_memory", False)),
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )
