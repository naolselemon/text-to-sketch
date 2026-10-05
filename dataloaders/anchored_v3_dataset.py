"""Manifest-driven reader for content-addressed anchored-V3 token shards."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from services.anchored_sketch_data.artifacts import validate_dataset


@dataclass(frozen=True)
class AnchoredV3SampleIndex:
    file_path: Path
    shard_index: int
    length: int
    sample_id: str
    source_sample_id: str
    augmented: bool


class AnchoredV3Dataset(Dataset):
    """Read only manifest-listed V3 samples and reject altered artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        max_cached_files: int = 2,
        validate_artifacts: bool = True,
        max_source_samples: int | None = None,
        subset_seed: int = 42,
        include_augmentations: bool = True,
        source_subset_strategy: str = "nested_hash",
    ) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be train, valid, or test")
        # Pin a content-addressed directory at construction time. Retargeting a
        # user-facing ``current`` symlink cannot change later uncached reads.
        self.root = Path(root).resolve(strict=True)
        self.split = split
        self.transform = transform
        self.max_cached_files = max(0, int(max_cached_files))
        if max_source_samples is not None and int(max_source_samples) <= 0:
            raise ValueError("max_source_samples must be positive")
        if max_source_samples is not None and split != "train":
            raise ValueError("max_source_samples is only valid for the train split")
        self.max_source_samples = (
            int(max_source_samples) if max_source_samples is not None else None
        )
        self.subset_seed = int(subset_seed)
        self.include_augmentations = bool(include_augmentations)
        if source_subset_strategy not in {"nested_hash", "length_stratified"}:
            raise ValueError(
                "source_subset_strategy must be nested_hash or length_stratified"
            )
        self.source_subset_strategy = source_subset_strategy
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self.metadata = (
            validate_dataset(self.root)
            if validate_artifacts
            else json.loads((self.root / "meta.json").read_text())
        )
        self.index = self._build_index()
        if not self.index:
            raise ValueError(f"Anchored V3 split {split} is empty under {self.root}")

    def _build_index(self) -> list[AnchoredV3SampleIndex]:
        values: list[AnchoredV3SampleIndex] = []
        with (self.root / "manifest.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                if entry.get("status") != "accepted" or entry.get("split") != self.split:
                    continue
                preprocessing = entry.get("preprocessing", {}) or {}
                sample_id = str(entry["sample_id"])
                values.append(
                    AnchoredV3SampleIndex(
                        file_path=self.root / str(entry["shard"]),
                        shard_index=int(entry["shard_index"]),
                        length=int(entry["token_length"]),
                        sample_id=sample_id,
                        source_sample_id=str(
                            preprocessing.get(
                                "source_sample_id",
                                sample_id.split("#aug-", 1)[0],
                            )
                        ),
                        augmented=bool(preprocessing.get("augmented", False)),
                    )
                )
        values.sort(key=lambda item: item.sample_id)
        if not self.include_augmentations:
            values = [entry for entry in values if not entry.augmented]
        if self.max_source_samples is None:
            return values

        originals = [entry for entry in values if not entry.augmented]
        original_ids = self._ordered_source_ids(originals)
        if len(original_ids) < self.max_source_samples:
            raise ValueError(
                f"Requested {self.max_source_samples} train source sketches, "
                f"but the manifest contains only {len(original_ids)}"
            )
        selected = set(original_ids[: self.max_source_samples])
        return [entry for entry in values if entry.source_sample_id in selected]

    def _ordered_source_ids(
        self,
        originals: list[AnchoredV3SampleIndex],
    ) -> list[str]:
        def stable_key(entry: AnchoredV3SampleIndex) -> tuple[str, str]:
            return (
                hashlib.sha256(
                    f"{self.subset_seed}:{entry.source_sample_id}".encode("utf-8")
                ).hexdigest(),
                entry.source_sample_id,
            )

        unique = {entry.source_sample_id: entry for entry in originals}
        if self.source_subset_strategy == "nested_hash":
            return [entry.source_sample_id for entry in sorted(unique.values(), key=stable_key)]

        buckets: list[list[AnchoredV3SampleIndex]] = [[], [], [], []]
        for entry in unique.values():
            bucket = (
                0
                if entry.length <= 512
                else 1
                if entry.length <= 1024
                else 2
                if entry.length <= 2048
                else 3
            )
            buckets[bucket].append(entry)
        for bucket in buckets:
            bucket.sort(key=stable_key)
        requested = int(self.max_source_samples or len(unique))
        base, remainder = divmod(requested, len(buckets))
        selected: list[AnchoredV3SampleIndex] = []
        for index, bucket in enumerate(buckets):
            selected.extend(bucket[: base + int(index < remainder)])
        selected_ids = {entry.source_sample_id for entry in selected}
        if len(selected) < requested:
            remaining = sorted(
                (
                    entry
                    for entry in unique.values()
                    if entry.source_sample_id not in selected_ids
                ),
                key=stable_key,
            )
            selected.extend(remaining[: requested - len(selected)])
        return [entry.source_sample_id for entry in selected]

    def __len__(self) -> int:
        return len(self.index)

    @property
    def lengths(self) -> list[int]:
        return [entry.length for entry in self.index]

    def __getitem__(self, item: int) -> dict[str, Any]:
        entry = self.index[item]
        shard = self._load_shard(entry.file_path)
        offsets = shard["offsets"]
        start = int(offsets[entry.shard_index])
        end = int(offsets[entry.shard_index + 1])
        tokens = np.asarray(shard["tokens"][start:end], dtype=np.int64)
        sample: dict[str, Any] = {
            "tokens": tokens,
            "label": 0,
            "length": len(tokens),
            "source_file": str(entry.file_path),
            "source_index": entry.shard_index,
            "sample_id": entry.sample_id,
        }
        return self.transform(sample) if self.transform is not None else sample

    def _load_shard(self, path: Path) -> dict[str, np.ndarray]:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        with np.load(path, allow_pickle=False) as archive:
            loaded = {name: archive[name] for name in archive.files}
        if self.max_cached_files > 0:
            self._cache[path] = loaded
            while len(self._cache) > self.max_cached_files:
                self._cache.popitem(last=False)
        return loaded
