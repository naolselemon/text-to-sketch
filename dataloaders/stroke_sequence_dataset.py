"""PyTorch dataset for Sketchformer-ready stroke3 or tok-dict chunks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset


@dataclass(frozen=True)
class StrokeSampleIndex:
    """Location metadata for one sketch inside a chunk file."""

    file_path: Path
    local_index: int
    length: int
    sample_id: str


class StrokeSequenceDataset(Dataset):
    """Load sketch sequences saved as ``x`` and ``y`` arrays in ``.npz`` chunks."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        train_pattern: str = "train_*.npz",
        valid_file: str = "valid.npz",
        test_file: str = "test.npz",
        metadata_file: str = "meta.npz",
        format_type: str = "stroke3",
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        max_cached_files: int = 2,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.metadata_file = metadata_file
        self.format_type = format_type
        self.transform = transform
        self.max_cached_files = max(0, int(max_cached_files))
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

        self.metadata = self._load_metadata()
        self._verify_metadata()
        self.files = self._resolve_files(train_pattern, valid_file, test_file)
        self.index = self._build_index()
        self._verify_index()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        entry = self.index[item]
        data = self._load_chunk(entry.file_path)
        sequence = data["x"][entry.local_index]
        label = int(np.asarray(data["y"][entry.local_index]).reshape(-1)[0])

        sample: dict[str, Any] = {
            "label": label,
            "length": int(len(sequence)),
            "source_file": str(entry.file_path),
            "source_index": int(entry.local_index),
            "sample_id": entry.sample_id,
        }
        if self.format_type in {"tok_dict", "token", "tokens"}:
            sample["tokens"] = np.asarray(sequence, dtype=np.int64)
        elif self.format_type == "stroke3":
            sample["stroke3"] = np.asarray(sequence, dtype=np.float32)
        else:
            raise ValueError("format_type must be one of: stroke3, tok_dict")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @property
    def lengths(self) -> list[int]:
        return [entry.length for entry in self.index]

    def _verify_metadata(self) -> None:
        meta_format = str(self.metadata.get("format", ""))
        is_tok_dict = (
            self.format_type in {"tok_dict", "token", "tokens"}
            or meta_format in {"tok_dict", "token", "tokens"}
        )
        if is_tok_dict or "is_published" in self.metadata:
            meta_path = self.root / self.metadata_file
            if not meta_path.is_file():
                raise RuntimeError(
                    f"Dataset at {self.root} is not published or build was interrupted. "
                    f"Missing metadata file {self.metadata_file}."
                )
            if not bool(self.metadata.get("is_published", False)):
                raise RuntimeError(
                    f"Dataset at {self.root} is not published or build was interrupted. "
                    "meta.npz is missing the verified publication marker."
                )
        # Older published archives did not record IDs; keep their legacy
        # loader contract, but never silently downgrade a versioned dataset.
        if meta_format in {"tok_dict", "token", "tokens"} and int(self.metadata.get("schema_version", 0)) >= 2:
            for split in ("train", "valid", "test"):
                if f"{split}_sample_ids" not in self.metadata:
                    raise ValueError(f"Missing {split}_sample_ids in published metadata")

    def _resolve_files(self, train_pattern: str, valid_file: str, test_file: str) -> list[Path]:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        if self.split == "train":
            if "train_chunks" in self.metadata:
                files = [self.root / str(name) for name in self.metadata["train_chunks"]]
            else:
                files = sorted(self.root.glob(train_pattern))
        elif self.split == "valid":
            if "valid_chunks" in self.metadata:
                files = [self.root / str(name) for name in self.metadata["valid_chunks"]]
            else:
                files = [self.root / valid_file]
        elif self.split == "test":
            if "test_chunks" in self.metadata:
                files = [self.root / str(name) for name in self.metadata["test_chunks"]]
            else:
                files = [self.root / test_file]
        else:
            raise ValueError("split must be one of: train, valid, test")

        for path in files:
            if not path.is_file():
                raise FileNotFoundError(f"Required chunk file does not exist: {path}")

        if not files:
            raise FileNotFoundError(f"No {self.split} files found under {self.root}")
        return files

    def _build_index(self) -> list[StrokeSampleIndex]:
        index: list[StrokeSampleIndex] = []
        for file_path in self.files:
            data = self._read_npz(file_path)
            self._validate_chunk(data, file_path)
            if f"{self.split}_sample_ids" in self.metadata:
                if "sample_ids" not in data:
                    raise ValueError(f"Missing sample_ids in published chunk {file_path}")
                if len(data["sample_ids"]) != len(data["x"]):
                    raise ValueError(f"Sample ID count mismatch in {file_path}")
            for local_index, stroke in enumerate(data["x"]):
                sample_id = (
                    str(data["sample_ids"][local_index])
                    if "sample_ids" in data
                    else f"{file_path.stem}:{local_index}"
                )
                index.append(
                    StrokeSampleIndex(
                        file_path=file_path,
                        local_index=local_index,
                        length=int(len(stroke)),
                        sample_id=sample_id,
                    )
                )
        if not index:
            raise ValueError(f"Split {self.split} is empty under {self.root}")
        return index

    def _verify_index(self) -> None:
        chunk_count_key = f"{self.split}_chunk_counts"
        if chunk_count_key in self.metadata:
            expected_counts = list(self.metadata[chunk_count_key])
            if len(expected_counts) != len(self.files):
                raise ValueError(
                    f"Chunk list length mismatch for split {self.split}: "
                    f"expected {len(expected_counts)} chunks, found {len(self.files)}"
                )
            for file_path, expected_count in zip(self.files, expected_counts):
                actual_count = sum(1 for entry in self.index if entry.file_path == file_path)
                if actual_count != int(expected_count):
                    raise ValueError(
                        f"Chunk count mismatch in {file_path}: "
                        f"expected {expected_count}, got {actual_count}"
                    )

        split_count_key = f"n_samples_{self.split}"
        if split_count_key in self.metadata:
            expected_total = int(self.metadata[split_count_key])
            if len(self.index) != expected_total:
                raise ValueError(
                    f"Total sample count mismatch for split '{self.split}': "
                    f"expected {expected_total}, got {len(self.index)}"
                )

        ids_key = f"{self.split}_sample_ids"
        if ids_key in self.metadata:
            expected_ids = [str(value) for value in self.metadata[ids_key]]
            actual_ids = [entry.sample_id for entry in self.index]
            if actual_ids != expected_ids:
                raise ValueError(
                    f"Sample IDs mismatch for split '{self.split}': "
                    f"expected {expected_ids}, got {actual_ids}"
                )

    def _load_metadata(self) -> dict[str, Any]:
        meta_path = self.root / self.metadata_file
        if not meta_path.exists():
            return {}

        metadata: dict[str, Any] = {}
        with np.load(meta_path, allow_pickle=True) as data:
            for key in data.files:
                value = data[key]
                metadata[key] = value.item() if value.shape == () else value
        return metadata

    def _load_chunk(self, file_path: Path) -> dict[str, np.ndarray]:
        if self.max_cached_files == 0:
            return self._read_npz(file_path)

        cached = self._cache.get(file_path)
        if cached is not None:
            self._cache.move_to_end(file_path)
            return cached

        loaded = self._read_npz(file_path)
        self._cache[file_path] = loaded
        while len(self._cache) > self.max_cached_files:
            self._cache.popitem(last=False)
        return loaded

    @staticmethod
    def _read_npz(file_path: Path) -> dict[str, np.ndarray]:
        with np.load(file_path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    @staticmethod
    def _validate_chunk(data: dict[str, np.ndarray], file_path: Path) -> None:
        if "x" not in data:
            raise ValueError(f"Missing x array in {file_path}")
        if "y" not in data:
            raise ValueError(f"Missing y array in {file_path}")
        if len(data["x"]) != len(data["y"]):
            raise ValueError(
                f"Mismatched x/y lengths in {file_path}: {len(data['x'])} != {len(data['y'])}"
            )
