"""PyTorch dataset for Sketchformer-ready stroke3 or tok-dict chunks."""

from __future__ import annotations

import hashlib
import io
import json
import warnings
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
        lineage: Any | None = None,
        verify_integrity: bool = True,
        allow_unverified: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.metadata_file = metadata_file
        self.format_type = format_type
        self.transform = transform
        self.max_cached_files = max(0, int(max_cached_files))
        self.lineage = lineage
        self.verify_integrity = bool(verify_integrity)
        self.allow_unverified = bool(allow_unverified)
        self._cache: OrderedDict[Path, tuple[dict[str, np.ndarray], float]] = OrderedDict()

        self.metadata = self._load_metadata()
        self._verify_metadata()
        self.declared_checksums: dict[str, str] = self._resolve_chunk_checksums()
        self.files = self._resolve_files(train_pattern, valid_file, test_file)
        self.index = self._build_index()
        self._verify_index()

    def _resolve_chunk_checksums(self) -> dict[str, str]:
        if self.lineage is not None:
            lineage_root = self.lineage.metadata.get("root") if hasattr(self.lineage, "metadata") and self.lineage.metadata else None
            if lineage_root:
                try:
                    if Path(lineage_root).resolve() != self.root.resolve():
                        raise ValueError(
                            f"DataLoader root mismatch: dataset initialized with {self.root} "
                            f"but lineage was verified for {lineage_root}"
                        )
                except Exception as e:
                    if isinstance(e, ValueError):
                        raise
            meta_build_id = str(self.metadata.get("build_id", "")).strip()
            lineage_build_id = str(getattr(self.lineage, "build_id", "")).strip()
            if meta_build_id and lineage_build_id and meta_build_id != lineage_build_id:
                raise ValueError(
                    f"DataLoader build_id mismatch: dataset has build_id '{meta_build_id}' "
                    f"but lineage has '{lineage_build_id}'"
                )
            if not getattr(self.lineage, "is_verified", False) and not self.allow_unverified:
                raise ValueError(
                    "Provided dataset lineage is unverified. Pass allow_unverified=True to permit loading."
                )

        declared_checksums: dict[str, str] = {}
        if self.lineage is not None and getattr(self.lineage, "chunk_checksums", None):
            declared_checksums = {str(k): str(v) for k, v in self.lineage.chunk_checksums.items()}
        elif "chunk_checksums" in self.metadata:
            raw = self.metadata["chunk_checksums"]
            if isinstance(raw, str):
                try:
                    declared_checksums = json.loads(raw)
                except Exception:
                    pass
            elif isinstance(raw, dict):
                declared_checksums = {str(k): str(v) for k, v in raw.items()}

        return declared_checksums

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
            sample["tokens"] = np.array(sequence, dtype=np.int64, copy=True)
        elif self.format_type == "stroke3":
            sample["stroke3"] = np.array(sequence, dtype=np.float32, copy=True)
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

    @staticmethod
    def _make_read_only(data: dict[str, np.ndarray]) -> None:
        """Mark arrays in chunk data as read-only to prevent in-place cache mutation."""
        for val in data.values():
            if isinstance(val, np.ndarray):
                try:
                    val.setflags(write=False)
                except (ValueError, AttributeError):
                    pass
                if val.dtype == object:
                    for item in val:
                        if isinstance(item, np.ndarray):
                            try:
                                item.setflags(write=False)
                            except (ValueError, AttributeError):
                                pass

    def _load_chunk(self, file_path: Path) -> dict[str, np.ndarray]:
        if self.max_cached_files == 0:
            return self._read_npz(file_path)

        cached = self._cache.get(file_path)
        if cached is not None:
            data, cached_mtime, cached_size = cached
            if self.verify_integrity and self.declared_checksums:
                st = file_path.stat() if file_path.exists() else None
                current_mtime = st.st_mtime if st else 0.0
                current_size = st.st_size if st else 0
                if current_mtime != cached_mtime or current_size != cached_size:
                    # File on disk changed since caching - re-read & re-verify
                    data = self._read_npz(file_path)
                    self._cache[file_path] = (data, current_mtime, current_size)
            self._cache.move_to_end(file_path)
            return data

        loaded = self._read_npz(file_path)
        st = file_path.stat() if file_path.exists() else None
        current_mtime = st.st_mtime if st else 0.0
        current_size = st.st_size if st else 0
        self._cache[file_path] = (loaded, current_mtime, current_size)
        while len(self._cache) > self.max_cached_files:
            self._cache.popitem(last=False)
        return loaded

    def _read_npz(self, file_path: Path) -> dict[str, np.ndarray]:
        if self.verify_integrity and self.declared_checksums:
            chunk_name = file_path.name
            expected_checksum = self.declared_checksums.get(chunk_name)
            if expected_checksum is None:
                if not self.allow_unverified:
                    raise ValueError(
                        f"Chunk file '{chunk_name}' has no declared checksum in verified lineage. "
                        "Rejecting unverified chunk."
                    )
                warnings.warn(
                    f"Reading chunk '{chunk_name}' without verified checksum.",
                    UserWarning,
                    stacklevel=2,
                )
                with np.load(file_path, allow_pickle=True) as data:
                    res = {key: data[key] for key in data.files}
                    self._make_read_only(res)
                    return res

            raw_bytes = file_path.read_bytes()
            actual_checksum = hashlib.sha256(raw_bytes).hexdigest()
            if actual_checksum != expected_checksum:
                raise ValueError(
                    f"Chunk integrity check failed for '{chunk_name}': "
                    f"expected sha256 {expected_checksum}, got {actual_checksum}. "
                    "Chunk contents have been modified or corrupted!"
                )
            with np.load(io.BytesIO(raw_bytes), allow_pickle=True) as data:
                res = {key: data[key] for key in data.files}
                self._make_read_only(res)
                return res

        with np.load(file_path, allow_pickle=True) as data:
            res = {key: data[key] for key in data.files}
            self._make_read_only(res)
            return res

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
