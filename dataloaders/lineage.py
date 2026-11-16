"""Dataset lineage tracking and verification for Text-to-Sketch / Sketchformer."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class DatasetLineage:
    """Provenance metadata identifying the exact dataset build and split membership."""

    build_id: str
    dictionary_fingerprint: str
    dataset_fingerprint: str
    split_counts: dict[str, int]
    split_hashes: dict[str, str]
    train_chunks: list[str]
    valid_chunks: list[str]
    test_chunks: list[str]
    chunk_checksums: dict[str, str] = field(default_factory=dict)
    format: str = "tok_dict"
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2
    is_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetLineage:
        split_counts = {str(k): int(v) for k, v in data.get("split_counts", {}).items()}
        split_hashes = {str(k): str(v) for k, v in data.get("split_hashes", {}).items()}
        train_chunks = [str(c) for c in data.get("train_chunks", [])]
        valid_chunks = [str(c) for c in data.get("valid_chunks", [])]
        test_chunks = [str(c) for c in data.get("test_chunks", [])]
        chunk_checksums = {str(k): str(v) for k, v in data.get("chunk_checksums", {}).items()}
        metadata = dict(data.get("metadata", {}))

        return cls(
            build_id=str(data.get("build_id", "")),
            dictionary_fingerprint=str(data.get("dictionary_fingerprint", "")),
            dataset_fingerprint=str(data.get("dataset_fingerprint", "")),
            split_counts=split_counts,
            split_hashes=split_hashes,
            train_chunks=train_chunks,
            valid_chunks=valid_chunks,
            test_chunks=test_chunks,
            chunk_checksums=chunk_checksums,
            format=str(data.get("format", "tok_dict")),
            created_at=str(data.get("created_at", "")),
            metadata=metadata,
            schema_version=int(data.get("schema_version", 2)),
            is_verified=_validate_is_verified(data.get("is_verified", False), default=False),
        )


def _validate_is_verified(val: Any, default: bool = False) -> bool:
    """Strictly validate that is_verified is genuinely a boolean.

    Accepts genuine Python booleans (bool), NumPy scalar booleans (np.bool_),
    or zero-dimensional scalar boolean NumPy arrays (np.ndarray of dtype bool and ndim 0).

    Rejects non-boolean types (e.g. strings, non-boolean integers, floats,
    arbitrary collections or multi-element arrays) instead of loosely coercing with bool().

    Raises ValueError if an invalid non-boolean value is provided.
    """
    if val is None:
        return default
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, np.ndarray) and val.dtype == bool and val.ndim == 0:
        return bool(val.item())
    raise ValueError(
        f"Invalid non-boolean value for 'is_verified': {val!r} (type {type(val).__name__}). "
        "Expected strict boolean True or False."
    )


def compute_chunk_checksum(path: str | Path, chunk_size: int = 65536) -> str:
    """Compute deterministic hex SHA-256 checksum of a chunk file artifact."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Chunk file not found for checksum calculation: {p}")
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_split_hash(sample_ids: Iterable[str]) -> str:
    """Compute deterministic SHA-256 hash over sorted unique sample IDs.

    Validates that:
    - No sample ID is empty or whitespace-only.
    - No duplicate sample IDs exist within the split.

    Returns empty string for an empty sample set.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    has_any = False

    for sid in sample_ids:
        s = str(sid).strip()
        if not s:
            raise ValueError("Sample IDs must not be empty or whitespace-only")
        has_any = True
        if s in seen:
            duplicates.append(s)
        seen.add(s)

    if duplicates:
        unique_dups = sorted(set(duplicates))
        raise ValueError(
            f"Duplicate sample IDs detected within split ({len(duplicates)} duplicate occurrences): {unique_dups[:10]}"
        )

    if not has_any:
        return ""

    content = "\n".join(sorted(seen))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_dataset_fingerprint(
    *,
    build_id: str,
    dictionary_fingerprint: str,
    split_hashes: Mapping[str, str],
    split_counts: Mapping[str, int],
    train_chunks: list[str],
    valid_chunks: list[str],
    test_chunks: list[str],
    chunk_checksums: Mapping[str, str] | None = None,
    format_type: str = "tok_dict",
    schema_version: int = 2,
) -> str:
    """Generate a single deterministic composite hash for the entire dataset build, including chunk artifacts."""
    clean_checksums = (
        {str(k): str(chunk_checksums[k]) for k in sorted(chunk_checksums)}
        if chunk_checksums is not None
        else {}
    )
    payload = {
        "build_id": str(build_id).strip(),
        "dictionary_fingerprint": str(dictionary_fingerprint).strip(),
        "format": str(format_type).strip(),
        "schema_version": int(schema_version),
        "split_counts": {k: int(split_counts[k]) for k in sorted(split_counts)},
        "split_hashes": {k: str(split_hashes[k]) for k in sorted(split_hashes)},
        "train_chunks": sorted(str(c) for c in train_chunks),
        "valid_chunks": sorted(str(c) for c in valid_chunks),
        "test_chunks": sorted(str(c) for c in test_chunks),
        "chunk_checksums": clean_checksums,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_dataset_lineage(
    dataset_root: str | Path,
    *,
    verify_chunks: bool = True,
    allow_unverified: bool = False,
    reject_unexpected_chunks: bool = True,
) -> DatasetLineage:
    """Load and verify dataset lineage metadata from a published dataset directory.

    Parameters
    ----------
    dataset_root : str | Path
        Root directory containing `meta.npz` and dataset chunks.
    verify_chunks : bool
        If True, strictly verifies that all chunk files exist, calculates chunk SHA-256
        checksums, verifies that declared chunk checksums match, ensures sample IDs are
        non-empty and unique within splits, verifies split counts match exactly, confirms
        no sample ID leaks across train/valid/test splits, rejects unexpected untracked chunks,
        and recomputes the dataset fingerprint.
        If False, fast inspection mode: verifies all declared chunk files exist on disk,
        but returns an unverified DatasetLineage with `is_verified=False`.
    allow_unverified : bool
        If True, permits loading legacy datasets without modern provenance or publication marker,
        marking is_verified=False without fabricating missing provenance fields.
    reject_unexpected_chunks : bool
        If True (default), raises ValueError when unexpected untracked .npz files are present on disk.

    Raises
    ------
    FileNotFoundError: If `meta.npz` or any declared chunk file does not exist.
    RuntimeError: If dataset is not published or build was interrupted.
    ValueError: If metadata is incomplete, counts mismatch, checksums mismatch, data leaks across splits,
                or unexpected untracked chunks are detected.
    """
    root = Path(dataset_root)
    meta_path = root / "meta.npz"
    if not meta_path.is_file():
        if not allow_unverified:
            raise FileNotFoundError(f"Dataset metadata file not found: {meta_path}")
        warnings.warn(
            f"Dataset at {root} lacks meta.npz. Loaded in unverified legacy mode.",
            UserWarning,
            stacklevel=2,
        )
        return DatasetLineage(
            build_id="",
            dictionary_fingerprint="",
            dataset_fingerprint="",
            split_counts={},
            split_hashes={},
            train_chunks=[],
            valid_chunks=[],
            test_chunks=[],
            chunk_checksums={},
            format="tok_dict",
            schema_version=0,
            is_verified=False,
            metadata={"is_legacy": True, "root": str(root)},
        )

    with np.load(meta_path, allow_pickle=True) as meta:
        is_published = bool(meta.get("is_published", False)) or bool(meta.get("verified", False))
        if not is_published:
            if not allow_unverified:
                raise RuntimeError(
                    f"Dataset at {root} is not published or build was interrupted. "
                    "meta.npz is missing the verified publication marker."
                )
            warnings.warn(
                f"Dataset at {root} lacks publication marker. Loaded in unverified legacy mode.",
                UserWarning,
                stacklevel=2,
            )

        build_id = str(meta.get("build_id", "")).strip()
        dictionary_fingerprint = str(meta.get("dictionary_fingerprint", "")).strip()
        format_type = str(meta.get("format", "tok_dict")).strip()

        train_chunks = [str(c) for c in meta.get("train_chunks", [])]
        valid_chunks = [str(c) for c in meta.get("valid_chunks", [])]
        test_chunks = [str(c) for c in meta.get("test_chunks", [])]

        if "split_counts" in meta:
            counts_arr = meta["split_counts"]
            if len(counts_arr) < 3:
                raise ValueError(
                    f"split_counts array in {meta_path} must have at least 3 elements (train, valid, test), got {len(counts_arr)}"
                )
            split_counts = {
                "train": int(counts_arr[0]),
                "valid": int(counts_arr[1]),
                "test": int(counts_arr[2]),
            }
        else:
            n_tr = meta.get("n_samples_train", None)
            n_val = meta.get("n_samples_valid", None)
            n_te = meta.get("n_samples_test", None)
            if n_tr is None or n_val is None or n_te is None:
                if not allow_unverified:
                    raise ValueError(f"Dataset metadata at {meta_path} is missing split counts")
                split_counts = {"train": 0, "valid": 0, "test": 0}
            else:
                split_counts = {
                    "train": int(n_tr),
                    "valid": int(n_val),
                    "test": int(n_te),
                }

        # Parse chunk checksums if stored
        declared_checksums: dict[str, str] = {}
        if "chunk_checksums" in meta:
            raw_ck = meta["chunk_checksums"]
            if isinstance(raw_ck, np.ndarray) and raw_ck.ndim == 0:
                raw_ck = raw_ck.item()
            if isinstance(raw_ck, str):
                try:
                    declared_checksums = json.loads(raw_ck)
                except Exception:
                    declared_checksums = {}
            elif isinstance(raw_ck, dict):
                declared_checksums = {str(k): str(v) for k, v in raw_ck.items()}

        declared_fingerprint = str(meta.get("dataset_fingerprint", "")).strip()

        # If missing critical lineage fields:
        if not build_id or not declared_fingerprint:
            if not allow_unverified:
                missing_fields = []
                if not build_id:
                    missing_fields.append("build_id")
                if not declared_fingerprint:
                    missing_fields.append("dataset_fingerprint")
                raise ValueError(
                    f"Dataset metadata at {meta_path} is missing required lineage fields: {missing_fields}. "
                    "Pass allow_unverified=True to load legacy datasets without modern provenance."
                )
            warnings.warn(
                f"Dataset at {root} lacks modern lineage fields. Loaded in unverified legacy mode.",
                UserWarning,
                stacklevel=2,
            )
            return DatasetLineage(
                build_id="",
                dictionary_fingerprint="",
                dataset_fingerprint="",
                split_counts=split_counts,
                split_hashes={},
                train_chunks=train_chunks,
                valid_chunks=valid_chunks,
                test_chunks=test_chunks,
                chunk_checksums=declared_checksums,
                format=format_type,
                schema_version=0,
                is_verified=False,
                metadata={"is_legacy": True, "root": str(root)},
            )

        missing_dict_fp = format_type in {"tok_dict", "token", "tokens"} and not dictionary_fingerprint
        if missing_dict_fp:
            if not allow_unverified:
                raise ValueError(f"Dataset metadata at {meta_path} is missing required dictionary_fingerprint")
            warnings.warn(
                f"Dataset at {root} is token-based but lacks verified dictionary_fingerprint. "
                "Loaded in unverified legacy mode.",
                UserWarning,
                stacklevel=2,
            )

        precomputed_train_hash = str(meta.get("split_hash_train", "")).strip()
        precomputed_valid_hash = str(meta.get("split_hash_valid", "")).strip()
        precomputed_test_hash = str(meta.get("split_hash_test", "")).strip()
        schema_version = int(meta.get("schema_version", 2 if declared_checksums else 1))

    # All declared chunk files MUST exist on disk regardless of verify_chunks
    chunk_map = {
        "train": train_chunks,
        "valid": valid_chunks,
        "test": test_chunks,
    }
    for split_name, chunk_names in chunk_map.items():
        for chunk_name in chunk_names:
            chunk_file = root / chunk_name
            if not chunk_file.is_file():
                raise FileNotFoundError(f"Required {split_name} chunk missing on disk: {chunk_file}")

    # Detect untracked .npz chunks on disk
    declared_set = set(train_chunks) | set(valid_chunks) | set(test_chunks) | {"meta.npz"}
    compatibility_aliases = {"valid.npz", "test.npz"}
    disk_npz = [
        f for f in root.glob("*.npz")
        if not f.name.startswith(".") and not f.name.endswith(".tmp.npz")
    ]
    unexpected_chunks = [
        f.name for f in disk_npz
        if f.name not in declared_set and f.name not in compatibility_aliases
    ]

    if unexpected_chunks:
        if verify_chunks and reject_unexpected_chunks and not allow_unverified:
            raise ValueError(
                f"Unexpected untracked dataset chunk(s) found in {root}: {sorted(unexpected_chunks)}. "
                f"Active metadata declares: {sorted(declared_set)}. Untracked chunks risk data contamination."
            )

    lineage_metadata: dict[str, Any] = {"root": str(root)}
    if unexpected_chunks:
        lineage_metadata["unexpected_chunks"] = sorted(unexpected_chunks)
    if missing_dict_fp:
        lineage_metadata["is_unverified_dictionary"] = True

    if not verify_chunks:
        return DatasetLineage(
            build_id=build_id,
            dictionary_fingerprint=dictionary_fingerprint,
            dataset_fingerprint=declared_fingerprint,
            split_counts=split_counts,
            split_hashes={
                "train": precomputed_train_hash,
                "valid": precomputed_valid_hash,
                "test": precomputed_test_hash,
            },
            train_chunks=train_chunks,
            valid_chunks=valid_chunks,
            test_chunks=test_chunks,
            chunk_checksums=declared_checksums,
            format=format_type,
            schema_version=schema_version,
            is_verified=False,
            metadata=lineage_metadata,
        )

    # Full verification: compute and check chunk checksums, sample IDs, leakage, and fingerprint
    actual_chunk_checksums: dict[str, str] = {}
    split_sample_ids: dict[str, list[str]] = {"train": [], "valid": [], "test": []}

    for split_name, chunk_names in chunk_map.items():
        for chunk_name in chunk_names:
            chunk_file = root / chunk_name
            actual_checksum = compute_chunk_checksum(chunk_file)
            actual_chunk_checksums[chunk_name] = actual_checksum

            if declared_checksums and chunk_name in declared_checksums:
                if actual_checksum != declared_checksums[chunk_name]:
                    raise ValueError(
                        f"Chunk checksum mismatch for {chunk_name}: declared {declared_checksums[chunk_name]}, "
                        f"actual {actual_checksum}. Chunk contents have been modified!"
                    )

            with np.load(chunk_file, allow_pickle=True) as cdata:
                if "sample_ids" in cdata:
                    raw_ids = cdata["sample_ids"]
                    ids = [str(sid).strip() for sid in raw_ids]
                else:
                    if schema_version >= 2:
                        raise ValueError(
                            f"Chunk {chunk_file} lacks required sample_ids in schema version {schema_version}"
                        )
                    chunk_len = len(cdata["x"]) if "x" in cdata else 0
                    ids = [f"legacy_generated:{chunk_file.stem}:{idx}" for idx in range(chunk_len)]

                split_sample_ids[split_name].extend(ids)

    # Split integrity: detect empty IDs and duplicate IDs within each split
    split_hashes = {
        "train": compute_split_hash(split_sample_ids["train"]),
        "valid": compute_split_hash(split_sample_ids["valid"]),
        "test": compute_split_hash(split_sample_ids["test"]),
    }

    # Data leakage check: Reject any split overlap across different splits
    train_set = set(split_sample_ids["train"])
    valid_set = set(split_sample_ids["valid"])
    test_set = set(split_sample_ids["test"])

    tv_overlap = train_set & valid_set
    tt_overlap = train_set & test_set
    vt_overlap = valid_set & test_set
    if tv_overlap or tt_overlap or vt_overlap:
        all_leaks = sorted(tv_overlap | tt_overlap | vt_overlap)
        raise ValueError(
            f"Data leakage detected: sample IDs overlap across splits. "
            f"Overlapping IDs ({len(all_leaks)}): {all_leaks[:10]}"
        )

    # Verify split counts match actual sample counts exactly (distinguishing zero from missing)
    for split_name in ("train", "valid", "test"):
        actual_count = len(split_sample_ids[split_name])
        recorded_count = split_counts[split_name]
        if actual_count != recorded_count:
            raise ValueError(
                f"Split sample count mismatch for {split_name}: "
                f"meta.npz records {recorded_count}, but chunks contain {actual_count}"
            )

    recomputed_fingerprint = compute_dataset_fingerprint(
        build_id=build_id,
        dictionary_fingerprint=dictionary_fingerprint,
        split_hashes=split_hashes,
        split_counts=split_counts,
        train_chunks=train_chunks,
        valid_chunks=valid_chunks,
        test_chunks=test_chunks,
        chunk_checksums=actual_chunk_checksums if declared_checksums else None,
        format_type=format_type,
        schema_version=schema_version,
    )

    if declared_fingerprint != recomputed_fingerprint:
        raise ValueError(
            f"Dataset fingerprint mismatch: meta.npz declared {declared_fingerprint}, "
            f"but recomputed verified fingerprint is {recomputed_fingerprint}"
        )

    return DatasetLineage(
        build_id=build_id,
        dictionary_fingerprint=dictionary_fingerprint,
        dataset_fingerprint=recomputed_fingerprint,
        split_counts=split_counts,
        split_hashes=split_hashes,
        train_chunks=train_chunks,
        valid_chunks=valid_chunks,
        test_chunks=test_chunks,
        chunk_checksums=actual_chunk_checksums,
        format=format_type,
        schema_version=schema_version,
        is_verified=not missing_dict_fp,
        metadata=lineage_metadata,
    )


def query_sample_membership(
    sample_id: str,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Query whether a sample ID was included in training, validation, or test.

    Returns dict with sample_id, split (or None if not found), chunk name, and build_id.
    Raises FileNotFoundError if any declared chunk is missing.
    Raises ValueError if sample ID appears in multiple splits (data leakage).
    """
    root = Path(dataset_root)
    target_id = str(sample_id).strip()
    if not target_id:
        raise ValueError("sample_id must not be empty or whitespace-only")

    lineage = load_dataset_lineage(root, verify_chunks=False)

    chunk_map = {
        "train": lineage.train_chunks,
        "valid": lineage.valid_chunks,
        "test": lineage.test_chunks,
    }

    matches: list[dict[str, Any]] = []

    for split_name, chunk_names in chunk_map.items():
        for chunk_name in chunk_names:
            chunk_file = root / chunk_name
            if not chunk_file.is_file():
                raise FileNotFoundError(f"Declared dataset chunk missing on disk: {chunk_file}")

            with np.load(chunk_file, allow_pickle=True) as cdata:
                if "sample_ids" in cdata:
                    raw_ids = [str(sid).strip() for sid in cdata["sample_ids"]]
                    for local_idx, sid in enumerate(raw_ids):
                        if sid == target_id:
                            matches.append(
                                {
                                    "split": split_name,
                                    "chunk": chunk_name,
                                    "chunk_file": chunk_name,
                                    "local_index": local_idx,
                                    "chunk_index": local_idx,
                                    "is_synthetic_id": sid.startswith("legacy_generated:"),
                                }
                            )
                else:
                    # Unindexed chunk fallback
                    chunk_len = len(cdata["x"]) if "x" in cdata else 0
                    for local_idx in range(chunk_len):
                        synthetic_id = f"legacy_generated:{chunk_file.stem}:{local_idx}"
                        if synthetic_id == target_id:
                            matches.append(
                                {
                                    "split": split_name,
                                    "chunk": chunk_name,
                                    "chunk_file": chunk_name,
                                    "local_index": local_idx,
                                    "chunk_index": local_idx,
                                    "is_synthetic_id": True,
                                }
                            )

    if not matches:
        return {
            "sample_id": target_id,
            "split": None,
            "chunk": None,
            "chunk_file": None,
            "local_index": None,
            "chunk_index": None,
            "is_synthetic_id": False,
            "build_id": lineage.build_id,
            "dataset_fingerprint": lineage.dataset_fingerprint,
        }

    # Verify if sample was found in multiple different splits (data leakage)
    splits_found = set(m["split"] for m in matches)
    if len(splits_found) > 1:
        raise ValueError(
            f"Data leakage detected for sample '{target_id}': found in multiple splits: {sorted(splits_found)}"
        )

    first = matches[0]
    return {
        "sample_id": target_id,
        "split": first["split"],
        "chunk": first["chunk"],
        "chunk_file": first["chunk_file"],
        "local_index": first["local_index"],
        "chunk_index": first["chunk_index"],
        "is_synthetic_id": first["is_synthetic_id"],
        "total_occurrences": len(matches),
        "build_id": lineage.build_id,
        "dataset_fingerprint": lineage.dataset_fingerprint,
    }
