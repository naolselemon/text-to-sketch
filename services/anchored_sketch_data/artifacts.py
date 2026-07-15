"""Atomic, content-addressed anchored V3 dataset artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .contract import (
    FORMAT_TYPE,
    FORMAT_VERSION,
    PRODUCTION_MIN_SOURCE_SKETCHES,
    TOKEN_LAYOUT,
    artifact_contract,
)
from .grammar import validate_tokens
from .tokenizer import validate_codebook


@dataclass(frozen=True)
class EncodedSample:
    sample_id: str
    source_path: str
    source_sha256: str
    perceptual_hash: int
    group_id: str
    split: str
    point_count: int
    stroke_count: int
    tokens: np.ndarray
    preprocessing: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RejectedSample:
    sample_id: str
    source_path: str
    source_sha256: str
    perceptual_hash: int | None
    rejection_reason: str
    preprocessing: dict[str, Any] = field(default_factory=dict)
    group_id: str | None = None
    split: str | None = None
    point_count: int | None = None
    stroke_count: int | None = None
    token_length: int | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_dataset_atomic(
    output_root: str | Path,
    samples: Iterable[EncodedSample],
    codebook: np.ndarray,
    *,
    preparation: dict[str, Any],
    rejected: Iterable[RejectedSample] = (),
    shard_size: int = 1024,
) -> Path:
    """Write a complete dataset then atomically publish its hash-named directory."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    centers = validate_codebook(codebook)
    accepted = sorted(samples, key=lambda item: (item.split, item.sample_id))
    rejected_values = sorted(rejected, key=lambda item: item.sample_id)
    _validate_samples_before_write(accepted)

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_conflicting_current_path(root)
    stale = sorted(root.glob(".anchored_v3-*.tmp"))
    if stale:
        raise FileExistsError(
            f"Refusing build while stale temporary target exists: {stale[0]}"
        )

    codebook_digest = hashlib.sha256(centers.tobytes(order="C")).hexdigest()
    content_records = [
        {
            "sample_id": sample.sample_id,
            "source_sha256": sample.source_sha256,
            "group_id": sample.group_id,
            "split": sample.split,
            "tokens_sha256": hashlib.sha256(
                np.asarray(sample.tokens, dtype=np.int32).tobytes(order="C")
            ).hexdigest(),
            "preprocessing": sample.preprocessing,
        }
        for sample in accepted
    ]
    content_digest = sha256_json(
        {
            "contract": artifact_contract(),
            "codebook_sha256": codebook_digest,
            "preparation": preparation,
            "accepted": content_records,
            "rejected": [asdict(item) for item in rejected_values],
        }
    )
    final_path = root / f"{FORMAT_TYPE}-{content_digest[:16]}"
    if final_path.exists():
        if any(final_path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty target {final_path}")
        final_path.rmdir()

    temporary = Path(
        tempfile.mkdtemp(prefix=".anchored_v3-", suffix=".tmp", dir=str(root))
    )
    try:
        np.save(temporary / "codebook.npy", centers, allow_pickle=False)
        written_codebook_digest = sha256_file(temporary / "codebook.npy")
        manifest_entries = _write_shards(
            temporary, accepted, shard_size=shard_size
        )
        manifest_entries.extend(_rejected_manifest_entry(item) for item in rejected_values)
        manifest_entries.sort(key=lambda item: (item["sample_id"], item["status"]))
        manifest_path = temporary / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for entry in manifest_entries:
                handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        manifest_digest = sha256_file(manifest_path)
        metadata = {
            **artifact_contract(),
            "dataset_content_sha256": content_digest,
            "codebook_sha256": written_codebook_digest,
            "codebook_values_sha256": codebook_digest,
            "manifest_sha256": manifest_digest,
            "preparation": preparation,
            "split_counts": {
                split: sum(sample.split == split for sample in accepted)
                for split in ("train", "valid", "test")
            },
            "rejected_count": len(rejected_values),
        }
        metadata["metadata_payload_sha256"] = sha256_json(metadata)
        (temporary / "meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_dataset(temporary)
        os.replace(temporary, final_path)
        _replace_current_symlink(root, final_path)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return final_path


def _refuse_conflicting_current_path(output_root: Path) -> None:
    current = output_root / "current"
    if os.path.lexists(current) and not current.is_symlink():
        raise FileExistsError(
            f"Refusing to replace conflicting real path at {current}; "
            "current must be absent or a symbolic link"
        )


def _replace_current_symlink(output_root: Path, dataset_path: Path) -> None:
    """Atomically point ``current`` at a sibling content-addressed dataset."""

    _refuse_conflicting_current_path(output_root)
    current = output_root / "current"
    temporary = output_root / f".current-{uuid.uuid4().hex}.tmp"
    try:
        os.symlink(dataset_path.name, temporary, target_is_directory=True)
        os.replace(temporary, current)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def validate_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Validate hashes, grammar, shard membership, and split isolation."""

    root = Path(dataset_dir)
    required = (root / "meta.json", root / "manifest.jsonl", root / "codebook.npy")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Dataset is missing required artifacts: {', '.join(missing)}")
    metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    metadata_without_hash = dict(metadata)
    metadata_digest = metadata_without_hash.pop("metadata_payload_sha256", None)
    if metadata_digest != sha256_json(metadata_without_hash):
        raise ValueError("Metadata payload SHA-256 mismatch")
    contract = artifact_contract()
    for key, expected in contract.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Metadata contract mismatch for {key}")
    if metadata.get("format_type") != FORMAT_TYPE or metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("Not an anchored V3 dataset")
    if sha256_file(root / "codebook.npy") != metadata.get("codebook_sha256"):
        raise ValueError("Codebook SHA-256 mismatch")
    codebook = np.load(root / "codebook.npy", allow_pickle=False)
    validate_codebook(codebook)
    if sha256_file(root / "manifest.jsonl") != metadata.get("manifest_sha256"):
        raise ValueError("Manifest SHA-256 mismatch")

    entries: list[dict[str, Any]] = []
    with (root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid manifest JSON on line {line_number}") from exc
    ids = [entry.get("sample_id") for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest sample IDs overlap")

    group_split: dict[str, str] = {}
    source_split: dict[str, str] = {}
    accepted = [entry for entry in entries if entry.get("status") == "accepted"]
    for entry in accepted:
        split = entry.get("split")
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split {split!r} in manifest")
        for field_name, seen in (
            ("group_id", group_split),
            ("source_sha256", source_split),
        ):
            value = str(entry.get(field_name, ""))
            previous = seen.setdefault(value, split)
            if previous != split:
                raise ValueError(f"{field_name} appears in more than one split")

    expected_shards = {str(entry["shard"]) for entry in accepted}
    actual_shards = {
        path.relative_to(root).as_posix() for path in (root / "shards").glob("*.npz")
    } if (root / "shards").is_dir() else set()
    if actual_shards != expected_shards:
        raise ValueError("Shard files do not exactly match the manifest")

    by_shard: dict[str, list[dict[str, Any]]] = {}
    for entry in accepted:
        by_shard.setdefault(str(entry["shard"]), []).append(entry)
    for relative_path, shard_entries in by_shard.items():
        path = root / relative_path
        expected_hashes = {str(entry["shard_sha256"]) for entry in shard_entries}
        if len(expected_hashes) != 1 or sha256_file(path) not in expected_hashes:
            raise ValueError(f"Shard SHA-256 mismatch for {relative_path}")
        with np.load(path, allow_pickle=False) as shard:
            sample_ids = shard["sample_ids"].astype(str).tolist()
            tokens = np.asarray(shard["tokens"], dtype=np.int32)
            offsets = np.asarray(shard["offsets"], dtype=np.int64)
        if len(offsets) != len(sample_ids) + 1 or offsets[0] != 0 or offsets[-1] != len(tokens):
            raise ValueError(f"Invalid token offsets in {relative_path}")
        entries_by_index = {int(entry["shard_index"]): entry for entry in shard_entries}
        if set(entries_by_index) != set(range(len(sample_ids))):
            raise ValueError(f"Shard indices are incomplete in {relative_path}")
        for index, sample_id in enumerate(sample_ids):
            entry = entries_by_index[index]
            if entry["sample_id"] != sample_id:
                raise ValueError(f"Shard sample order mismatch in {relative_path}")
            sequence = tokens[offsets[index] : offsets[index + 1]]
            if len(sequence) != int(entry["token_length"]):
                raise ValueError(f"Token length mismatch for {sample_id}")
            summary = validate_tokens(sequence)
            if summary.stroke_count != int(entry["stroke_count"]):
                raise ValueError(f"Stroke count mismatch for {sample_id}")

    split_counts = {
        split: sum(entry.get("split") == split for entry in accepted)
        for split in ("train", "valid", "test")
    }
    if metadata.get("split_counts") != split_counts:
        raise ValueError("Metadata split counts do not match manifest")
    return metadata


def require_minimum_source_sketches(
    metadata: Mapping[str, Any],
) -> int:
    """Enforce the immutable production-data floor."""

    preparation = metadata.get("preparation")
    if not isinstance(preparation, Mapping):
        raise ValueError("Anchored V3 metadata has no preparation contract")
    count = int(preparation.get("accepted_source_sketches", -1))
    if count < PRODUCTION_MIN_SOURCE_SKETCHES:
        raise ValueError(
            "Anchored V3 training/evaluation requires at least "
            f"{PRODUCTION_MIN_SOURCE_SKETCHES} cleaned "
            f"source sketches; artifact contains {count}"
        )
    return count


def _validate_samples_before_write(samples: list[EncodedSample]) -> None:
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Encoded sample IDs must be unique")
    group_splits: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    for sample in samples:
        if sample.split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split {sample.split!r} for {sample.sample_id}")
        summary = validate_tokens(sample.tokens)
        if summary.stroke_count != int(sample.stroke_count):
            raise ValueError(f"Stroke count disagrees with tokens for {sample.sample_id}")
        for value, seen, name in (
            (sample.group_id, group_splits, "group_id"),
            (sample.source_sha256, source_splits, "source_sha256"),
        ):
            previous = seen.setdefault(value, sample.split)
            if previous != sample.split:
                raise ValueError(f"{name} appears in more than one split")


def _write_shards(
    root: Path,
    samples: list[EncodedSample],
    *,
    shard_size: int,
) -> list[dict[str, Any]]:
    shard_dir = root / "shards"
    shard_dir.mkdir()
    entries: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        split_samples = [sample for sample in samples if sample.split == split]
        for chunk_index, start in enumerate(range(0, len(split_samples), shard_size)):
            chunk = split_samples[start : start + shard_size]
            arrays = [np.asarray(sample.tokens, dtype=np.int32) for sample in chunk]
            offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
            if arrays:
                offsets[1:] = np.cumsum([len(array) for array in arrays])
                flat = np.concatenate(arrays)
            else:
                flat = np.empty(0, dtype=np.int32)
            relative = Path("shards") / f"{split}-{chunk_index:05d}.npz"
            path = root / relative
            np.savez_compressed(
                path,
                sample_ids=np.asarray([sample.sample_id for sample in chunk]),
                tokens=flat,
                offsets=offsets,
            )
            shard_digest = sha256_file(path)
            for shard_index, sample in enumerate(chunk):
                entries.append(
                    {
                        "schema_version": FORMAT_VERSION,
                        "status": "accepted",
                        "sample_id": sample.sample_id,
                        "source_path": sample.source_path,
                        "source_sha256": sample.source_sha256,
                        "perceptual_hash": f"{int(sample.perceptual_hash):016x}",
                        "group_id": sample.group_id,
                        "split": sample.split,
                        "point_count": int(sample.point_count),
                        "stroke_count": int(sample.stroke_count),
                        "token_length": int(len(sample.tokens)),
                        "preprocessing": sample.preprocessing,
                        "shard": relative.as_posix(),
                        "shard_index": shard_index,
                        "shard_sha256": shard_digest,
                    }
                )
    return entries


def _rejected_manifest_entry(sample: RejectedSample) -> dict[str, Any]:
    return {
        "schema_version": FORMAT_VERSION,
        "status": "rejected",
        "sample_id": sample.sample_id,
        "source_path": sample.source_path,
        "source_sha256": sample.source_sha256,
        "perceptual_hash": (
            None if sample.perceptual_hash is None else f"{int(sample.perceptual_hash):016x}"
        ),
        "group_id": sample.group_id,
        "split": sample.split,
        "point_count": sample.point_count,
        "stroke_count": sample.stroke_count,
        "token_length": sample.token_length,
        "rejection_reason": sample.rejection_reason,
        "preprocessing": sample.preprocessing,
    }
