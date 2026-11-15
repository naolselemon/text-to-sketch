"""Prepare tok-dict chunks for native Sketchformer fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dataloaders.lineage import (
    compute_chunk_checksum,
    compute_dataset_fingerprint,
    compute_split_hash,
)
from prep_data.prepare_sketchformer import build_splits, chunk_sketch_data
from prep_data.sketch_token.create_token_dict import (
    compute_codebook_artifact_checksum,
    get_token_dict_fingerprint,
    load_codebook_from_dir,
)
from utils.io import load_token_sequence_with_metadata
from utils.paths import DEFAULT_SKETCH_TOKEN_DIR, DEFAULT_STROKE5_DIR, PROCESSED_DATA_DIR
from utils.tokenizer import encode_stroke5


def load_stroke5_file(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if len(data.files) != 1:
        raise ValueError(f"Expected one array in {path}, found {data.files}")
    return np.asarray(data[data.files[0]], dtype=np.float32)


def load_token_file(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "tokens" in data.files:
        values = data["tokens"]
    elif len(data.files) == 1:
        values = data[data.files[0]]
    else:
        raise ValueError(f"Expected a tokens array in {path}, found {data.files}")
    tokens = np.asarray(values, dtype=np.int64)
    if tokens.ndim != 1:
        raise ValueError(f"Expected one-dimensional tokens in {path}, got {tokens.shape}")
    return tokens


def truncate_tokens(
    tokens: np.ndarray,
    *,
    max_length: int | None,
    sep_token_id: int,
    eos_token_id: int,
) -> np.ndarray:
    sequence = np.asarray(tokens, dtype=np.int64)
    if max_length is None or len(sequence) <= max_length:
        return sequence
    if max_length <= 0:
        raise ValueError("--max-length must be positive")

    truncated = np.array(sequence[:max_length], copy=True, dtype=np.int64)
    if max_length == 1:
        truncated[-1] = int(eos_token_id)
    else:
        truncated[-2:] = [int(sep_token_id), int(eos_token_id)]
    return truncated


def save_chunk(
    path: Path,
    token_sequences: list[np.ndarray],
    labels: list[int],
    sample_ids: list[str] | None = None,
) -> None:
    payload = {
        "x": np.asarray(token_sequences, dtype=object),
        "y": np.asarray(labels, dtype=np.int32),
        "token_lengths": np.asarray([len(tokens) for tokens in token_sequences], dtype=np.int32),
    }
    if sample_ids is not None:
        payload["sample_ids"] = np.asarray(sample_ids, dtype=object)
    np.savez_compressed(
        path,
        **payload,
    )


def _snapshot_previous_build(target_path: Path) -> None:
    meta_path = target_path / "meta.npz"
    if not meta_path.is_file():
        return
    prev_build_id: str | None = None
    chunk_names: set[str] = set()
    try:
        with np.load(meta_path, allow_pickle=True) as prev_meta:
            if "build_id" in prev_meta:
                prev_build_id = str(prev_meta["build_id"])
            if "train_chunks" in prev_meta:
                chunk_names.update(str(c) for c in prev_meta["train_chunks"])
            if "valid_chunks" in prev_meta:
                chunk_names.update(str(c) for c in prev_meta["valid_chunks"])
            if "test_chunks" in prev_meta:
                chunk_names.update(str(c) for c in prev_meta["test_chunks"])
    except Exception:
        pass

    if not prev_build_id:
        prev_build_id = f"prev_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    safe_prev_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", prev_build_id)
    versions_dir = target_path / "versions" / safe_prev_id
    versions_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(meta_path, versions_dir / "meta.npz")
    report_path = target_path / "preparation_report.json"
    if report_path.is_file():
        shutil.copy2(report_path, versions_dir / "preparation_report.json")

    if not chunk_names:
        for p in target_path.glob("train_*.npz"):
            chunk_names.add(p.name)
        for p in target_path.glob("valid*.npz"):
            chunk_names.add(p.name)
        for p in target_path.glob("test*.npz"):
            chunk_names.add(p.name)

    for name in chunk_names:
        src = target_path / name
        if src.is_file():
            shutil.copy2(src, versions_dir / Path(name).name)


def discover_manifest(source_path: Path, manifest_arg: str | Path | None = None) -> Path | None:
    if manifest_arg:
        p = Path(manifest_arg)
        if not p.is_file():
            raise FileNotFoundError(f"Specified manifest file not found: {p}")
        return p
    candidates = [
        source_path / "preprocessing_manifest.jsonl",
        source_path.parent / "preprocessing_manifest.jsonl",
        source_path / "preprocessing_manifest.json",
        source_path.parent / "preprocessing_manifest.json",
        source_path / "manifest.jsonl",
        source_path.parent / "manifest.jsonl",
        source_path / "manifest.json",
        source_path.parent / "manifest.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    content = manifest_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    if content.startswith("["):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                records = parsed
        except json.JSONDecodeError:
            pass
    if not records:
        for line_no, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Invalid JSON on line {line_no} in manifest {manifest_path}: {err}"
                ) from err

    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for r in records:
        sid = r.get("sample_id") or r.get("drawing_id") or r.get("id")
        if sid is not None:
            sid_str = str(sid)
            if sid_str in seen_ids:
                duplicate_ids.append(sid_str)
            seen_ids.add(sid_str)
    if duplicate_ids:
        raise ValueError(f"Duplicate drawing IDs detected in manifest: {sorted(set(duplicate_ids))}")

    return records


def _resolve_sample_file(source_path: Path, sample_id: str, record: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    if record.get("tokens_path"):
        cand = Path(record["tokens_path"])
        candidates.append(cand)
        if not cand.is_absolute():
            candidates.append(source_path / cand)
    candidates.append(source_path / f"{sample_id}.npz")
    candidates.append(source_path / sample_id)
    for c in candidates:
        if c.is_file():
            return c
    return source_path / f"{sample_id}.npz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare anime stroke5 data as tok-dict Sketchformer chunks."
    )
    parser.add_argument("--source-dir", default=DEFAULT_STROKE5_DIR)
    parser.add_argument(
        "--tokens-dir",
        default=None,
        help="Use pre-encoded token .npz files instead of re-encoding stroke5.",
    )
    parser.add_argument(
        "--token-dict-dir",
        default=DEFAULT_SKETCH_TOKEN_DIR,
        help="Directory containing codebook.npy and metadata.json.",
    )
    parser.add_argument(
        "--target-dir",
        default=PROCESSED_DATA_DIR / "sketchformer-ready-data" / "tok-dict",
    )
    parser.add_argument("--n-chunks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--overlength-policy",
        choices=("error", "reject", "truncate"),
        default=None,
        help=(
            "error/reject preserve complete sketches. Defaults to error for "
            "--tokens-dir and truncate for legacy stroke5 input."
        ),
    )
    parser.add_argument(
        "--error-feedback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use cumulative-error-correcting quantization for stroke5 input.",
    )
    parser.add_argument("--n-classes", type=int, default=1)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest JSONL from preprocessing run.",
    )
    parser.add_argument(
        "--build-id",
        default=None,
        help="Build ID for dataset lineage tracking and verification.",
    )
    parser.add_argument(
        "--reject-unexpected-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject any .npz files on disk not recorded in the run manifest.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_chunks <= 0:
        raise ValueError("--n-chunks must be positive")

    source_path = Path(args.tokens_dir) if args.tokens_dir else Path(args.source_dir)
    source_type = "preencoded_tokens" if args.tokens_dir else "stroke5"
    overlength_policy = args.overlength_policy or (
        "error" if args.tokens_dir else "truncate"
    )
    error_feedback = True if args.tokens_dir else bool(args.error_feedback)
    target_path = Path(args.target_dir)
    token_dict_dir = Path(args.token_dict_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    if args.build_id:
        target_meta = target_path / "meta.npz"
        if target_meta.is_file():
            try:
                with np.load(target_meta, allow_pickle=True) as existing_meta:
                    existing_bid = str(existing_meta.get("build_id", "")).strip()
                    if existing_bid and existing_bid == str(args.build_id).strip():
                        raise ValueError(
                            f"Dataset build '{args.build_id}' already exists at {target_path}. "
                            "Dataset builds are immutable; to create a new build, use a distinct build ID."
                        )
            except ValueError:
                raise
            except Exception:
                pass
        safe_bid = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(args.build_id).strip())
        if (target_path / "versions" / safe_bid).is_dir():
            raise ValueError(
                f"Dataset build '{args.build_id}' already exists in versions archive at {target_path / 'versions' / safe_bid}. "
                "Dataset builds are immutable; to create a new build, use a distinct build ID."
            )

    codebook, codebook_metadata = load_codebook_from_dir(token_dict_dir)
    current_dict_fingerprint = get_token_dict_fingerprint(token_dict_dir)
    current_codebook_checksum = compute_codebook_artifact_checksum(codebook)
    codebook_size = int(len(codebook))
    motion_token_offset = 1
    pad_token_id = 0
    sep_token_id = codebook_size + 1
    sos_token_id = codebook_size + 2
    eos_token_id = codebook_size + 3
    vocab_size = codebook_size + 4

    manifest_path = discover_manifest(source_path, args.manifest)
    manifest_records: list[dict[str, Any]] | None = None
    if manifest_path is not None:
        manifest_records = load_manifest_records(manifest_path)

    disk_files = sorted(source_path.rglob("*.npz")) if source_path.exists() else []

    accepted_build_ids: set[str] = set()
    if manifest_records is not None:
        for r in manifest_records:
            status = r.get("status")
            is_accepted = (status == "accepted") or (r.get("accepted") is True)
            if is_accepted:
                bid = r.get("build_id")
                if bid is not None and str(bid).strip():
                    accepted_build_ids.add(str(bid).strip())

        if len(accepted_build_ids) > 1:
            raise ValueError(
                f"Multiple distinct build IDs detected among accepted records in manifest: "
                f"{sorted(accepted_build_ids)}. All accepted records must share a single consistent build_id."
            )

        if args.build_id:
            for r in manifest_records:
                status = r.get("status")
                is_accepted = (status == "accepted") or (r.get("accepted") is True)
                if is_accepted:
                    bid = r.get("build_id")
                    if bid is not None and str(bid).strip() and str(bid).strip() != str(args.build_id):
                        sid = r.get("sample_id") or r.get("drawing_id") or r.get("id")
                        raise ValueError(
                            f"Build ID mismatch for accepted sample '{sid}': "
                            f"manifest recorded '{bid}', expected '{args.build_id}'"
                        )

        manifest_by_id: dict[str, dict[str, Any]] = {}
        for r in manifest_records:
            sid = r.get("sample_id") or r.get("drawing_id") or r.get("id")
            if sid is not None:
                manifest_by_id[str(sid)] = r

        if args.tokens_dir and args.reject_unexpected_files:
            for disk_file in disk_files:
                sample_id = disk_file.relative_to(source_path).with_suffix("").as_posix()
                if sample_id not in manifest_by_id and disk_file.stem not in manifest_by_id:
                    raise ValueError(
                        f"Unexpected untracked token file found on disk not listed in manifest: {disk_file}"
                    )

        items_to_process: list[tuple[Path, str, dict[str, Any]]] = []
        for r in manifest_records:
            status = r.get("status")
            is_accepted = (status == "accepted") or (r.get("accepted") is True)
            if not is_accepted:
                continue

            sid = str(r.get("sample_id") or r.get("drawing_id") or r.get("id"))
            expected_file = _resolve_sample_file(source_path, sid, r)
            if not expected_file.is_file():
                raise FileNotFoundError(
                    f"Missing token file for accepted sample '{sid}': {expected_file}"
                )
            items_to_process.append((expected_file, sid, r))
    else:
        if args.tokens_dir and args.reject_unexpected_files:
            raise ValueError(
                f"No manifest found in or around {source_path} and --reject-unexpected-files is enabled. "
                "A manifest is required to prevent data leakage."
            )
        if not disk_files:
            raise FileNotFoundError(f"No .npz files found in {source_path}")
        items_to_process = [
            (f, f.relative_to(source_path).with_suffix("").as_posix(), {})
            for f in disk_files
        ]

    prepared: list[tuple[np.ndarray, str]] = []
    rejected: list[dict[str, object]] = []
    for path, sample_id, record in items_to_process:
        expected_tokens_sha256 = record.get("tokens_sha256") or record.get("token_sha256")
        if expected_tokens_sha256:
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha256 != str(expected_tokens_sha256):
                raise ValueError(
                    f"SHA-256 hash mismatch for token file {path}: "
                    f"expected {expected_tokens_sha256}, got {actual_sha256}"
                )

        rec_dict_sha256 = (
            record.get("token_dictionary_sha256")
            or record.get("token_dict_sha256")
            or record.get("codebook_sha256")
        )
        if rec_dict_sha256 and (current_dict_fingerprint or current_codebook_checksum):
            val_str = str(rec_dict_sha256).strip()
            if val_str != current_dict_fingerprint and val_str != current_codebook_checksum:
                raise ValueError(
                    f"Token dictionary fingerprint mismatch in manifest record for {sample_id}: "
                    f"manifest recorded {rec_dict_sha256}, current codebook is {current_dict_fingerprint}"
                )

        if args.build_id and record.get("build_id"):
            if str(record["build_id"]) != str(args.build_id):
                raise ValueError(
                    f"Build ID mismatch for sample {sample_id}: "
                    f"manifest recorded {record['build_id']}, expected {args.build_id}"
                )

        if args.tokens_dir:
            tokens, embedded_meta = load_token_sequence_with_metadata(path)
            if embedded_meta:
                if "tokens_sha256" in embedded_meta:
                    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                    if actual_sha256 != str(embedded_meta["tokens_sha256"]):
                        raise ValueError(
                            f"SHA-256 hash mismatch in embedded metadata for {path}: "
                            f"embedded {embedded_meta['tokens_sha256']}, actual {actual_sha256}"
                        )
                if "token_dict_sha256" in embedded_meta and (current_dict_fingerprint or current_codebook_checksum):
                    val_str = str(embedded_meta["token_dict_sha256"]).strip()
                    if val_str != current_dict_fingerprint and val_str != current_codebook_checksum:
                        raise ValueError(
                            f"Token dictionary fingerprint mismatch in token file {path}: "
                            f"embedded {embedded_meta['token_dict_sha256']}, current codebook is {current_dict_fingerprint}"
                        )
                expected_build = args.build_id or record.get("build_id")
                if "build_id" in embedded_meta and expected_build:
                    if str(embedded_meta["build_id"]) != str(expected_build):
                        raise ValueError(
                            f"Build ID mismatch in token file {path}: "
                            f"embedded {embedded_meta['build_id']}, expected {expected_build}"
                        )
        else:
            tokens = encode_stroke5(
                load_stroke5_file(path),
                codebook,
                error_feedback=error_feedback,
            )

        if len(tokens) < 2 or int(tokens[0]) != sos_token_id or int(tokens[-1]) != eos_token_id:
            raise ValueError(f"{sample_id} must start with SOS and end with EOS")
        if int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size:
            raise ValueError(f"{sample_id} contains token IDs outside [0, {vocab_size})")
        if args.max_length is not None and len(tokens) > args.max_length:
            rejection = {
                "sample_id": sample_id,
                "source_path": str(path),
                "token_length": int(len(tokens)),
                "max_length": int(args.max_length),
                "reason": "overlength",
            }
            rejected.append(rejection)
            if overlength_policy == "error":
                raise ValueError(
                    f"{sample_id} has {len(tokens)} tokens, exceeding "
                    f"max_length={args.max_length}; preprocessing refuses to truncate"
                )
            if overlength_policy == "reject":
                continue
            tokens = truncate_tokens(
                tokens,
                max_length=args.max_length,
                sep_token_id=sep_token_id,
                eos_token_id=eos_token_id,
            )
        prepared.append((tokens, sample_id))

    if not prepared:
        raise ValueError("No token sequences remained after overlength filtering")

    train_records, valid_records, test_records = build_splits(
        prepared,
        seed=args.seed,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
    )

    train_tokens = [tokens for tokens, _ in train_records]
    valid_tokens = [tokens for tokens, _ in valid_records]
    test_tokens = [tokens for tokens, _ in test_records]
    train_ids = [sample_id for _, sample_id in train_records]
    valid_ids = [sample_id for _, sample_id in valid_records]
    test_ids = [sample_id for _, sample_id in test_records]

    train_labels = [0] * len(train_tokens)
    valid_labels = [0] * len(valid_tokens)
    test_labels = [0] * len(test_tokens)

    resolved_build_id = args.build_id
    if not resolved_build_id and accepted_build_ids:
        resolved_build_id = next(iter(accepted_build_ids))
    if not resolved_build_id:
        resolved_build_id = f"build_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    safe_build_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", resolved_build_id)

    # Immutability check: Reject reusing an existing build ID
    meta_path = target_path / "meta.npz"
    if meta_path.is_file():
        try:
            with np.load(meta_path, allow_pickle=True) as existing_meta:
                existing_bid = str(existing_meta.get("build_id", "")).strip()
                if existing_bid and existing_bid == resolved_build_id:
                    raise ValueError(
                        f"Dataset build '{resolved_build_id}' already exists at {target_path}. "
                        "Dataset builds are immutable; to create a new build, use a distinct build ID."
                    )
        except ValueError:
            raise
        except Exception:
            pass

    existing_version_dir = target_path / "versions" / safe_build_id
    if existing_version_dir.is_dir():
        raise ValueError(
            f"Dataset build '{resolved_build_id}' already exists in versions archive at {existing_version_dir}. "
            "Dataset builds are immutable; to create a new build, use a distinct build ID."
        )

    staging_dir = target_path / f".staging_{safe_build_id}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        train_chunks = chunk_sketch_data(
            list(zip(train_tokens, train_ids)),
            train_labels,
            args.n_chunks,
        )
        train_chunk_names: list[str] = []
        train_chunk_counts: list[int] = []
        for chunk_idx, (chunk_records, chunk_labels) in enumerate(train_chunks):
            chunk_tokens = [tokens for tokens, _ in chunk_records]
            chunk_ids = [sample_id for _, sample_id in chunk_records]
            chunk_name = f"train_{chunk_idx:03}_{safe_build_id}.npz"
            train_chunk_names.append(chunk_name)
            train_chunk_counts.append(len(chunk_tokens))
            save_chunk(
                staging_dir / chunk_name,
                chunk_tokens,
                chunk_labels.tolist(),
                chunk_ids,
            )

        valid_chunk_name = f"valid_{safe_build_id}.npz"
        test_chunk_name = f"test_{safe_build_id}.npz"
        save_chunk(staging_dir / valid_chunk_name, valid_tokens, valid_labels, valid_ids)
        save_chunk(staging_dir / test_chunk_name, test_tokens, test_labels, test_ids)

        for chunk_name, expected_count in zip(train_chunk_names, train_chunk_counts):
            staged_file = staging_dir / chunk_name
            if not staged_file.is_file():
                raise RuntimeError(f"Failed to generate staged chunk: {staged_file}")
            with np.load(staged_file, allow_pickle=True) as check_data:
                if len(check_data["x"]) != expected_count:
                    raise RuntimeError(
                        f"Staged chunk count mismatch in {staged_file}: expected {expected_count}, got {len(check_data['x'])}"
                    )

        for name, expected_count in [(valid_chunk_name, len(valid_tokens)), (test_chunk_name, len(test_tokens))]:
            staged_file = staging_dir / name
            if not staged_file.is_file():
                raise RuntimeError(f"Failed to generate staged chunk: {staged_file}")
            with np.load(staged_file, allow_pickle=True) as check_data:
                if len(check_data["x"]) != expected_count:
                    raise RuntimeError(
                        f"Staged chunk count mismatch in {staged_file}: expected {expected_count}, got {len(check_data['x'])}"
                    )

        # Compute chunk checksums in staging area before publication
        chunk_checksums: dict[str, str] = {}
        for cname in train_chunk_names:
            chunk_checksums[cname] = compute_chunk_checksum(staging_dir / cname)
        chunk_checksums[valid_chunk_name] = compute_chunk_checksum(staging_dir / valid_chunk_name)
        chunk_checksums[test_chunk_name] = compute_chunk_checksum(staging_dir / test_chunk_name)

        if (target_path / "meta.npz").is_file():
            _snapshot_previous_build(target_path)

        for staged_file in staging_dir.iterdir():
            if staged_file.is_file() and staged_file.name.endswith(".npz"):
                dest = target_path / staged_file.name
                os.replace(staged_file, dest)
                if staged_file.name in chunk_checksums:
                    actual_ck = compute_chunk_checksum(dest)
                    if actual_ck != chunk_checksums[staged_file.name]:
                        raise RuntimeError(
                            f"Checksum mismatch after moving {staged_file.name} to {dest}: "
                            f"expected {chunk_checksums[staged_file.name]}, got {actual_ck}"
                        )

        split_hashes = {
            "train": compute_split_hash(train_ids),
            "valid": compute_split_hash(valid_ids),
            "test": compute_split_hash(test_ids),
        }
        dataset_fingerprint = compute_dataset_fingerprint(
            build_id=resolved_build_id,
            dictionary_fingerprint=current_dict_fingerprint or "",
            split_hashes=split_hashes,
            split_counts={
                "train": len(train_tokens),
                "valid": len(valid_tokens),
                "test": len(test_tokens),
            },
            train_chunks=train_chunk_names,
            valid_chunks=[valid_chunk_name],
            test_chunks=[test_chunk_name],
            format_type="tok_dict",
            chunk_checksums=chunk_checksums,
            schema_version=2,
        )

        temp_meta = target_path / f".meta_{safe_build_id}.tmp.npz"
        np.savez(
            temp_meta,
            is_published=True,
            build_id=resolved_build_id,
            dictionary_fingerprint=current_dict_fingerprint or "",
            dataset_fingerprint=dataset_fingerprint,
            split_hash_train=split_hashes["train"],
            split_hash_valid=split_hashes["valid"],
            split_hash_test=split_hashes["test"],
            train_chunks=np.asarray(train_chunk_names, dtype=object),
            train_chunk_counts=np.asarray(train_chunk_counts, dtype=np.int32),
            valid_chunks=np.asarray([valid_chunk_name], dtype=object),
            valid_chunk_counts=np.asarray([len(valid_tokens)], dtype=np.int32),
            test_chunks=np.asarray([test_chunk_name], dtype=object),
            test_chunk_counts=np.asarray([len(test_tokens)], dtype=np.int32),
            chunk_checksums=json.dumps(chunk_checksums),
            schema_version=2,
            format="tok_dict",
            codebook_size=codebook_size,
            motion_token_offset=motion_token_offset,
            pad_token_id=pad_token_id,
            sep_token_id=sep_token_id,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            codebook_metadata=np.asarray(codebook_metadata, dtype=object),
            class_names=np.asarray(["anime"], dtype=object),
            n_classes=int(args.n_classes),
            n_samples_train=len(train_tokens),
            n_samples_valid=len(valid_tokens),
            n_samples_test=len(test_tokens),
            max_length=int(args.max_length),
            overlength_policy=overlength_policy,
            error_feedback=error_feedback,
            source_type=source_type,
            n_samples_rejected=len(rejected),
        )
        os.replace(temp_meta, target_path / "meta.npz")

        # Copy valid and test chunks as valid.npz and test.npz for backwards compatibility with unconfigured tools
        try:
            shutil.copy2(target_path / valid_chunk_name, target_path / f".valid_{safe_build_id}.tmp.npz")
            os.replace(target_path / f".valid_{safe_build_id}.tmp.npz", target_path / "valid.npz")
            shutil.copy2(target_path / test_chunk_name, target_path / f".test_{safe_build_id}.tmp.npz")
            os.replace(target_path / f".test_{safe_build_id}.tmp.npz", target_path / "test.npz")
        except Exception:
            pass

        report = {
            "schema_version": 2,
            "build_id": resolved_build_id,
            "source_dir": str(source_path),
            "source_type": source_type,
            "target_dir": str(target_path),
            "max_length": int(args.max_length),
            "overlength_policy": overlength_policy,
            "error_feedback": error_feedback,
            "accepted": len(prepared),
            "rejected": rejected,
            "split_counts": {
                "train": len(train_tokens),
                "valid": len(valid_tokens),
                "test": len(test_tokens),
            },
            "split_hashes": split_hashes,
            "dataset_fingerprint": dataset_fingerprint,
            "train_chunks": train_chunk_names,
            "train_chunk_counts": train_chunk_counts,
            "valid_chunks": [valid_chunk_name],
            "test_chunks": [test_chunk_name],
            "chunk_checksums": chunk_checksums,
            "dictionary_fingerprint": current_dict_fingerprint,
        }
        temp_report = target_path / f".report_{safe_build_id}.tmp.json"
        temp_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_report, target_path / "preparation_report.json")
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Saved tok-dict Sketchformer data (build_id={resolved_build_id}) to {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

