"""Prepare tok-dict chunks for native Sketchformer fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

from dataloaders.lineage import (
    compute_chunk_checksum,
    compute_dataset_fingerprint,
    compute_split_hash,
)
from prep_data.prepare_sketchformer import build_splits, chunk_sketch_data
from prep_data.sketch_token.create_token_dict import (
<<<<<<< HEAD
    compute_codebook_artifact_checksum,
=======
>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c
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
    """Load legacy single-array or named token archives without requantizing."""
    tokens, _ = load_token_sequence_with_metadata(path)
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


def load_manifest_records(path: Path) -> dict[str, dict[str, object]]:
    """Index accepted preprocessing records by their stable sample IDs."""
    text = path.read_text(encoding="utf-8").strip()
    rows = json.loads(text) if text.startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    records: dict[str, dict[str, object]] = {}
    seen_ids: set[str] = set()
    for record in rows:
        sample_id = str(
            record.get("sample_id") or record.get("drawing_id") or record.get("id") or ""
        ).strip()
        if not sample_id:
            raise ValueError(f"Manifest record in {path} lacks a sample ID")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate drawing ID in manifest {path}: {sample_id}")
        seen_ids.add(sample_id)
        if record.get("status") == "accepted" or record.get("accepted") is True:
            records[sample_id] = record
    return records


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--manifest", type=Path, help="Preprocessing manifest JSONL or JSON array.")
    parser.add_argument("--build-id", help="Unique dataset build identifier.")
    parser.add_argument(
        "--allow-unverified-codebook",
        action="store_true",
        help="Allow a legacy codebook, but mark the resulting dataset unverified.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

<<<<<<< HEAD
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
=======
    if args.build_id and not args.build_id.strip():
        raise ValueError("--build-id must not be empty")
    if (target_path / "meta.npz").exists():
        with np.load(target_path / "meta.npz", allow_pickle=False) as previous:
            if args.build_id and str(previous.get("build_id", "")) == args.build_id:
                raise ValueError(
                    f"Dataset build '{args.build_id}' already exists at {target_path}. "
                    "Dataset builds are immutable; to create a new build, use a distinct build ID."
                )
            if bool(previous.get("is_published", False)) or bool(previous.get("verified", False)):
                raise ValueError(
                    f"Verified dataset already exists at {target_path}; use a new target directory."
                )

    codebook, codebook_metadata = load_codebook_from_dir(
        token_dict_dir, allow_unverified=args.allow_unverified_codebook
    )
    current_dict_fingerprint = get_token_dict_fingerprint(
        token_dict_dir, allow_unverified=args.allow_unverified_codebook
    )
    current_codebook_checksum = codebook_metadata.get("codebook_sha256")
    current_codebook_file_checksum = hashlib.sha256(
        (token_dict_dir / "codebook.npy").read_bytes()
    ).hexdigest()
>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c
    codebook_size = int(len(codebook))
    motion_token_offset = 1
    pad_token_id = 0
    sep_token_id = codebook_size + 1
    sos_token_id = codebook_size + 2
    eos_token_id = codebook_size + 3
    vocab_size = codebook_size + 4

    file_list = sorted(source_path.rglob("*.npz"))
    if not file_list:
        raise FileNotFoundError(f"No .npz files found in {source_path}")

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = next(
            (
                candidate
                for directory in (source_path, source_path.parent)
                for name in ("preprocessing_manifest.jsonl", "preprocessing_manifest.json")
                if (candidate := directory / name).is_file()
            ),
            None,
        )
    if manifest_path is not None and not manifest_path.is_file():
        raise FileNotFoundError(f"Preprocessing manifest not found: {manifest_path}")
    records = load_manifest_records(manifest_path) if manifest_path else {}
    if manifest_path:
        file_ids = {path.relative_to(source_path).with_suffix("").as_posix() for path in file_list}
        if file_ids != records.keys():
            raise ValueError(
                f"Manifest/source mismatch: untracked={sorted(file_ids - records.keys())}, "
                f"missing={sorted(records.keys() - file_ids)}"
            )
    accepted_build_ids = {
        str(record["build_id"]).strip()
        for record in records.values() if record.get("build_id")
    }
    if len(accepted_build_ids) > 1 or (
        args.build_id and accepted_build_ids and accepted_build_ids != {args.build_id}
    ):
        raise ValueError(f"Manifest build ID mismatch: {sorted(accepted_build_ids)}")
    resolved_build_id = args.build_id or (
        next(iter(accepted_build_ids)) if accepted_build_ids else None
    )

    prepared: list[tuple[np.ndarray, str]] = []
    rejected: list[dict[str, object]] = []
<<<<<<< HEAD
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
=======
    sources_attested = True
    for path in file_list:
        sample_id = path.relative_to(source_path).with_suffix("").as_posix()
        record = records.get(sample_id, {})
        recorded_path = record.get("tokens_path" if args.tokens_dir else "stroke5_path")
        if recorded_path is not None and Path(str(recorded_path)).resolve() != path.resolve():
            raise ValueError(f"Manifest source path mismatch for {sample_id}: {recorded_path}")
        expected_hash = record.get("tokens_sha256") or record.get("token_sha256")
        if args.tokens_dir and expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != str(expected_hash):
            raise ValueError(f"SHA-256 hash mismatch for source file {path}")
        for field in (
            "token_dictionary_sha256", "token_dict_sha256", "codebook_sha256",
            "dictionary_fingerprint",
        ):
            value = record.get(field)
            if value and str(value).strip() not in {
                current_dict_fingerprint, current_codebook_checksum,
                current_codebook_file_checksum,
            }:
                raise ValueError(f"Token dictionary fingerprint mismatch for {sample_id}: {field}={value}")
        if args.tokens_dir:
            tokens, embedded_meta = load_token_sequence_with_metadata(path)
            if embedded_meta.get("sample_id") and embedded_meta["sample_id"] != sample_id:
                raise ValueError(f"Sample ID mismatch in token file {path}")
            embedded_build_id = embedded_meta.get("build_id")
            if embedded_build_id:
                if resolved_build_id and embedded_build_id != resolved_build_id:
                    raise ValueError(f"Build ID mismatch in token file {path}")
                resolved_build_id = embedded_build_id
            if embedded_meta.get("tokens_sha256") and (
                hashlib.sha256(path.read_bytes()).hexdigest() != embedded_meta["tokens_sha256"]
            ):
                raise ValueError(f"SHA-256 hash mismatch in token file {path}")
            for field in (
                "token_dictionary_sha256", "token_dict_sha256", "codebook_sha256",
                "dictionary_fingerprint",
            ):
                value = embedded_meta.get(field)
                if value and value.strip() not in {
                    current_dict_fingerprint, current_codebook_checksum,
                    current_codebook_file_checksum,
                }:
                    raise ValueError(f"Token dictionary fingerprint mismatch in token file {path}: {field}={value}")
            sources_attested &= any(
                embedded_meta.get(key) or record.get(key)
                for key in (
                    "token_dictionary_sha256", "token_dict_sha256",
                    "codebook_sha256", "dictionary_fingerprint",
                )
            )
>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c
        else:
            tokens = encode_stroke5(
                load_stroke5_file(path),
                codebook,
                error_feedback=error_feedback,
            )
        if not sample_id:
            raise ValueError(f"Empty sample ID for {path}")
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
    resolved_build_id = resolved_build_id or uuid.uuid4().hex
    if (target_path / "meta.npz").exists():
        with np.load(target_path / "meta.npz", allow_pickle=False) as previous:
            if str(previous.get("build_id", "")) == resolved_build_id:
                raise ValueError(
                    f"Dataset build '{resolved_build_id}' already exists at {target_path}. "
                    "Dataset builds are immutable; to create a new build, use a distinct build ID."
                )

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

<<<<<<< HEAD
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
=======
    train_chunks = chunk_sketch_data(
        list(zip(train_tokens, train_ids)),
        train_labels,
        args.n_chunks,
    )
    train_chunk_names = [f"train_{idx:03}.npz" for idx in range(len(train_chunks))]
    chunk_names = train_chunk_names + ["valid.npz", "test.npz"]
    unexpected = {
        path.name for path in target_path.glob("*.npz")
        if path.name not in {*chunk_names, "meta.npz"}
    }
    if unexpected:
        raise ValueError(f"Unexpected untracked dataset chunks at {target_path}: {sorted(unexpected)}")
    for chunk_idx, (chunk_records, chunk_labels) in enumerate(train_chunks):
        chunk_tokens = [tokens for tokens, _ in chunk_records]
        chunk_ids = [sample_id for _, sample_id in chunk_records]
        chunk_name = train_chunk_names[chunk_idx]
        save_chunk(
            target_path / chunk_name,
            chunk_tokens,
            chunk_labels.tolist(),
            chunk_ids,
>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c
        )

<<<<<<< HEAD
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
=======
    save_chunk(target_path / "valid.npz", valid_tokens, valid_labels, valid_ids)
    save_chunk(target_path / "test.npz", test_tokens, test_labels, test_ids)
>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c

    chunk_checksums = {
        name: compute_chunk_checksum(target_path / name) for name in chunk_names
    }
    split_hashes = {
        "train": compute_split_hash(train_ids),
        "valid": compute_split_hash(valid_ids),
        "test": compute_split_hash(test_ids),
    }
    split_counts = {
        "train": len(train_tokens),
        "valid": len(valid_tokens),
        "test": len(test_tokens),
    }
    dataset_fingerprint = compute_dataset_fingerprint(
        build_id=resolved_build_id,
        dictionary_fingerprint=current_dict_fingerprint,
        split_hashes=split_hashes,
        split_counts=split_counts,
        train_chunks=train_chunk_names,
        valid_chunks=["valid.npz"],
        test_chunks=["test.npz"],
        format_type="tok_dict",
        chunk_checksums=chunk_checksums,
        schema_version=2,
    )
    metadata_path = target_path / f".meta_{uuid.uuid4().hex}.tmp.npz"
    np.savez(
        metadata_path,
        is_published=bool(current_dict_fingerprint) and sources_attested,
        build_id=resolved_build_id,
        dictionary_fingerprint=current_dict_fingerprint,
        dataset_fingerprint=dataset_fingerprint,
        split_hash_train=split_hashes["train"],
        split_hash_valid=split_hashes["valid"],
        split_hash_test=split_hashes["test"],
        train_chunks=np.asarray(train_chunk_names, dtype=object),
        train_chunk_counts=np.asarray([len(chunk[0]) for chunk in train_chunks], dtype=np.int32),
        valid_chunks=np.asarray(["valid.npz"], dtype=object),
        valid_chunk_counts=np.asarray([len(valid_tokens)], dtype=np.int32),
        test_chunks=np.asarray(["test.npz"], dtype=object),
        test_chunk_counts=np.asarray([len(test_tokens)], dtype=np.int32),
        chunk_checksums=json.dumps(chunk_checksums, sort_keys=True),
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
    os.replace(metadata_path, target_path / "meta.npz")
    report = {
        "schema_version": 2,
        "build_id": resolved_build_id,
        "dictionary_fingerprint": current_dict_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "is_published": bool(current_dict_fingerprint) and sources_attested,
        "source_dir": str(source_path),
        "source_type": source_type,
        "manifest": str(manifest_path) if manifest_path else None,
        "target_dir": str(target_path),
        "max_length": int(args.max_length),
        "overlength_policy": overlength_policy,
        "error_feedback": error_feedback,
        "accepted": len(prepared),
        "rejected": rejected,
        "split_counts": split_counts,
        "split_hashes": split_hashes,
        "train_chunks": train_chunk_names,
        "valid_chunks": ["valid.npz"],
        "test_chunks": ["test.npz"],
        "chunk_checksums": chunk_checksums,
    }
    (target_path / "preparation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved tok-dict Sketchformer data (build_id={resolved_build_id}) to {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
