"""Prepare tok-dict chunks for native Sketchformer fine-tuning."""

from __future__ import annotations

import argparse
import filecmp
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

from prep_data.prepare_sketchformer import build_splits, chunk_sketch_data
from prep_data.sketch_token.create_token_dict import get_token_dict_fingerprint, load_codebook_from_dir
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
    with np.load(meta_path, allow_pickle=True) as prev_meta:
        if not bool(prev_meta["is_published"]) or "build_id" not in prev_meta:
            raise ValueError(f"Cannot snapshot unpublished or unidentified dataset: {meta_path}")
        prev_build_id = str(prev_meta["build_id"])
        report_name = (
            str(prev_meta["report_file"])
            if "report_file" in prev_meta
            else "preparation_report.json"
        )
        chunk_names = {
            str(name)
            for split in ("train", "valid", "test")
            for name in prev_meta[f"{split}_chunks"]
        }
    for name in chunk_names:
        if Path(name).name != name:
            raise ValueError(f"Unsafe chunk name in previous metadata: {name}")
        chunk_path = target_path / name
        if not chunk_path.is_file():
            raise FileNotFoundError(f"Cannot snapshot missing published chunk: {chunk_path}")

    report_path = target_path / report_name
    if not report_path.is_file():
        raise FileNotFoundError(f"Cannot snapshot missing published report: {report_path}")
    with report_path.open(encoding="utf-8") as report_file:
        report_build_id = json.load(report_file).get("build_id")
    if str(report_build_id) != prev_build_id:
        raise ValueError(
            f"Cannot snapshot inconsistent report {report_path}: "
            f"report build {report_build_id}, metadata build {prev_build_id}"
        )

    safe_prev_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", prev_build_id)
    versions_dir = target_path / "versions" / safe_prev_id
    versions_dir.mkdir(parents=True, exist_ok=True)
    # Chunks are immutable. Retain only metadata and the report, using the
    # target as the root when reopening an archived metadata version.
    for source, archived in (
        (meta_path, versions_dir / "meta.npz"),
        (report_path, versions_dir / "preparation_report.json"),
    ):
        if archived.exists():
            if not archived.is_file() or not filecmp.cmp(source, archived, shallow=False):
                raise FileExistsError(f"Conflicting published snapshot: {archived}")
        else:
            shutil.copy2(source, archived)

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


def _resolve_sample_file(
    source_path: Path, sample_id: str, record: dict[str, Any], *,
    token_mode: bool, manifest_path: Path,
) -> Path:
    key = "tokens_path" if token_mode else "stroke5_path"
    declared = record.get(key)
    if declared:
        path = Path(declared)
        if path.is_absolute():
            return path
        # Pipeline manifests may contain paths relative to the caller's working
        # directory, while manually supplied manifests often use local paths.
        candidates = (path, manifest_path.parent / path, source_path / path)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]
    if int(record.get("schema_version", 0)) >= 2:
        raise ValueError(f"Accepted versioned sample '{sample_id}' is missing {key}")
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

    codebook, codebook_metadata = load_codebook_from_dir(token_dict_dir)
    current_dict_fingerprint = get_token_dict_fingerprint(token_dict_dir)
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
            if r.get("status") != "accepted" and r.get("accepted") is not True:
                continue
            sid = r.get("sample_id") or r.get("drawing_id") or r.get("id")
            if sid is None or not str(sid).strip():
                raise ValueError("Accepted manifest record is missing sample_id")
            if int(r.get("schema_version", 0)) >= 2:
                if not r.get("sample_id"):
                    raise ValueError("Accepted versioned manifest record is missing sample_id")
                required = ["build_id", "token_dictionary_sha256"]
                if args.tokens_dir:
                    required.append("tokens_sha256")
                for key in required:
                    if not r.get(key):
                        raise ValueError(f"Accepted versioned sample '{sid}' is missing {key}")
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
        if accepted_build_ids and any(
            (r.get("status") == "accepted" or r.get("accepted") is True)
            and not str(r.get("build_id") or "").strip()
            for r in manifest_records
        ):
            raise ValueError("Accepted records mix identified and unidentified build IDs")

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
            run_build = args.build_id or next(iter(accepted_build_ids), None)
            for disk_file in disk_files:
                sample_id = disk_file.relative_to(source_path).with_suffix("").as_posix()
                if sample_id not in manifest_by_id:
                    raise ValueError(
                        f"Unexpected untracked token file found on disk not listed in manifest: {disk_file}"
                    )
                _, embedded = load_token_sequence_with_metadata(disk_file)
                if run_build and embedded.get("build_id") and str(embedded["build_id"]) != str(run_build):
                    raise ValueError(
                        f"Mixed token builds in shared directory: {disk_file} belongs to "
                        f"{embedded['build_id']}, expected {run_build}"
                    )

        items_to_process: list[tuple[Path, str, dict[str, Any]]] = []
        for r in manifest_records:
            status = r.get("status")
            is_accepted = (status == "accepted") or (r.get("accepted") is True)
            if not is_accepted:
                continue

            sid = str(r.get("sample_id") or r.get("drawing_id") or r.get("id"))
            expected_file = _resolve_sample_file(
                source_path, sid, r, token_mode=bool(args.tokens_dir), manifest_path=manifest_path,
            )
            if not expected_file.is_file():
                raise FileNotFoundError(
                    f"Missing {'token' if args.tokens_dir else 'stroke5'} file for accepted sample '{sid}': {expected_file}"
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
        if args.tokens_dir and expected_tokens_sha256:
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
        if rec_dict_sha256 and current_dict_fingerprint:
            if str(rec_dict_sha256) != current_dict_fingerprint:
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
            versioned = int(record.get("schema_version", 0)) >= 2
            if versioned:
                for key in ("sample_id", "build_id", "token_dict_sha256"):
                    if not embedded_meta.get(key):
                        raise ValueError(f"Versioned token file {path} is missing {key}")
            expected_fields = {
                "sample_id": sample_id,
                "build_id": record.get("build_id") or args.build_id,
                "token_dict_sha256": rec_dict_sha256 or current_dict_fingerprint,
            }
            for key, expected in expected_fields.items():
                if key in embedded_meta and expected and str(embedded_meta[key]) != str(expected):
                    raise ValueError(
                        f"{'Token dictionary fingerprint' if key == 'token_dict_sha256' else key} "
                        f"mismatch in token file {path}: embedded {embedded_meta[key]}, expected {expected}"
                    )
            if "tokens_sha256" in embedded_meta:
                actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual_sha256 != str(embedded_meta["tokens_sha256"]):
                    raise ValueError(
                        f"SHA-256 hash mismatch in embedded metadata for {path}: "
                        f"embedded {embedded_meta['tokens_sha256']}, actual {actual_sha256}"
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
    report_filename = f"preparation_report_{safe_build_id}.json"
    active_meta = target_path / "meta.npz"
    if active_meta.is_file():
        with np.load(active_meta, allow_pickle=True) as previous:
            if "build_id" not in previous or not bool(previous["is_published"]):
                raise ValueError(f"Cannot replace unidentified or unpublished dataset: {active_meta}")
            if str(previous["build_id"]) == resolved_build_id:
                raise ValueError(
                    f"Build ID {resolved_build_id} is already published; use a new build ID "
                    "rather than overwriting active chunks"
                )
    else:
        for name in ("valid.npz", "test.npz", "preparation_report.json"):
            if (target_path / name).exists():
                raise FileExistsError(
                    f"Unpublished output exists: {target_path / name}; refusing to overwrite user data"
                )
    staging_dir = target_path / f".staging_{safe_build_id}_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)
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

        chunk_names = [*train_chunk_names, valid_chunk_name, test_chunk_name]
        for name in chunk_names:
            if (target_path / name).exists():
                raise FileExistsError(
                    f"Chunk name already exists: {target_path / name}; "
                    "choose a distinct build ID to preserve previous data"
                )
        if (target_path / report_filename).exists():
            raise FileExistsError(f"Published report already exists: {target_path / report_filename}")

        temp_meta = staging_dir / "meta.npz"
        np.savez(
            temp_meta,
            schema_version=2,
            is_published=True,
            build_id=resolved_build_id,
            report_file=report_filename,
            dictionary_fingerprint=current_dict_fingerprint or "",
            train_chunks=np.asarray(train_chunk_names, dtype=object),
            train_sample_ids=np.asarray(train_ids, dtype=object),
            train_chunk_counts=np.asarray(train_chunk_counts, dtype=np.int32),
            valid_chunks=np.asarray([valid_chunk_name], dtype=object),
            valid_chunk_counts=np.asarray([len(valid_tokens)], dtype=np.int32),
            valid_sample_ids=np.asarray(valid_ids, dtype=object),
            test_chunks=np.asarray([test_chunk_name], dtype=object),
            test_chunk_counts=np.asarray([len(test_tokens)], dtype=np.int32),
            test_sample_ids=np.asarray(test_ids, dtype=object),
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

        report = {
            "schema_version": 2,
            "build_id": resolved_build_id,
            "source_dir": str(source_path),
            "source_type": source_type,
            "manifest_path": str(manifest_path.resolve()) if manifest_path else None,
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
            "train_chunks": train_chunk_names,
            "train_chunk_counts": train_chunk_counts,
            "valid_chunks": [valid_chunk_name],
            "test_chunks": [test_chunk_name],
            "dictionary_fingerprint": current_dict_fingerprint,
        }
        temp_report = staging_dir / "preparation_report.json"
        temp_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if active_meta.is_file():
            _snapshot_previous_build(target_path)

        # New chunks and their report have unique filenames. Only the final
        # metadata swap makes this verified version visible to readers.
        for name in chunk_names:
            os.link(staging_dir / name, target_path / name)
        os.link(temp_report, target_path / report_filename)
        os.replace(temp_meta, active_meta)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Saved tok-dict Sketchformer data (build_id={resolved_build_id}) to {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

