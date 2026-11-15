"""Prepare tok-dict chunks for native Sketchformer fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prep_data.prepare_sketchformer import build_splits, chunk_sketch_data
from prep_data.sketch_token.create_token_dict import load_codebook_from_dir
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

    codebook, codebook_metadata = load_codebook_from_dir(token_dict_dir)
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

    prepared: list[tuple[np.ndarray, str]] = []
    rejected: list[dict[str, object]] = []
    for path in file_list:
        if args.tokens_dir:
            tokens = load_token_file(path)
        else:
            tokens = encode_stroke5(
                load_stroke5_file(path),
                codebook,
                error_feedback=error_feedback,
            )
        sample_id = path.relative_to(source_path).with_suffix("").as_posix()
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

    train_chunks = chunk_sketch_data(
        list(zip(train_tokens, train_ids)),
        train_labels,
        args.n_chunks,
    )
    for chunk_idx, (chunk_records, chunk_labels) in enumerate(train_chunks):
        chunk_tokens = [tokens for tokens, _ in chunk_records]
        chunk_ids = [sample_id for _, sample_id in chunk_records]
        save_chunk(
            target_path / f"train_{chunk_idx:03}.npz",
            chunk_tokens,
            chunk_labels.tolist(),
            chunk_ids,
        )

    save_chunk(target_path / "valid.npz", valid_tokens, valid_labels, valid_ids)
    save_chunk(target_path / "test.npz", test_tokens, test_labels, test_ids)

    np.savez(
        target_path / "meta.npz",
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
    }
    (target_path / "preparation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved tok-dict Sketchformer data to {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
