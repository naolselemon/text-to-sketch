import argparse
import json
from pathlib import Path

import numpy as np

from utils.paths import DEFAULT_STROKE5_DIR, PROCESSED_DATA_DIR


def load_stroke5_file(path):
    data = np.load(path, allow_pickle=True)
    if len(data.files) != 1:
        raise ValueError(f"Expected one array in {path}, found {data.files}")
    return data[data.files[0]]


def stroke5_to_stroke3(stroke5):
    if stroke5.ndim != 2 or stroke5.shape[1] != 5:
        raise ValueError(f"Expected stroke5 shape (N,5), got {stroke5.shape}")

    if stroke5.shape[0] < 2:
        raise ValueError("Stroke5 sketch is too short to contain sentinel row")

    stroke5 = stroke5[:-1]  # drop final sentinel/padding row
    dx_dy = stroke5[:, :2]
    pen_state = stroke5[:, 3]
    return np.concatenate([dx_dy, pen_state[:, None]], axis=1).astype(np.float32)


def build_splits(sketches, seed, train_frac=0.8, valid_frac=0.1, test_frac=0.1):
    num = len(sketches)
    if not np.isclose(train_frac + valid_frac + test_frac, 1.0):
        raise ValueError("train/valid/test fractions must sum to 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num)

    train_end = int(num * train_frac)
    valid_end = train_end + int(num * valid_frac)

    train_idx = indices[:train_end]
    valid_idx = indices[train_end:valid_end]
    test_idx = indices[valid_end:]

    return (
        [sketches[i] for i in train_idx],
        [sketches[i] for i in valid_idx],
        [sketches[i] for i in test_idx],
    )


def chunk_sketch_data(sketches, labels, n_chunks):
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")
    if len(sketches) != len(labels):
        raise ValueError("Sketch and label counts do not match")

    sketches = list(sketches)
    labels = np.array(labels, dtype=np.int32)
    total = len(sketches)
    chunk_size = int(np.ceil(total / n_chunks))

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        if start >= end:
            break
        chunks.append((sketches[start:end], labels[start:end]))
    return chunks


def save_chunk(path, sketches, labels):
    np.savez_compressed(path,
                         x=np.array(sketches, dtype=object),
                         y=np.array(labels, dtype=np.int32))


def compute_scale_stats(sketches):
    all_deltas = np.concatenate([sk[:, :2] for sk in sketches], axis=0)
    return float(np.std(all_deltas)), float(np.mean(all_deltas))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare anime Stroke-5 data for Sketchformer Stroke-3 chunked loading."
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_STROKE5_DIR,
        help="Source directory containing data/processed/stroke5/*.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSONL audit manifest path. Reads inputs exclusively from accepted entries.",
    )
    parser.add_argument(
        "--accepted-ids",
        type=str,
        default=None,
        help="Path to an accepted-IDs file or comma-separated list of sketch IDs.",
    )
    parser.add_argument(
        "--target-dir",
        default=PROCESSED_DATA_DIR / "sketchformer-ready-data" / "stroke3",
        help="Target directory to save train/valid/test chunks and meta.npz",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=10,
        help="Number of train chunks to write",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/valid/test splitting",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of data used for training",
    )
    parser.add_argument(
        "--valid-frac",
        type=float,
        default=0.1,
        help="Fraction of data used for validation",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of data used for testing",
    )
    return parser.parse_args()


def _parse_accepted_ids(accepted_ids_arg: str) -> list[str]:
    raw_path = Path(accepted_ids_arg)
    if raw_path.is_file():
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        ids = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    else:
        ids = [item.strip() for item in accepted_ids_arg.split(",") if item.strip()]

    seen: set[str] = set()
    for item in ids:
        if item in seen:
            raise ValueError(f"Duplicate sample ID in accepted-ids: {item!r}")
        seen.add(item)
    return ids


def main():
    args = parse_args()

    if args.n_chunks <= 0:
        raise ValueError("--n-chunks must be positive")

    if not args.manifest and not args.accepted_ids:
        raise ValueError(
            "An explicit input list is required: specify --manifest <path/to/manifest.jsonl> "
            "or --accepted-ids <path_or_ids>. Directory-scanning fallback is disabled to prevent data leakage."
        )

    source_path = Path(args.source_dir)
    target_path = Path(args.target_dir)

    # Validate target directory is an isolated version directory
    resolved_target = target_path.resolve()
    unstructured_shared = [
        Path.cwd().resolve(),
        source_path.resolve(),
        Path(DEFAULT_STROKE5_DIR).resolve(),
    ]
    if resolved_target in unstructured_shared:
        raise ValueError(
            f"Target directory {args.target_dir} is an unstructured shared directory. "
            "Dataset builds must use an isolated version subfolder (e.g. --target-dir data/processed/build_<version>)."
        )

    # Clean existing chunks and metadata in target directory before build
    target_path.mkdir(parents=True, exist_ok=True)
    for pattern in ("train_*.npz", "valid*.npz", "test*.npz", "meta.npz", "meta.json", "preparation_report.json"):
        for old_file in target_path.glob(pattern):
            old_file.unlink(missing_ok=True)

    accepted_ids_filter: list[str] | None = None
    if args.accepted_ids:
        accepted_ids_filter = _parse_accepted_ids(args.accepted_ids)
    accepted_filter_set = set(accepted_ids_filter) if accepted_ids_filter is not None else None

    # Load targets strictly from manifest or accepted IDs
    seen_ids: set[str] = set()
    items_to_process: list[tuple[str, Path]] = []

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                record = json.loads(raw)
                sample_id = record.get("sample_id")
                if not sample_id:
                    raise ValueError(f"Manifest line {line_no} is missing 'sample_id': {raw}")
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate sample ID in manifest: {sample_id!r}")
                seen_ids.add(sample_id)

                status = record.get("status", "accepted")
                if status != "accepted":
                    continue

                if accepted_filter_set is not None and sample_id not in accepted_filter_set:
                    continue

                explicit_path = Path(record["stroke5_path"]) if "stroke5_path" in record else None
                candidate = explicit_path if explicit_path and explicit_path.is_file() else source_path / f"{sample_id}.npz"
                items_to_process.append((sample_id, candidate))
    else:
        assert accepted_ids_filter is not None
        for sample_id in accepted_ids_filter:
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID in accepted-ids: {sample_id!r}")
            seen_ids.add(sample_id)
            candidate = source_path / f"{sample_id}.npz"
            items_to_process.append((sample_id, candidate))

    if not items_to_process:
        raise ValueError("No accepted sketches found in the provided manifest/accepted list")

    sketches = []
    sample_ids = []
    for sample_id, path in items_to_process:
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Expected stroke5 file does not exist for sample {sample_id!r}: {path}")
        stroke5 = load_stroke5_file(path)
        sketch = stroke5_to_stroke3(stroke5)
        sketches.append(sketch)
        sample_ids.append(sample_id)

    train_sketches, valid_sketches, test_sketches = build_splits(
        sketches,
        seed=args.seed,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
    )

    train_labels = [0] * len(train_sketches)
    valid_labels = [0] * len(valid_sketches)
    test_labels = [0] * len(test_sketches)

    train_chunks_list = chunk_sketch_data(train_sketches, train_labels, args.n_chunks)
    for chunk_idx, (chunk_sketches, chunk_labels) in enumerate(train_chunks_list):
        chunk_file = target_path / f"train_{chunk_idx:03}.npz"
        print(f"Saving train chunk {chunk_idx + 1}/{len(train_chunks_list)}: {chunk_file}")
        save_chunk(chunk_file, chunk_sketches, chunk_labels)

    print(f"Saving valid set ({len(valid_sketches)} sketches)")
    save_chunk(target_path / "valid.npz", valid_sketches, valid_labels)
    print(f"Saving test set ({len(test_sketches)} sketches)")
    save_chunk(target_path / "test.npz", test_sketches, test_labels)

    # Verify chunk completion and counts
    total_chunk_items = sum(len(c[0]) for c in train_chunks_list) + len(valid_sketches) + len(test_sketches)
    if total_chunk_items != len(sketches):
        raise AssertionError(
            f"Chunked example count {total_chunk_items} does not match prepared count {len(sketches)}"
        )

    std, mean = compute_scale_stats(train_sketches)

    # Atomic publication of meta.npz and meta.json
    tmp_meta_npz = target_path / ".meta_tmp.npz"
    tmp_meta_json = target_path / ".meta_tmp.json"
    np.savez(
        tmp_meta_npz,
        status="complete",
        published=True,
        std=std,
        mean=mean,
        class_names=np.array(["anime"], dtype=object),
        n_classes=1,
        example_count=len(sketches),
        total_count=len(sketches),
        n_samples_train=len(train_sketches),
        n_samples_valid=len(valid_sketches),
        n_samples_test=len(test_sketches),
    )
    tmp_meta_npz.replace(target_path / "meta.npz")

    meta_json_data = {
        "status": "complete",
        "published": True,
        "format": "stroke3",
        "example_count": len(sketches),
        "total_count": len(sketches),
        "split_counts": {
            "train": len(train_sketches),
            "valid": len(valid_sketches),
            "test": len(test_sketches),
        },
        "std": std,
        "mean": mean,
        "source_dir": str(source_path),
        "target_dir": str(target_path),
    }
    tmp_meta_json.write_text(
        json.dumps(meta_json_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_meta_json.replace(target_path / "meta.json")

    print(f"Saved meta file: {target_path / 'meta.npz'}")
    return 0
 
