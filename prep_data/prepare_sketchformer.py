import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from dataloaders.lineage import (
    compute_chunk_checksum,
    compute_dataset_fingerprint,
    compute_split_hash,
)
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


def build_splits(sketches, seed, train_frac=0.8, valid_frac=0.1, test_frac=0.1, sample_ids=None):
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

    train_sk = [sketches[i] for i in train_idx]
    valid_sk = [sketches[i] for i in valid_idx]
    test_sk = [sketches[i] for i in test_idx]

    if sample_ids is not None:
        train_id = [sample_ids[i] for i in train_idx]
        valid_id = [sample_ids[i] for i in valid_idx]
        test_id = [sample_ids[i] for i in test_idx]
        return train_sk, valid_sk, test_sk, train_id, valid_id, test_id

    return train_sk, valid_sk, test_sk


def chunk_sketch_data(sketches, labels, n_chunks, sample_ids=None):
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")
    if len(sketches) != len(labels):
        raise ValueError("Sketch and label counts do not match")

    sketches = list(sketches)
    labels = np.array(labels, dtype=np.int32)
    sids = list(sample_ids) if sample_ids is not None else None
    total = len(sketches)
    chunk_size = int(np.ceil(total / n_chunks))

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        if start >= end:
            break
        if sids is not None:
            chunks.append((sketches[start:end], labels[start:end], sids[start:end]))
        else:
            chunks.append((sketches[start:end], labels[start:end]))
    return chunks


def save_chunk(path, sketches, labels, sample_ids=None):
    kwargs: dict[str, Any] = {
        "x": np.array(sketches, dtype=object),
        "y": np.array(labels, dtype=np.int32),
    }
    if sample_ids is not None:
        kwargs["sample_ids"] = np.array(sample_ids, dtype=object)
    np.savez_compressed(path, **kwargs)


def compute_scale_stats(sketches):
    all_deltas = np.concatenate([sk[:, :2] for sk in sketches], axis=0)
    return float(np.std(all_deltas)), float(np.mean(all_deltas))


def migrate_legacy_stroke3_dataset(
    target_dir: str | Path,
    build_id: str | None = None,
) -> dict[str, Any]:
    """Migrate a legacy Stroke-3 dataset to modern lineage schema version 2.

    Generates synthetic sample IDs prefixed with 'legacy_generated:' for chunks lacking them.
    Original historical source sample identities were not recorded in legacy builds and
    cannot be recovered; these generated identifiers provide deterministic chunk-position
    identity for split tracking without claiming historical provenance.
    Computes deterministic chunk checksums and split hashes, and writes an updated meta.npz
    with full lineage and explicit legacy migration markers.
    """
    target_path = Path(target_dir)
    meta_path = target_path / "meta.npz"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.npz not found in {target_path}")

    with np.load(meta_path, allow_pickle=True) as meta_data:
        meta = {k: meta_data[k] for k in meta_data.files}

    if "train_chunks" in meta:
        train_chunk_names = [str(c) for c in meta["train_chunks"]]
    else:
        train_chunk_names = sorted(f.name for f in target_path.glob("train_*.npz"))

    valid_chunk_names = [str(c) for c in meta.get("valid_chunks", ["valid.npz"])]
    test_chunk_names = [str(c) for c in meta.get("test_chunks", ["test.npz"])]

    effective_build_id = str(build_id or meta.get("build_id") or f"stroke3-migrated-{uuid.uuid4().hex[:8]}")
    chunk_checksums: dict[str, str] = {}
    split_sample_ids: dict[str, list[str]] = {"train": [], "valid": [], "test": []}

    chunk_map = {
        "train": train_chunk_names,
        "valid": valid_chunk_names,
        "test": test_chunk_names,
    }

    for split_name, chunk_names in chunk_map.items():
        for chunk_name in chunk_names:
            chunk_file = target_path / chunk_name
            if not chunk_file.is_file():
                raise FileNotFoundError(f"Required chunk not found: {chunk_file}")

            with np.load(chunk_file, allow_pickle=True) as cdata:
                chunk_dict = {k: cdata[k] for k in cdata.files}

            needs_save = False
            if "sample_ids" not in chunk_dict:
                count = len(chunk_dict["x"])
                chunk_dict["sample_ids"] = np.array(
                    [f"legacy_generated:{chunk_file.stem}:{idx}" for idx in range(count)],
                    dtype=object,
                )
                needs_save = True

            if needs_save:
                np.savez_compressed(chunk_file, **chunk_dict)

            actual_checksum = compute_chunk_checksum(chunk_file)
            chunk_checksums[chunk_name] = actual_checksum
            ids = [str(sid).strip() for sid in chunk_dict["sample_ids"]]
            split_sample_ids[split_name].extend(ids)

    split_hashes = {
        "train": compute_split_hash(split_sample_ids["train"]),
        "valid": compute_split_hash(split_sample_ids["valid"]),
        "test": compute_split_hash(split_sample_ids["test"]),
    }
    split_counts = {
        "train": len(split_sample_ids["train"]),
        "valid": len(split_sample_ids["valid"]),
        "test": len(split_sample_ids["test"]),
    }

    fingerprint = compute_dataset_fingerprint(
        build_id=effective_build_id,
        dictionary_fingerprint="",
        split_hashes=split_hashes,
        split_counts=split_counts,
        train_chunks=train_chunk_names,
        valid_chunks=valid_chunk_names,
        test_chunks=test_chunk_names,
        chunk_checksums=chunk_checksums,
        format_type="stroke3",
        schema_version=2,
    )

    meta["build_id"] = effective_build_id
    meta["dataset_fingerprint"] = fingerprint
    meta["dictionary_fingerprint"] = ""
    meta["split_hash_train"] = split_hashes["train"]
    meta["split_hash_valid"] = split_hashes["valid"]
    meta["split_hash_test"] = split_hashes["test"]
    meta["chunk_checksums"] = json.dumps(chunk_checksums)
    meta["format"] = "stroke3"
    meta["schema_version"] = 2
    meta["is_published"] = True
    meta["sample_ids_are_generated"] = True
    meta["has_original_sample_ids"] = False
    meta["is_migrated_legacy"] = True

    np.savez(meta_path, **meta)
    return {
        "build_id": effective_build_id,
        "dataset_fingerprint": fingerprint,
        "split_counts": split_counts,
        "chunk_checksums": chunk_checksums,
        "sample_ids_are_generated": True,
        "has_original_sample_ids": False,
        "is_migrated_legacy": True,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare anime Stroke-5 data for Sketchformer Stroke-3 chunked loading.")
    parser.add_argument("--source-dir", default=DEFAULT_STROKE5_DIR,
                        help="Source directory containing data/processed/stroke5/*.npz")
    parser.add_argument("--target-dir", default=PROCESSED_DATA_DIR / "sketchformer-ready-data" / "stroke3",
                        help="Target directory to save train/valid/test chunks and meta.npz")
    parser.add_argument("--n-chunks", type=int, default=10,
                        help="Number of train chunks to write")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/valid/test splitting")
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="Fraction of data used for training")
    parser.add_argument("--valid-frac", type=float, default=0.1,
                        help="Fraction of data used for validation")
    parser.add_argument("--test-frac", type=float, default=0.1,
                        help="Fraction of data used for testing")
    parser.add_argument("--build-id", default=None,
                        help="Explicit build ID for modern dataset lineage.")
    parser.add_argument("--migrate-legacy", action="store_true",
                        help="Migrate existing legacy Stroke-3 dataset in --target-dir to schema v2.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.migrate_legacy:
        res = migrate_legacy_stroke3_dataset(args.target_dir, build_id=args.build_id)
        print(f"Migrated legacy dataset at {args.target_dir}: build_id={res['build_id']}, fingerprint={res['dataset_fingerprint']}")
        return

    source_path = Path(args.source_dir)
    target_path = Path(args.target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    file_list = sorted(source_path.glob("*.npz"))
    if not file_list:
        raise FileNotFoundError(f"No .npz files found in {source_path}")

    print(f"Found {len(file_list)} source files in {source_path}")

    sketches = []
    sample_ids = []
    for path in file_list:
        stroke5 = load_stroke5_file(path)
        sketch = stroke5_to_stroke3(stroke5)
        sketches.append(sketch)
        sample_ids.append(path.stem)

    train_sketches, valid_sketches, test_sketches, train_ids, valid_ids, test_ids = build_splits(
        sketches,
        seed=args.seed,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        sample_ids=sample_ids,
    )

    train_labels = [0] * len(train_sketches)
    valid_labels = [0] * len(valid_sketches)
    test_labels = [0] * len(test_sketches)

    train_chunks_list = chunk_sketch_data(train_sketches, train_labels, args.n_chunks, sample_ids=train_ids)
    train_chunk_names = []
    train_chunk_counts = []
    for chunk_idx, (chunk_sketches, chunk_labels, chunk_sids) in enumerate(train_chunks_list):
        chunk_name = f"train_{chunk_idx:03}.npz"
        chunk_file = target_path / chunk_name
        print(f"Saving train chunk {chunk_idx + 1}/{len(train_chunks_list)}: {chunk_file}")
        save_chunk(chunk_file, chunk_sketches, chunk_labels, sample_ids=chunk_sids)
        train_chunk_names.append(chunk_name)
        train_chunk_counts.append(len(chunk_sketches))

    print(f"Saving valid set ({len(valid_sketches)} sketches)")
    save_chunk(target_path / "valid.npz", valid_sketches, valid_labels, sample_ids=valid_ids)
    print(f"Saving test set ({len(test_sketches)} sketches)")
    save_chunk(target_path / "test.npz", test_sketches, test_labels, sample_ids=test_ids)

    # Compute modern lineage checksums and hashes
    chunk_checksums: dict[str, str] = {}
    for c_name in train_chunk_names + ["valid.npz", "test.npz"]:
        chunk_checksums[c_name] = compute_chunk_checksum(target_path / c_name)

    split_hashes = {
        "train": compute_split_hash(train_ids),
        "valid": compute_split_hash(valid_ids),
        "test": compute_split_hash(test_ids),
    }
    split_counts = {
        "train": len(train_sketches),
        "valid": len(valid_sketches),
        "test": len(test_sketches),
    }

    effective_build_id = str(args.build_id or f"stroke3-{uuid.uuid4().hex[:8]}")
    dataset_fingerprint = compute_dataset_fingerprint(
        build_id=effective_build_id,
        dictionary_fingerprint="",
        split_hashes=split_hashes,
        split_counts=split_counts,
        train_chunks=train_chunk_names,
        valid_chunks=["valid.npz"],
        test_chunks=["test.npz"],
        chunk_checksums=chunk_checksums,
        format_type="stroke3",
        schema_version=2,
    )

    std, mean = compute_scale_stats(train_sketches)
    meta_file = target_path / "meta.npz"
    np.savez(
        meta_file,
        std=std,
        mean=mean,
        class_names=np.array(["anime"], dtype=object),
        n_classes=1,
        n_samples_train=len(train_sketches),
        n_samples_valid=len(valid_sketches),
        n_samples_test=len(test_sketches),
        train_chunks=np.asarray(train_chunk_names, dtype=object),
        train_chunk_counts=np.asarray(train_chunk_counts, dtype=np.int32),
        valid_chunks=np.asarray(["valid.npz"], dtype=object),
        valid_chunk_counts=np.asarray([len(valid_sketches)], dtype=np.int32),
        test_chunks=np.asarray(["test.npz"], dtype=object),
        test_chunk_counts=np.asarray([len(test_sketches)], dtype=np.int32),
        is_published=True,
        build_id=effective_build_id,
        dataset_fingerprint=dataset_fingerprint,
        dictionary_fingerprint="",
        split_hash_train=split_hashes["train"],
        split_hash_valid=split_hashes["valid"],
        split_hash_test=split_hashes["test"],
        chunk_checksums=json.dumps(chunk_checksums),
        format="stroke3",
        schema_version=2,
    )
    print(f"Saved meta file: {meta_file} (build_id={effective_build_id}, fingerprint={dataset_fingerprint})")


if __name__ == "__main__":
    main()
 
