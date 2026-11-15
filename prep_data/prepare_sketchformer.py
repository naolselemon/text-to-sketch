import argparse
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
    return parser.parse_args()


def main():
    args = parse_args()

    source_path = Path(args.source_dir)
    target_path = Path(args.target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    file_list = sorted(source_path.glob("*.npz"))
    if not file_list:
        raise FileNotFoundError(f"No .npz files found in {source_path}")

    print(f"Found {len(file_list)} source files in {source_path}")

    sketches = []
    for path in file_list:
        stroke5 = load_stroke5_file(path)
        sketch = stroke5_to_stroke3(stroke5)
        sketches.append(sketch)

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

    std, mean = compute_scale_stats(train_sketches)
    meta_file = target_path / "meta.npz"
    np.savez(meta_file,
             std=std,
             mean=mean,
             class_names=np.array(["anime"], dtype=object),    
             n_classes=1,
             n_samples_train=len(train_sketches),
             n_samples_valid=len(valid_sketches),
             n_samples_test=len(test_sketches))
    print(f"Saved meta file: {meta_file}")


if __name__ == "__main__":
    main()
 
