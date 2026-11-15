"""Export real <=200-token teacher-forced logits from the original TF checkpoint.

CPU example (TensorFlow + original sketchformer runtime required for ``export-tf``)::

  python -m scripts.sketchformer.tf_reference_export select --config configs/train.yaml --experiment anime_tok_dict_long --split valid --samples 8 --original-dictionary sketchformer/prep_data/sketch_token/token_dict.pkl --input /tmp/short-input.npz
  python -m scripts.sketchformer.tf_reference_export export-tf --input /tmp/short-input.npz --checkpoint weights/pretrained/sketch-transformer-tf2-cvpr_tform_tok_dict/weights/ckpt-12 --original-config weights/pretrained/sketch-transformer-tf2-cvpr_tform_tok_dict/config.json --original-dictionary sketchformer/prep_data/sketch_token/token_dict.pkl --output /tmp/tf-short.npz

``select`` only uses prepared tok-dict data (never synthetic data); input
artifacts record hashed dataset files and source rows that are checked again
before either model export and the final comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evals.conversion_fidelity import _file_hash

LAYOUT = {"codebook_size": 1000, "pad_token_id": 0, "motion_token_offset": 1,
          "sep_token_id": 1001, "sos_token_id": 1002, "eos_token_id": 1003,
          "vocab_size": 1004}
WIDTH = 200


def original_codebook_sha256(original_dictionary: Path, codebook: Path) -> tuple[str, str]:
    """Verify the native .npy codebook is the released pickle's ordered centers."""
    from prep_data.sketch_token.create_token_dict import load_original_token_dict_centers

    original_dictionary, codebook = Path(original_dictionary), Path(codebook)
    centers = load_original_token_dict_centers(original_dictionary)
    exported = np.load(codebook, allow_pickle=False)
    if centers.shape != (1000, 2) or not np.array_equal(centers, exported):
        raise ValueError("Codebook differs from original token_dict.pkl ordered cluster centers")
    return _file_hash(original_dictionary), _file_hash(codebook)


def _verify_real_samples(meta: dict, tokens: np.ndarray, lengths: np.ndarray, ids: list[str]) -> None:
    """Match each parity input to its exact row in the unchanged prepared split."""
    from dataloaders.transforms import TokenSequenceTransform

    rows = meta.get("source_rows")
    hashes = meta.get("dataset_files_sha256")
    if not isinstance(rows, list) or len(rows) != len(ids) or not isinstance(hashes, dict):
        raise ValueError("Real inputs require source rows and hashed prepared dataset files")
    root = Path(meta["dataset_source"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    root = root.resolve()
    by_file: dict[str, list[tuple[int, int]]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Invalid real sample source row")
        name, index = row.get("file"), row.get("index")
        if not isinstance(name, str) or not name or Path(name).name != name or type(index) is not int or index < 0:
            raise ValueError("Invalid real sample source file or row index")
        by_file.setdefault(name, []).append((position, index))
    transform = TokenSequenceTransform(
        split=meta["dataset_split"], max_length=WIDTH, truncate_long_sequences=False,
        sos_token_id=LAYOUT["sos_token_id"], eos_token_id=LAYOUT["eos_token_id"],
    )
    for name, positions in by_file.items():
        path = root / name
        if not path.is_file() or path.resolve().parent != root or hashes.get(name) != _file_hash(path):
            raise ValueError(f"Prepared dataset source changed or is missing: {path}")
        with np.load(path, allow_pickle=True) as source:
            if "x" not in source.files or "y" not in source.files:
                raise ValueError(f"Invalid prepared tok-dict dataset: {path}")
            raw, labels = source["x"], source["y"]
            if len(raw) != len(labels):
                raise ValueError(f"Prepared dataset x/y lengths differ: {path}")
            source_ids = source["sample_ids"] if "sample_ids" in source.files else None
            for position, index in positions:
                if index >= len(raw):
                    raise ValueError(f"Prepared dataset row out of range: {path}:{index}")
                expected_id = str(source_ids[index]) if source_ids is not None else f"{path.stem}:{index}"
                stored = np.asarray(raw[index], dtype=np.int64)
                expected = transform({"tokens": stored[stored != LAYOUT["pad_token_id"]]})["tokens"]
                if expected_id != ids[position] or not np.array_equal(expected, tokens[position, :lengths[position]]):
                    raise ValueError(f"Parity input differs from real prepared sample: {path}:{index}")


def validate_input(path: Path, *, original_dictionary: Path, codebook: Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {"tokens", "lengths", "sample_ids", "metadata"}
        if not required.issubset(payload.files):
            raise ValueError(f"Real input artifact requires {sorted(required)}")
        tokens = payload["tokens"]
        lengths = payload["lengths"]
        sample_ids = payload["sample_ids"]
        meta = json.loads(str(payload["metadata"].item()))
    dictionary_sha, codebook_sha = original_codebook_sha256(original_dictionary, codebook)
    if meta.get("original_dictionary_sha256") != dictionary_sha or meta.get("codebook_sha256") != codebook_sha:
        raise ValueError("Input dictionary/codebook identity does not match original dictionary")
    if meta.get("token_layout") != LAYOUT or meta.get("preprocessing") != "prepared_tok_dict_no_augmentation_sos_eos_pad200":
        raise ValueError("Input token layout or preprocessing is not the original short-sequence contract")
    if not meta.get("dataset_split") or not meta.get("dataset_source"):
        raise ValueError("Real input artifact requires dataset split and source provenance")
    if tokens.ndim != 2 or tokens.shape[1] != WIDTH or tokens.dtype.kind not in "iu":
        raise ValueError("Input tokens must be integer [N,200]")
    n = len(tokens)
    if not n or lengths.shape != (n,) or lengths.dtype.kind not in "iu" or sample_ids.shape != (n,) or sample_ids.dtype.kind not in "US":
        raise ValueError("Input requires nonempty aligned string sample IDs and integer lengths")
    ids = [str(value) for value in sample_ids]
    if any(not value for value in ids) or len(set(ids)) != n:
        raise ValueError("Sample IDs must be nonempty and unique")
    if np.any((lengths < 3) | (lengths > WIDTH)) or np.any(tokens < 0) or np.any(tokens >= LAYOUT["vocab_size"]):
        raise ValueError("Invalid short sequence length or token ID")
    valid = np.arange(WIDTH)[None, :] < lengths[:, None]
    if np.any(tokens[:, 0] != LAYOUT["sos_token_id"]) or np.any(tokens[np.arange(n), lengths - 1] != LAYOUT["eos_token_id"]):
        raise ValueError("Sequences require SOS at start and EOS at last valid position")
    if np.any(tokens[valid] == LAYOUT["pad_token_id"]) or np.any(tokens[~valid] != LAYOUT["pad_token_id"]):
        raise ValueError("Incorrect padding inside or outside valid sequence")
    if np.any(tokens[:, 1:][valid[:, 1:]] == LAYOUT["sos_token_id"]):
        raise ValueError("Interior SOS token")
    interior = np.arange(WIDTH - 1)[None, :] < (lengths - 1)[:, None]
    if np.any(tokens[:, :-1][interior] == LAYOUT["eos_token_id"]):
        raise ValueError("Interior EOS token")
    _verify_real_samples(meta, tokens, lengths, ids)
    return {"tokens": tokens.astype(np.int64, copy=False), "lengths": lengths.astype(np.int64, copy=False),
            "sample_ids": np.asarray(ids), "metadata": meta}


def select_real_sequences(config: str, experiment: str, split: str, samples: int,
                          original_dictionary: Path, codebook: Path, output: Path) -> None:
    from scripts.sketchformer.config import compose_training_config
    from dataloaders import StrokeSequenceDataModule
    from evals.benchmark_eval import BenchmarkTokenTransform

    if samples < 1:
        raise ValueError("--samples must be positive")
    if split not in ("train", "valid", "test"):
        raise ValueError("Unsupported dataset split")
    cfg = compose_training_config(config, experiment=experiment)
    data = json.loads(json.dumps(cfg["data"]))
    if data["format"]["type"] != "tok_dict" or data.get("augmentation", {}).get("enabled", False):
        raise ValueError("Parity requires a prepared tok-dict dataset without augmentation")
    data["sequence"]["max_length"] = WIDTH
    data["sequence"]["truncate_long_sequences"] = False
    module = StrokeSequenceDataModule(data, project_root=Path(__file__).resolve().parents[2])
    dataset = module._make_dataset(split)
    dataset.transform.max_length = None
    dataset.transform = BenchmarkTokenTransform(
        dataset.transform, LAYOUT["pad_token_id"]
    )
    chosen = []
    for index in range(len(dataset)):
        record = dataset[index]
        if 3 <= len(record["tokens"]) <= WIDTH:
            chosen.append(record)
            if len(chosen) == samples:
                break
    if len(chosen) != samples:
        raise ValueError(f"Only {len(chosen)} eligible real short sequences found (requested {samples})")
    original_sha, codebook_sha = original_codebook_sha256(original_dictionary, codebook)
    tokens = np.zeros((samples, WIDTH), dtype=np.int64)
    lengths = np.array([len(record["tokens"]) for record in chosen], dtype=np.int64)
    for row, record in enumerate(chosen):
        tokens[row, :lengths[row]] = record["tokens"]
    metadata = {
        "original_dictionary_sha256": original_sha,
        "codebook_sha256": codebook_sha,
        "token_layout": LAYOUT,
        "preprocessing": "prepared_tok_dict_no_augmentation_sos_eos_pad200",
        "dataset_split": split,
        "dataset_source": str(data["dataset"]["root"]),
        "dataset_files_sha256": {
            path.name: _file_hash(path)
            for path in {Path(record["source_file"]) for record in chosen}
        },
        "source_rows": [
            {"file": Path(record["source_file"]).name, "index": int(record["source_index"])}
            for record in chosen
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, tokens=tokens, lengths=lengths,
             sample_ids=np.asarray([str(record["sample_id"]) for record in chosen]),
             metadata=json.dumps(metadata, sort_keys=True))
    validate_input(output, original_dictionary=original_dictionary, codebook=codebook)


def tensorflow_logits(inputs: dict, checkpoint: Path, original_config: Path) -> np.ndarray:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Original TensorFlow reference requires TensorFlow and original sketchformer runtime; install the original TF dependencies in a CPU environment") from exc
    tf.config.set_visible_devices([], "GPU")
    import sys
    from types import ModuleType, SimpleNamespace

    source = Path(__file__).resolve().parents[2] / "sketchformer"
    # The original release uses absolute imports (core/utils/builders/metrics).
    # Replace native package bindings only in this dedicated TF-export process.
    for prefix in ("core", "utils", "builders", "metrics"):
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
    sys.path.insert(0, str(source))
    package = ModuleType("original_sketchformer_models")
    package.__path__ = [str(source / "models")]
    sys.modules[package.__name__] = package
    try:
        from original_sketchformer_models.sketchformer import Transformer
        import builders as original_builders
    except ImportError as exc:
        raise RuntimeError(
            "Original TensorFlow runtime dependencies are missing; run export-tf "
            "in the released sketchformer TensorFlow environment on CPU"
        ) from exc

    hps = json.loads(Path(original_config).read_text(encoding="utf-8"))
    if hps.get("max_seq_len") != WIDTH or hps.get("token_type") != "dictionary" or hps.get("use_continuous_data"):
        raise ValueError("Original model config must be the released 200-position dictionary configuration")
    class InferenceTransformer(Transformer):
        def prepare_model_trainer(self):
            pass

    model = InferenceTransformer.__new__(InferenceTransformer)
    tf.keras.Model.__init__(model)
    model.hps = hps
    model.dataset = SimpleNamespace(hps={"use_continuous_data": False},
                                    tokenizer=SimpleNamespace(VOCAB_SIZE=LAYOUT["vocab_size"]), n_classes=345)
    model.vocab_size = LAYOUT["vocab_size"]
    model.seq_len = WIDTH
    model.losses_manager = original_builders.losses.LossManager()
    model.metrics_manager = original_builders.keras_metrics.MetricManager()
    model.build_model()
    data = tf.convert_to_tensor(inputs["tokens"], dtype=tf.int64)
    masks = original_builders.utils.create_masks(data, data[:, :-1])
    _ = model.call(data, data[:, :-1], False, *masks)  # materialize every inference weight
    status = tf.train.Checkpoint(transformer=model).restore(str(checkpoint))
    status.assert_existing_objects_matched()  # optimizer slots are not required for inference
    return model.call(data, data[:, :-1], False, *masks)["recon"].numpy().astype(np.float32)


def checkpoint_identity(prefix: Path) -> str:
    from models.sketchformer.pretrained import fingerprint_tensorflow_checkpoint

    return fingerprint_tensorflow_checkpoint(prefix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="Select untruncated short sequences from prepared data")
    select.add_argument("--config", default="configs/train.yaml")
    select.add_argument("--experiment", default="anime_tok_dict_long")
    select.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    select.add_argument("--samples", type=int, default=8)
    for sub in (select, commands.add_parser("export-tf", help="Run original TF checkpoint on saved real tokens")):
        sub.add_argument("--original-dictionary", type=Path, required=True)
        sub.add_argument("--codebook", type=Path, default=Path("data/processed/sketch_token/codebook.npy"))
    select.add_argument("--input", type=Path, required=True)
    export = commands.choices["export-tf"]
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--original-config", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        select_real_sequences(args.config, args.experiment, args.split, args.samples,
                              args.original_dictionary, args.codebook, args.input)
    else:
        inputs = validate_input(args.input, original_dictionary=args.original_dictionary, codebook=args.codebook)
        identity = checkpoint_identity(args.checkpoint)
        logits = tensorflow_logits(inputs, args.checkpoint, args.original_config)
        if logits.shape != (len(inputs["lengths"]), WIDTH - 1, LAYOUT["vocab_size"]):
            raise ValueError(f"Unexpected TF logit shape: {logits.shape}")
        meta = {"framework": "tensorflow_original", "input_sha256": _file_hash(args.input),
                "source_tf_checkpoint_sha256": identity, "original_config_sha256": _file_hash(args.original_config),
                "codebook_sha256": inputs["metadata"]["codebook_sha256"],
                "original_dictionary_sha256": inputs["metadata"]["original_dictionary_sha256"],
                "token_layout_json": json.dumps(LAYOUT, sort_keys=True, separators=(",", ":")),
                "logit_layout": "tokens[:, :-1] predict tokens[:, 1:]"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, logits=logits, tokens=inputs["tokens"], lengths=inputs["lengths"],
                 sample_ids=inputs["sample_ids"], metadata=json.dumps(meta, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
