"""Export converted PyTorch logits and gate against an original TensorFlow export.

Reconvert the original TensorFlow checkpoint so its codebook and source identity
are embedded (the older converted checkpoint cannot pass this gate)::

  python -m scripts.sketchformer.convert_checkpoint --experiment anime_tok_dict_long --source weights/pretrained/sketch-transformer-tf2-cvpr_tform_tok_dict/weights/ckpt-12 --output weights/pretrained/sketchformer_tok_dict_4096_identity.safetensors

CPU example, after ``tf_reference_export select`` and ``export-tf``::

  python -m scripts.sketchformer.parity_check export-torch --input /tmp/short-input.npz --checkpoint weights/pretrained/sketchformer_tok_dict_4096_identity.safetensors --original-checkpoint weights/pretrained/sketch-transformer-tf2-cvpr_tform_tok_dict/weights/ckpt-12 --original-dictionary sketchformer/prep_data/sketch_token/token_dict.pkl --output /tmp/torch-short.npz
  python -m scripts.sketchformer.parity_check compare --input /tmp/short-input.npz --reference /tmp/tf-short.npz --candidate /tmp/torch-short.npz --checkpoint weights/pretrained/sketchformer_tok_dict_4096_identity.safetensors --original-checkpoint weights/pretrained/sketch-transformer-tf2-cvpr_tform_tok_dict/weights/ckpt-12 --original-config weights/pretrained/sketch-transformer-tf2-cvpr_tform_tok_dict/config.json --original-dictionary sketchformer/prep_data/sketch_token/token_dict.pkl

The released 200-position TensorFlow model must be the reference. No random
tokens, native-model reference, or unverified logit file can pass this gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evals.conversion_fidelity import _file_hash, evaluate_verified_parity
from scripts.sketchformer.tf_reference_export import LAYOUT, WIDTH, validate_input


def export_converted(
    input_path: Path,
    checkpoint: Path,
    original_checkpoint: Path,
    original_dictionary: Path,
    codebook: Path,
    output_path: Path,
    *,
    config: str = "configs/train.yaml",
    experiment: str = "anime_tok_dict_long",
) -> None:
    import torch
    from safetensors import safe_open
    from models.sketchformer.pretrained import fingerprint_tensorflow_checkpoint
    from builders import build_model
    from dataloaders.masks import build_sequence_masks
    from scripts.sketchformer.config import compose_training_config
    from scripts.sketchformer.train import _load_state_dict_file

    input_path, checkpoint, output_path = map(Path, (input_path, checkpoint, output_path))
    inputs = validate_input(
        input_path, original_dictionary=original_dictionary, codebook=codebook
    )
    source_hash = fingerprint_tensorflow_checkpoint(original_checkpoint)
    layout = json.dumps(LAYOUT, sort_keys=True, separators=(",", ":"))
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
    expected = {
        "codebook_sha256": inputs["metadata"]["codebook_sha256"],
        "source_tf_checkpoint_sha256": source_hash,
        "token_layout_json": layout,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Converted checkpoint lacks matching original checkpoint/codebook/token-layout identity")
    cfg = compose_training_config(config, experiment=experiment)
    model_config = cfg["model"]
    if model_config["input"]["type"] != "tok_dict" or int(model_config["architecture"]["latent_expander_base_length"]) != WIDTH:
        raise ValueError("Candidate must have the 200-position TF-compatible tok-dict memory base")
    if any(int(model_config["input"]["token_dictionary"][key]) != value for key, value in LAYOUT.items()):
        raise ValueError("Candidate token dictionary layout differs from original")
    model = build_model(model_config)
    model.load_state_dict(_load_state_dict_file(checkpoint), strict=True)
    model.eval().cpu()
    tokens = torch.from_numpy(inputs["tokens"]).long()
    batch = {
        "tokens": tokens,
        "targets": tokens,
        **build_sequence_masks(
            inputs["lengths"].tolist(), max_length=WIDTH, device=tokens.device,
        ),
    }
    with torch.no_grad():
        prediction = model(batch).reconstruction.token_logits
    logits = prediction.cpu().float().numpy()
    if logits.shape != (len(tokens), WIDTH - 1, LAYOUT["vocab_size"]):
        raise ValueError(f"Unexpected candidate logits shape: {logits.shape}")
    export_metadata = {
        "framework": "pytorch_converted",
        "input_sha256": _file_hash(input_path),
        "source_tf_checkpoint_sha256": source_hash,
        "converted_checkpoint_sha256": _file_hash(checkpoint),
        "original_dictionary_sha256": inputs["metadata"]["original_dictionary_sha256"],
        "codebook_sha256": inputs["metadata"]["codebook_sha256"],
        "token_layout_json": layout,
        "logit_layout": "tokens[:, :-1] predict tokens[:, 1:]",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, logits=logits, tokens=inputs["tokens"],
             lengths=inputs["lengths"], sample_ids=inputs["sample_ids"],
             metadata=json.dumps(export_metadata, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export-torch", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--original-checkpoint", type=Path, required=True)
        command.add_argument("--original-dictionary", type=Path, required=True)
        command.add_argument("--codebook", type=Path, default=Path("data/processed/sketch_token/codebook.npy"))
    commands.choices["compare"].add_argument("--original-config", type=Path, required=True)
    export = commands.choices["export-torch"]
    export.add_argument("--config", default="configs/train.yaml")
    export.add_argument("--experiment", default="anime_tok_dict_long")
    export.add_argument("--output", type=Path, required=True)
    compare = commands.choices["compare"]
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path)
    compare.add_argument("--min-cosine", type=float, default=0.999)
    compare.add_argument("--min-agreement", type=float, default=0.99)
    compare.add_argument("--max-abs-diff", type=float, default=0.01)
    args = parser.parse_args()
    if args.command == "export-torch":
        export_converted(
            args.input, args.checkpoint, args.original_checkpoint,
            args.original_dictionary, args.codebook, args.output,
            config=args.config, experiment=args.experiment,
        )
        return 0
    report = evaluate_verified_parity(
        args.candidate, args.reference, args.input,
        candidate_checkpoint=args.checkpoint,
        original_checkpoint=args.original_checkpoint,
        original_config=args.original_config,
        original_dictionary=args.original_dictionary,
        codebook=args.codebook,
        min_cosine=args.min_cosine,
        min_agreement=args.min_agreement,
        max_abs_diff=args.max_abs_diff,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
