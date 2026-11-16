"""Export native Sketchformer weights from a training checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()

import json

import torch

from builders import build_model
from core import load_checkpoint, validate_checkpoint_compatibility
from scripts.sketchformer.config import compose_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-unverified-checkpoint",
        action="store_true",
        help="Allow exporting legacy checkpoints without dataset lineage metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = compose_training_config(args.config, experiment=args.experiment)
    ckpt_path = PROJECT_ROOT / args.checkpoint

    from core import load_checkpoint_metadata, validate_checkpoint_compatibility

    ckpt_meta = load_checkpoint_metadata(ckpt_path)
    checkpoint_lineage = ckpt_meta.get("lineage", {})
    checkpoint_config = ckpt_meta.get("config", {})

    data_root_str = config.get("data", {}).get("dataset", {}).get("root")
    dataset_lineage = None
    if data_root_str:
        resolved_data_root = Path(data_root_str)
        if not resolved_data_root.is_absolute():
            resolved_data_root = PROJECT_ROOT / resolved_data_root
        if (resolved_data_root / "meta.npz").is_file():
            from dataloaders.lineage import load_dataset_lineage

            dataset_lineage = load_dataset_lineage(
                resolved_data_root,
                verify_chunks=True,
                allow_unverified=args.allow_unverified_checkpoint,
            )

    validate_checkpoint_compatibility(
        checkpoint_lineage,
        dataset_lineage,
        checkpoint_config=checkpoint_config,
        active_config=config,
        allow_unverified=args.allow_unverified_checkpoint,
    )

    model = build_model(config["model"])
    load_result = load_checkpoint(ckpt_path, model, strict=False)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()

    if output_path.suffix == ".safetensors":
        from safetensors.torch import save_file

        metadata = {
            "config": json.dumps(config),
            "lineage": json.dumps(load_result.lineage),
            "epoch": str(load_result.epoch),
            "step": str(load_result.step),
            "build_id": str(load_result.lineage.get("build_id", "")),
            "dataset_fingerprint": str(load_result.lineage.get("dataset_fingerprint", "")),
            "dictionary_fingerprint": str(load_result.lineage.get("dictionary_fingerprint", "")),
        }
        save_file(state_dict, str(output_path), metadata=metadata)
    else:
        torch.save(
            {
                "state_dict": state_dict,
                "config": config,
                "lineage": load_result.lineage,
                "epoch": load_result.epoch,
                "step": load_result.step,
            },
            output_path,
        )

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
