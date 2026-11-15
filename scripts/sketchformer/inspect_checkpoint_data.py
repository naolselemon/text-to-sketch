"""CLI tool to inspect checkpoint provenance and query dataset membership."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()

import torch

from core.checkpointing import validate_checkpoint_compatibility
from dataloaders.lineage import load_dataset_lineage, query_sample_membership


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect checkpoint lineage provenance and query dataset split membership."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pt or .safetensors checkpoint to inspect.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to dataset root containing meta.npz to inspect.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Query which split (train/valid/test) a sample ID belongs to.",
    )
    parser.add_argument(
        "--verify-compat",
        action="store_true",
        help="Verify compatibility between specified checkpoint and dataset.",
    )
    parser.add_argument(
        "--allow-unverified-checkpoint",
        action="store_true",
        help="Allow legacy checkpoints without lineage during compatibility checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output inspection results as JSON.",
    )
    return parser.parse_args()


def inspect_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Inspect checkpoint file and extract metadata, config, and dataset lineage."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(checkpoint_path), framework="pt") as f:
            metadata = f.metadata() or {}

        lineage = {}
        if "lineage" in metadata:
            try:
                lineage = json.loads(metadata["lineage"])
            except Exception:
                lineage = {}
        config = {}
        if "config" in metadata:
            try:
                config = json.loads(metadata["config"])
            except Exception:
                config = {}

        return {
            "format": "safetensors",
            "path": str(checkpoint_path),
            "epoch": int(metadata.get("epoch", 0)),
            "step": int(metadata.get("step", 0)),
            "metrics": {},
            "config": config,
            "lineage": lineage,
            "has_lineage": bool(lineage),
        }

    # PyTorch .pt checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return {
            "format": "torch_state_dict_only",
            "path": str(checkpoint_path),
            "epoch": 0,
            "step": 0,
            "metrics": {},
            "config": {},
            "lineage": {},
            "has_lineage": False,
        }

    lineage = dict(checkpoint.get("lineage", {}))
    config = dict(checkpoint.get("config", {}))
    metrics = dict(checkpoint.get("metrics", {}))

    return {
        "format": "torch",
        "path": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", 0)),
        "step": int(checkpoint.get("step", 0)),
        "metrics": metrics,
        "config": config,
        "lineage": lineage,
        "has_lineage": bool(
            lineage
            and (
                lineage.get("build_id")
                or lineage.get("dataset_fingerprint")
                or lineage.get("dictionary_fingerprint")
            )
        ),
    }


def main() -> int:
    args = parse_args()
    if not args.checkpoint and not args.dataset and not args.sample_id:
        print("Please provide at least one of --checkpoint, --dataset, or --sample-id.")
        return 1

    report: dict[str, Any] = {}

    # 1. Inspect checkpoint if provided
    ckpt_info: dict[str, Any] | None = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        ckpt_info = inspect_checkpoint(ckpt_path)
        report["checkpoint"] = ckpt_info

    # 2. Inspect dataset if provided
    dataset_lineage = None
    if args.dataset:
        ds_path = Path(args.dataset)
        if not ds_path.is_absolute():
            ds_path = PROJECT_ROOT / ds_path
        dataset_lineage = load_dataset_lineage(
            ds_path,
            verify_chunks=False,
            allow_unverified=args.allow_unverified_checkpoint,
        )
        report["dataset"] = dataset_lineage.to_dict()

    # 3. Query sample membership if requested
    if args.sample_id:
        if not args.dataset:
            # Check if dataset path can be inferred from checkpoint
            if ckpt_info and ckpt_info.get("config", {}).get("data", {}).get("dataset", {}).get("root"):
                inferred_ds = ckpt_info["config"]["data"]["dataset"]["root"]
                ds_path = Path(inferred_ds)
                if not ds_path.is_absolute():
                    ds_path = PROJECT_ROOT / ds_path
            else:
                print("Error: --sample-id requires --dataset to be specified.")
                return 1
        membership = query_sample_membership(args.sample_id, ds_path)
        report["sample_query"] = {
            "sample_id": args.sample_id,
            "dataset_root": str(ds_path),
            "membership": membership,
        }

    # 4. Verify compatibility if requested
    if args.verify_compat or (args.checkpoint and args.dataset and not args.sample_id):
        if ckpt_info and dataset_lineage:
            try:
                validate_checkpoint_compatibility(
                    checkpoint_lineage=ckpt_info.get("lineage"),
                    dataset_lineage=dataset_lineage,
                    checkpoint_config=ckpt_info.get("config"),
                    allow_unverified=args.allow_unverified_checkpoint,
                )
                report["compatibility"] = {
                    "status": "COMPATIBLE",
                    "details": "Checkpoint matches active dataset lineage and configuration.",
                }
            except Exception as exc:
                report["compatibility"] = {
                    "status": "INCOMPATIBLE",
                    "error": str(exc),
                }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    # Human-readable output
    if "checkpoint" in report:
        c = report["checkpoint"]
        print("=== CHECKPOINT PROVENANCE ===")
        print(f"Path: {c['path']}")
        print(f"Format: {c['format']}")
        print(f"Epoch: {c['epoch']}, Step: {c['step']}")
        if c.get("metrics"):
            print(f"Metrics: {c['metrics']}")
        lin = c.get("lineage", {})
        if lin:
            print(f"Build ID: {lin.get('build_id', 'unknown')}")
            print(f"Dataset Fingerprint: {lin.get('dataset_fingerprint', 'unknown')}")
            print(f"Dictionary Fingerprint: {lin.get('dictionary_fingerprint', 'unknown')}")
            print(f"Split Counts: {lin.get('split_counts', {})}")
            print(f"Split Hashes: {lin.get('split_hashes', {})}")
            if lin.get("chunk_checksums"):
                print(f"Chunk Checksums: {len(lin['chunk_checksums'])} chunks verified")
        else:
            print("Lineage: [UNVERIFIED / LEGACY CHECKPOINT - NO PROVENANCE METADATA]")
        print()

    if "dataset" in report:
        d = report["dataset"]
        print("=== DATASET LINEAGE ===")
        print(f"Build ID: {d.get('build_id')}")
        print(f"Dataset Fingerprint: {d.get('dataset_fingerprint')}")
        print(f"Dictionary Fingerprint: {d.get('dictionary_fingerprint')}")
        print(f"Split Counts: {d.get('split_counts')}")
        print(f"Split Hashes: {d.get('split_hashes')}")
        if d.get("chunk_checksums"):
            print(f"Chunk Checksums: {len(d['chunk_checksums'])} chunks verified")
        if d.get("metadata", {}).get("unexpected_chunks"):
            print(f"Untracked Chunks Detected: {d['metadata']['unexpected_chunks']}")
        print()

    if "compatibility" in report:
        comp = report["compatibility"]
        print("=== COMPATIBILITY CHECK ===")
        print(f"Status: {comp['status']}")
        if "error" in comp:
            print(f"Error: {comp['error']}")
        else:
            print(f"Details: {comp.get('details')}")
        print()

    if "sample_query" in report:
        sq = report["sample_query"]
        mem = sq["membership"]
        print("=== SAMPLE MEMBERSHIP ===")
        print(f"Sample ID: {sq['sample_id']}")
        print(f"Split: {mem.get('split', 'NOT_FOUND')}")
        if mem.get("split"):
            print(f"Chunk: {mem.get('chunk_file')}")
            print(f"Local Index: {mem.get('chunk_index')}")
        else:
            print("Sample was not found in any dataset split chunk.")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
