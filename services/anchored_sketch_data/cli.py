"""Command-line interface for anchored V3 dataset preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import validate_dataset
from .builder import BuilderConfig, build_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tts-prepare-sketchformer-v3",
        description="Build an atomic, content-addressed anchored V3 sketch dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)
    build = subparsers.add_parser("build", help="Prepare images and publish a dataset")
    build.add_argument("--source-dir", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--calibration-size", type=int, default=256)
    build.add_argument("--augmentation-copies", type=int, default=1)
    build.add_argument(
        "--minimum-accepted-source-sketches",
        type=int,
        default=1,
        help=(
            "Fail unless this chosen number of original source sketches survives "
            "cleaning and encoding (default: 1)."
        ),
    )
    build.add_argument("--shard-size", type=int, default=1024)
    validate = subparsers.add_parser("validate", help="Validate an existing dataset")
    validate.add_argument("dataset_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required: build or validate")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "validate":
        metadata = validate_dataset(args.dataset_dir)
        print(
            f"Validated {args.dataset_dir}: "
            f"manifest={metadata['manifest_sha256']} codebook={metadata['codebook_sha256']}"
        )
        return 0
    destination = build_dataset(
        args.source_dir,
        args.output_root,
        config=BuilderConfig(
            seed=args.seed,
            calibration_size=args.calibration_size,
            train_augmentation_copies=args.augmentation_copies,
            minimum_accepted_source_sketches=args.minimum_accepted_source_sketches,
            shard_size=args.shard_size,
        ),
    )
    print(f"Published anchored V3 dataset to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
