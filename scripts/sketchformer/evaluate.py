"""Evaluate the native in-repo Sketchformer model."""

from __future__ import annotations

import argparse
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
import numpy as np

from builders import build_loss, build_model, maybe_compile_model
from builders.config_utils import get_nested
from core import average_logs, load_checkpoint, move_to_device
from core.metrics import reconstruction_metrics
from dataloaders import StrokeSequenceDataModule
from metrics.sketchformer.reconstruction import (
    attach_reconstruction_render_metadata,
    collect_generated_reconstruction_examples,
    collect_reconstruction_examples,
    load_reconstruction_render_metadata,
    write_metrics_report,
)
from metrics.sketchformer.free_running import (
    aggregate_free_running_records,
    free_running_reconstruction_records,
)
from prep_data.sketch_token.create_token_dict import load_codebook_from_dir
from scripts.sketchformer.config import (
    compose_training_config,
    format_logs,
    limited,
    parse_batch_limit,
    resolve_device,
)
from scripts.sketchformer.train import (
    _autocast_context,
    _configure_torch_runtime,
    _resolve_precision_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--limit-batches", type=parse_batch_limit, default=1.0)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--plots-output-dir", default=None)
    parser.add_argument("--num-plots", type=int, default=8)
    parser.add_argument(
        "--decode-mode",
        choices=("free-running", "teacher-forced"),
        default="free-running",
    )
    parser.add_argument("--max-generation-length", type=int, default=None)
    parser.add_argument(
        "--preprocessing-manifest",
        default=None,
        help=(
            "JSONL manifest containing source_path and Stroke5Transform metadata. "
            "When omitted, nearby preprocessing_manifest.jsonl files are detected."
        ),
    )
    parser.add_argument(
        "--source-images-root",
        default=None,
        help=(
            "Optional replacement root for manifest source_relative_path values, "
            "useful when source images moved after preprocessing."
        ),
    )
    parser.add_argument(
        "--enforce-long-sequence-gates",
        action="store_true",
        help="Require a non-empty 2049-4096 bucket with median geometry F1 >= 0.90.",
    )
    parser.add_argument(
        "--allow-unverified-checkpoint",
        action="store_true",
        help="Allow evaluating legacy checkpoints without dataset lineage metadata.",
    )
    parser.add_argument(
        "--allow-random-weights",
        action="store_true",
        help="Allow evaluating randomly initialized weights without a checkpoint (for testing/debug only).",
    )
    return parser.parse_args()


def _load_codebook_for_plots(config: dict[str, Any]) -> Any:
    if str(get_nested(config, "data.format.type", "stroke3")) not in {
        "tok_dict",
        "token",
        "tokens",
    }:
        return None

    codebook_dir = get_nested(config, "data.format.token_dictionary.codebook_dir")
    codebook_path = get_nested(config, "data.format.token_dictionary.codebook_path")
    if codebook_dir:
        codebook, _metadata = load_codebook_from_dir(PROJECT_ROOT / codebook_dir)
        return codebook
    if codebook_path:
        path = Path(codebook_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return np.load(path)
    raise ValueError(
        "plots for tok-dict evaluation require "
        "data.format.token_dictionary.codebook_dir or codebook_path"
    )


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _discover_preprocessing_manifest(
    config: dict[str, Any],
    explicit_path: str | None,
) -> Path | None:
    if explicit_path:
        manifest = _resolve_project_path(explicit_path)
        if not manifest.is_file():
            raise FileNotFoundError(f"Preprocessing manifest does not exist: {manifest}")
        return manifest

    dataset_root = _resolve_project_path(
        str(get_nested(config, "data.dataset.root"))
    )
    candidates: list[Path] = []
    report_path = dataset_root / "preparation_report.json"
    if report_path.is_file():
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))
        source_dir = report.get("source_dir")
        if source_dir:
            candidates.append(
                _resolve_project_path(str(source_dir)).parent
                / "preprocessing_manifest.jsonl"
            )
    candidates.extend(
        [
            dataset_root.parent / "preprocessing_manifest.jsonl",
            dataset_root.parent.parent / "preprocessing_manifest.jsonl",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def main() -> int:
    args = parse_args()
    if args.enforce_long_sequence_gates and args.decode_mode != "free-running":
        raise ValueError("--enforce-long-sequence-gates requires --decode-mode free-running")
    config = compose_training_config(args.config, experiment=args.experiment)
    if not args.checkpoint and not args.allow_random_weights:
        raise ValueError(
            "Evaluation requires a trained model checkpoint via --checkpoint. "
            "To evaluate uninitialized random weights for testing/debugging, pass --allow-random-weights."
        )

    if args.data_root:
        config["data"]["dataset"]["root"] = args.data_root
    if args.precision:
        config.setdefault("trainer", {}).setdefault("runtime", {})["precision"] = args.precision

    data_root_str = config.get("data", {}).get("dataset", {}).get("root")
    dataset_lineage = None
    if data_root_str:
        resolved_data_root = Path(data_root_str)
        if not resolved_data_root.is_absolute():
            resolved_data_root = PROJECT_ROOT / resolved_data_root
        meta_path = resolved_data_root / "meta.npz"
        if not meta_path.is_file():
            if not args.allow_unverified_checkpoint:
                raise RuntimeError(
                    f"Dataset metadata not found at {meta_path}. Cannot verify dataset provenance. "
                    "Pass --allow-unverified-checkpoint to bypass verification for legacy datasets."
                )
        else:
            from dataloaders.lineage import load_dataset_lineage

            dataset_lineage = load_dataset_lineage(
                resolved_data_root,
                verify_chunks=True,
                allow_unverified=args.allow_unverified_checkpoint,
            )

    load_result = None
    if args.checkpoint:
        from core import load_checkpoint_metadata, validate_checkpoint_compatibility

        ckpt_path = PROJECT_ROOT / args.checkpoint
        ckpt_meta = load_checkpoint_metadata(ckpt_path)

        validate_checkpoint_compatibility(
            ckpt_meta.get("lineage"),
            dataset_lineage,
            checkpoint_config=ckpt_meta.get("config"),
            active_config=config,
            split=args.split,
            allow_unverified=args.allow_unverified_checkpoint,
        )

    device = resolve_device(args.device)
    _configure_torch_runtime(config, device)
    precision = _resolve_precision_runtime(config, device)
    datamodule = StrokeSequenceDataModule(
        config["data"],
        project_root=PROJECT_ROOT,
        lineage=dataset_lineage,
        allow_unverified=args.allow_unverified_checkpoint,
    )
    datamodule.setup("test" if args.split == "test" else "fit")
    loader = datamodule.test_dataloader() if args.split == "test" else datamodule.val_dataloader()

    raw_model = build_model(config["model"])
    if args.checkpoint:
        ckpt_path = PROJECT_ROOT / args.checkpoint
        load_result = load_checkpoint(ckpt_path, raw_model, strict=False)
    else:
        print("[warning] evaluating randomly initialized model (--allow-random-weights passed)")
    model = maybe_compile_model(raw_model.to(device), config["model"])

    loss_fn = build_loss(config["optimizer"]).to(device)
    model.eval()
    needs_codebook = args.plots_output_dir or args.decode_mode == "free-running"
    codebook = _load_codebook_for_plots(config) if needs_codebook else None
    render_metadata = {}
    render_manifest = None
    if args.plots_output_dir:
        render_manifest = _discover_preprocessing_manifest(
            config,
            args.preprocessing_manifest,
        )
        if render_manifest is not None:
            render_metadata = load_reconstruction_render_metadata(
                render_manifest,
                project_root=PROJECT_ROOT,
                source_images_root=args.source_images_root,
            )
            print(
                f"[plots] preprocessing_manifest={render_manifest} "
                f"lookup_keys={len(render_metadata)}"
            )
        else:
            print(
                "[plots] no preprocessing manifest found; "
                "using a shared normalized raster canvas"
            )

    logs: list[dict[str, torch.Tensor]] = []
    free_running_records: list[dict[str, float | int | str]] = []
    examples = []
    with torch.no_grad():
        for batch in limited(loader, args.limit_batches):
            batch = move_to_device(batch, device)
            with _autocast_context(device, precision):
                if args.decode_mode == "teacher-forced":
                    output = model(batch)
                    loss_output = loss_fn(output, batch)
                    metric_output = reconstruction_metrics(output, batch)
                    step_logs = loss_output.as_log_dict(prefix=args.split)
                    step_logs.update(metric_output.as_log_dict(prefix=args.split))
                    generation = None
                else:
                    if codebook is None:
                        raise ValueError("free-running evaluation requires the token codebook")
                    generation = raw_model.generate(
                        batch,
                        max_length=args.max_generation_length,
                        use_cache=True,
                    )
                    batch_records = free_running_reconstruction_records(
                        generation.tokens,
                        generation.lengths,
                        batch,
                        codebook,
                        eos_token_id=int(
                            get_nested(config, "data.format.token_dictionary.eos_token_id")
                        ),
                    )
                    free_running_records.extend(batch_records)
                    step_logs = {}
            if step_logs:
                logs.append(step_logs)

            remaining_examples = args.num_plots - len(examples)
            if args.plots_output_dir and remaining_examples > 0:
                if args.decode_mode == "free-running":
                    assert generation is not None and codebook is not None
                    examples.extend(
                        collect_generated_reconstruction_examples(
                            generation,
                            batch,
                            max_examples=remaining_examples,
                            codebook=codebook,
                        )
                    )
                else:
                    examples.extend(
                        collect_reconstruction_examples(
                            output,
                            batch,
                            max_examples=remaining_examples,
                            codebook=codebook,
                        )
                    )

    if args.decode_mode == "free-running":
        free_summary = aggregate_free_running_records(
            free_running_records,
            device=device,
        )
        summary = {f"{args.split}/{key}": value for key, value in free_summary.items()}
    else:
        weight_key = f"{args.split}/valid_tokens"
        summary = average_logs(
            logs,
            weight_key=weight_key if logs and weight_key in logs[0] else None,
        )
    print(format_logs(summary))

    if args.enforce_long_sequence_gates:
        metric_prefix = f"{args.split}/free_running"
        count = float(summary[f"{metric_prefix}/count_length_2049_4096"])
        median_f1 = float(
            summary[f"{metric_prefix}/geometry_f1_2px_median_length_2049_4096"]
        )
        if count <= 0 or median_f1 < 0.90:
            raise SystemExit(
                "Long-sequence free-running gate failed: "
                f"count_2049_4096={count:.0f} median_f1={median_f1:.4f}"
            )

    if args.metrics_output:
        metrics_path = PROJECT_ROOT / args.metrics_output
        write_metrics_report(
            metrics_path,
            summary,
            metadata={
                "experiment": args.experiment,
                "split": args.split,
                "checkpoint": args.checkpoint,
                "checkpoint_lineage": load_result.lineage if load_result else {},
                "data_root": config["data"]["dataset"]["root"],
                "device": str(device),
                "precision": precision.effective,
                "limit_batches": args.limit_batches,
                "decode_mode": args.decode_mode,
                "max_generation_length": args.max_generation_length,
                "enforce_long_sequence_gates": args.enforce_long_sequence_gates,
                "preprocessing_manifest": (
                    str(render_manifest) if render_manifest is not None else None
                ),
                "source_images_root": args.source_images_root,
            },
        )
        print(f"[metrics] wrote {metrics_path}")

    if args.plots_output_dir and examples:
        from metrics.sketchformer.visualisation import save_reconstruction_examples

        if render_metadata:
            examples = attach_reconstruction_render_metadata(
                examples,
                render_metadata,
            )
            matched = sum(
                example.canvas_transform is not None for example in examples
            )
            print(f"[plots] manifest_matches={matched}/{len(examples)}")
        plot_dir = PROJECT_ROOT / args.plots_output_dir
        saved = save_reconstruction_examples(examples, plot_dir)
        print(f"[plots] wrote {len(saved)} reconstruction plots to {plot_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
