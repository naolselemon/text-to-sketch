"""Evaluate the native in-repo Sketchformer model."""

from __future__ import annotations

import argparse
import hashlib
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
from core import (
    average_logs,
    checkpoint_compatibility_config,
    load_checkpoint,
    move_to_device,
)
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
from scripts.sketchformer.config import (
    compose_training_config,
    format_logs,
    limited,
    parse_batch_limit,
    pin_anchored_v3_artifacts,
    resolve_device,
)
from scripts.sketchformer.train import (
    _autocast_context,
    _configure_torch_runtime,
    _resolve_precision_runtime,
)
from services.anchored_sketch_data.contract import (
    TOKEN_LAYOUT as ANCHORED_V3_TOKEN_LAYOUT,
    validate_anchored_v3_runtime_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="valid",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--train-source-limit",
        type=int,
        default=None,
        help="Match a nested training-subset value stored in a strict checkpoint.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--limit-batches", type=parse_batch_limit, default=1.0)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--plots-output-dir", default=None)
    parser.add_argument("--num-plots", type=int, default=8)
    parser.add_argument(
        "--human-review-template",
        default=None,
        help="Write the fixed 100-plot V3 manual-review template.",
    )
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
        "--enforce-v3-gates",
        action="store_true",
        help="Enforce automated target-faithful V3 release thresholds.",
    )
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help=(
            "Allow a trusted pre-contract V2 checkpoint. Tensor keys still load "
            "strictly; artifact/config/hash validation is skipped. V3 rejects this flag."
        ),
    )
    return parser.parse_args()


def _codebook_path(config: dict[str, Any]) -> Path | None:
    if str(get_nested(config, "data.format.type", "stroke3")) not in {
        "tok_dict",
        "token",
        "tokens",
        "anchored_v3",
    }:
        return None

    direct_path = get_nested(config, "data.format.codebook_path")
    codebook_dir = get_nested(config, "data.format.token_dictionary.codebook_dir")
    codebook_path = get_nested(config, "data.format.token_dictionary.codebook_path")
    dataset_root = _project_path(get_nested(config, "data.dataset.root"))
    if direct_path:
        return _project_path(direct_path)
    if codebook_path:
        return _project_path(codebook_path)
    if codebook_dir:
        return _project_path(codebook_dir) / "codebook.npy"
    candidate = dataset_root / "codebook.npy"
    if candidate.is_file():
        return candidate
    raise ValueError(
        "token evaluation requires data.format.codebook_path, "
        "data.format.token_dictionary.codebook_path/codebook_dir, or dataset/codebook.npy"
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


def _load_codebook(config: dict[str, Any]) -> tuple[np.ndarray | None, Path | None]:
    path = _codebook_path(config)
    if path is None:
        return None, None
    if not path.is_file():
        raise FileNotFoundError(f"Codebook does not exist: {path}")
    return np.load(path), path


def _token_layout(config: dict[str, Any]) -> dict[str, Any]:
    format_config = dict(get_nested(config, "data.format", {}) or {})
    nested_layout = format_config.get("token_layout")
    if isinstance(nested_layout, dict):
        layout = dict(nested_layout)
    else:
        layout = dict(format_config.get("token_dictionary", {}) or {})
    layout.setdefault("type", format_config.get("type", "stroke3"))
    if "version" in format_config:
        layout.setdefault("version", format_config["version"])
    if "token_layout_version" in format_config:
        layout.setdefault("version", format_config["token_layout_version"])
    return layout


def _manifest_path(config: dict[str, Any]) -> Path:
    manifest_path = get_nested(config, "data.dataset.manifest_path")
    dataset_root = _project_path(get_nested(config, "data.dataset.root"))
    if manifest_path:
        path = _project_path(manifest_path)
    else:
        manifest_file = get_nested(config, "data.dataset.manifest_file", "manifest.jsonl")
        path = dataset_root / str(manifest_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"Strict checkpoint validation requires the dataset manifest: {path}"
        )
    return path


def _expected_checkpoint_contract(
    config: dict[str, Any],
    *,
    codebook_path: Path | None,
) -> dict[str, Any]:
    layout = _token_layout(config)
    format_config = dict(get_nested(config, "data.format", {}) or {})
    version = format_config.get(
        "token_layout_version",
        format_config.get("version", layout.get("version", "legacy_v2")),
    )
    manifest_path = _manifest_path(config)
    return {
        "schema_version": 1,
        "compatibility_config": checkpoint_compatibility_config(config),
        "token_layout_version": version,
        "codebook_sha256": (
            _sha256_file(codebook_path) if codebook_path is not None else None
        ),
        "dataset_manifest_sha256": _sha256_file(manifest_path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _token_id(layout: dict[str, Any], name: str, fallback: int | None = None) -> int:
    value = layout.get(name, fallback)
    if value is None:
        raise ValueError(f"Token layout does not define {name}")
    return int(value)


def main() -> int:
    args = parse_args()
    if (args.enforce_long_sequence_gates or args.enforce_v2_gates or args.enforce_v3_gates) and args.decode_mode != "free-running":
        raise ValueError("release gates require --decode-mode free-running")
    config = compose_training_config(args.config, experiment=args.experiment)
    if args.data_root:
        config["data"]["dataset"]["root"] = args.data_root
        if str(get_nested(config, "data.format.type")) == "anchored_v3":
            config["data"]["format"]["token_dictionary"]["codebook_path"] = str(
                Path(args.data_root) / "codebook.npy"
            )
    if args.train_source_limit is not None:
        if args.train_source_limit <= 0:
            raise ValueError("--train-source-limit must be positive")
        config["data"]["dataset"]["train_source_limit"] = args.train_source_limit
    if args.precision:
        config.setdefault("trainer", {}).setdefault("runtime", {})["precision"] = args.precision
    _validate_evaluation_request(args, config)
    pin_anchored_v3_artifacts(config, project_root=PROJECT_ROOT)

    device = resolve_device(args.device)
    _configure_torch_runtime(config, device)
    precision = _resolve_precision_runtime(config, device)
    codebook, codebook_path = _load_codebook(config)
    token_layout = _token_layout(config)
    datamodule = StrokeSequenceDataModule(config["data"], project_root=PROJECT_ROOT)
    datamodule.setup("test" if args.split == "test" else "fit")
    if args.split == "test":
        loader = datamodule.test_dataloader()
    elif args.split == "train":
        loader = datamodule.train_dataloader()
    else:
        loader = datamodule.val_dataloader()
    if str(get_nested(config, "data.format.type")) == "anchored_v3":
        from services.anchored_sketch_data.artifacts import (
            require_minimum_source_sketches,
        )

        metadata = getattr(loader.dataset, "metadata", None)
        if not isinstance(metadata, dict):
            raise ValueError("Anchored V3 evaluation dataset has no validated metadata")
        require_minimum_source_sketches(
            metadata,
        )

    raw_model = build_model(config["model"])
    if args.checkpoint:
        checkpoint_result = load_checkpoint(
            _project_path(args.checkpoint),
            raw_model,
            strict=True,
            expected_contract=(
                None
                if args.allow_legacy_checkpoint
                else _expected_checkpoint_contract(config, codebook_path=codebook_path)
            ),
            require_contract=not args.allow_legacy_checkpoint,
        )
        if checkpoint_result.legacy:
            print("[warning] trusted legacy checkpoint loaded without artifact contract")
    else:
        print("[warning] evaluating randomly initialized model; pass --checkpoint for trained weights")
    model = maybe_compile_model(raw_model.to(device), config["model"])

    loss_fn = build_loss(
        config["optimizer"],
        data_config=config["data"],
        project_root=PROJECT_ROOT,
    ).to(device)
    model.eval()
    needs_codebook = args.plots_output_dir or args.decode_mode == "free-running"
    codebook = _load_codebook_for_plots(config) if needs_codebook else None
    if args.decode_mode == "free-running" and codebook is None:
        raise ValueError("token reconstruction evaluation requires a codebook")
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
                        eos_token_id=_token_id(token_layout, "eos_token_id"),
                        token_layout=token_layout,
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
                            token_layout=token_layout,
                            records=batch_records,
                        )
                    )
                else:
                    examples.extend(
                        collect_reconstruction_examples(
                            output,
                            batch,
                            max_examples=remaining_examples,
                            codebook=codebook,
                            token_layout=token_layout,
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

    if args.enforce_v3_gates:
        _enforce_v3_gates(summary, split=args.split)

    if args.metrics_output:
        metrics_path = PROJECT_ROOT / args.metrics_output
        write_metrics_report(
            metrics_path,
            summary,
            metadata={
                "experiment": args.experiment,
                "split": args.split,
                "checkpoint": args.checkpoint,
                "data_root": config["data"]["dataset"]["root"],
                "device": str(device),
                "precision": precision.effective,
                "limit_batches": args.limit_batches,
                "format_type": get_nested(config, "data.format.type"),
                "format_version": get_nested(config, "data.format.version"),
                "token_layout": (
                    ANCHORED_V3_TOKEN_LAYOUT.to_dict()
                    if str(get_nested(config, "data.format.type")) == "anchored_v3"
                    else None
                ),
                "decoder_memory_source": get_nested(
                    config, "model.decoder.memory_source"
                ),
                "decode_mode": args.decode_mode,
                "max_generation_length": args.max_generation_length,
                "enforce_long_sequence_gates": args.enforce_long_sequence_gates,
                "enforce_v2_gates": args.enforce_v2_gates,
                "enforce_v3_gates": args.enforce_v3_gates,
                "allow_legacy_checkpoint": args.allow_legacy_checkpoint,
                "human_review_template": args.human_review_template,
                "checkpoint_contract": (
                    checkpoint_result.contract if args.checkpoint else None
                ),
                "preprocessing_manifest": (
                    str(render_manifest) if render_manifest is not None else None
                ),
                "source_images_root": args.source_images_root,
            },
            records=(free_running_records if args.decode_mode == "free-running" else None),
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
        saved = save_reconstruction_examples(
            examples,
            plot_dir,
            prefix="reconstruction",
        )
        print(f"[plots] wrote {len(saved)} reconstruction plots to {plot_dir}")
        if args.human_review_template:
            if len(examples) != 100:
                raise RuntimeError(
                    f"Human review requires 100 plots, but only {len(examples)} were created"
                )
            from scripts.sketchformer.human_review import (
                build_review_template,
                sha256_file,
                write_json_atomic,
            )

            template = build_review_template(
                [str(example.sample_id) for example in examples],
                evaluation_report_sha256=sha256_file(metrics_path),
                plot_records=[
                    {
                        "sample_id": str(example.sample_id),
                        "plot_path": str(plot_path.resolve()),
                        "plot_sha256": sha256_file(plot_path),
                    }
                    for example, plot_path in zip(examples, saved)
                ],
            )
            template_path = write_json_atomic(
                _project_path(args.human_review_template),
                template,
            )
            print(f"[human-review] wrote template {template_path}")

    return 0


def _validate_evaluation_request(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    """Keep V3 release evaluation strict and impossible to downgrade silently."""

    is_v3 = str(get_nested(config, "data.format.type")) == "anchored_v3"
    if is_v3:
        validate_anchored_v3_runtime_config(config)
    if is_v3 and args.allow_legacy_checkpoint:
        raise ValueError("--allow-legacy-checkpoint is restricted to trusted V2 artifacts")
    if args.enforce_v3_gates and not is_v3:
        raise ValueError("--enforce-v3-gates requires data.format.type=anchored_v3")
    if args.enforce_v3_gates:
        failures: list[str] = []
        if args.split != "test":
            failures.append("--split test")
        if args.decode_mode != "free-running":
            failures.append("--decode-mode free-running")
        if not args.checkpoint:
            failures.append("--checkpoint")
        if not isinstance(args.limit_batches, float) or args.limit_batches != 1.0:
            failures.append("--limit-batches 1.0")
        if args.max_generation_length is not None:
            failures.append("no --max-generation-length override")
        if not args.metrics_output:
            failures.append("--metrics-output")
        if failures:
            raise ValueError(
                "V3 release gates require the complete held-out test contract: "
                + ", ".join(failures)
            )
    if args.human_review_template:
        failures = []
        if not is_v3:
            failures.append("anchored_v3 format")
        if not args.enforce_v3_gates:
            failures.append("--enforce-v3-gates")
        if not args.plots_output_dir:
            failures.append("--plots-output-dir")
        if args.num_plots != 100:
            failures.append("--num-plots 100")
        if not args.metrics_output:
            failures.append("--metrics-output")
        if failures:
            raise ValueError(
                "--human-review-template requires: " + ", ".join(failures)
            )


def _enforce_v3_gates(
    summary: dict[str, torch.Tensor],
    *,
    split: str,
) -> None:
    prefix = f"{split}/free_running"
    checks = {
        "median_f1_2px": (
            float(summary[f"{prefix}/geometry_f1_2px_median"]),
            ">=",
            0.95,
        ),
        "long_median_f1_2px": (
            float(summary[f"{prefix}/geometry_f1_2px_median_length_2049_4096"]),
            ">=",
            0.90,
        ),
        "p10_f1_2px": (
            float(summary[f"{prefix}/geometry_f1_2px_p10"]),
            ">=",
            0.85,
        ),
        "p95_chamfer_px": (
            float(summary[f"{prefix}/symmetric_chamfer_px_p95"]),
            "<=",
            3.0,
        ),
        "premature_eos_rate": (
            float(summary[f"{prefix}/premature_eos_rate"]),
            "<=",
            0.02,
        ),
        "max_length_hit_rate": (
            float(summary[f"{prefix}/max_length_hit_rate"]),
            "<=",
            0.02,
        ),
    }
    long_count = float(summary[f"{prefix}/count_length_2049_4096"])
    failures = [] if long_count > 0 else ["long_length_count=0"]
    for name, (value, operator, threshold) in checks.items():
        passed = value >= threshold if operator == ">=" else value <= threshold
        if not passed:
            failures.append(f"{name}={value:.6g} expected {operator}{threshold:.6g}")
    if failures:
        raise SystemExit("V3 automated release gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    raise SystemExit(main())
