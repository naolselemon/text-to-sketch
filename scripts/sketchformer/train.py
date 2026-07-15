"""Train the native in-repo Sketchformer model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
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

from builders import (
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
    maybe_compile_model,
)
from builders.config_utils import get_nested
from core import (
    CheckpointCallback,
    CheckpointContract,
    CheckpointContractError,
    average_logs,
    current_git_commit,
    move_to_device,
    set_seed,
)
from core.metrics import reconstruction_metrics
from dataloaders import StrokeSequenceDataModule
from scripts.sketchformer.config import (
    batch_limit,
    compose_training_config,
    format_logs,
    limited,
    parse_batch_limit,
    pin_anchored_v3_artifacts,
    resolve_device,
)
from scripts.sketchformer.curriculum import (
    build_stage_parameter_groups,
    parse_curriculum,
    resume_epoch_for_stage,
)
from scripts.sketchformer.exposure import (
    AnchoredTokenRanges,
    corrupt_decoder_prefixes,
    scheduled_sampling_probability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--train-source-limit",
        type=int,
        default=None,
        help="Use a deterministic nested subset of this many original train sketches.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=parse_batch_limit, default=None)
    parser.add_argument("--limit-val-batches", type=parse_batch_limit, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overfit-gate-report", default=None)
    return parser.parse_args()


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
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
    if args.output_dir:
        config.setdefault("experiment", {}).setdefault("run", {})[
            "output_dir"
        ] = args.output_dir
    if args.pretrained:
        pretrained = config.setdefault("experiment", {}).setdefault("pretrained", {})
        pretrained["use_converted_sketchformer_weights"] = True
        pretrained["path"] = args.pretrained
    if args.resume:
        config.setdefault("experiment", {}).setdefault("run", {})[
            "resume_from_checkpoint"
        ] = args.resume
    if args.precision:
        config.setdefault("trainer", {}).setdefault("runtime", {})["precision"] = args.precision
    if args.max_epochs is not None:
        config["trainer"]["training"]["max_epochs"] = args.max_epochs
    if args.limit_train_batches is not None:
        config["trainer"]["training"]["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        config["trainer"]["training"]["limit_val_batches"] = args.limit_val_batches
    if args.overfit_gate_report:
        config.setdefault("trainer", {}).setdefault("gates", {})[
            "overfit_report"
        ] = args.overfit_gate_report


def _checkpoint_dir(config: dict[str, Any]) -> Path:
    output_dir = get_nested(config, "experiment.run.output_dir")
    if output_dir:
        return PROJECT_ROOT / output_dir

    output_root = get_nested(config, "paths.output_root", "weights/finetuned")
    run_name = get_nested(config, "experiment.name", "sketchformer-run")
    return PROJECT_ROOT / output_root / run_name


def _total_optimizer_steps(config: dict[str, Any], train_loader: Any) -> int:
    max_epochs = int(get_nested(config, "trainer.training.max_epochs", 1))
    limit_train = get_nested(config, "trainer.training.limit_train_batches", 1.0)
    accumulate = int(get_nested(config, "trainer.training.accumulate_grad_batches", 1))
    batches = batch_limit(train_loader, limit_train)
    return max(1, math.ceil(batches / max(1, accumulate)) * max_epochs)


def _divide_gradients(model: torch.nn.Module, divisor: float) -> None:
    if divisor <= 0:
        raise ValueError("gradient divisor must be positive")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(divisor)


_TRAINING_STATE_SCHEMA_VERSION = 1


def _full_validation_training_state(
    *,
    stage_index: int,
    stage_name: str,
    completed_stage_epochs: int,
    monitor: str,
    mode: str,
    best: float | None,
    non_improving: int,
) -> dict[str, Any]:
    """Build the strict resume state written after a full validation."""

    if mode not in {"min", "max"}:
        raise ValueError("checkpoint mode must be 'min' or 'max'")
    if non_improving < 0:
        raise ValueError("non_improving must be non-negative")
    return {
        "schema_version": _TRAINING_STATE_SCHEMA_VERSION,
        "curriculum_stage": {
            "index": int(stage_index),
            "name": str(stage_name),
            "completed_epochs": int(completed_stage_epochs),
        },
        "full_validation_early_stopping": {
            "monitor": str(monitor),
            "mode": mode,
            "best": None if best is None else float(best),
            "non_improving": int(non_improving),
        },
    }


def _restore_full_validation_early_stopping(
    training_state: dict[str, Any] | None,
    *,
    stage_index: int,
    stage_name: str,
    completed_stage_epochs: int,
    monitor: str,
    mode: str,
) -> tuple[float, int]:
    """Strictly restore the final-stage full-validation stopping history."""

    if not isinstance(training_state, dict):
        raise CheckpointContractError(
            "Resume checkpoint has no mapping training_state"
        )
    if training_state.get("schema_version") != _TRAINING_STATE_SCHEMA_VERSION:
        raise CheckpointContractError(
            "Resume checkpoint training_state schema_version mismatch"
        )
    stage = training_state.get("curriculum_stage")
    if not isinstance(stage, dict):
        raise CheckpointContractError(
            "Resume checkpoint training_state has no curriculum_stage"
        )
    expected_stage = {
        "index": int(stage_index),
        "name": str(stage_name),
        "completed_epochs": int(completed_stage_epochs),
    }
    actual_stage = {
        "index": stage.get("index"),
        "name": stage.get("name"),
        "completed_epochs": stage.get("completed_epochs"),
    }
    if actual_stage != expected_stage:
        raise CheckpointContractError(
            "Resume checkpoint curriculum stage mismatch: "
            f"checkpoint={actual_stage!r} runtime={expected_stage!r}"
        )
    early_stopping = training_state.get("full_validation_early_stopping")
    if not isinstance(early_stopping, dict):
        raise CheckpointContractError(
            "Resume checkpoint training_state has no "
            "full_validation_early_stopping"
        )
    if early_stopping.get("monitor") != monitor:
        raise CheckpointContractError(
            "Resume checkpoint full-validation monitor mismatch"
        )
    if early_stopping.get("mode") != mode:
        raise CheckpointContractError(
            "Resume checkpoint full-validation mode mismatch"
        )
    best = early_stopping.get("best")
    if isinstance(best, bool) or not isinstance(best, (int, float)):
        raise CheckpointContractError(
            "Resume checkpoint full-validation best must be numeric"
        )
    best = float(best)
    if not math.isfinite(best):
        raise CheckpointContractError(
            "Resume checkpoint full-validation best must be finite"
        )
    non_improving = early_stopping.get("non_improving")
    if (
        isinstance(non_improving, bool)
        or not isinstance(non_improving, int)
        or non_improving < 0
    ):
        raise CheckpointContractError(
            "Resume checkpoint full-validation non_improving must be a "
            "non-negative integer"
        )
    return best, non_improving


@dataclass(frozen=True)
class PrecisionRuntime:
    """Autocast and scaler settings resolved from trainer.runtime.precision."""

    requested: str
    effective: str
    autocast_dtype: torch.dtype | None
    scaler: Any


def _configure_torch_runtime(config: dict[str, Any], device: torch.device) -> None:
    """Apply device-level runtime settings that matter on RTX-class GPUs."""

    if device.type != "cuda":
        return

    benchmark = bool(get_nested(config, "trainer.runtime.benchmark", True))
    allow_tf32 = bool(get_nested(config, "trainer.runtime.allow_tf32", True))
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _resolve_precision_runtime(
    config: dict[str, Any],
    device: torch.device,
) -> PrecisionRuntime:
    requested = str(get_nested(config, "trainer.runtime.precision", "32-true")).lower()
    effective = requested
    autocast_dtype: torch.dtype | None = None

    if requested in {"32", "32-true", "fp32", "float32"}:
        effective = "32-true"
    elif requested in {"16", "16-mixed", "fp16", "float16"}:
        if device.type != "cuda":
            print(f"[precision] {requested} requested on {device.type}; using 32-true")
            effective = "32-true"
        else:
            effective = "16-mixed"
            autocast_dtype = torch.float16
    elif requested in {"bf16", "bf16-mixed", "bfloat16"}:
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            print("[precision] bf16 is not supported on this CUDA device; using 16-mixed")
            effective = "16-mixed"
            autocast_dtype = torch.float16
        elif device.type in {"cuda", "cpu"}:
            effective = "bf16-mixed"
            autocast_dtype = torch.bfloat16
        else:
            print(f"[precision] bf16 requested on {device.type}; using 32-true")
            effective = "32-true"
    else:
        raise ValueError(
            "trainer.runtime.precision must be one of 32-true, 16-mixed, or bf16-mixed"
        )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and effective == "16-mixed"),
    )
    return PrecisionRuntime(
        requested=requested,
        effective=effective,
        autocast_dtype=autocast_dtype,
        scaler=scaler,
    )


def _autocast_context(device: torch.device, precision: PrecisionRuntime):
    if precision.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=precision.autocast_dtype)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load_state_dict_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path)))

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a mapping: {path}")
    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _load_pretrained_if_configured(model: torch.nn.Module, config: dict[str, Any]) -> None:
    use_pretrained = bool(
        get_nested(
            config,
            "experiment.pretrained.use_converted_sketchformer_weights",
            False,
        )
    ) or bool(get_nested(config, "model.checkpoint.load_converted_weights", False))
    if not use_pretrained:
        return

    path = get_nested(
        config,
        "experiment.pretrained.path",
        get_nested(config, "model.checkpoint.converted_weights_path"),
    )
    if not path:
        raise ValueError("Pretrained loading is enabled, but no pretrained path is configured")

    checkpoint_path = _resolve_project_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Configured pretrained weights do not exist: "
            f"{checkpoint_path}. Convert TensorFlow weights first or disable pretrained loading."
        )

    initialization_mode = str(
        get_nested(config, "experiment.pretrained.mode", "strict")
    )
    if initialization_mode == "transformer_blocks":
        if str(get_nested(config, "data.format.type")) == "anchored_v3":
            raise ValueError(
                "Anchored V3 partial transfer is allowed only through "
                "tts-initialize-sketchformer-v3; train from its complete .pt output"
            )
        from scripts.sketchformer.initialize_v3 import initialize_transformer_blocks

        report = initialize_transformer_blocks(
            model,
            _load_state_dict_file(checkpoint_path),
        )
        print(
            "[pretrained] transformer-block initialization loaded={} "
            "reinitialized={} skipped_shape={} ignored_source={}".format(
                len(report.loaded),
                len(report.reinitialized),
                len(report.skipped_shape),
                len(report.ignored_source),
            )
        )
        return
    if initialization_mode != "strict":
        raise ValueError("experiment.pretrained.mode must be strict or transformer_blocks")

    strict = bool(
        get_nested(
            config,
            "experiment.pretrained.strict",
            get_nested(config, "model.checkpoint.strict", False),
        )
    )
    incompatible = model.load_state_dict(_load_state_dict_file(checkpoint_path), strict=strict)
    print(
        "[pretrained] loaded={} missing_keys={} unexpected_keys={}".format(
            checkpoint_path,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
    )


def _validate_teacher_forced(
    model,
    valid_loader,
    loss_fn,
    config: dict[str, Any],
    device: torch.device,
    precision: PrecisionRuntime,
) -> dict[str, torch.Tensor]:
    model.eval()
    logs = []
    limit_val = get_nested(config, "trainer.training.limit_val_batches", 1.0)
    with torch.no_grad():
        for batch in limited(valid_loader, limit_val):
            batch = move_to_device(batch, device)
            with _autocast_context(device, precision):
                output = model(batch)
                loss_output = loss_fn(output, batch)
                metric_output = reconstruction_metrics(output, batch)
            step_logs = loss_output.as_log_dict(prefix="val")
            step_logs.update(metric_output.as_log_dict(prefix="val"))
            logs.append(step_logs)
    weight_key = "val/valid_tokens" if logs and "val/valid_tokens" in logs[0] else None
    return average_logs(logs, weight_key=weight_key)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _token_layout(config: dict[str, Any]) -> dict[str, Any]:
    format_config = dict(get_nested(config, "data.format", {}) or {})
    layout = dict(format_config.get("token_dictionary", {}) or {})
    layout.setdefault("type", format_config.get("type", "stroke3"))
    if "version" in format_config:
        layout.setdefault("version", format_config["version"])
    return layout


def _codebook_path(config: dict[str, Any]) -> Path:
    configured = get_nested(config, "data.format.token_dictionary.codebook_path")
    if configured:
        return _project_path(configured)
    candidate = _project_path(get_nested(config, "data.dataset.root")) / "codebook.npy"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError("V3 training requires a codebook.npy artifact")


def _manifest_path(config: dict[str, Any]) -> Path:
    manifest_path = get_nested(config, "data.dataset.manifest_path")
    dataset_root = _project_path(get_nested(config, "data.dataset.root"))
    if manifest_path:
        path = _project_path(manifest_path)
    else:
        manifest_file = get_nested(config, "data.dataset.manifest_file", "manifest.jsonl")
        path = dataset_root / str(manifest_file)
    if not path.is_file():
        raise FileNotFoundError(f"V3 training requires dataset manifest: {path}")
    return path


def _checkpoint_contract(config: dict[str, Any]) -> CheckpointContract | None:
    if str(get_nested(config, "data.format.type")) != "anchored_v3":
        return None
    codebook_path = _codebook_path(config)
    manifest_path = _manifest_path(config)
    return CheckpointContract(
        config=config,
        token_layout_version=get_nested(config, "data.format.version", 3),
        codebook_sha256=_sha256_file(codebook_path),
        dataset_manifest_sha256=_sha256_file(manifest_path),
        git_commit=current_git_commit(PROJECT_ROOT),
    )


def _validate_overfit_gate(config: dict[str, Any]) -> None:
    gates = get_nested(config, "trainer.gates", {}) or {}
    if not bool(gates.get("require_overfit_report", False)):
        return
    report_value = gates.get("overfit_report")
    if not report_value:
        raise RuntimeError(
            "Full V3 training is blocked until trainer.gates.overfit_report is configured"
        )
    report_path = _project_path(report_value)
    if not report_path.is_file():
        raise FileNotFoundError(f"Overfit gate report does not exist: {report_path}")
    report = json.loads(report_path.read_text())
    metadata = report.get("metadata", {}) or {}
    metrics = report.get("metrics", {}) or {}
    checks = {
        "schema_version": int(report.get("schema_version", -1)) == 1,
        "passed": bool(report.get("passed", False)),
        "sample_count": int(metrics.get("sample_count", -1)) == 32,
        "teacher_forced_token_accuracy": float(report.get("teacher_forced_token_accuracy", 0.0))
        >= 0.995,
        "free_running_geometry_f1_2px_median": float(
            report.get("free_running_geometry_f1_2px_median", 0.0)
        )
        >= 0.99,
        "cached_uncached_exact_match": bool(report.get("cached_uncached_exact_match", False)),
        "codebook_sha256": metadata.get("codebook_sha256")
        == _sha256_file(_codebook_path(config)),
        "manifest_sha256": metadata.get("manifest_sha256")
        == _sha256_file(_manifest_path(config)),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("Overfit gate failed: " + ", ".join(failures))


def _validate_free_running(
    model,
    valid_loader,
    config: dict[str, Any],
    device: torch.device,
    *,
    max_samples: int | None,
    metric_prefix: str = "val",
) -> dict[str, torch.Tensor]:
    import numpy as np

    from metrics.sketchformer.free_running import (
        aggregate_free_running_records,
        free_running_reconstruction_records,
    )

    codebook = np.load(_codebook_path(config))
    layout = _token_layout(config)
    records: list[dict[str, float | int | str]] = []
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            batch = move_to_device(batch, device)
            generation = model.generate(batch, use_cache=True)
            records.extend(
                free_running_reconstruction_records(
                    generation.tokens,
                    generation.lengths,
                    batch,
                    codebook,
                    eos_token_id=int(layout["eos_token_id"]),
                    token_layout=layout,
                )
            )
            if max_samples is not None and len(records) >= max_samples:
                records = records[:max_samples]
                break
    summary = aggregate_free_running_records(records, device=device)
    return {f"{metric_prefix}/{name}": value for name, value in summary.items()}


def _require_production_v3_dataset(config: dict[str, Any], datamodule: Any) -> None:
    if str(get_nested(config, "data.format.type")) != "anchored_v3":
        return
    from services.anchored_sketch_data.artifacts import (
        require_minimum_source_sketches,
    )

    train_dataset = datamodule.train_dataset
    metadata = getattr(train_dataset, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("Anchored V3 training dataset has no validated metadata")
    require_minimum_source_sketches(
        metadata,
    )


def _require_fixed_length_stratified_validation(loader: Any, expected: int) -> None:
    lengths = [int(value) for value in loader.dataset.lengths]
    if len(lengths) != int(expected):
        raise ValueError(
            f"V3 fixed validation requires exactly {expected} sketches; got {len(lengths)}"
        )
    counts = [
        sum(1 <= length <= 512 for length in lengths),
        sum(513 <= length <= 1024 for length in lengths),
        sum(1025 <= length <= 2048 for length in lengths),
        sum(2049 <= length <= 4096 for length in lengths),
    ]
    if any(count == 0 for count in counts):
        raise ValueError(
            "V3 fixed validation must cover all four length buckets; "
            f"counts={counts}"
        )


def _v3_decoder_inputs(
    output: Any,
    batch: dict[str, Any],
    config: dict[str, Any],
    *,
    scheduled_probability: float,
) -> torch.Tensor:
    if output.reconstruction is None or output.reconstruction.token_logits is None:
        raise ValueError("scheduled sampling requires token reconstruction logits")
    ranges = AnchoredTokenRanges.from_mapping(_token_layout(config))
    return corrupt_decoder_prefixes(
        batch["targets"],
        output.reconstruction.token_logits.detach(),
        ranges,
        scheduled_probability=scheduled_probability,
        mask_probability=float(
            get_nested(config, "trainer.exposure.decoder_mask_probability", 0.10)
        ),
    )


def main() -> int:
    args = parse_args()
    config = compose_training_config(args.config, experiment=args.experiment)
    _apply_cli_overrides(config, args)
    if str(get_nested(config, "data.format.type")) == "anchored_v3":
        from services.anchored_sketch_data.contract import (
            validate_anchored_v3_runtime_config,
        )

        validate_anchored_v3_runtime_config(config)

    seed = int(get_nested(config, "project.seed", 42))
    deterministic = bool(get_nested(config, "trainer.runtime.deterministic", False))
    set_seed(seed, deterministic=deterministic)

    device = resolve_device(args.device)
    _configure_torch_runtime(config, device)
    precision = _resolve_precision_runtime(config, device)
    checkpoint_dir = _checkpoint_dir(config)
    curriculum_stages = parse_curriculum(
        config["trainer"],
        default_max_length=int(get_nested(config, "data.sequence.max_length")),
    )

    if args.dry_run:
        print(f"experiment={get_nested(config, 'experiment.name')}")
        print(f"data_root={get_nested(config, 'data.dataset.root')}")
        print(f"train_source_limit={get_nested(config, 'data.dataset.train_source_limit')}")
        print(f"model={get_nested(config, 'model.name')}")
        print(f"device={device}")
        print(f"precision={precision.effective}")
        print(f"checkpoint_dir={checkpoint_dir}")
        print(f"max_epochs={get_nested(config, 'trainer.training.max_epochs')}")
        print(
            "curriculum="
            + ",".join(
                f"{stage.name}:{stage.max_length}x{stage.epochs}:{stage.trainable}"
                for stage in curriculum_stages
            )
        )
        return 0

    pin_anchored_v3_artifacts(config, project_root=PROJECT_ROOT)

    datamodule = StrokeSequenceDataModule(
        config["data"],
        project_root=PROJECT_ROOT,
        seed=seed,
    )
    datamodule.setup("fit")
    _require_production_v3_dataset(config, datamodule)
    _validate_overfit_gate(config)
    checkpoint_contract = _checkpoint_contract(config)
    is_anchored_v3 = checkpoint_contract is not None
    raw_model = build_model(config["model"])
    _load_pretrained_if_configured(raw_model, config)
    raw_model = raw_model.to(device)
    model = maybe_compile_model(raw_model, config["model"])
    loss_fn = build_loss(
        config["optimizer"],
        data_config=config["data"],
        project_root=PROJECT_ROOT,
    ).to(device)
    resume_path = get_nested(config, "experiment.run.resume_from_checkpoint")
    resume_result = None
    if resume_path:
        from core import load_checkpoint

        resume_result = load_checkpoint(
            _resolve_project_path(resume_path),
            raw_model,
            strict=True,
            expected_contract=checkpoint_contract,
            require_contract=checkpoint_contract is not None,
            require_training_state=is_anchored_v3,
        )
        print(
            f"[resume] restored epoch={resume_result.epoch} "
            f"step={resume_result.step}"
        )

    checkpoint_callback = CheckpointCallback(
        checkpoint_dir,
        monitor=str(
            get_nested(config, "trainer.checkpointing.monitor", "val/token_loss")
        ),
        mode=str(get_nested(config, "trainer.checkpointing.mode", "min")),
        save_last=bool(get_nested(config, "trainer.checkpointing.save_last", True)),
        contract=checkpoint_contract,
        require_contract=checkpoint_contract is not None,
    )
    if resume_result is not None:
        restored_best = resume_result.metrics.get(
            "checkpoint/best_metric",
            resume_result.metrics.get(checkpoint_callback.monitor),
        )
        if restored_best is not None:
            checkpoint_callback.tracker.best = float(restored_best)

    log_every = int(get_nested(config, "trainer.training.log_every_n_steps", 10))
    accumulate = max(
        1,
        int(get_nested(config, "trainer.training.accumulate_grad_batches", 1)),
    )
    grad_clip = get_nested(config, "optimizer.gradient.clip_norm", None)
    limit_train = get_nested(config, "trainer.training.limit_train_batches", 1.0)
    target_tokens = int(
        get_nested(config, "trainer.training.target_tokens_per_step", 0) or 0
    )
    early_stopping_patience = int(
        get_nested(config, "trainer.curriculum.early_stopping_patience", 0) or 0
    )
    global_step = resume_result.step if resume_result is not None else 0
    global_epoch = resume_result.epoch if resume_result is not None else 0
    exposure_total_epochs = sum(
        item.epochs for item in curriculum_stages if item.max_length > 1024
    )

    for stage_index, stage in enumerate(curriculum_stages):
        exposure_epochs_before = sum(
            item.epochs
            for item in curriculum_stages[:stage_index]
            if item.max_length > 1024
        )
        start_stage_epoch = resume_epoch_for_stage(
            curriculum_stages,
            stage_index,
            global_epoch,
        )
        if start_stage_epoch >= stage.epochs:
            print(f"[curriculum] skip completed stage={stage.name}")
            continue
        train_loader = datamodule.train_dataloader(stage.max_length)
        overfit_mode = bool(get_nested(config, "trainer.gates.overfit_mode", False))
        if overfit_mode:
            valid_loader = datamodule.train_dataloader(stage.max_length)
            fixed_free_valid_loader = valid_loader
            complete_valid_loader = valid_loader
        else:
            valid_loader = datamodule.val_dataloader(stage.max_length)
            fixed_sample_count = int(
                get_nested(
                    config,
                    "trainer.free_running_validation.max_samples",
                    256,
                )
            )
            fixed_free_valid_loader = datamodule.val_dataloader(
                max_samples=fixed_sample_count,
                stratify_by_length=bool(
                    get_nested(
                        config,
                        "trainer.free_running_validation.stratify_by_length",
                        True,
                    )
                ),
            )
            _require_fixed_length_stratified_validation(
                fixed_free_valid_loader,
                fixed_sample_count,
            )
            complete_valid_loader = datamodule.val_dataloader()
        parameter_groups = build_stage_parameter_groups(raw_model, stage)
        trainable_parameters = sum(
            parameter.numel()
            for group in parameter_groups
            for parameter in group["params"]
        )
        optimizer_config = copy.deepcopy(config["optimizer"])
        if not math.isnan(stage.learning_rate):
            optimizer_config.setdefault("optimizer", {})["lr"] = stage.learning_rate
        optimizer = build_optimizer(parameter_groups, optimizer_config)
        if target_tokens > 0:
            dataset_tokens = sum(int(length) for length in train_loader.dataset.lengths)
            total_steps = max(1, math.ceil(dataset_tokens / target_tokens) * stage.epochs)
        else:
            stage_config = copy.deepcopy(config)
            stage_config["trainer"]["training"]["max_epochs"] = stage.epochs
            total_steps = _total_optimizer_steps(stage_config, train_loader)
        scheduler = build_scheduler(
            optimizer,
            optimizer_config,
            total_steps=total_steps,
        )
        if resume_path and start_stage_epoch > 0:
            from core import load_checkpoint

            load_checkpoint(
                _resolve_project_path(resume_path),
                raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                strict=True,
                expected_contract=checkpoint_contract,
                require_contract=checkpoint_contract is not None,
                require_training_state=is_anchored_v3,
            )
        is_final_stage = stage_index == len(curriculum_stages) - 1
        early_stop_monitor = checkpoint_callback.monitor.replace(
            "val/", "val_full/", 1
        )
        non_improving = 0
        stage_validation_best: float | None = None
        if (
            resume_result is not None
            and is_anchored_v3
            and is_final_stage
            and start_stage_epoch > 0
        ):
            stage_validation_best, non_improving = (
                _restore_full_validation_early_stopping(
                    resume_result.training_state,
                    stage_index=stage_index,
                    stage_name=stage.name,
                    completed_stage_epochs=start_stage_epoch,
                    monitor=early_stop_monitor,
                    mode=checkpoint_callback.mode,
                )
            )
        learning_rate_summary = ",".join(
            f"{group.get('name', 'group')}:{group['lr']:.8g}"
            for group in optimizer.param_groups
        )
        print(
            f"[curriculum] stage={stage.name} max_length={stage.max_length} "
            f"epochs={stage.epochs} trainable={stage.trainable} "
            f"parameters={trainable_parameters} "
            f"lrs={learning_rate_summary}"
        )
        if (
            is_final_stage
            and early_stopping_patience > 0
            and non_improving >= early_stopping_patience
        ):
            print(
                f"[early-stopping] stage={stage.name} "
                f"patience={early_stopping_patience} already reached in resume checkpoint"
            )
            continue

        for stage_epoch in range(start_stage_epoch, stage.epochs):
            global_epoch += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_logs = []
            num_batches = batch_limit(train_loader, limit_train)
            accumulated_tokens = 0

            for batch_index, batch in enumerate(limited(train_loader, limit_train)):
                batch = move_to_device(batch, device)
                decoder_inputs = None
                exposure_enabled = (
                    str(get_nested(config, "data.format.type")) == "anchored_v3"
                    and stage.max_length > 1024
                    and bool(get_nested(config, "trainer.exposure.enabled", True))
                )
                if exposure_enabled:
                    progress = (
                        exposure_epochs_before
                        + stage_epoch
                        + batch_index / max(1, num_batches)
                    ) / max(1, exposure_total_epochs)
                    probability = scheduled_sampling_probability(
                        progress,
                        maximum=float(
                            get_nested(config, "trainer.exposure.maximum_probability", 0.25)
                        ),
                    )
                    with torch.no_grad(), _autocast_context(device, precision):
                        first_pass = model(batch)
                    decoder_inputs = _v3_decoder_inputs(
                        first_pass,
                        batch,
                        config,
                        scheduled_probability=probability,
                    )
                with _autocast_context(device, precision):
                    output = model(
                        batch,
                        targets=batch["targets"],
                        decoder_inputs=decoder_inputs,
                    )
                    loss_output = loss_fn(output, batch)
                    batch_tokens = int(
                        loss_output.valid_tokens.item()
                        if loss_output.valid_tokens is not None
                        else batch["valid_mask"].sum().item()
                    )
                    scaled_loss = (
                        loss_output.total * batch_tokens
                        if target_tokens > 0
                        else loss_output.total / accumulate
                    )

                precision.scaler.scale(scaled_loss).backward()
                accumulated_tokens += batch_tokens
                train_logs.append(loss_output.as_log_dict(prefix="train"))

                should_step = (
                    (target_tokens > 0 and accumulated_tokens >= target_tokens)
                    or (target_tokens <= 0 and (batch_index + 1) % accumulate == 0)
                    or (batch_index + 1) == num_batches
                )
                if should_step:
                    precision.scaler.unscale_(optimizer)
                    if target_tokens > 0:
                        _divide_gradients(raw_model, float(accumulated_tokens))
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(grad_clip))
                    precision.scaler.step(optimizer)
                    precision.scaler.update()
                    if scheduler.scheduler is not None:
                        scheduler.scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_tokens = 0
                    global_step += 1

                if (batch_index + 1) % log_every == 0:
                    print(
                        f"stage={stage.name} epoch={stage_epoch + 1} "
                        f"batch={batch_index + 1} {format_logs(train_logs[-1])}"
                    )

            train_weight_key = (
                "train/valid_tokens"
                if train_logs and "train/valid_tokens" in train_logs[0]
                else None
            )
            train_epoch_logs = average_logs(train_logs, weight_key=train_weight_key)
            val_logs = _validate_teacher_forced(
                model,
                valid_loader,
                loss_fn,
                config,
                device,
                precision,
            )
            if is_anchored_v3:
                fixed_free_logs = _validate_free_running(
                    raw_model,
                    fixed_free_valid_loader,
                    config,
                    device,
                    max_samples=None,
                )
                val_logs.update(fixed_free_logs)
                full_stage_validation = is_final_stage or stage_epoch + 1 == stage.epochs
                if full_stage_validation:
                    complete_free_logs = _validate_free_running(
                        raw_model,
                        complete_valid_loader,
                        config,
                        device,
                        max_samples=None,
                        metric_prefix="val_full",
                    )
                    val_logs.update(complete_free_logs)
            monitored_value = val_logs.get(early_stop_monitor) if is_final_stage else None
            if monitored_value is not None:
                current_value = float(monitored_value.detach().cpu())
                stage_improved = (
                    stage_validation_best is None
                    or (
                        checkpoint_callback.mode == "max"
                        and current_value > stage_validation_best
                    )
                    or (
                        checkpoint_callback.mode == "min"
                        and current_value < stage_validation_best
                    )
                )
                if stage_improved:
                    stage_validation_best = current_value
                    non_improving = 0
                else:
                    non_improving += 1

            training_state = None
            if is_anchored_v3:
                training_state = _full_validation_training_state(
                    stage_index=stage_index,
                    stage_name=stage.name,
                    completed_stage_epochs=stage_epoch + 1,
                    monitor=early_stop_monitor,
                    mode=checkpoint_callback.mode,
                    best=stage_validation_best,
                    non_improving=non_improving,
                )
            checkpoint_callback.on_validation_end(
                raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=global_epoch,
                step=global_step,
                metrics={key: float(value.detach().cpu()) for key, value in val_logs.items()},
                training_state=training_state,
            )

            print(
                f"stage={stage.name} epoch={stage_epoch + 1} "
                f"train {format_logs(train_epoch_logs)}"
            )
            print(
                f"stage={stage.name} epoch={stage_epoch + 1} "
                f"valid {format_logs(val_logs)}"
            )
            if (
                is_final_stage
                and early_stopping_patience > 0
                and non_improving >= early_stopping_patience
            ):
                print(
                    f"[early-stopping] stage={stage.name} "
                    f"patience={early_stopping_patience}"
                )
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
