"""Run and enforce the mandatory anchored-V3 32-sketch overfit gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()

import numpy as np
import torch

from builders import build_model
from builders.config_utils import get_nested
from core import checkpoint_compatibility_config, load_checkpoint, move_to_device, set_seed
from dataloaders import StrokeSequenceDataModule
from metrics.sketchformer.free_running import free_running_reconstruction_records
from scripts.sketchformer.config import (
    apply_minimum_source_override,
    compose_training_config,
    configured_minimum_source_sketches,
    pin_anchored_v3_artifacts,
    resolve_device,
)
from services.anchored_sketch_data.artifacts import (
    require_minimum_source_sketches,
    validate_dataset,
)
from services.anchored_sketch_data.contract import (
    validate_anchored_v3_runtime_config,
)


OVERFIT_GATE_SCHEMA_VERSION = 1
TEACHER_ACCURACY_THRESHOLD = 0.995
FREE_RUNNING_F1_THRESHOLD = 0.99
EXPECTED_SAMPLE_COUNT = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default="anime_anchored_v3_overfit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--train-source-limit", type=int, default=None)
    parser.add_argument(
        "--minimum-source-sketches",
        type=int,
        default=None,
        help="Match the cleaned-source minimum selected for this V3 artifact.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default="data/processed/evaluations/anchored_v3_overfit_gate.json",
    )
    parser.add_argument("--expected-samples", type=int, default=EXPECTED_SAMPLE_COUNT)
    return parser.parse_args()


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    apply_minimum_source_override(config, args.minimum_source_sketches)
    if args.data_root:
        config["data"]["dataset"]["root"] = args.data_root
        config["data"]["format"]["token_dictionary"]["codebook_path"] = str(
            Path(args.data_root) / "codebook.npy"
        )
    if args.train_source_limit is not None:
        if args.train_source_limit <= 0:
            raise ValueError("--train-source-limit must be positive")
        config["data"]["dataset"]["train_source_limit"] = args.train_source_limit


def evaluate_overfit_model(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    codebook: np.ndarray,
    token_layout: Mapping[str, Any],
    device: torch.device | str,
    expected_samples: int = EXPECTED_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Measure teacher forcing, free generation, and cache equivalence exactly."""

    if expected_samples <= 0:
        raise ValueError("expected_samples must be positive")
    eos_token_id = int(token_layout["eos_token_id"])
    teacher_correct = 0
    teacher_tokens = 0
    free_records: list[dict[str, float | int | str]] = []
    cache_mismatch_sample_ids: list[str] = []

    model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_to_device(raw_batch, device)
            output = model(batch)
            if output.reconstruction is None:
                raise ValueError("Overfit gate requires a reconstruction output")
            logits = getattr(output.reconstruction, "token_logits", None)
            if logits is None or output.loss_targets is None:
                raise ValueError("Overfit gate requires autoregressive token logits")
            valid_mask = output.loss_valid_mask
            if valid_mask is None:
                valid_mask = output.loss_targets != int(token_layout["pad_token_id"])
            predictions = logits.argmax(dim=-1)
            teacher_correct += int(
                ((predictions == output.loss_targets) & valid_mask).sum().item()
            )
            teacher_tokens += int(valid_mask.sum().item())

            cached = model.generate(batch, use_cache=True)
            uncached = model.generate(batch, use_cache=False)
            cache_mismatch_sample_ids.extend(
                _cache_mismatch_ids(cached, uncached, batch)
            )
            free_records.extend(
                free_running_reconstruction_records(
                    cached.tokens,
                    cached.lengths,
                    batch,
                    codebook,
                    eos_token_id=eos_token_id,
                    token_layout=token_layout,
                )
            )

    if teacher_tokens == 0:
        raise ValueError("Overfit gate found no valid teacher-forced target tokens")
    sample_ids = [str(record["sample_id"]) for record in free_records]
    if len(sample_ids) != expected_samples:
        raise ValueError(
            f"Overfit gate requires exactly {expected_samples} sketches, "
            f"but evaluated {len(sample_ids)}"
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Overfit gate sample IDs must be unique")

    geometry_f1 = [float(record["geometry_f1_2px"]) for record in free_records]
    return {
        "sample_count": len(sample_ids),
        "teacher_forced_correct_tokens": teacher_correct,
        "teacher_forced_valid_tokens": teacher_tokens,
        "teacher_forced_token_accuracy": teacher_correct / teacher_tokens,
        "free_running_geometry_f1_2px_median": float(np.median(geometry_f1)),
        "cached_uncached_exact_match": not cache_mismatch_sample_ids,
        "cached_uncached_mismatch_count": len(cache_mismatch_sample_ids),
        "cached_uncached_mismatch_sample_ids": cache_mismatch_sample_ids,
    }


def build_gate_report(
    metrics: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable gate report consumed by full V3 training."""

    teacher_accuracy = float(metrics["teacher_forced_token_accuracy"])
    free_running_f1 = float(metrics["free_running_geometry_f1_2px_median"])
    cache_equal = bool(metrics["cached_uncached_exact_match"])
    checks = {
        "teacher_forced_token_accuracy": {
            "actual": teacher_accuracy,
            "operator": ">=",
            "threshold": TEACHER_ACCURACY_THRESHOLD,
            "passed": teacher_accuracy >= TEACHER_ACCURACY_THRESHOLD,
        },
        "free_running_geometry_f1_2px_median": {
            "actual": free_running_f1,
            "operator": ">=",
            "threshold": FREE_RUNNING_F1_THRESHOLD,
            "passed": free_running_f1 >= FREE_RUNNING_F1_THRESHOLD,
        },
        "cached_uncached_exact_match": {
            "actual": cache_equal,
            "operator": "==",
            "threshold": True,
            "passed": cache_equal,
        },
    }
    failures = [name for name, check in checks.items() if not check["passed"]]
    return {
        "schema_version": OVERFIT_GATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "failures": failures,
        # These top-level values are the stable training-gate interface.
        "teacher_forced_token_accuracy": teacher_accuracy,
        "free_running_geometry_f1_2px_median": free_running_f1,
        "cached_uncached_exact_match": cache_equal,
        "checks": checks,
        "metrics": _json_safe_mapping(metrics),
        "metadata": _json_safe_mapping(metadata or {}),
    }


def write_gate_report_atomic(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Durably publish one complete report or leave the old report untouched."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _cache_mismatch_ids(
    cached: Any,
    uncached: Any,
    batch: Mapping[str, Any],
) -> list[str]:
    cached_lengths = cached.lengths.detach().cpu().long()
    uncached_lengths = uncached.lengths.detach().cpu().long()
    cached_tokens = cached.tokens.detach().cpu().long()
    uncached_tokens = uncached.tokens.detach().cpu().long()
    sample_ids = batch.get("sample_ids")
    mismatches: list[str] = []
    for row in range(len(cached_lengths)):
        cached_length = int(cached_lengths[row])
        uncached_length = int(uncached_lengths[row])
        equal = cached_length == uncached_length
        if equal:
            equal = torch.equal(
                cached_tokens[row, :cached_length],
                uncached_tokens[row, :uncached_length],
            )
        if not equal:
            mismatches.append(
                str(sample_ids[row]) if sample_ids is not None else str(row)
            )
    return mismatches


def _validate_v3_config(config: Mapping[str, Any]) -> None:
    validate_anchored_v3_runtime_config(config)
    if str(get_nested(config, "data.format.type")) != "anchored_v3":
        raise ValueError("Overfit gate requires data.format.type=anchored_v3")
    if int(get_nested(config, "data.format.version", -1)) != 3:
        raise ValueError("Overfit gate requires data.format.version=3")
    if str(get_nested(config, "model.decoder.memory_source")) != "encoder":
        raise ValueError("Overfit gate requires model.decoder.memory_source=encoder")
    if int(get_nested(config, "data.dataset.train_source_limit", -1)) != 32:
        raise ValueError("Overfit gate requires data.dataset.train_source_limit=32")
    if bool(get_nested(config, "data.dataset.include_augmentations", True)):
        raise ValueError("Overfit gate must exclude precomputed train augmentations")
    if str(get_nested(config, "data.dataset.source_subset_strategy")) != "length_stratified":
        raise ValueError("Overfit gate requires a length-stratified representative subset")
    if not bool(get_nested(config, "trainer.gates.overfit_mode", False)):
        raise ValueError("Overfit gate requires trainer.gates.overfit_mode=true")
    precision = str(get_nested(config, "trainer.runtime.precision", "")).lower()
    if precision not in {"32", "32-true", "fp32", "float32"}:
        raise ValueError("The 32-sketch overfit gate must run in FP32")

    token_dictionary = dict(
        get_nested(config, "data.format.token_dictionary", {}) or {}
    )
    exact_values = {
        "pad_token_id": 0,
        "codebook_size": 2048,
        "motion_token_offset": 1,
        "x_token_offset": 2049,
        "y_token_offset": 2305,
        "coordinate_bins": 256,
        "stroke_start_token_id": 2561,
        "stroke_end_token_id": 2562,
        "sos_token_id": 2563,
        "eos_token_id": 2564,
        "mask_token_id": 2565,
        "vocab_size": 2566,
    }
    mismatches = [
        f"{name}={token_dictionary.get(name)!r} expected {expected}"
        for name, expected in exact_values.items()
        if token_dictionary.get(name) != expected
    ]
    if token_dictionary.get("sep_token_id") is not None:
        mismatches.append("sep_token_id must be null")
    if mismatches:
        raise ValueError("Anchored V3 token layout mismatch: " + "; ".join(mismatches))


def _token_layout(config: Mapping[str, Any]) -> dict[str, Any]:
    layout = dict(get_nested(config, "data.format.token_dictionary", {}) or {})
    layout["type"] = str(get_nested(config, "data.format.type"))
    layout["version"] = int(get_nested(config, "data.format.version"))
    return layout


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _dataset_root(config: Mapping[str, Any]) -> Path:
    return _project_path(get_nested(config, "data.dataset.root"))


def _codebook_path(config: Mapping[str, Any]) -> Path:
    configured = get_nested(config, "data.format.token_dictionary.codebook_path")
    return _project_path(configured) if configured else _dataset_root(config) / "codebook.npy"


def _manifest_path(config: Mapping[str, Any]) -> Path:
    configured_path = get_nested(config, "data.dataset.manifest_path")
    if configured_path:
        return _project_path(configured_path)
    filename = get_nested(config, "data.dataset.manifest_file", "manifest.jsonl")
    path = Path(filename)
    return path if path.is_absolute() else _dataset_root(config) / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_checkpoint_contract(
    config: Mapping[str, Any],
    *,
    codebook_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "compatibility_config": checkpoint_compatibility_config(config),
        "token_layout_version": 3,
        "codebook_sha256": _sha256_file(codebook_path),
        "dataset_manifest_sha256": _sha256_file(manifest_path),
    }


def _json_safe_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in values.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Report tensor {name!r} must be scalar")
            result[str(name)] = value.detach().cpu().item()
        elif isinstance(value, np.generic):
            result[str(name)] = value.item()
        elif isinstance(value, Path):
            result[str(name)] = str(value)
        else:
            result[str(name)] = value
    return result


def main() -> int:
    args = parse_args()
    config = compose_training_config(args.config, experiment=args.experiment)
    _apply_cli_overrides(config, args)
    pin_anchored_v3_artifacts(config, project_root=PROJECT_ROOT)
    _validate_v3_config(config)

    seed = int(get_nested(config, "project.seed", 42))
    set_seed(seed, deterministic=True)
    device = resolve_device(args.device)
    dataset_root = _dataset_root(config)
    metadata = validate_dataset(dataset_root)
    require_minimum_source_sketches(
        metadata,
        minimum=configured_minimum_source_sketches(config),
    )
    codebook_path = _codebook_path(config)
    manifest_path = _manifest_path(config)
    if not codebook_path.is_file():
        raise FileNotFoundError(f"Overfit codebook does not exist: {codebook_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Overfit manifest does not exist: {manifest_path}")
    codebook = np.load(codebook_path, allow_pickle=False)

    datamodule = StrokeSequenceDataModule(
        config["data"],
        project_root=PROJECT_ROOT,
        seed=seed,
    )
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    model = build_model(config["model"]).to(device)
    checkpoint_path = _project_path(args.checkpoint)
    checkpoint_result = load_checkpoint(
        checkpoint_path,
        model,
        strict=True,
        expected_contract=_expected_checkpoint_contract(
            config,
            codebook_path=codebook_path,
            manifest_path=manifest_path,
        ),
        require_contract=True,
    )
    metrics = evaluate_overfit_model(
        model,
        loader,
        codebook=codebook,
        token_layout=_token_layout(config),
        device=device,
        expected_samples=args.expected_samples,
    )
    report = build_gate_report(
        metrics,
        metadata={
            "experiment": args.experiment,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_result.epoch,
            "checkpoint_step": checkpoint_result.step,
            "dataset_root": str(dataset_root),
            "manifest_sha256": _sha256_file(manifest_path),
            "codebook_sha256": _sha256_file(codebook_path),
            "device": str(device),
            "precision": "32-true",
            "expected_samples": args.expected_samples,
            "minimum_source_sketches": configured_minimum_source_sketches(config),
        },
    )
    output_path = write_gate_report_atomic(_project_path(args.output), report)
    print(
        "overfit_gate="
        f"{'PASS' if report['passed'] else 'FAIL'} "
        f"teacher_accuracy={metrics['teacher_forced_token_accuracy']:.6f} "
        f"free_running_f1_2px_median="
        f"{metrics['free_running_geometry_f1_2px_median']:.6f} "
        f"cache_equal={metrics['cached_uncached_exact_match']}"
    )
    print(f"[report] wrote {output_path}")
    if report["passed"]:
        return 0
    print("failed checks: " + ", ".join(report["failures"]), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
