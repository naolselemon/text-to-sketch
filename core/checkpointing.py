"""Checkpoint helpers with explicit, verifiable artifact contracts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import copy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONTRACT_FIELDS = (
    "schema_version",
    "config",
    "compatibility_config",
    "token_layout_version",
    "codebook_sha256",
    "dataset_manifest_sha256",
    "git_commit",
    "monitored_metrics",
)


class CheckpointContractError(ValueError):
    """Raised when a checkpoint does not match its declared artifact contract."""


@dataclass(frozen=True)
class CheckpointContract:
    """Reproducibility metadata stored with every current-format checkpoint."""

    config: Mapping[str, Any]
    token_layout_version: str | int
    codebook_sha256: str | None
    dataset_manifest_sha256: str
    git_commit: str
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def as_payload(
        self,
        monitored_metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"] = _normalise_value(self.config)
        payload["compatibility_config"] = checkpoint_compatibility_config(
            self.config
        )
        payload["monitored_metrics"] = _normalise_value(monitored_metrics or {})
        return payload


def checkpoint_compatibility_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove run-location metadata that cannot affect model behavior.

    The complete composed config remains in every checkpoint. This projection
    is the strict resume/evaluation comparison surface, allowing a checkpoint
    to move directories or be resumed from its own path without weakening any
    model, data, optimizer, scheduler, precision, or curriculum checks.
    """

    projected = copy.deepcopy(_normalise_value(config))
    experiment = projected.get("experiment")
    if isinstance(experiment, dict):
        run = experiment.get("run")
        if isinstance(run, dict):
            run.pop("output_dir", None)
            run.pop("resume_from_checkpoint", None)
        experiment.pop("pretrained", None)
    trainer = projected.get("trainer")
    if isinstance(trainer, dict):
        gates = trainer.get("gates")
        if isinstance(gates, dict):
            gates.pop("overfit_report", None)
    return projected


@dataclass
class CheckpointLoadResult:
    """Metadata returned after loading a checkpoint."""

    epoch: int = 0
    step: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    training_state: dict[str, Any] | None = None
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    contract: dict[str, Any] | None = None
    legacy: bool = False


def current_git_commit(project_root: str | Path) -> str:
    """Return the checked-out commit without mutating the repository."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CheckpointContractError(
            f"Could not determine git commit for {project_root}"
        ) from exc
    commit = result.stdout.strip()
    if not commit:
        raise CheckpointContractError("git rev-parse returned an empty commit")
    return commit


def validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | CheckpointContract | None = None,
    require_contract: bool = True,
) -> dict[str, Any] | None:
    """Validate the embedded contract and optional runtime expectations.

    ``require_contract=False`` is the only supported compatibility path for
    checkpoints created before schema version 1. Tensor loading can and should
    remain strict even when this legacy metadata exception is used.
    """

    raw_contract = checkpoint.get("contract")
    if raw_contract is None:
        if require_contract:
            raise CheckpointContractError(
                "Checkpoint has no artifact contract. Use the explicit legacy "
                "compatibility option only for a trusted pre-contract checkpoint."
            )
        return None
    if not isinstance(raw_contract, Mapping):
        raise CheckpointContractError("Checkpoint contract must be a mapping")

    missing = [field for field in _CONTRACT_FIELDS if field not in raw_contract]
    if missing:
        raise CheckpointContractError(
            "Checkpoint contract is missing fields: " + ", ".join(missing)
        )
    contract = _normalise_value(raw_contract)
    if contract["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointContractError(
            "Unsupported checkpoint schema version: "
            f"{contract['schema_version']!r}; expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    if not isinstance(contract["config"], Mapping) or not contract["config"]:
        raise CheckpointContractError("Checkpoint contract config must be non-empty")
    if (
        not isinstance(contract["compatibility_config"], Mapping)
        or not contract["compatibility_config"]
    ):
        raise CheckpointContractError(
            "Checkpoint contract compatibility_config must be non-empty"
        )
    if not isinstance(contract["token_layout_version"], (str, int)):
        raise CheckpointContractError(
            "Checkpoint contract token_layout_version must be a string or integer"
        )
    _validate_sha256("codebook_sha256", contract["codebook_sha256"], nullable=True)
    _validate_sha256(
        "dataset_manifest_sha256",
        contract["dataset_manifest_sha256"],
        nullable=False,
    )
    if not isinstance(contract["git_commit"], str) or not contract["git_commit"].strip():
        raise CheckpointContractError("Checkpoint contract git_commit must be non-empty")
    if not isinstance(contract["monitored_metrics"], Mapping):
        raise CheckpointContractError(
            "Checkpoint contract monitored_metrics must be a mapping"
        )

    if expected is not None:
        expected_payload = (
            expected.as_payload()
            if isinstance(expected, CheckpointContract)
            else _normalise_value(expected)
        )
        for key, expected_value in expected_payload.items():
            if key == "monitored_metrics":
                continue
            if (
                key == "config"
                and isinstance(expected, CheckpointContract)
                and "compatibility_config" in contract
            ):
                continue
            if key not in contract:
                raise CheckpointContractError(
                    f"Checkpoint contract does not contain expected field {key!r}"
                )
            if _canonical_json(contract[key]) != _canonical_json(expected_value):
                raise CheckpointContractError(
                    f"Checkpoint contract mismatch for {key}: "
                    f"checkpoint={contract[key]!r} runtime={expected_value!r}"
                )
    return dict(contract)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
    contract: CheckpointContract | Mapping[str, Any] | None = None,
    require_contract: bool = False,
) -> Path:
    """Atomically save a training checkpoint.

    ``config`` remains accepted for legacy callers. New training paths must pass
    ``contract`` and set ``require_contract=True``.
    """

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metric_values = dict(metrics or {})
    contract_payload: dict[str, Any] | None
    if isinstance(contract, CheckpointContract):
        contract_payload = contract.as_payload(metric_values)
    elif contract is not None:
        contract_payload = _normalise_value(contract)
        contract_payload["monitored_metrics"] = _normalise_value(metric_values)
    else:
        contract_payload = None
    if require_contract and contract_payload is None:
        raise CheckpointContractError(
            "Current-format checkpoint saving requires a CheckpointContract"
        )

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "metrics": metric_values,
        "config": dict(config or {}),
    }
    if training_state is not None and not isinstance(training_state, Mapping):
        raise CheckpointContractError("Checkpoint training_state must be a mapping")
    if training_state is not None:
        payload["training_state"] = _normalise_value(training_state)
    if contract_payload is not None:
        payload["contract"] = contract_payload
        validate_checkpoint_contract(payload, require_contract=True)
        # Retain the historical top-level config for simple inspection.
        payload["config"] = contract_payload["config"]
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    scheduler_obj = getattr(scheduler, "scheduler", scheduler)
    if scheduler_obj is not None and hasattr(scheduler_obj, "state_dict"):
        payload["scheduler"] = scheduler_obj.state_dict()

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        dir=checkpoint_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary_path, checkpoint_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
    expected_contract: Mapping[str, Any] | CheckpointContract | None = None,
    require_contract: bool = False,
    require_training_state: bool = False,
) -> CheckpointLoadResult:
    """Load model state and validate the artifact contract before mutation."""

    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")

    contract = validate_checkpoint_contract(
        checkpoint,
        expected=expected_contract,
        require_contract=require_contract or expected_contract is not None,
    )
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_dict, Mapping):
        raise CheckpointContractError("Checkpoint model state must be a mapping")
    raw_training_state = checkpoint.get("training_state")
    if raw_training_state is None:
        if require_training_state:
            raise CheckpointContractError(
                "Resume checkpoint is missing training_state"
            )
        training_state = None
    elif not isinstance(raw_training_state, Mapping):
        raise CheckpointContractError("Checkpoint training_state must be a mapping")
    else:
        training_state = dict(_normalise_value(raw_training_state))
    if strict:
        expected_state = model.state_dict()
        missing = sorted(set(expected_state) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(expected_state))
        if missing or unexpected:
            raise CheckpointContractError(
                "Strict model state mismatch: "
                f"missing={missing} unexpected={unexpected}"
            )
        shape_mismatches = [
            name
            for name, expected_tensor in expected_state.items()
            if not hasattr(state_dict[name], "shape")
            or tuple(state_dict[name].shape) != tuple(expected_tensor.shape)
        ]
        if shape_mismatches:
            raise CheckpointContractError(
                "Strict model state shape mismatch: " + ", ".join(shape_mismatches)
            )
    if optimizer is not None and "optimizer" not in checkpoint:
        raise CheckpointContractError("Resume checkpoint is missing optimizer state")
    scheduler_obj = getattr(scheduler, "scheduler", scheduler)
    if scheduler_obj is not None and "scheduler" not in checkpoint:
        raise CheckpointContractError("Resume checkpoint is missing scheduler state")

    incompatible = model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler_obj is not None:
        scheduler_obj.load_state_dict(checkpoint["scheduler"])

    return CheckpointLoadResult(
        epoch=int(checkpoint.get("epoch", 0)),
        step=int(checkpoint.get("step", 0)),
        metrics=dict(checkpoint.get("metrics", {})),
        training_state=training_state,
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
        contract=contract,
        legacy=contract is None,
    )


def latest_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Path | None:
    """Return the most recently modified checkpoint in a directory."""

    checkpoint_dir = Path(directory)
    candidates = sorted(checkpoint_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _validate_sha256(name: str, value: Any, *, nullable: bool) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value.lower()) is None:
        raise CheckpointContractError(f"{name} must be a 64-character SHA-256 hex digest")


def _normalise_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise CheckpointContractError("Contract tensors must be scalar")
        return value.detach().cpu().item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _normalise_value(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointContractError(
            f"Checkpoint contract value is not canonical JSON: {value!r}"
        ) from exc
