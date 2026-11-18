"""Checkpoint helpers for model training and recovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class CheckpointLoadResult:
    """Metadata returned after loading a checkpoint."""

    epoch: int = 0
    step: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)


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
    training_origin: str | None = None,
) -> Path:
    """Save a training checkpoint."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "global_step": int(step),
        "metrics": dict(metrics or {}),
        "config": dict(config or {}),
    }
    if training_origin is not None:
        payload['training_origin'] = training_origin
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    scheduler_obj = getattr(scheduler, "scheduler", scheduler)
    if scheduler_obj is not None and hasattr(scheduler_obj, "state_dict"):
        payload["scheduler"] = scheduler_obj.state_dict()

    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> CheckpointLoadResult:
    """Load model and optional optimizer/scheduler state."""

    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")

    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler_obj = getattr(scheduler, "scheduler", scheduler)
    if scheduler_obj is not None and "scheduler" in checkpoint:
        scheduler_obj.load_state_dict(checkpoint["scheduler"])

    return CheckpointLoadResult(
        epoch=int(checkpoint.get("epoch", 0)),
        step=int(checkpoint.get("step", 0)),
        metrics=dict(checkpoint.get("metrics", {})),
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
    )


def latest_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Path | None:
    """Return the most recently modified checkpoint in a directory."""

    checkpoint_dir = Path(directory)
    candidates = sorted(checkpoint_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None
