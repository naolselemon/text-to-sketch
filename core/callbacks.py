"""Small callback-style helpers for lightweight training loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from core.checkpointing import CheckpointContract, CheckpointContractError, save_checkpoint


@dataclass
class BestMetricTracker:
    """Track whether a monitored metric improved."""

    monitor: str = "val/token_loss"
    mode: str = "min"
    best: float | None = None

    def is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best
        if self.mode == "max":
            return value > self.best
        raise ValueError("mode must be 'min' or 'max'")

    def update(self, metrics: dict[str, Any]) -> bool:
        if self.monitor not in metrics:
            return False
        value = float(metrics[self.monitor])
        if not self.is_better(value):
            return False
        self.best = value
        return True


@dataclass
class CheckpointCallback:
    """Save last and best checkpoints from a simple training loop."""

    directory: str | Path
    monitor: str = "val/token_loss"
    mode: str = "min"
    save_last: bool = True
    contract: CheckpointContract | dict[str, Any] | None = None
    require_contract: bool = True

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.tracker = BestMetricTracker(monitor=self.monitor, mode=self.mode)

    def on_validation_end(
        self,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        epoch: int,
        step: int,
        metrics: dict[str, Any],
        training_state: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        if self.require_contract and self.contract is None:
            raise CheckpointContractError(
                "CheckpointCallback requires an artifact contract. Pass the composed "
                "config, token layout version, codebook hash, manifest hash, and git commit."
            )
        saved: dict[str, Path] = {}
        improved = self.tracker.update(metrics)
        checkpoint_metrics = dict(metrics)
        if self.tracker.best is not None:
            checkpoint_metrics["checkpoint/best_metric"] = self.tracker.best
        if self.save_last:
            saved["last"] = save_checkpoint(
                self.directory / "last.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                metrics=checkpoint_metrics,
                training_state=training_state,
                contract=self.contract,
                require_contract=self.require_contract,
            )
        if improved:
            saved["best"] = save_checkpoint(
                self.directory / "best.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                metrics=checkpoint_metrics,
                training_state=training_state,
                contract=self.contract,
                require_contract=self.require_contract,
            )
        return saved
