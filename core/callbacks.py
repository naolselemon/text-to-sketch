"""Small callback-style helpers for lightweight training loops."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from core.checkpointing import save_checkpoint


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
    lineage: Any = None
    config: Mapping[str, Any] | None = None

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
        lineage: Any = None,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        saved: dict[str, Path] = {}
        active_lineage = lineage if lineage is not None else self.lineage
        active_config = config if config is not None else self.config
        if self.save_last:
            saved["last"] = save_checkpoint(
                self.directory / "last.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                config=active_config,
                metrics=metrics,
                lineage=active_lineage,
            )
        if self.tracker.update(metrics):
            saved["best"] = save_checkpoint(
                self.directory / "best.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                config=active_config,
                metrics=metrics,
                lineage=active_lineage,
            )
        return saved
