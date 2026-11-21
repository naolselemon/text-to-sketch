"""Small callback-style helpers for lightweight training loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from core.checkpointing import save_checkpoint
from core.candidates import check_unregistered, retain_candidate


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
    retain_candidates: bool = False
    config: dict | None = None
    training_origin: str | None = None

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.tracker = BestMetricTracker(monitor=self.monitor, mode=self.mode)
        if self.training_origin is None and self.config is not None:
            from evals.report_evidence import training_origin_from_config

            self.training_origin = training_origin_from_config(self.config)
        if self.retain_candidates:
            check_unregistered(self.directory)

    def on_validation_end(
        self,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        epoch: int,
        step: int,
        metrics: dict[str, Any],
        stage: str = "default",
        stage_max_length: int | None = None,
    ) -> dict[str, Path]:
        saved: dict[str, Path] = {}
        if self.save_last:
            saved["last"] = save_checkpoint(
                self.directory / "last.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                metrics=metrics,
                config=self.config,
                training_origin=self.training_origin,
            )
        if self.tracker.update(metrics):
            saved["best"] = save_checkpoint(
                self.directory / "best.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                metrics=metrics,
                config=self.config,
                training_origin=self.training_origin,
            )
        if self.retain_candidates:
            saved["candidate"] = retain_candidate(
                self.directory, model,
                config=self.config or {}, training_origin=self.training_origin,
                stage=stage, stage_max_length=stage_max_length, epoch=epoch, step=step,
                validation_loss=float(metrics["val/token_loss"]),
            )
        return saved
