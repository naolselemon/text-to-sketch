"""Factories for learning-rate schedulers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import math
import torch
from torch.optim.lr_scheduler import LambdaLR


@dataclass(frozen=True)
class SchedulerBundle:
    """Scheduler plus metadata needed by a trainer loop."""

    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    interval: str = "step"
    frequency: int = 1


def _cosine_with_warmup_lambda(
    *,
    warmup_steps: int,
    total_steps: int | None,
    min_lr_ratio: float,
):
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), min_lr_ratio)

        if total_steps is None or total_steps <= warmup_steps:
            return 1.0

        progress = min(
            1.0,
            float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)),
        )
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return schedule


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    *,
    total_steps: int | None = None,
) -> SchedulerBundle:
    """Build a scheduler bundle from optimizer config."""

    scheduler_config = config.get("scheduler", config)
    scheduler_type = str(scheduler_config.get("type", "none")).lower()
    interval = str(scheduler_config.get("interval", "step"))
    frequency = int(scheduler_config.get("frequency", 1))

    if scheduler_type in {"none", "null"}:
        return SchedulerBundle(None, interval=interval, frequency=frequency)

    if scheduler_type == "cosine_with_warmup":
        warmup_steps = int(scheduler_config.get("warmup_steps", 0))
        if "warmup_fraction" in scheduler_config:
            if total_steps is None:
                raise ValueError("warmup_fraction requires total_steps")
            fraction = float(scheduler_config["warmup_fraction"])
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("warmup_fraction must be between 0 and 1")
            warmup_steps = math.ceil(total_steps * fraction)
            warmup_steps = min(
                warmup_steps,
                int(scheduler_config.get("warmup_cap_steps", warmup_steps)),
            )
        base_lr = float(optimizer.param_groups[0]["lr"])
        min_lr = float(scheduler_config.get("min_lr", 0.0))
        min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0
        scheduler = LambdaLR(
            optimizer,
            _cosine_with_warmup_lambda(
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
            ),
        )
        return SchedulerBundle(scheduler, interval=interval, frequency=frequency)

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
