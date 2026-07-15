"""Factories for PyTorch optimizers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn


def trainable_parameters(model_or_parameters: nn.Module | Iterable[nn.Parameter]):
    """Return trainable parameters from a model or parameter iterable."""

    if isinstance(model_or_parameters, nn.Module):
        return (param for param in model_or_parameters.parameters() if param.requires_grad)
    return (param for param in model_or_parameters if param.requires_grad)


def build_optimizer(
    model_or_parameters: nn.Module | Iterable[nn.Parameter] | Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Build a PyTorch optimizer from optimizer config."""

    optimizer_config = config.get("optimizer", config)
    optimizer_type = str(optimizer_config.get("type", "adamw")).lower()
    if (
        isinstance(model_or_parameters, Sequence)
        and model_or_parameters
        and isinstance(model_or_parameters[0], dict)
    ):
        params: Any = list(model_or_parameters)
    else:
        params = list(trainable_parameters(model_or_parameters))
    if not params:
        raise ValueError("No trainable parameters were provided to the optimizer")

    lr = float(optimizer_config.get("lr", 1e-4))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    eps = float(optimizer_config.get("eps", 1e-8))

    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(optimizer_config.get("betas", (0.9, 0.999))),
            eps=eps,
        )
    if optimizer_type == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(optimizer_config.get("betas", (0.9, 0.999))),
            eps=eps,
        )
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(optimizer_config.get("momentum", 0.9)),
        )

    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
