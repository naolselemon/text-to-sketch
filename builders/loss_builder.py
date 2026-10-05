"""Loss configuration builders for Sketchformer fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class LossWeights:
    """Weights for the loss terms used by the training loop."""

    reconstruction: float = 1.0
    token: float = 1.0
    pen_state: float = 1.0
    classification: float = 0.0
    kl: float = 0.0


def build_loss_weights(config: Mapping[str, Any]) -> LossWeights:
    """Build structured loss weights from optimizer/training config."""

    weights = config.get("loss_weights", config)
    return LossWeights(
        reconstruction=float(weights.get("reconstruction", 1.0)),
        token=float(weights.get("token", weights.get("reconstruction", 1.0))),
        pen_state=float(weights.get("pen_state", 1.0)),
        classification=float(weights.get("classification", 0.0)),
        kl=float(weights.get("kl", 0.0)),
    )


def build_loss(
    config: Mapping[str, Any],
    *,
    data_config: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
):
    """Build the concrete loss object once ``core.losses`` is available."""

    try:
        from core.losses import SketchformerLoss
    except ImportError as exc:
        raise ImportError(
            "core.losses.SketchformerLoss is required before build_loss can "
            "create the training loss object."
        ) from exc

    anchored_objective = None
    if data_config is not None and str(data_config.get("format", {}).get("type")) == "anchored_v3":
        from core.anchored_v3_objective import (
            AnchoredV3Objective,
            AnchoredV3ObjectiveConfig,
        )

        token_dictionary = data_config["format"]["token_dictionary"]
        codebook_path = Path(token_dictionary["codebook_path"])
        if not codebook_path.is_absolute():
            codebook_path = Path(project_root or Path.cwd()) / codebook_path
        codebook = torch.from_numpy(np.load(codebook_path).astype(np.float32, copy=False))
        anchored_objective = AnchoredV3Objective(
            AnchoredV3ObjectiveConfig.from_mapping(
                token_dictionary,
                config.get("loss_weights", config),
            ),
            codebook,
        )
    return SketchformerLoss(
        build_loss_weights(config),
        anchored_objective=anchored_objective,
    )
