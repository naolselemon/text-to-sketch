"""Losses for native in-repo Sketchformer fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SketchformerLossOutput:
    """Structured loss values returned by ``SketchformerLoss``."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    pen_state: torch.Tensor
    classification: torch.Tensor
    xy_mse: torch.Tensor
    token_accuracy: torch.Tensor | None = None
    token_perplexity: torch.Tensor | None = None
    valid_tokens: torch.Tensor | None = None
    cumulative_point: torch.Tensor | None = None
    stroke_endpoint: torch.Tensor | None = None
    kind: str = "continuous"

    def as_log_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        name = f"{prefix}/" if prefix else ""
        logs = {
            f"{name}loss": self.total.detach(),
            f"{name}reconstruction_loss": self.reconstruction.detach(),
            f"{name}classification_loss": self.classification.detach(),
        }
        if self.kind == "tok_dict":
            logs[f"{name}token_loss"] = self.reconstruction.detach()
            if self.token_accuracy is not None:
                logs[f"{name}token_accuracy"] = self.token_accuracy.detach()
            if self.token_perplexity is not None:
                logs[f"{name}token_perplexity"] = self.token_perplexity.detach()
            if self.valid_tokens is not None:
                logs[f"{name}valid_tokens"] = self.valid_tokens.detach()
            if self.cumulative_point is not None:
                logs[f"{name}cumulative_point_loss"] = self.cumulative_point.detach()
            if self.stroke_endpoint is not None:
                logs[f"{name}stroke_endpoint_loss"] = self.stroke_endpoint.detach()
        else:
            logs[f"{name}pen_state_loss"] = self.pen_state.detach()
            logs[f"{name}xy_mse"] = self.xy_mse.detach()
        return logs


def _weight(weights: Any, name: str, default: float) -> float:
    if isinstance(weights, Mapping):
        return float(weights.get(name, default))
    return float(getattr(weights, name, default))


def masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid sequence positions only."""

    mask = valid_mask.to(dtype=values.dtype, device=values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    numerator = torch.sum(values * mask)
    denominator = torch.sum(mask).clamp_min(1.0)
    return numerator / denominator


def gaussian_mixture_nll(reconstruction: Any, target_xy: torch.Tensor) -> torch.Tensor:
    """Bivariate Gaussian-mixture negative log likelihood per time step."""

    if reconstruction.mixture_logits is None:
        raise ValueError("Gaussian mixture output is missing mixture_logits")
    if reconstruction.mu is None or reconstruction.log_sigma is None or reconstruction.rho is None:
        raise ValueError("Gaussian mixture output is missing mu/log_sigma/rho")

    target = target_xy.unsqueeze(-2)
    mu = reconstruction.mu
    log_sigma = reconstruction.log_sigma.clamp(min=-7.0, max=7.0)
    sigma = torch.exp(log_sigma).clamp_min(1e-5)
    rho = reconstruction.rho.clamp(min=-0.999, max=0.999)

    norm_x = (target[..., 0] - mu[..., 0]) / sigma[..., 0]
    norm_y = (target[..., 1] - mu[..., 1]) / sigma[..., 1]
    one_minus_rho2 = (1.0 - rho.square()).clamp_min(1e-5)
    z = norm_x.square() + norm_y.square() - 2.0 * rho * norm_x * norm_y

    log_norm = (
        -torch.log(torch.tensor(2.0 * torch.pi, device=target_xy.device, dtype=target_xy.dtype))
        - log_sigma[..., 0]
        - log_sigma[..., 1]
        - 0.5 * torch.log(one_minus_rho2)
    )
    log_prob_components = log_norm - z / (2.0 * one_minus_rho2)
    log_mix = F.log_softmax(reconstruction.mixture_logits, dim=-1)
    log_prob = torch.logsumexp(log_mix + log_prob_components, dim=-1)
    return -log_prob


class SketchformerLoss(nn.Module):
    """Mask-aware reconstruction, pen-state, and optional class loss."""

    def __init__(self, weights: Any, *, anchored_objective: nn.Module | None = None) -> None:
        super().__init__()
        self.reconstruction_weight = _weight(weights, "reconstruction", 1.0)
        self.token_weight = _weight(weights, "token", self.reconstruction_weight)
        self.pen_state_weight = _weight(weights, "pen_state", 1.0)
        self.classification_weight = _weight(weights, "classification", 0.0)
        self.anchored_objective = anchored_objective

    def forward(self, output: Any, batch: Mapping[str, torch.Tensor]) -> SketchformerLossOutput:
        if output.reconstruction is None:
            raise ValueError("Model output does not include reconstruction predictions")

        targets = output.loss_targets if getattr(output, "loss_targets", None) is not None else batch["targets"]
        valid_mask = batch.get("valid_mask")
        if getattr(output, "loss_valid_mask", None) is not None:
            valid_mask = output.loss_valid_mask
        elif valid_mask is None:
            valid_mask = torch.ones(targets.shape[:2], dtype=torch.bool, device=targets.device)
        elif valid_mask.shape[:2] != targets.shape[:2]:
            valid_mask = valid_mask[:, : targets.shape[1]]
        else:
            valid_mask = valid_mask.to(device=targets.device, dtype=torch.bool)

        reconstruction = output.reconstruction
        token_logits = getattr(reconstruction, "token_logits", None)
        if token_logits is not None:
            return self._token_loss(output, batch, token_logits, targets, valid_mask)

        target_xy = targets[..., :2]

        if reconstruction.mixture_logits is not None:
            reconstruction_per_step = gaussian_mixture_nll(reconstruction, target_xy)
            reconstruction_loss = masked_mean(reconstruction_per_step, valid_mask)
        else:
            reconstruction_loss = masked_mean(
                F.mse_loss(reconstruction.xy, target_xy, reduction="none"),
                valid_mask,
            )

        xy_mse = masked_mean(
            F.mse_loss(reconstruction.xy, target_xy, reduction="none"),
            valid_mask,
        )
        pen_state_loss = self._pen_state_loss(reconstruction.pen_logits, targets, valid_mask)
        classification_loss = self._classification_loss(output, batch, targets)

        total = (
            self.reconstruction_weight * reconstruction_loss
            + self.pen_state_weight * pen_state_loss
            + self.classification_weight * classification_loss
        )
        return SketchformerLossOutput(
            total=total,
            reconstruction=reconstruction_loss,
            pen_state=pen_state_loss,
            classification=classification_loss,
            xy_mse=xy_mse,
        )

    def _token_loss(
        self,
        output: Any,
        batch: Mapping[str, torch.Tensor],
        token_logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> SketchformerLossOutput:
        token_targets = targets.to(device=token_logits.device, dtype=torch.long)
        token_mask = valid_mask.to(device=token_logits.device, dtype=torch.bool)
        if token_logits.shape[:2] != token_targets.shape:
            raise ValueError(
                "Token logits and targets must agree on batch/sequence shape: "
                f"{token_logits.shape[:2]} != {token_targets.shape}"
            )

        cumulative_point = None
        stroke_endpoint = None
        if self.anchored_objective is not None:
            token_loss, cumulative_point, stroke_endpoint = self.anchored_objective(
                token_logits,
                token_targets,
                token_mask,
            )
        else:
            loss = F.cross_entropy(
                token_logits.reshape(-1, token_logits.shape[-1]),
                token_targets.reshape(-1),
                reduction="none",
            ).view_as(token_targets)
            token_loss = masked_mean(loss, token_mask)
        token_predictions = torch.argmax(token_logits, dim=-1)
        token_accuracy = masked_mean(
            (token_predictions == token_targets).to(dtype=torch.float32),
            token_mask,
        )
        token_perplexity = torch.exp(token_loss.detach().clamp(max=50.0))
        classification_loss = self._classification_loss(output, batch, token_logits)
        zero = token_loss.new_tensor(0.0)

        total = (
            self.token_weight * token_loss
            + self.classification_weight * classification_loss
        )
        if cumulative_point is not None and stroke_endpoint is not None:
            total = (
                total
                + float(getattr(self.anchored_objective.config, "point_weight"))
                * cumulative_point
                + float(getattr(self.anchored_objective.config, "endpoint_weight"))
                * stroke_endpoint
            )
        return SketchformerLossOutput(
            total=total,
            reconstruction=token_loss,
            pen_state=zero,
            classification=classification_loss,
            xy_mse=zero,
            token_accuracy=token_accuracy,
            token_perplexity=token_perplexity,
            valid_tokens=token_mask.sum().to(dtype=torch.float32),
            cumulative_point=cumulative_point,
            stroke_endpoint=stroke_endpoint,
            kind="tok_dict",
        )

    @staticmethod
    def _pen_state_loss(
        pen_logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        pen_targets = targets[..., 2].round().long().clamp(min=0, max=pen_logits.shape[-1] - 1)
        loss = F.cross_entropy(
            pen_logits.reshape(-1, pen_logits.shape[-1]),
            pen_targets.reshape(-1),
            reduction="none",
        ).view_as(pen_targets)
        return masked_mean(loss, valid_mask)

    def _classification_loss(
        self,
        output: Any,
        batch: Mapping[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.classification_weight <= 0.0 or output.class_logits is None:
            return targets.new_tensor(0.0)
        labels = batch.get("labels")
        if labels is None:
            raise ValueError("Classification loss is enabled but batch has no labels")
        return F.cross_entropy(output.class_logits, labels.to(output.class_logits.device).long())
