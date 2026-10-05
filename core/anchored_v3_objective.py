"""Differentiable token and geometry objective for anchored-V3 sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AnchoredV3ObjectiveConfig:
    motion_start: int
    motion_end: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    stroke_start_id: int
    stroke_end_id: int
    sos_id: int
    eos_id: int
    point_weight: float = 2.0
    endpoint_weight: float = 1.0
    structural_weight: float = 2.0
    label_smoothing: float = 0.05
    canvas_scale: float = 255.0

    @classmethod
    def from_mapping(
        cls,
        token_dictionary: Mapping[str, Any],
        loss_weights: Mapping[str, Any],
    ) -> "AnchoredV3ObjectiveConfig":
        motion_start = int(token_dictionary["motion_token_offset"])
        codebook_size = int(token_dictionary["codebook_size"])
        coordinate_bins = int(token_dictionary.get("coordinate_bins", 256))
        x_start = int(token_dictionary["x_token_offset"])
        y_start = int(token_dictionary["y_token_offset"])
        return cls(
            motion_start=motion_start,
            motion_end=motion_start + codebook_size,
            x_start=x_start,
            x_end=x_start + coordinate_bins,
            y_start=y_start,
            y_end=y_start + coordinate_bins,
            stroke_start_id=int(token_dictionary["stroke_start_token_id"]),
            stroke_end_id=int(token_dictionary["stroke_end_token_id"]),
            sos_id=int(token_dictionary["sos_token_id"]),
            eos_id=int(token_dictionary["eos_token_id"]),
            point_weight=float(loss_weights.get("cumulative_point", 2.0)),
            endpoint_weight=float(loss_weights.get("stroke_endpoint", 1.0)),
            structural_weight=float(loss_weights.get("structural_token", 2.0)),
            label_smoothing=float(loss_weights.get("label_smoothing", 0.05)),
            canvas_scale=float(coordinate_bins - 1),
        )


class AnchoredV3Objective(nn.Module):
    """Weighted token CE plus expected cumulative-position losses."""

    def __init__(
        self,
        config: AnchoredV3ObjectiveConfig,
        codebook: torch.Tensor,
    ) -> None:
        super().__init__()
        if codebook.ndim != 2 or codebook.shape != (config.motion_end - config.motion_start, 2):
            raise ValueError(
                "anchored-V3 codebook shape does not match configured motion vocabulary"
            )
        self.config = config
        self.register_buffer("codebook", codebook.to(dtype=torch.float32))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.shape[:2] != targets.shape or targets.shape != valid_mask.shape:
            raise ValueError("anchored-V3 logits, targets, and mask shapes do not align")
        token_loss = self._token_loss(logits, targets, valid_mask)
        point_loss, endpoint_loss = self._geometry_loss(logits, targets, valid_mask)
        return token_loss, point_loss, endpoint_loss

    def _token_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        log_prob = F.log_softmax(logits, dim=-1)
        nll = -log_prob.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        content = torch.zeros_like(valid_mask, dtype=torch.bool)
        smoothed = nll
        for start, end in (
            (config.motion_start, config.motion_end),
            (config.x_start, config.x_end),
            (config.y_start, config.y_end),
        ):
            category = (targets >= start) & (targets < end)
            category_mean = -log_prob[..., start:end].mean(dim=-1)
            category_loss = (
                (1.0 - config.label_smoothing) * nll
                + config.label_smoothing * category_mean
            )
            smoothed = torch.where(category, category_loss, smoothed)
            content |= category

        weights = torch.where(
            content,
            torch.ones_like(smoothed),
            torch.full_like(smoothed, config.structural_weight),
        )
        weights = weights * valid_mask.to(dtype=weights.dtype)
        return (smoothed * weights).sum() / weights.sum().clamp_min(1.0)

    def _geometry_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        dtype = logits.dtype
        codebook = self.codebook.to(device=logits.device, dtype=dtype) / config.canvas_scale

        motion_probability = F.softmax(
            logits[..., config.motion_start : config.motion_end],
            dim=-1,
        )
        predicted_delta = motion_probability @ codebook
        x_values = torch.arange(
            config.x_end - config.x_start,
            device=logits.device,
            dtype=dtype,
        ) / config.canvas_scale
        y_values = torch.arange(
            config.y_end - config.y_start,
            device=logits.device,
            dtype=dtype,
        ) / config.canvas_scale
        predicted_x = (
            F.softmax(logits[..., config.x_start : config.x_end], dim=-1) * x_values
        ).sum(dim=-1)
        predicted_y = (
            F.softmax(logits[..., config.y_start : config.y_end], dim=-1) * y_values
        ).sum(dim=-1)

        motion_mask = (
            (targets >= config.motion_start)
            & (targets < config.motion_end)
            & valid_mask
        )
        x_mask = (targets >= config.x_start) & (targets < config.x_end) & valid_mask
        anchor_mask = (targets >= config.y_start) & (targets < config.y_end) & valid_mask

        safe_motion = (targets - config.motion_start).clamp(
            min=0,
            max=codebook.shape[0] - 1,
        )
        target_delta = F.embedding(safe_motion, codebook)
        predicted_delta = predicted_delta * motion_mask.unsqueeze(-1)
        target_delta = target_delta * motion_mask.unsqueeze(-1)

        previous_predicted_x = F.pad(predicted_x[:, :-1], (1, 0))
        target_x_value = (
            (targets - config.x_start).clamp(
                min=0,
                max=config.x_end - config.x_start - 1,
            ).to(dtype)
            / config.canvas_scale
        )
        previous_target_x = F.pad(target_x_value[:, :-1], (1, 0))
        target_y_value = (
            (targets - config.y_start).clamp(
                min=0,
                max=config.y_end - config.y_start - 1,
            ).to(dtype)
            / config.canvas_scale
        )
        predicted_anchor = torch.stack((previous_predicted_x, predicted_y), dim=-1)
        target_anchor = torch.stack((previous_target_x, target_y_value), dim=-1)
        predicted_anchor = predicted_anchor * anchor_mask.unsqueeze(-1)
        target_anchor = target_anchor * anchor_mask.unsqueeze(-1)

        predicted_position = self._positions(
            predicted_delta,
            predicted_anchor,
            anchor_mask,
        )
        target_position = self._positions(target_delta, target_anchor, anchor_mask)
        point_mask = (motion_mask | anchor_mask) & valid_mask
        point_error = F.smooth_l1_loss(
            predicted_position,
            target_position,
            reduction="none",
        ).mean(dim=-1)
        point_loss = self._masked_mean(point_error, point_mask)

        endpoint_mask = (targets == config.stroke_end_id) & valid_mask
        previous_predicted_position = F.pad(
            predicted_position[:, :-1],
            (0, 0, 1, 0),
        )
        previous_target_position = F.pad(target_position[:, :-1], (0, 0, 1, 0))
        endpoint_error = F.smooth_l1_loss(
            previous_predicted_position,
            previous_target_position,
            reduction="none",
        ).mean(dim=-1)
        endpoint_loss = self._masked_mean(endpoint_error, endpoint_mask)
        return point_loss, endpoint_loss

    @staticmethod
    def _positions(
        deltas: torch.Tensor,
        anchors: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        cumulative = torch.cumsum(deltas, dim=1)
        segment = torch.cumsum(anchor_mask.to(dtype=torch.long), dim=1)
        batch, length = segment.shape
        slots = length + 1
        index = segment.unsqueeze(-1).expand(batch, length, 2)
        baseline_table = torch.zeros(
            batch,
            slots,
            2,
            device=deltas.device,
            dtype=deltas.dtype,
        ).scatter_add(1, index, cumulative * anchor_mask.unsqueeze(-1))
        anchor_table = torch.zeros_like(baseline_table).scatter_add(
            1,
            index,
            anchors,
        )
        gathered_baseline = baseline_table.gather(1, index)
        gathered_anchor = anchor_table.gather(1, index)
        return gathered_anchor + cumulative - gathered_baseline

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)
