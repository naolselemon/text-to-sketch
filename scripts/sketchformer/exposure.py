"""Decoder-prefix corruption for anchored-V3 exposure-bias training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class AnchoredTokenRanges:
    motion_start: int
    motion_end: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    mask_token_id: int

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AnchoredTokenRanges":
        motion_start = int(mapping["motion_token_offset"])
        x_start = int(mapping["x_token_offset"])
        y_start = int(mapping["y_token_offset"])
        return cls(
            motion_start=motion_start,
            motion_end=motion_start + int(mapping["codebook_size"]),
            x_start=x_start,
            x_end=x_start + int(mapping.get("coordinate_bins", 256)),
            y_start=y_start,
            y_end=y_start + int(mapping.get("coordinate_bins", 256)),
            mask_token_id=int(mapping["mask_token_id"]),
        )


def scheduled_sampling_probability(
    progress: float,
    *,
    maximum: float = 0.25,
) -> float:
    """Linear scheduled-sampling ramp with explicit bounds."""

    return float(maximum) * min(1.0, max(0.0, float(progress)))


def corrupt_decoder_prefixes(
    targets: torch.Tensor,
    first_pass_logits: torch.Tensor,
    token_ranges: AnchoredTokenRanges,
    *,
    scheduled_probability: float,
    mask_probability: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build a second-pass prefix without changing source or loss targets.

    Predicted replacements are constrained to the true grammar category for
    that position. Structural tokens are never replaced or masked.
    """

    if targets.ndim != 2 or first_pass_logits.ndim != 3:
        raise ValueError("targets must be [B,L] and logits must be [B,L-1,V]")
    if first_pass_logits.shape[:2] != (targets.shape[0], targets.shape[1] - 1):
        raise ValueError("first-pass logits do not align with shifted targets")
    for name, value in (
        ("scheduled_probability", scheduled_probability),
        ("mask_probability", mask_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    corrupted = targets.clone()
    target_content = targets[:, 1:]
    replacement = target_content.clone()
    eligible = torch.zeros_like(target_content, dtype=torch.bool)

    categories = (
        (token_ranges.motion_start, token_ranges.motion_end),
        (token_ranges.x_start, token_ranges.x_end),
        (token_ranges.y_start, token_ranges.y_end),
    )
    for start, end in categories:
        category_mask = (target_content >= start) & (target_content < end)
        category_prediction = torch.argmax(
            first_pass_logits[..., start:end],
            dim=-1,
        ) + start
        replacement = torch.where(category_mask, category_prediction, replacement)
        eligible |= category_mask

    random_shape = target_content.shape
    scheduled_draw = torch.rand(
        random_shape,
        device=targets.device,
        generator=generator,
    )
    mask_draw = torch.rand(
        random_shape,
        device=targets.device,
        generator=generator,
    )
    use_mask = eligible & (mask_draw < mask_probability)
    use_prediction = eligible & ~use_mask & (scheduled_draw < scheduled_probability)
    second_pass = torch.where(use_prediction, replacement, target_content)
    second_pass = torch.where(
        use_mask,
        torch.full_like(second_pass, token_ranges.mask_token_id),
        second_pass,
    )
    corrupted[:, 1:] = second_pass
    return corrupted
