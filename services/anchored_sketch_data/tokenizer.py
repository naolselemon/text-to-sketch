"""Anchored V3 stroke tokenizer with per-stroke residual reset."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .contract import CANVAS_SIZE, CODEBOOK_SIZE, TOKEN_LAYOUT, TokenLayout
from .grammar import AnchoredGrammar

Point = tuple[float, float]
Stroke = list[Point]


def validate_codebook(codebook: np.ndarray, *, exact_size: bool = True) -> np.ndarray:
    centers = np.asarray(codebook, dtype=np.float32)
    expected_rows = CODEBOOK_SIZE if exact_size else "K"
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"Expected codebook shape ({expected_rows}, 2), got {centers.shape}")
    if exact_size and len(centers) != CODEBOOK_SIZE:
        raise ValueError(
            f"Anchored V3 requires exactly {CODEBOOK_SIZE} centers, got {len(centers)}"
        )
    if len(centers) == 0 or not np.isfinite(centers).all():
        raise ValueError("Codebook must be non-empty and contain only finite values")
    return centers


class AnchoredTokenizer:
    """Encode fixed-canvas strokes using anchors and one token per later point."""

    def __init__(
        self,
        codebook: np.ndarray,
        *,
        layout: TokenLayout = TOKEN_LAYOUT,
        exact_codebook_size: bool = True,
    ) -> None:
        self.codebook = validate_codebook(codebook, exact_size=exact_codebook_size)
        if len(self.codebook) != layout.motion_token_end - layout.motion_token_start + 1:
            raise ValueError("Codebook size must equal the number of motion token IDs")
        self.layout = layout
        self.index = cKDTree(self.codebook)
        self.grammar = AnchoredGrammar(layout)

    def encode(self, strokes: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
        tokens: list[int] = [self.layout.sos_token_id]
        stroke_count = 0
        for raw_stroke in strokes:
            points = _coerce_stroke(raw_stroke)
            if len(points) < 2:
                raise ValueError("Every anchored V3 stroke must contain at least two points")
            if np.any(points < 0.0) or np.any(points > CANVAS_SIZE - 1):
                raise ValueError(f"Stroke coordinates must lie inside the {CANVAS_SIZE}x{CANVAS_SIZE} canvas")

            anchor = np.rint(points[0]).astype(np.int64)
            tokens.extend(
                (
                    self.layout.stroke_start_token_id,
                    self.layout.x_token_start + int(anchor[0]),
                    self.layout.y_token_start + int(anchor[1]),
                )
            )

            # Both accumulators deliberately restart at the absolute anchor.
            # Residual error from one stroke can never influence the next one.
            decoded_position = anchor.astype(np.float32)
            for desired_position in points[1:]:
                residual = desired_position - decoded_position
                cluster_id = int(self.index.query(residual)[1])
                tokens.append(self.layout.motion_token_start + cluster_id)
                decoded_position = decoded_position + self.codebook[cluster_id]
            tokens.append(self.layout.stroke_end_token_id)
            stroke_count += 1

        if stroke_count == 0:
            raise ValueError("Anchored V3 requires at least one drawable stroke")
        tokens.append(self.layout.eos_token_id)
        encoded = np.asarray(tokens, dtype=np.int32)
        self.grammar.validate(encoded)
        return encoded

    def decode(self, tokens: Sequence[int] | np.ndarray, *, validate: bool = True) -> list[Stroke]:
        values = np.asarray(tokens, dtype=np.int64)
        if validate:
            self.grammar.validate(values)

        strokes: list[Stroke] = []
        current: Stroke | None = None
        pending_x: float | None = None
        position: np.ndarray | None = None
        for raw_token in values:
            token = int(raw_token)
            if token in {self.layout.sos_token_id, self.layout.pad_token_id}:
                continue
            if token == self.layout.eos_token_id:
                break
            if token == self.layout.stroke_start_token_id:
                current = []
                pending_x = None
                position = None
            elif self.layout.is_x(token):
                pending_x = float(self.layout.x_coordinate(token))
            elif self.layout.is_y(token):
                if current is None or pending_x is None:
                    if validate:
                        raise ValueError("Y token appeared without a stroke anchor")
                    continue
                position = np.asarray(
                    [pending_x, float(self.layout.y_coordinate(token))], dtype=np.float32
                )
                current.append((float(position[0]), float(position[1])))
            elif self.layout.is_motion(token):
                if current is None or position is None:
                    if validate:
                        raise ValueError("Motion token appeared without an absolute anchor")
                    continue
                position = position + self.codebook[self.layout.motion_index(token)]
                current.append((float(position[0]), float(position[1])))
            elif token == self.layout.stroke_end_token_id:
                if current is not None:
                    strokes.append(current)
                current = None
                pending_x = None
                position = None
        return strokes


def encode_strokes(
    strokes: Sequence[Sequence[Sequence[float]]],
    codebook: np.ndarray,
) -> np.ndarray:
    return AnchoredTokenizer(codebook).encode(strokes)


def decode_tokens(
    tokens: Sequence[int] | np.ndarray,
    codebook: np.ndarray,
    *,
    validate: bool = True,
) -> list[Stroke]:
    return AnchoredTokenizer(codebook).decode(tokens, validate=validate)


def _coerce_stroke(stroke: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(stroke, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected stroke shape (N, 2), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Stroke coordinates must be finite")
    if len(points) == 0:
        return points
    keep = np.ones(len(points), dtype=bool)
    if len(points) > 1:
        keep[1:] = np.any(points[1:] != points[:-1], axis=1)
    return points[keep]

