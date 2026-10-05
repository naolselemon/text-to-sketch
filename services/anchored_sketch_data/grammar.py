"""Finite-state grammar for anchored V3 token sequences."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np

from .contract import TOKEN_LAYOUT, TokenLayout


class GrammarState(str, Enum):
    EXPECT_SOS = "expect_sos"
    EXPECT_STROKE_OR_EOS = "expect_stroke_or_eos"
    EXPECT_X = "expect_x"
    EXPECT_Y = "expect_y"
    EXPECT_MOTION = "expect_motion"
    EXPECT_MOTION_OR_END = "expect_motion_or_end"
    COMPLETE = "complete"


@dataclass(frozen=True)
class GrammarSummary:
    stroke_count: int
    motion_count: int
    eos_index: int
    unpadded_length: int


class AnchoredGrammar:
    """Validate and constrain ``SOS (START X Y MOTION+ END)+ EOS``."""

    def __init__(self, layout: TokenLayout = TOKEN_LAYOUT) -> None:
        self.layout = layout

    def allowed_token_ids(
        self,
        state: GrammarState,
        *,
        stroke_count: int = 0,
    ) -> tuple[int, ...] | range:
        layout = self.layout
        if state == GrammarState.EXPECT_SOS:
            return (layout.sos_token_id,)
        if state == GrammarState.EXPECT_STROKE_OR_EOS:
            if stroke_count == 0:
                return (layout.stroke_start_token_id,)
            return (layout.stroke_start_token_id, layout.eos_token_id)
        if state == GrammarState.EXPECT_X:
            return range(layout.x_token_start, layout.x_token_end + 1)
        if state == GrammarState.EXPECT_Y:
            return range(layout.y_token_start, layout.y_token_end + 1)
        if state == GrammarState.EXPECT_MOTION:
            return range(layout.motion_token_start, layout.motion_token_end + 1)
        if state == GrammarState.EXPECT_MOTION_OR_END:
            return tuple(range(layout.motion_token_start, layout.motion_token_end + 1)) + (
                layout.stroke_end_token_id,
            )
        return (layout.pad_token_id,)

    def allowed_token_mask(
        self,
        state: GrammarState,
        *,
        stroke_count: int = 0,
    ) -> np.ndarray:
        mask = np.zeros(self.layout.vocab_size, dtype=bool)
        allowed = self.allowed_token_ids(state, stroke_count=stroke_count)
        if isinstance(allowed, range):
            mask[allowed.start : allowed.stop] = True
        else:
            mask[np.asarray(allowed, dtype=np.int64)] = True
        return mask

    def validate(self, tokens: Sequence[int] | np.ndarray) -> GrammarSummary:
        values = np.asarray(tokens)
        if values.ndim != 1:
            raise ValueError(f"Expected one-dimensional tokens, got {values.shape}")
        if values.size == 0:
            raise ValueError("Anchored V3 token sequence cannot be empty")
        if not np.issubdtype(values.dtype, np.integer):
            if not np.all(np.equal(values, np.floor(values))):
                raise ValueError("Token sequence contains non-integer values")
        integers = values.astype(np.int64, copy=False)
        if int(integers.min()) < 0 or int(integers.max()) >= self.layout.vocab_size:
            raise ValueError(
                f"Token IDs must be in [0, {self.layout.vocab_size}), "
                f"got [{int(integers.min())}, {int(integers.max())}]"
            )

        state = GrammarState.EXPECT_SOS
        stroke_count = 0
        motion_count = 0
        eos_index = -1
        for index, token_value in enumerate(integers):
            token = int(token_value)
            if state == GrammarState.COMPLETE:
                if token != self.layout.pad_token_id:
                    raise ValueError(f"Only PAD is valid after EOS, got {token} at index {index}")
                continue

            if not self._is_allowed(token, state, stroke_count=stroke_count):
                raise ValueError(
                    f"Token {token} at index {index} is invalid while grammar is {state.value}"
                )

            if state == GrammarState.EXPECT_SOS:
                state = GrammarState.EXPECT_STROKE_OR_EOS
            elif state == GrammarState.EXPECT_STROKE_OR_EOS:
                if token == self.layout.eos_token_id:
                    eos_index = index
                    state = GrammarState.COMPLETE
                else:
                    state = GrammarState.EXPECT_X
            elif state == GrammarState.EXPECT_X:
                state = GrammarState.EXPECT_Y
            elif state == GrammarState.EXPECT_Y:
                state = GrammarState.EXPECT_MOTION
            elif state == GrammarState.EXPECT_MOTION:
                motion_count += 1
                state = GrammarState.EXPECT_MOTION_OR_END
            elif state == GrammarState.EXPECT_MOTION_OR_END:
                if token == self.layout.stroke_end_token_id:
                    stroke_count += 1
                    state = GrammarState.EXPECT_STROKE_OR_EOS
                else:
                    motion_count += 1

        if state != GrammarState.COMPLETE:
            raise ValueError(f"Token sequence ended while grammar was {state.value}")
        return GrammarSummary(
            stroke_count=stroke_count,
            motion_count=motion_count,
            eos_index=eos_index,
            unpadded_length=eos_index + 1,
        )

    def _is_allowed(
        self,
        token: int,
        state: GrammarState,
        *,
        stroke_count: int,
    ) -> bool:
        """Check one transition without allocating a vocabulary-sized mask."""

        layout = self.layout
        if state == GrammarState.EXPECT_SOS:
            return token == layout.sos_token_id
        if state == GrammarState.EXPECT_STROKE_OR_EOS:
            return token == layout.stroke_start_token_id or (
                stroke_count > 0 and token == layout.eos_token_id
            )
        if state == GrammarState.EXPECT_X:
            return layout.is_x(token)
        if state == GrammarState.EXPECT_Y:
            return layout.is_y(token)
        if state == GrammarState.EXPECT_MOTION:
            return layout.is_motion(token)
        if state == GrammarState.EXPECT_MOTION_OR_END:
            return layout.is_motion(token) or token == layout.stroke_end_token_id
        return token == layout.pad_token_id


def validate_tokens(tokens: Sequence[int] | np.ndarray) -> GrammarSummary:
    return AnchoredGrammar().validate(tokens)
