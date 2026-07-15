"""Public token and artifact contract for anchored V3 sketches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


FORMAT_TYPE = "anchored_v3"
FORMAT_VERSION = 3
TOKEN_LAYOUT_VERSION = 3
CANVAS_SIZE = 256
CANVAS_MARGIN = 8
MAX_SEQUENCE_LENGTH = 4096
CODEBOOK_SIZE = 2048
PRODUCTION_MIN_SOURCE_SKETCHES = 25_000


CANONICAL_TOKEN_DICTIONARY = {
    "pad_token_id": 0,
    "codebook_size": 2048,
    "motion_token_offset": 1,
    "x_token_offset": 2049,
    "y_token_offset": 2305,
    "coordinate_bins": 256,
    "stroke_start_token_id": 2561,
    "stroke_end_token_id": 2562,
    "sos_token_id": 2563,
    "eos_token_id": 2564,
    "mask_token_id": 2565,
    "vocab_size": 2566,
}


@dataclass(frozen=True)
class TokenLayout:
    """Stable IDs shared by preprocessing, models, checkpoints, and metrics."""

    version: int = TOKEN_LAYOUT_VERSION
    vocab_size: int = 2566
    pad_token_id: int = 0
    motion_token_start: int = 1
    motion_token_end: int = 2048
    x_token_start: int = 2049
    x_token_end: int = 2304
    y_token_start: int = 2305
    y_token_end: int = 2560
    stroke_start_token_id: int = 2561
    stroke_end_token_id: int = 2562
    sos_token_id: int = 2563
    eos_token_id: int = 2564
    mask_token_id: int = 2565

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TokenLayout":
        layout = cls(**{field: int(value) for field, value in values.items()})
        if layout != TOKEN_LAYOUT:
            raise ValueError("Token layout does not match anchored V3")
        return layout

    def is_motion(self, token: int) -> bool:
        return self.motion_token_start <= int(token) <= self.motion_token_end

    def is_x(self, token: int) -> bool:
        return self.x_token_start <= int(token) <= self.x_token_end

    def is_y(self, token: int) -> bool:
        return self.y_token_start <= int(token) <= self.y_token_end

    def motion_index(self, token: int) -> int:
        if not self.is_motion(token):
            raise ValueError(f"Token {token} is not a motion token")
        return int(token) - self.motion_token_start

    def x_coordinate(self, token: int) -> int:
        if not self.is_x(token):
            raise ValueError(f"Token {token} is not an absolute X token")
        return int(token) - self.x_token_start

    def y_coordinate(self, token: int) -> int:
        if not self.is_y(token):
            raise ValueError(f"Token {token} is not an absolute Y token")
        return int(token) - self.y_token_start


TOKEN_LAYOUT = TokenLayout()


def validate_anchored_v3_runtime_config(config: Mapping[str, Any]) -> None:
    """Reject configurations that only claim to implement anchored V3."""

    data = config.get("data")
    model = config.get("model")
    if not isinstance(data, Mapping) or not isinstance(model, Mapping):
        raise ValueError("Anchored V3 requires composed data and model configuration")
    format_config = data.get("format")
    if not isinstance(format_config, Mapping):
        raise ValueError("Anchored V3 requires data.format configuration")

    failures: list[str] = []
    if format_config.get("type") != FORMAT_TYPE:
        failures.append(f"format.type={format_config.get('type')!r}")
    if format_config.get("version") != FORMAT_VERSION:
        failures.append(f"format.version={format_config.get('version')!r}")
    token_dictionary = format_config.get("token_dictionary")
    if not isinstance(token_dictionary, Mapping):
        failures.append("format.token_dictionary is missing")
    else:
        failures.extend(
            f"{name}={token_dictionary.get(name)!r} expected {expected!r}"
            for name, expected in CANONICAL_TOKEN_DICTIONARY.items()
            if token_dictionary.get(name) != expected
        )
        if token_dictionary.get("sep_token_id") is not None:
            failures.append("sep_token_id must be null")

    decoder = model.get("decoder")
    if not isinstance(decoder, Mapping):
        failures.append("model.decoder is missing")
    else:
        if decoder.get("memory_source") != "encoder":
            failures.append("decoder.memory_source must be encoder")
        if decoder.get("tie_token_weights") is not True:
            failures.append("decoder.tie_token_weights must be true")
        if decoder.get("generation_grammar") != FORMAT_TYPE:
            failures.append("decoder.generation_grammar must be anchored_v3")

    if failures:
        raise ValueError("Anchored V3 runtime contract mismatch: " + "; ".join(failures))


def artifact_contract() -> dict[str, Any]:
    """Return the immutable portion of every anchored V3 metadata file."""

    return {
        "format_type": FORMAT_TYPE,
        "format_version": FORMAT_VERSION,
        "token_layout_version": TOKEN_LAYOUT_VERSION,
        "canvas_size": CANVAS_SIZE,
        "canvas_margin": CANVAS_MARGIN,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "codebook_size": CODEBOOK_SIZE,
        "token_layout": TOKEN_LAYOUT.to_dict(),
    }
