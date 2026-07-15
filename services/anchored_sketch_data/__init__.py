"""Anchored V3 sketch dataset service."""

from .artifacts import (
    EncodedSample,
    RejectedSample,
    require_minimum_source_sketches,
    validate_dataset,
    write_dataset_atomic,
)
from .builder import BuilderConfig, build_dataset
from .codebook import TrainingStrokeSample, fit_training_codebook, within_stroke_deltas
from .contract import (
    CANONICAL_TOKEN_DICTIONARY,
    CANVAS_MARGIN,
    CANVAS_SIZE,
    CODEBOOK_SIZE,
    FORMAT_TYPE,
    FORMAT_VERSION,
    MAX_SEQUENCE_LENGTH,
    PRODUCTION_MIN_SOURCE_SKETCHES,
    TOKEN_LAYOUT,
    TOKEN_LAYOUT_VERSION,
    TokenLayout,
    validate_anchored_v3_runtime_config,
)
from .grammar import AnchoredGrammar, GrammarState, GrammarSummary, validate_tokens
from .tokenizer import AnchoredTokenizer, decode_tokens, encode_strokes

__all__ = [
    "AnchoredGrammar",
    "AnchoredTokenizer",
    "BuilderConfig",
    "CANONICAL_TOKEN_DICTIONARY",
    "CANVAS_MARGIN",
    "CANVAS_SIZE",
    "CODEBOOK_SIZE",
    "EncodedSample",
    "FORMAT_TYPE",
    "FORMAT_VERSION",
    "GrammarState",
    "GrammarSummary",
    "MAX_SEQUENCE_LENGTH",
    "PRODUCTION_MIN_SOURCE_SKETCHES",
    "RejectedSample",
    "require_minimum_source_sketches",
    "TOKEN_LAYOUT",
    "TOKEN_LAYOUT_VERSION",
    "TokenLayout",
    "TrainingStrokeSample",
    "build_dataset",
    "decode_tokens",
    "encode_strokes",
    "fit_training_codebook",
    "validate_dataset",
    "validate_anchored_v3_runtime_config",
    "validate_tokens",
    "within_stroke_deltas",
    "write_dataset_atomic",
]
