"""Evaluation commands and lightweight reusable metric helpers."""

from metrics.sketchformer import (
    ReconstructionExample,
    collect_generated_reconstruction_examples,
    collect_reconstruction_examples,
    decode_token_sequence,
    prediction_to_stroke3,
    tensor_logs_to_floats,
    write_metrics_report,
    aggregate_free_running_records,
    free_running_reconstruction_metrics,
    free_running_reconstruction_records,
    stroke_geometry_metrics,
)

__all__ = [
    "ReconstructionExample",
    "collect_generated_reconstruction_examples",
    "collect_reconstruction_examples",
    "prediction_to_stroke3",
    "tensor_logs_to_floats",
    "write_metrics_report",
    "decode_token_sequence",
    "free_running_reconstruction_metrics",
    "free_running_reconstruction_records",
    "aggregate_free_running_records",
    "stroke_geometry_metrics",
]
