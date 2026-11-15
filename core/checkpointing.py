"""Checkpoint helpers for model training and recovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import warnings

import torch
from torch import nn


class IncompatibleCheckpointError(ValueError):
    """Raised when a checkpoint is incompatible with the target dataset or configuration."""


class UnverifiedCheckpointError(IncompatibleCheckpointError):
    """Raised when a checkpoint lacks verified dataset provenance."""


class CorruptedProvenanceError(IncompatibleCheckpointError):
    """Raised when a checkpoint's provenance metadata is incomplete or corrupted."""


@dataclass
class CheckpointLoadResult:
    """Metadata returned after loading a checkpoint."""

    epoch: int = 0
    step: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)


def load_checkpoint_metadata(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Inspect checkpoint metadata without instantiating model architecture or weights.

    Returns a dictionary containing 'epoch', 'step', 'metrics', 'config', and 'lineage'.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint at {ckpt_path} must be a mapping, got {type(checkpoint).__name__}")

    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "step": int(checkpoint.get("step", 0)),
        "metrics": dict(checkpoint.get("metrics", {})),
        "config": dict(checkpoint.get("config", {})),
        "lineage": dict(checkpoint.get("lineage", {})),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    lineage: Any | None = None,
) -> Path:
    """Save a training checkpoint with config and dataset lineage metadata."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    lineage_dict: dict[str, Any] = {}
    if lineage is not None:
        if hasattr(lineage, "to_dict"):
            lineage_dict = lineage.to_dict()
        elif isinstance(lineage, Mapping):
            lineage_dict = dict(lineage)

        if lineage_dict:
            required_provenance = ["build_id", "dictionary_fingerprint", "dataset_fingerprint"]
            missing_prov = [k for k in required_provenance if not str(lineage_dict.get(k, "")).strip()]
            if missing_prov:
                raise CorruptedProvenanceError(
                    f"Cannot save checkpoint with incomplete lineage metadata: missing {missing_prov}"
                )

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "metrics": dict(metrics or {}),
        "config": dict(config or {}),
        "lineage": lineage_dict,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    scheduler_obj = getattr(scheduler, "scheduler", scheduler)
    if scheduler_obj is not None and hasattr(scheduler_obj, "state_dict"):
        payload["scheduler"] = scheduler_obj.state_dict()

    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> CheckpointLoadResult:
    """Load model and optional optimizer/scheduler state and provenance."""

    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")

    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler_obj = getattr(scheduler, "scheduler", scheduler)
    if scheduler_obj is not None and "scheduler" in checkpoint:
        scheduler_obj.load_state_dict(checkpoint["scheduler"])

    return CheckpointLoadResult(
        epoch=int(checkpoint.get("epoch", 0)),
        step=int(checkpoint.get("step", 0)),
        metrics=dict(checkpoint.get("metrics", {})),
        config=dict(checkpoint.get("config", {})),
        lineage=dict(checkpoint.get("lineage", {})),
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
    )


def validate_checkpoint_compatibility(
    checkpoint_lineage: Mapping[str, Any] | Any | None,
    dataset_lineage: Mapping[str, Any] | Any | None = None,
    *,
    checkpoint_config: Mapping[str, Any] | None = None,
    active_config: Mapping[str, Any] | None = None,
    split: str | None = None,
    allow_unverified: bool = False,
) -> bool:
    """Validate that a checkpoint matches the active dataset lineage and model configuration.

    Args:
        checkpoint_lineage: Lineage mapping or DatasetLineage from checkpoint.
        dataset_lineage: Active DatasetLineage or lineage mapping from dataset root.
        checkpoint_config: Configuration mapping saved in checkpoint.
        active_config: Current configuration mapping for evaluation or export.
        split: Optional split name ("train", "valid", "test") to verify specifically.
        allow_unverified: If True, permit legacy checkpoints without provenance metadata.

    Returns:
        True if compatible or permitted by allow_unverified.

    Raises:
        UnverifiedCheckpointError: If checkpoint lacks dataset provenance and allow_unverified is False.
        CorruptedProvenanceError: If checkpoint's provenance metadata is incomplete or corrupted.
        IncompatibleCheckpointError: If fingerprints, build IDs, splits, or architecture mismatch.
    """
    ckpt_lin: dict[str, Any] = {}
    if checkpoint_lineage is not None:
        if hasattr(checkpoint_lineage, "to_dict"):
            ckpt_lin = checkpoint_lineage.to_dict()
        elif isinstance(checkpoint_lineage, Mapping):
            ckpt_lin = dict(checkpoint_lineage)

    if not ckpt_lin:
        if not allow_unverified:
            raise UnverifiedCheckpointError(
                "Checkpoint lacks dataset lineage and provenance metadata. "
                "Pass --allow-unverified-checkpoint to bypass verification for legacy models."
            )
        warnings.warn(
            "Checkpoint lacks dataset lineage metadata; continuing because unverified checkpoints are allowed.",
            UserWarning,
            stacklevel=2,
        )
    else:
        # Checkpoint declared provenance: verify it is complete and uncorrupted
        required_provenance = ["build_id", "dictionary_fingerprint", "dataset_fingerprint"]
        missing_provenance = [k for k in required_provenance if not str(ckpt_lin.get(k, "")).strip()]
        if missing_provenance:
            raise CorruptedProvenanceError(
                f"Checkpoint has incomplete or corrupted provenance metadata: missing {missing_provenance}. "
                "Corrupted provenance cannot be bypassed with --allow-unverified-checkpoint."
            )

        ds_lin: dict[str, Any] = {}
        if dataset_lineage is not None:
            if hasattr(dataset_lineage, "to_dict"):
                ds_lin = dataset_lineage.to_dict()
            elif isinstance(dataset_lineage, Mapping):
                ds_lin = dict(dataset_lineage)

        if ds_lin:
            # 1. Token dictionary fingerprint verification
            ckpt_dict_fp = str(ckpt_lin.get("dictionary_fingerprint", "")).strip()
            ds_dict_fp = str(ds_lin.get("dictionary_fingerprint", "")).strip()
            if ckpt_dict_fp and ds_dict_fp and ckpt_dict_fp != ds_dict_fp:
                raise IncompatibleCheckpointError(
                    f"Token dictionary fingerprint mismatch: checkpoint was trained with '{ckpt_dict_fp}', "
                    f"but target dataset uses '{ds_dict_fp}'."
                )

            # 2. Dataset build ID verification
            ckpt_build = str(ckpt_lin.get("build_id", "")).strip()
            ds_build = str(ds_lin.get("build_id", "")).strip()
            if ckpt_build and ds_build and ckpt_build != ds_build:
                raise IncompatibleCheckpointError(
                    f"Dataset build_id mismatch: checkpoint was trained on build '{ckpt_build}', "
                    f"but target dataset is build '{ds_build}'."
                )

            # 3. Overall dataset fingerprint verification
            ckpt_ds_fp = str(ckpt_lin.get("dataset_fingerprint", "")).strip()
            ds_ds_fp = str(ds_lin.get("dataset_fingerprint", "")).strip()
            if ckpt_ds_fp and ds_ds_fp and ckpt_ds_fp != ds_ds_fp:
                raise IncompatibleCheckpointError(
                    f"Dataset fingerprint mismatch: checkpoint was trained on dataset '{ckpt_ds_fp}', "
                    f"but target dataset fingerprint is '{ds_ds_fp}'."
                )

            # 4. Split hash verification
            ckpt_splits = ckpt_lin.get("split_hashes", {})
            ds_splits = ds_lin.get("split_hashes", {})
            if split:
                if split in ckpt_splits and split in ds_splits:
                    if ckpt_splits[split] != ds_splits[split]:
                        raise IncompatibleCheckpointError(
                            f"Dataset split '{split}' hash mismatch: checkpoint has '{ckpt_splits[split]}', "
                            f"but target dataset has '{ds_splits[split]}'."
                        )
            else:
                for s in ("train", "valid", "test"):
                    if s in ckpt_splits and s in ds_splits:
                        if ckpt_splits[s] != ds_splits[s]:
                            raise IncompatibleCheckpointError(
                                f"Dataset split '{s}' hash mismatch: checkpoint has '{ckpt_splits[s]}', "
                                f"but target dataset has '{ds_splits[s]}'."
                            )

    if checkpoint_config and active_config:
        try:
            from models.sketchformer.config import SketchformerConfig

            ckpt_raw = checkpoint_config.get("model", checkpoint_config)
            act_raw = active_config.get("model", active_config)
            ckpt_sf = SketchformerConfig.from_mapping(ckpt_raw)
            act_sf = SketchformerConfig.from_mapping(act_raw)

            arch_checks = [
                ("name", ckpt_sf.name, act_sf.name),
                ("input_mode", ckpt_sf.input_mode, act_sf.input_mode),
                ("stroke_dim", ckpt_sf.stroke_dim, act_sf.stroke_dim),
                ("pen_classes", ckpt_sf.pen_classes, act_sf.pen_classes),
                ("d_model", ckpt_sf.d_model, act_sf.d_model),
                ("latent_dim", ckpt_sf.latent_dim, act_sf.latent_dim),
                ("num_encoder_layers", ckpt_sf.num_encoder_layers, act_sf.num_encoder_layers),
                ("num_decoder_layers", ckpt_sf.num_decoder_layers, act_sf.num_decoder_layers),
                ("num_heads", ckpt_sf.num_heads, act_sf.num_heads),
                ("dim_feedforward", ckpt_sf.dim_feedforward, act_sf.dim_feedforward),
                ("pooling_mode", ckpt_sf.pooling_mode, act_sf.pooling_mode),
                ("pool_hidden_dim", ckpt_sf.pool_hidden_dim, act_sf.pool_hidden_dim),
                ("latent_expander_mode", ckpt_sf.latent_expander_mode, act_sf.latent_expander_mode),
                ("pen_embedding_dim", ckpt_sf.pen_embedding_dim, act_sf.pen_embedding_dim),
                ("combine_method", ckpt_sf.combine_method, act_sf.combine_method),
                ("positional_encoding.type", ckpt_sf.positional_encoding.type, act_sf.positional_encoding.type),
                ("positional_encoding.max_length", ckpt_sf.positional_encoding.max_length, act_sf.positional_encoding.max_length),
                ("reconstruction.enabled", ckpt_sf.reconstruction.enabled, act_sf.reconstruction.enabled),
                ("reconstruction.target", ckpt_sf.reconstruction.target, act_sf.reconstruction.target),
                ("reconstruction.xy_distribution", ckpt_sf.reconstruction.xy_distribution, act_sf.reconstruction.xy_distribution),
                ("reconstruction.num_mixtures", ckpt_sf.reconstruction.num_mixtures, act_sf.reconstruction.num_mixtures),
                ("reconstruction.predict_pen_state", ckpt_sf.reconstruction.predict_pen_state, act_sf.reconstruction.predict_pen_state),
                ("classification.enabled", ckpt_sf.classification.enabled, act_sf.classification.enabled),
                ("classification.num_classes", ckpt_sf.classification.num_classes, act_sf.classification.num_classes),
            ]
            if ckpt_sf.input_mode in {"tok_dict", "token", "tokens"} or act_sf.input_mode in {"tok_dict", "token", "tokens"}:
                tok_checks = [
                    ("vocab_size", ckpt_sf.token_dictionary.vocab_size, act_sf.token_dictionary.vocab_size),
                    ("codebook_size", ckpt_sf.token_dictionary.codebook_size, act_sf.token_dictionary.codebook_size),
                    ("motion_token_offset", ckpt_sf.token_dictionary.motion_token_offset, act_sf.token_dictionary.motion_token_offset),
                    ("pad_token_id", ckpt_sf.token_dictionary.pad_token_id, act_sf.token_dictionary.pad_token_id),
                    ("sep_token_id", ckpt_sf.token_dictionary.sep_token_id, act_sf.token_dictionary.sep_token_id),
                    ("sos_token_id", ckpt_sf.token_dictionary.sos_token_id, act_sf.token_dictionary.sos_token_id),
                    ("eos_token_id", ckpt_sf.token_dictionary.eos_token_id, act_sf.token_dictionary.eos_token_id),
                ]
                arch_checks.extend(tok_checks)

            for param, ckpt_val, act_val in arch_checks:
                if ckpt_val != act_val:
                    raise IncompatibleCheckpointError(
                        f"Model architecture mismatch: checkpoint has '{param}={ckpt_val}', "
                        f"but active config expects '{param}={act_val}'."
                    )
        except IncompatibleCheckpointError:
            raise
        except Exception:
            # Fallback for arbitrary config dictionaries
            ckpt_model = checkpoint_config.get("model", checkpoint_config)
            act_model = active_config.get("model", active_config)

            ckpt_name = ckpt_model.get("name")
            act_name = act_model.get("name")
            if (ckpt_name is not None or act_name is not None) and ckpt_name != act_name:
                raise IncompatibleCheckpointError(
                    f"Model architecture name mismatch: checkpoint model is '{ckpt_name}', "
                    f"but active config expects '{act_name}'."
                )

            ckpt_input = ckpt_model.get("input", {})
            act_input = act_model.get("input", {})
            ckpt_in_mode = ckpt_input.get("type", ckpt_input.get("mode"))
            act_in_mode = act_input.get("type", act_input.get("mode"))
            if (ckpt_in_mode is not None or act_in_mode is not None) and ckpt_in_mode != act_in_mode:
                raise IncompatibleCheckpointError(
                    f"Model input mode mismatch: checkpoint input mode is '{ckpt_in_mode}', "
                    f"but active config expects '{act_in_mode}'."
                )

            ckpt_arch = ckpt_model.get("architecture", {})
            act_arch = act_model.get("architecture", {})
            for param in (
                "d_model",
                "dim_feedforward",
                "latent_dim",
                "num_encoder_layers",
                "num_decoder_layers",
                "num_heads",
                "pooling_mode",
                "latent_expander_mode",
                "pool_hidden_dim",
            ):
                ckpt_v = ckpt_arch.get(param)
                act_v = act_arch.get(param)
                if (ckpt_v is not None or act_v is not None) and ckpt_v != act_v:
                    raise IncompatibleCheckpointError(
                        f"Model architecture mismatch: checkpoint has '{param}={ckpt_v}', "
                        f"but active config has '{param}={act_v}'."
                    )

            ckpt_tok = ckpt_input.get("token_dictionary", {})
            act_tok = act_input.get("token_dictionary", {})
            ckpt_vocab = ckpt_tok.get("vocab_size")
            act_vocab = act_tok.get("vocab_size")
            if (ckpt_vocab is not None or act_vocab is not None) and ckpt_vocab != act_vocab:
                raise IncompatibleCheckpointError(
                    f"Model vocab_size mismatch: checkpoint has vocab_size={ckpt_vocab}, "
                    f"but active config expects vocab_size={act_vocab}."
                )

    return True


def latest_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Path | None:
    """Return the most recently modified checkpoint in a directory."""

    checkpoint_dir = Path(directory)
    candidates = sorted(checkpoint_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None
