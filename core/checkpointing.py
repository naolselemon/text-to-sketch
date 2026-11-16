"""Checkpoint helpers for model training and recovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

<<<<<<< HEAD
import re
import warnings

import numpy as np
=======
import warnings

>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c
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


<<<<<<< HEAD
STROKE_FORMAT_NAMES = {
    "stroke3",
    "anime_stroke3",
    "continuous",
    "stroke_3",
    "stroke-3",
    "sketchformer_continuous",
}
TOKEN_FORMAT_NAMES = {
    "tok_dict",
    "token",
    "tokens",
    "anime_tok_dict",
    "sketchformer_tok_dict",
}


def _validate_is_verified(val: Any, default: bool = True) -> bool:
    """Strictly validate that is_verified is genuinely a boolean.

    Accepts genuine Python booleans (bool), NumPy scalar booleans (np.bool_),
    or zero-dimensional scalar boolean NumPy arrays (np.ndarray of dtype bool and ndim 0).

    Rejects non-boolean types (e.g. strings, non-boolean integers, floats,
    arbitrary collections or multi-element arrays) instead of loosely coercing with bool().

    Raises CorruptedProvenanceError if an invalid non-boolean value is provided.
    """
    if val is None:
        return default
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, np.ndarray) and val.dtype == bool and val.ndim == 0:
        return bool(val.item())
    raise CorruptedProvenanceError(
        f"Invalid non-boolean value for 'is_verified': {val!r} (type {type(val).__name__}). "
        "Expected strict boolean True or False."
    )


def _classify_format_token(val: Any) -> str | None:
    """Classify a candidate format string into 'stroke3' or 'tok_dict'."""
    if val is None or not isinstance(val, (str, bytes)):
        return None
    s = str(val).strip().lower()
    if not s:
        return None
    if s in STROKE_FORMAT_NAMES:
        return "stroke3"
    if s in TOKEN_FORMAT_NAMES:
        return "tok_dict"

    # Compound name matching with word delimiters or explicit suffixes
    tokens = set(re.split(r"[-_.\s]+", s))
    has_stroke = bool(
        tokens & {"stroke3", "continuous"}
        or s.endswith(("_stroke3", "-stroke3", "_continuous", "-continuous"))
    )
    has_token = bool(
        tokens & {"tok_dict", "tokens"}
        or (tokens & {"token"} and not tokens & {"count", "dim", "size", "offset", "id", "embedding"})
    )

    if has_stroke and not has_token:
        return "stroke3"
    if has_token and not has_stroke:
        return "tok_dict"
    return None


def _resolve_dataset_format(
    lineage_data: Mapping[str, Any] | None = None,
    config_data: Mapping[str, Any] | None = None,
    *,
    is_verified: bool = True,
) -> str | None:
    """Resolve whether dataset format is continuous 'stroke3' or discrete 'tok_dict'.

    Determines the format exclusively from:
    1. Explicit, recognized lineage metadata ('format' key or nested metadata['format']).
    2. Explicit, recognized configuration fields ('data.format.type', 'data.name',
       'model.input.type', 'model.input.mode', 'model.name').

    Integrity guarantees:
    - Rejects explicit but unrecognized or unsupported formats in verified lineage with CorruptedProvenanceError.
    - Never infers Stroke-3 solely from an empty or missing dictionary fingerprint.
    - If lineage metadata and configuration explicitly declare conflicting formats,
      raises IncompatibleCheckpointError.
    - If neither lineage nor configuration establishes a recognized format:
        * If a non-empty dictionary fingerprint is present in lineage, resolves to 'tok_dict'.
        * If dictionary fingerprint is empty or missing and is_verified is True,
          fails closed with CorruptedProvenanceError (ambiguous provenance).
        * If is_verified is False, returns None (allowing unverified legacy checkpoints
          under explicit opt-in).
    """
    lineage_fmt: str | None = None
    if lineage_data is not None and isinstance(lineage_data, Mapping):
        fmt = lineage_data.get("format")
        if fmt is not None and str(fmt).strip():
            f_cls = _classify_format_token(fmt)
            if f_cls is None and is_verified:
                raise CorruptedProvenanceError(
                    f"Unrecognized or unsupported dataset format '{fmt}' in verified lineage metadata. "
                    "Expected recognized format ('stroke3' or 'tok_dict')."
                )
            lineage_fmt = f_cls

        meta = lineage_data.get("metadata")
        if isinstance(meta, Mapping):
            meta_fmt = meta.get("format")
            if meta_fmt is not None and str(meta_fmt).strip():
                m_cls = _classify_format_token(meta_fmt)
                if m_cls is None and is_verified:
                    raise CorruptedProvenanceError(
                        f"Unrecognized or unsupported dataset format '{meta_fmt}' in verified lineage metadata['format']. "
                        "Expected recognized format ('stroke3' or 'tok_dict')."
                    )
                if m_cls is not None:
                    if lineage_fmt is not None and lineage_fmt != m_cls:
                        raise IncompatibleCheckpointError(
                            f"Conflicting dataset format declarations within lineage metadata: "
                            f"'{fmt}' vs metadata['format']='{meta_fmt}'."
                        )
                    lineage_fmt = m_cls

    config_fmt: str | None = None
    if config_data is not None and isinstance(config_data, Mapping):
        config_candidates: list[Any] = []
        data_cfg = config_data.get("data")
        if isinstance(data_cfg, Mapping):
            data_fmt = data_cfg.get("format")
            if isinstance(data_fmt, Mapping):
                config_candidates.append(data_fmt.get("type"))
            elif isinstance(data_fmt, str):
                config_candidates.append(data_fmt)
            config_candidates.append(data_cfg.get("name"))

        model_cfg = config_data.get("model")
        if isinstance(model_cfg, Mapping):
            input_cfg = model_cfg.get("input")
            if isinstance(input_cfg, Mapping):
                config_candidates.append(input_cfg.get("type"))
                config_candidates.append(input_cfg.get("mode"))
            config_candidates.append(model_cfg.get("name"))

        for cand in config_candidates:
            c_cls = _classify_format_token(cand)
            if c_cls is not None:
                if config_fmt is not None and config_fmt != c_cls:
                    raise IncompatibleCheckpointError(
                        f"Conflicting dataset format declarations within configuration: "
                        f"'{config_fmt}' vs '{c_cls}'."
                    )
                config_fmt = c_cls

    # Validate cross-source conflict
    if lineage_fmt is not None and config_fmt is not None:
        if lineage_fmt != config_fmt:
            raise IncompatibleCheckpointError(
                f"Conflicting dataset format declarations: lineage declares '{lineage_fmt}', "
                f"but configuration declares '{config_fmt}'."
            )

    resolved_fmt = lineage_fmt or config_fmt

    if resolved_fmt is None:
        dict_fp = ""
        if lineage_data is not None and isinstance(lineage_data, Mapping):
            dict_fp = str(lineage_data.get("dictionary_fingerprint", "")).strip()

        if dict_fp:
            resolved_fmt = "tok_dict"
        elif is_verified:
            raise CorruptedProvenanceError(
                "Cannot determine dataset format for verified provenance: neither lineage metadata "
                "nor configuration specifies a recognized format ('stroke3' or 'tok_dict'), "
                "and no dictionary fingerprint is present."
            )

    return resolved_fmt


def _is_stroke_lineage(
    lineage_data: Mapping[str, Any] | None = None,
    config_data: Mapping[str, Any] | None = None,
) -> bool:
    """Determine whether a lineage mapping or configuration represents continuous Stroke-3 format."""
    try:
        fmt = _resolve_dataset_format(lineage_data, config_data, is_verified=False)
        return fmt == "stroke3"
    except (CorruptedProvenanceError, IncompatibleCheckpointError):
        raise


=======
>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c
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
<<<<<<< HEAD
            is_verified = _validate_is_verified(lineage_dict.get("is_verified", True), default=True)
            resolved_fmt = _resolve_dataset_format(
                lineage_dict,
                config,
                is_verified=is_verified,
            )
            is_stroke = (resolved_fmt == "stroke3")
            if resolved_fmt and not str(lineage_dict.get("format", "")).strip():
                lineage_dict["format"] = resolved_fmt

            if is_verified:
                required_provenance = ["build_id", "dataset_fingerprint"]
                if not is_stroke:
                    required_provenance.append("dictionary_fingerprint")
                missing_prov = [k for k in required_provenance if not str(lineage_dict.get(k, "")).strip()]
                if missing_prov:
                    raise CorruptedProvenanceError(
                        f"Cannot save checkpoint with incomplete lineage metadata: missing {missing_prov}"
                    )
=======
            required_provenance = ["build_id", "dictionary_fingerprint", "dataset_fingerprint"]
            missing_prov = [k for k in required_provenance if not str(lineage_dict.get(k, "")).strip()]
            if missing_prov:
                raise CorruptedProvenanceError(
                    f"Cannot save checkpoint with incomplete lineage metadata: missing {missing_prov}"
                )
>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c

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
<<<<<<< HEAD
    require_dataset: bool = False,
=======
>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c
) -> bool:
    """Validate that a checkpoint matches the active dataset lineage and model configuration.

    Args:
        checkpoint_lineage: Lineage mapping or DatasetLineage from checkpoint.
        dataset_lineage: Active DatasetLineage or lineage mapping from dataset root.
        checkpoint_config: Configuration mapping saved in checkpoint.
        active_config: Current configuration mapping for evaluation or export.
        split: Optional split name ("train", "valid", "test") to verify specifically.
        allow_unverified: If True, permit legacy checkpoints without provenance metadata.
<<<<<<< HEAD
        require_dataset: If True, enforce that the active dataset has verified lineage metadata;
            rejects missing dataset lineage with UnverifiedCheckpointError unless allow_unverified is True.
=======
>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c

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

<<<<<<< HEAD
    ckpt_is_stroke = False
    ckpt_fmt: str | None = None
    if not ckpt_lin:
        # Checkpoint lacks lineage metadata: attempt to resolve format from checkpoint configuration
        ckpt_fmt = _resolve_dataset_format(
            None,
            checkpoint_config,
            is_verified=False,
        )
        ckpt_is_stroke = (ckpt_fmt == "stroke3")

=======
    if not ckpt_lin:
>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c
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
<<<<<<< HEAD
        # Checkpoint declared provenance: verify format and verification status
        ckpt_is_verified = _validate_is_verified(ckpt_lin.get("is_verified", True), default=True)
        ckpt_fmt = _resolve_dataset_format(
            ckpt_lin,
            checkpoint_config,
            is_verified=ckpt_is_verified,
        )
        ckpt_is_stroke = (ckpt_fmt == "stroke3")

        if not ckpt_is_verified:
            if not allow_unverified:
                raise UnverifiedCheckpointError(
                    "Checkpoint has unverified dataset lineage. "
                    "Pass --allow-unverified-checkpoint to bypass verification."
                )
            warnings.warn(
                "Checkpoint has unverified dataset lineage; continuing because unverified checkpoints are allowed.",
                UserWarning,
                stacklevel=2,
            )
        else:
            required_provenance = ["build_id", "dataset_fingerprint"]
            if not ckpt_is_stroke:
                required_provenance.append("dictionary_fingerprint")
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

    ds_is_stroke = False
    ds_fmt: str | None = None
    if not ds_lin:
        # Active dataset lacks lineage metadata: attempt to resolve format from active configuration
        ds_fmt = _resolve_dataset_format(
            None,
            active_config,
            is_verified=False,
        )
        ds_is_stroke = (ds_fmt == "stroke3")

        if require_dataset:
            if not allow_unverified:
                raise UnverifiedCheckpointError(
                    "Active dataset lacks verified lineage metadata (meta.npz not found or missing provenance). "
                    "Pass --allow-unverified-checkpoint to bypass verification for unverified datasets."
                )
            warnings.warn(
                "Active dataset lacks verified lineage metadata; continuing because unverified checkpoints are allowed.",
                UserWarning,
                stacklevel=2,
            )
    else:
        ds_is_verified = _validate_is_verified(ds_lin.get("is_verified", True), default=True)
        ds_fmt = _resolve_dataset_format(
            ds_lin,
            active_config,
            is_verified=ds_is_verified,
        )
        ds_is_stroke = (ds_fmt == "stroke3")

        if not ds_is_verified:
            if not allow_unverified:
                raise UnverifiedCheckpointError(
                    "Active dataset lineage is unverified. "
                    "Pass --allow-unverified-checkpoint to bypass verification for unverified datasets."
                )
            warnings.warn(
                "Active dataset lineage is unverified; continuing because unverified datasets are allowed.",
                UserWarning,
                stacklevel=2,
            )
        else:
            required_ds_prov = ["build_id", "dataset_fingerprint"]
            if not ds_is_stroke:
                required_ds_prov.append("dictionary_fingerprint")
            missing_ds_prov = [k for k in required_ds_prov if not str(ds_lin.get(k, "")).strip()]
            if missing_ds_prov:
                raise CorruptedProvenanceError(
                    f"Active dataset has incomplete or corrupted provenance metadata: missing {missing_ds_prov}. "
                    "Corrupted provenance cannot be bypassed with --allow-unverified-checkpoint."
                )

    # Cross-format conflict check: runs whenever both formats are known,
    # even if lineage is missing on either or both sides!
    if ckpt_fmt is not None and ds_fmt is not None:
        if ckpt_is_stroke != ds_is_stroke:
            ckpt_fmt_name = "stroke" if ckpt_is_stroke else "token"
            ds_fmt_name = "stroke" if ds_is_stroke else "token"
            raise IncompatibleCheckpointError(
                f"Dataset format mismatch: checkpoint was trained on {ckpt_fmt_name} dataset, "
                f"but target dataset is {ds_fmt_name} format."
            )

    if ckpt_lin and ds_lin:

        # 1. Token dictionary fingerprint verification
        if not ckpt_is_stroke and not ds_is_stroke:
=======
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
>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c
            ckpt_dict_fp = str(ckpt_lin.get("dictionary_fingerprint", "")).strip()
            ds_dict_fp = str(ds_lin.get("dictionary_fingerprint", "")).strip()
            if ckpt_dict_fp and ds_dict_fp and ckpt_dict_fp != ds_dict_fp:
                raise IncompatibleCheckpointError(
                    f"Token dictionary fingerprint mismatch: checkpoint was trained with '{ckpt_dict_fp}', "
                    f"but target dataset uses '{ds_dict_fp}'."
                )

<<<<<<< HEAD
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
        ckpt_splits = ckpt_lin.get("split_hashes") or {}
        ds_splits = ds_lin.get("split_hashes") or {}
        if split:
            if ckpt_is_verified and ds_is_verified:
                if split not in ckpt_splits:
                    raise IncompatibleCheckpointError(
                        f"Requested evaluation split '{split}' is missing from checkpoint lineage split_hashes."
                    )
                if split not in ds_splits:
                    raise IncompatibleCheckpointError(
                        f"Requested evaluation split '{split}' is missing from active dataset lineage split_hashes."
                    )
                if ckpt_splits[split] != ds_splits[split]:
                    raise IncompatibleCheckpointError(
                        f"Dataset split '{split}' hash mismatch: checkpoint has '{ckpt_splits[split]}', "
                        f"but target dataset has '{ds_splits[split]}'."
                    )
            elif split in ckpt_splits and split in ds_splits:
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
=======
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
>>>>>>> 59b0fe3f0e28f0e53ffba5988a6b9911698e1d8c

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
