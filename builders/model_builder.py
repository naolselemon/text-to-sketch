"""Factory for native in-repo Sketchformer models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build a model from the model config mapping."""

    name = str(config.get("name", "sketchformer_tok_dict"))
    if name not in {
        "sketchformer_tok_dict",
        "sketchformer_continuous",
        "sketchformer_anchored_v3_direct",
        "sketchformer_anchored_v3_wide",
    }:
        raise ValueError(f"Unsupported model config name: {name}")

    try:
        from models.sketchformer import SketchformerConfig, SketchformerModel
    except ImportError as exc:
        raise ImportError(
            "models.sketchformer is required before build_model can instantiate "
            "the native Sketchformer model."
        ) from exc

    return SketchformerModel(SketchformerConfig.from_mapping(config))


def maybe_compile_model(model: nn.Module, config: Mapping[str, Any]) -> nn.Module:
    """Apply ``torch.compile`` when enabled by config."""

    compile_config = config.get("compile", {})
    if not bool(compile_config.get("enabled", False)):
        return model

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")

    mode = compile_config.get("mode", "default")
    return torch.compile(model, mode=mode)


def build_model_from_config(config: Mapping[str, Any]) -> nn.Module:
    """Build and optionally compile a model from config."""

    model = build_model(config)
    return maybe_compile_model(model, config)
