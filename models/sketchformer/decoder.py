"""Transformer decoder blocks for Sketchformer reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.sketchformer.config import SketchformerConfig
from models.sketchformer.encoder import FeedForward, SDPAAttention


class LatentExpander(nn.Module):
    """Expand a pooled latent vector into decoder memory states."""

    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        max_length: int,
        *,
        mode: str = "projected_position",
        base_length: int | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.max_length = int(max_length)
        self.output_dim = d_model if mode == "projected_position" else latent_dim
        if mode == "projected_position":
            self.latent_projection = nn.Linear(latent_dim, d_model)
            self.position = nn.Embedding(max_length, d_model)
        elif mode == "tf_dense":
            self.base_length = int(base_length or max_length)
            self.expand_layer = nn.Linear(1, self.base_length)
            if self.base_length < self.max_length:
                self.long_weight = nn.Parameter(torch.zeros(self.max_length, 1))
                self.long_bias = nn.Parameter(torch.zeros(self.max_length))
            else:
                self.register_parameter("long_weight", None)
                self.register_parameter("long_bias", None)
        else:
            raise ValueError("mode must be one of: projected_position, tf_dense")

    def forward(self, latent: torch.Tensor, sequence_length: int) -> torch.Tensor:
        if sequence_length > self.max_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds latent expander capacity {self.max_length}"
            )
        if self.mode == "projected_position":
            positions = torch.arange(sequence_length, device=latent.device)
            memory = self.latent_projection(latent).unsqueeze(1)
            return memory + self.position(positions).unsqueeze(0)

        expanded = self.expand_layer(latent.unsqueeze(2)).transpose(1, 2)
        if sequence_length != self.base_length:
            expanded = F.interpolate(
                expanded.transpose(1, 2),
                size=sequence_length,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        if self.long_weight is not None and self.long_bias is not None:
            weight = _resize_position_parameter(self.long_weight, sequence_length)
            bias = _resize_position_parameter(self.long_bias, sequence_length)
            residual = F.linear(latent.unsqueeze(2), weight, bias).transpose(1, 2)
            expanded = expanded + residual
        return expanded


def _resize_position_parameter(value: torch.Tensor, length: int) -> torch.Tensor:
    if value.shape[0] == length:
        return value
    if value.ndim == 1:
        return F.interpolate(
            value.view(1, 1, -1),
            size=length,
            mode="linear",
            align_corners=False,
        ).view(length)
    return F.interpolate(
        value.T.unsqueeze(0),
        size=length,
        mode="linear",
        align_corners=False,
    ).squeeze(0).T


@dataclass
class DecoderLayerCache:
    """Projected attention states reused during autoregressive generation."""

    self_key_value: tuple[torch.Tensor, torch.Tensor] | None = None
    cross_key_value: tuple[torch.Tensor, torch.Tensor] | None = None


class DecoderBlock(nn.Module):
    """Pre-norm transformer decoder block with self and cross attention."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.norm_first = config.norm_first
        memory_dim = (
            config.d_model
            if config.decoder_memory_source == "encoder"
            else (
                config.pool_output_dim
                if config.latent_expander_mode == "tf_dense"
                else config.d_model
            )
        )
        self.self_attn = SDPAAttention(config.d_model, config.num_heads, config.dropout)
        self.cross_attn = SDPAAttention(
            config.d_model,
            config.num_heads,
            config.dropout,
            key_dim=memory_dim,
            value_dim=memory_dim,
        )
        self.ffn = FeedForward(
            config.d_model,
            config.dim_feedforward,
            config.dropout,
            config.activation,
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        self_attention_is_causal: bool = False,
    ) -> torch.Tensor:
        if self.norm_first:
            norm_x = self.norm1(x)
            x = x + self.dropout1(
                self.self_attn(
                    norm_x,
                    norm_x,
                    norm_x,
                    self_attention_mask,
                    is_causal=self_attention_is_causal,
                )
            )
            x = x + self.dropout2(
                self.cross_attn(self.norm2(x), memory, memory, cross_attention_mask)
            )
            x = x + self.dropout3(self.ffn(self.norm3(x)))
            return x

        self_attn = self.self_attn(
            x,
            x,
            x,
            self_attention_mask,
            is_causal=self_attention_is_causal,
        )
        x = self.norm1(x + self.dropout1(self_attn))
        cross_attn = self.cross_attn(x, memory, memory, cross_attention_mask)
        x = self.norm2(x + self.dropout2(cross_attn))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x

    def forward_step(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        cache: DecoderLayerCache | None = None,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DecoderLayerCache]:
        """Decode new positions using cached self and cross-attention states."""

        cache = cache or DecoderLayerCache()
        if self.norm_first:
            norm_x = self.norm1(x)
            self_output, self_cache = self.self_attn.forward_cached(
                norm_x,
                norm_x,
                norm_x,
                cache=cache.self_key_value,
            )
            x = x + self.dropout1(self_output)
            cross_query = self.norm2(x)
            cross_output, cross_cache = self.cross_attn.forward_cached(
                cross_query,
                memory,
                memory,
                cache=cache.cross_key_value,
                static_key_value=True,
                attention_mask=cross_attention_mask,
            )
            x = x + self.dropout2(cross_output)
            x = x + self.dropout3(self.ffn(self.norm3(x)))
        else:
            self_output, self_cache = self.self_attn.forward_cached(
                x,
                x,
                x,
                cache=cache.self_key_value,
            )
            x = self.norm1(x + self.dropout1(self_output))
            cross_output, cross_cache = self.cross_attn.forward_cached(
                x,
                memory,
                memory,
                cache=cache.cross_key_value,
                static_key_value=True,
                attention_mask=cross_attention_mask,
            )
            x = self.norm2(x + self.dropout2(cross_output))
            x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x, DecoderLayerCache(self_cache, cross_cache)


class StrokeDecoder(nn.Module):
    """Stack of SDPA transformer decoder blocks."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_decoder_layers)]
        )
        self.final_norm = (
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
            if config.use_final_norm
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        self_attention_is_causal: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(
                    layer,
                    x,
                    memory,
                    self_attention_mask,
                    cross_attention_mask,
                    self_attention_is_causal,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x,
                    memory,
                    self_attention_mask,
                    cross_attention_mask,
                    self_attention_is_causal,
                )
        return self.final_norm(x)

    def forward_step(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        caches: list[DecoderLayerCache] | None = None,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[DecoderLayerCache]]:
        """Decode incremental positions with one cache per decoder layer."""

        if caches is None:
            caches = [DecoderLayerCache() for _ in self.layers]
        if len(caches) != len(self.layers):
            raise ValueError("Decoder cache count does not match decoder layer count")
        updated: list[DecoderLayerCache] = []
        for layer, cache in zip(self.layers, caches):
            x, layer_cache = layer.forward_step(
                x,
                memory,
                cache,
                cross_attention_mask,
            )
            updated.append(layer_cache)
        return self.final_norm(x), updated
