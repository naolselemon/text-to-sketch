"""Native PyTorch Sketchformer-style autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from models.sketchformer.config import SketchformerConfig
from models.sketchformer.decoder import LatentExpander, StrokeDecoder
from models.sketchformer.embeddings import (
    DecoderQueryEmbedding,
    Stroke3Embedding,
    TokenEmbedding,
)
from models.sketchformer.encoder import AttentionPool, StrokeEncoder
from models.sketchformer.heads import (
    ClassificationHead,
    ContinuousReconstructionHead,
    ReconstructionOutput,
    TokenReconstructionHead,
    TokenReconstructionOutput,
)


@dataclass
class SketchformerOutput:
    """Forward-pass output consumed by losses, metrics, and visualization."""

    embedding: torch.Tensor
    encoded: torch.Tensor
    decoded: torch.Tensor
    reconstruction: ReconstructionOutput | TokenReconstructionOutput | None
    class_logits: torch.Tensor | None
    loss_targets: torch.Tensor | None = None
    loss_valid_mask: torch.Tensor | None = None


@dataclass
class GenerationOutput:
    """Free-running token reconstruction returned by ``generate``."""

    tokens: torch.Tensor
    lengths: torch.Tensor
    embedding: torch.Tensor


class SketchformerModel(nn.Module):
    """Long-sequence-capable Sketchformer-style model implemented in PyTorch."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        if self._uses_token_input:
            self.input_embedding = TokenEmbedding(config)
            self.target_embedding = (
                TokenEmbedding(config)
                if config.decoder_autoregressive
                else DecoderQueryEmbedding(config)
            )
        else:
            self.input_embedding = Stroke3Embedding(config)
            self.target_embedding = Stroke3Embedding(config)
        self.encoder = StrokeEncoder(config)
        self.pool = AttentionPool(
            config.d_model,
            config.latent_dim,
            mode=config.pooling_mode,
            hidden_dim=config.pool_hidden_dim,
        )
        self.latent_expander = (
            LatentExpander(
                config.pool_output_dim,
                config.d_model,
                config.max_seq_len,
                mode=config.latent_expander_mode,
                base_length=config.latent_expander_base_length,
            )
            if config.decoder_memory_source == "latent_expander"
            else None
        )
        self.decoder = StrokeDecoder(config)

        self.reconstruction_head = self._build_reconstruction_head(config)
        if config.tie_token_weights:
            self._tie_token_weights()
        self.classification_head = (
            ClassificationHead(config) if config.classification.enabled else None
        )

    @property
    def _uses_token_input(self) -> bool:
        return self.config.input_mode in {"tok_dict", "token", "tokens"}

    @staticmethod
    def _build_reconstruction_head(
        config: SketchformerConfig,
    ) -> ContinuousReconstructionHead | TokenReconstructionHead | None:
        if not config.reconstruction.enabled:
            return None
        if config.reconstruction.target in {"tok_dict", "token", "tokens"}:
            return TokenReconstructionHead(config)
        return ContinuousReconstructionHead(config)

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "SketchformerModel":
        return cls(SketchformerConfig.from_mapping(config))

    def forward(
        self,
        strokes: torch.Tensor | dict[str, Any],
        *,
        targets: torch.Tensor | None = None,
        decoder_inputs: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> SketchformerOutput:
        if isinstance(strokes, dict):
            batch = strokes
            targets = targets if targets is not None else batch.get("targets")
            decoder_inputs = (
                decoder_inputs
                if decoder_inputs is not None
                else batch.get("decoder_inputs")
            )
            valid_mask = batch.get("valid_mask", valid_mask)
            attention_mask = batch.get("sdpa_mask", attention_mask)
            strokes = batch["tokens"] if self._uses_token_input else batch["strokes"]

        if targets is None:
            targets = strokes
        if valid_mask is None:
            valid_mask = torch.ones(
                strokes.shape[:2],
                dtype=torch.bool,
                device=strokes.device,
            )

        if self.config.decoder_memory_source == "encoder" and attention_mask is None:
            attention_mask = self._token_decoder_attention_mask(valid_mask)
        encoded = self.encode(strokes, attention_mask=attention_mask)
        embedding = self.pool(encoded, valid_mask=valid_mask)
        decoder_targets = targets
        decoder_valid_mask = valid_mask
        loss_targets = None
        loss_valid_mask = None
        if self._uses_token_input and self.config.decoder_autoregressive:
            if targets.shape[1] < 2:
                raise ValueError("autoregressive token reconstruction requires sequence length >= 2")
            decoder_sequence = targets if decoder_inputs is None else decoder_inputs
            if decoder_sequence.shape != targets.shape:
                raise ValueError("decoder_inputs must have the same shape as targets")
            decoder_targets = decoder_sequence[:, :-1]
            decoder_valid_mask = (
                decoder_targets != self.config.token_dictionary.pad_token_id
            )
            loss_targets = targets[:, 1:]
            loss_valid_mask = loss_targets != self.config.token_dictionary.pad_token_id

        autoregressive_token_decode = (
            self._uses_token_input and self.config.decoder_autoregressive
        )
        decoded = self.decode(
            embedding,
            decoder_targets,
            memory_length=strokes.shape[1],
            self_attention_mask=(
                self._token_decoder_attention_mask(decoder_valid_mask)
                if autoregressive_token_decode
                else attention_mask
            ),
            valid_mask=valid_mask,
            self_attention_is_causal=(
                self._uses_token_input and self.config.decoder_autoregressive
            ),
            encoder_memory=encoded,
        )

        reconstruction = (
            self.reconstruction_head(decoded)
            if self.reconstruction_head is not None
            else None
        )
        class_logits = (
            self.classification_head(embedding)
            if self.classification_head is not None
            else None
        )

        return SketchformerOutput(
            embedding=embedding,
            encoded=encoded,
            decoded=decoded,
            reconstruction=reconstruction,
            class_logits=class_logits,
            loss_targets=loss_targets,
            loss_valid_mask=loss_valid_mask,
        )

    def encode(
        self,
        strokes: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded_input = self.input_embedding(strokes)
        return self.encoder(encoded_input, attention_mask=attention_mask)

    def decode(
        self,
        embedding: torch.Tensor,
        targets: torch.Tensor,
        *,
        memory_length: int | None = None,
        self_attention_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        self_attention_is_causal: bool = False,
        encoder_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._uses_token_input:
            if self.config.decoder_autoregressive:
                target_input = self.target_embedding(targets)
            else:
                target_input = self.target_embedding(
                    targets.shape[0],
                    targets.shape[1],
                    device=embedding.device,
                )
        else:
            target_input = self.target_embedding(targets)
        if self.config.decoder_memory_source == "encoder":
            if encoder_memory is None:
                raise ValueError(
                    "decoder.memory_source=encoder requires encoder_memory"
                )
            memory = encoder_memory
            cross_attention_mask = self._cross_attention_mask(
                valid_mask,
                target_input.shape[1],
                memory.shape[1],
            )
        else:
            if (
                self.config.latent_expander_mode == "tf_dense"
                and self.config.latent_expander_base_length is None
            ):
                resolved_memory_length = self.config.max_seq_len
            else:
                resolved_memory_length = int(memory_length or target_input.shape[1])
            if self.latent_expander is None:
                raise RuntimeError("latent expander is not available")
            memory = self.latent_expander(embedding, resolved_memory_length)
            cross_attention_mask = (
                None
                if self.config.blind_decoder_mask
                else self._cross_attention_mask(
                    valid_mask,
                    target_input.shape[1],
                    memory.shape[1],
                )
            )
        return self.decoder(
            target_input,
            memory,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            self_attention_is_causal=self_attention_is_causal,
        )

    @torch.no_grad()
    def generate(
        self,
        strokes: torch.Tensor | dict[str, Any],
        *,
        valid_mask: torch.Tensor | None = None,
        max_length: int | None = None,
        use_cache: bool = True,
    ) -> GenerationOutput:
        """Reconstruct token sketches without feeding previous target tokens."""

        if not self._uses_token_input or not self.config.decoder_autoregressive:
            raise ValueError("generate requires autoregressive tok-dict configuration")
        if isinstance(strokes, dict):
            batch = strokes
            valid_mask = batch.get("valid_mask", valid_mask)
            tokens = batch["tokens"]
            attention_mask = batch.get("sdpa_mask")
        else:
            tokens = strokes
            attention_mask = None
        if valid_mask is None:
            valid_mask = tokens != self.config.token_dictionary.pad_token_id

        if self.config.decoder_memory_source == "encoder" and attention_mask is None:
            attention_mask = self._token_decoder_attention_mask(valid_mask)
        encoded = self.encode(tokens, attention_mask=attention_mask)
        embedding = self.pool(encoded, valid_mask=valid_mask)
        source_lengths = valid_mask.sum(dim=1).to(dtype=torch.long)
        if bool((source_lengths < 2).any()):
            raise ValueError("generation inputs must contain at least two valid tokens")

        generation_limits: torch.Tensor | None = None
        if self.config.decoder_memory_source == "encoder":
            generation_limits = source_lengths.clamp(max=self.config.max_seq_len)
            if max_length is not None:
                generation_limits = generation_limits.clamp(max=int(max_length))
            generation_length = int(generation_limits.max().item())
        else:
            input_length = int(source_lengths.max().item())
            generation_length = int(max_length or input_length)
            generation_length = min(generation_length, self.config.max_seq_len)
        if generation_length < 2:
            raise ValueError("generation max_length must be at least 2")
        generated = self._generate_from_embedding(
            embedding,
            max_length=generation_length,
            use_cache=use_cache,
            encoder_memory=(
                encoded if self.config.decoder_memory_source == "encoder" else None
            ),
            memory_valid_mask=(
                valid_mask
                if self.config.decoder_memory_source == "encoder"
                else None
            ),
            generation_limits=generation_limits,
        )
        return GenerationOutput(
            tokens=generated,
            lengths=self._generated_lengths(generated),
            embedding=embedding,
        )

    def _generate_from_embedding(
        self,
        embedding: torch.Tensor,
        *,
        max_length: int,
        use_cache: bool,
        encoder_memory: torch.Tensor | None = None,
        memory_valid_mask: torch.Tensor | None = None,
        generation_limits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_config = self.config.token_dictionary
        batch_size = embedding.shape[0]
        device = embedding.device
        generated = torch.full(
            (batch_size, 1),
            token_config.sos_token_id,
            dtype=torch.long,
            device=device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if self.config.decoder_memory_source == "encoder":
            if encoder_memory is None or memory_valid_mask is None:
                raise ValueError(
                    "encoder generation requires memory and its valid mask"
                )
            memory = encoder_memory
        else:
            if self.latent_expander is None:
                raise RuntimeError("latent expander is not available")
            memory = self.latent_expander(embedding, max_length)
        if generation_limits is None:
            generation_limits = torch.full(
                (batch_size,),
                max_length,
                dtype=torch.long,
                device=device,
            )
        else:
            generation_limits = generation_limits.to(device=device, dtype=torch.long)
        caches = None

        for position in range(max_length - 1):
            if use_cache:
                decoder_input = self.target_embedding(
                    generated[:, -1:],
                    position_offset=position,
                )
                cross_attention_mask = self._cross_attention_mask(
                    memory_valid_mask,
                    1,
                    memory.shape[1],
                )
                decoded, caches = self.decoder.forward_step(
                    decoder_input,
                    memory,
                    caches,
                    cross_attention_mask,
                )
            else:
                decoder_input = self.target_embedding(generated)
                cross_attention_mask = self._cross_attention_mask(
                    memory_valid_mask,
                    decoder_input.shape[1],
                    memory.shape[1],
                )
                decoded = self.decoder(
                    decoder_input,
                    memory,
                    cross_attention_mask=cross_attention_mask,
                    self_attention_is_causal=True,
                )
                decoded = decoded[:, -1:]
            assert self.reconstruction_head is not None
            reconstruction = self.reconstruction_head(decoded)
            logits = reconstruction.token_logits[:, -1].clone()
            logits = self._apply_generation_constraints(
                logits,
                generated[:, -1],
            )
            next_token = torch.argmax(logits, dim=-1)
            within_limit = (position + 1) < generation_limits
            next_token = torch.where(
                finished | ~within_limit,
                torch.full_like(next_token, token_config.pad_token_id),
                next_token,
            )
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            reached_limit = (position + 2) >= generation_limits
            finished = (
                finished
                | (next_token == token_config.eos_token_id)
                | reached_limit
            )
            if bool(finished.all()):
                break
        return generated

    def _generated_lengths(self, tokens: torch.Tensor) -> torch.Tensor:
        eos = tokens == self.config.token_dictionary.eos_token_id
        positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        sentinel = torch.full_like(positions, tokens.shape[1])
        first_eos = torch.where(eos, positions, sentinel).min(dim=1).values + 1
        padding = tokens == self.config.token_dictionary.pad_token_id
        first_padding = torch.where(padding, positions, sentinel).min(dim=1).values
        full_length = torch.full_like(first_eos, tokens.shape[1])
        return torch.minimum(torch.minimum(first_eos, first_padding), full_length)

    def _tie_token_weights(self) -> None:
        if not isinstance(self.input_embedding, TokenEmbedding):
            raise ValueError("tied token weights require token input embeddings")
        if not isinstance(self.target_embedding, TokenEmbedding):
            raise ValueError("tied token weights require an autoregressive token decoder")
        if not isinstance(self.reconstruction_head, TokenReconstructionHead):
            raise ValueError("tied token weights require a token reconstruction head")
        shared_weight = self.input_embedding.token_embedding.weight
        self.target_embedding.token_embedding.weight = shared_weight
        self.reconstruction_head.projection.weight = shared_weight
        if self.reconstruction_head.projection.bias is not None:
            nn.init.zeros_(self.reconstruction_head.projection.bias)

    def _apply_generation_constraints(
        self,
        logits: torch.Tensor,
        previous_tokens: torch.Tensor,
    ) -> torch.Tensor:
        token_config = self.config.token_dictionary
        minimum = torch.finfo(logits.dtype).min
        if self.config.resolved_generation_grammar == "anchored_v3":
            allowed = self._anchored_v3_allowed_tokens(previous_tokens)
            return logits.masked_fill(~allowed, minimum)

        logits[:, token_config.pad_token_id] = minimum
        logits[:, token_config.sos_token_id] = minimum
        if token_config.sep_token_id is not None:
            previous_is_sep = previous_tokens == token_config.sep_token_id
            if previous_is_sep.any():
                logits[previous_is_sep, token_config.sep_token_id] = minimum
        return logits

    def _anchored_v3_allowed_tokens(
        self,
        previous_tokens: torch.Tensor,
    ) -> torch.Tensor:
        token_config = self.config.token_dictionary
        if not token_config.has_anchored_layout:
            raise RuntimeError("anchored_v3 token layout is incomplete")
        assert token_config.x_token_offset is not None
        assert token_config.y_token_offset is not None
        assert token_config.coordinate_bins is not None
        assert token_config.stroke_start_token_id is not None
        assert token_config.stroke_end_token_id is not None

        token_ids = torch.arange(
            token_config.vocab_size,
            device=previous_tokens.device,
        )
        motion = (
            (token_ids >= token_config.motion_token_offset)
            & (
                token_ids
                < token_config.motion_token_offset + token_config.codebook_size
            )
        )
        x_coordinate = (
            (token_ids >= token_config.x_token_offset)
            & (token_ids < token_config.x_token_offset + token_config.coordinate_bins)
        )
        y_coordinate = (
            (token_ids >= token_config.y_token_offset)
            & (token_ids < token_config.y_token_offset + token_config.coordinate_bins)
        )

        after_sos = previous_tokens == token_config.sos_token_id
        after_stroke_end = previous_tokens == token_config.stroke_end_token_id
        after_start = previous_tokens == token_config.stroke_start_token_id
        after_x = (
            (previous_tokens >= token_config.x_token_offset)
            & (
                previous_tokens
                < token_config.x_token_offset + token_config.coordinate_bins
            )
        )
        after_y = (
            (previous_tokens >= token_config.y_token_offset)
            & (
                previous_tokens
                < token_config.y_token_offset + token_config.coordinate_bins
            )
        )
        after_motion = (
            (previous_tokens >= token_config.motion_token_offset)
            & (
                previous_tokens
                < token_config.motion_token_offset + token_config.codebook_size
            )
        )

        allowed = torch.zeros(
            (previous_tokens.shape[0], token_config.vocab_size),
            dtype=torch.bool,
            device=previous_tokens.device,
        )
        allowed[:, token_config.stroke_start_token_id] |= after_sos | after_stroke_end
        # EOS remains observable immediately after SOS. This deliberately
        # exposes collapsed empty generations as premature-EOS failures rather
        # than hiding them behind the grammar mask.
        allowed[:, token_config.eos_token_id] |= after_sos | after_stroke_end
        allowed |= after_start[:, None] & x_coordinate[None, :]
        allowed |= after_x[:, None] & y_coordinate[None, :]
        allowed |= after_y[:, None] & motion[None, :]
        allowed |= after_motion[:, None] & motion[None, :]
        allowed[:, token_config.stroke_end_token_id] |= after_motion

        known_state = (
            after_sos | after_stroke_end | after_start | after_x | after_y | after_motion
        )
        allowed[:, token_config.eos_token_id] |= ~known_state
        return allowed

    @staticmethod
    def _token_decoder_attention_mask(valid_mask: torch.Tensor) -> torch.Tensor:
        """Combine causal and key-padding visibility into one SDPA mask."""

        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.to(dtype=torch.bool)
        if valid_mask.ndim != 2:
            raise ValueError("valid_mask must have shape (batch, sequence)")

        sequence_length = valid_mask.shape[1]
        causal_visibility = torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=valid_mask.device,
        ).tril()
        padding_visibility = valid_mask[:, None, None, :]
        return (padding_visibility & causal_visibility).contiguous()

    @staticmethod
    def _cross_attention_mask(
        valid_mask: torch.Tensor | None,
        target_length: int,
        source_length: int,
    ) -> torch.Tensor | None:
        if valid_mask is None:
            return None
        batch_size = valid_mask.shape[0]
        source_mask = valid_mask
        if source_mask.shape[1] < source_length:
            pad = torch.zeros(
                (batch_size, source_length - source_mask.shape[1]),
                dtype=torch.bool,
                device=source_mask.device,
            )
            source_mask = torch.cat([source_mask, pad], dim=1)
        elif source_mask.shape[1] > source_length:
            source_mask = source_mask[:, :source_length]
        return source_mask.unsqueeze(1).expand(batch_size, target_length, source_length).unsqueeze(1)
