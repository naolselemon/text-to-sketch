"""Configuration objects for the native PyTorch Sketchformer model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _get(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = config.get(key, default)
    return default if value is None else value


@dataclass(frozen=True)
class AttentionConfig:
    implementation: str = "sdpa"
    preferred_backend: str = "flash"
    allow_flash: bool = True
    allow_memory_efficient: bool = True
    allow_math_fallback: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "AttentionConfig":
        config = config or {}
        return cls(
            implementation=str(_get(config, "implementation", "sdpa")),
            preferred_backend=str(_get(config, "preferred_backend", "flash")),
            allow_flash=bool(_get(config, "allow_flash", True)),
            allow_memory_efficient=bool(_get(config, "allow_memory_efficient", True)),
            allow_math_fallback=bool(_get(config, "allow_math_fallback", True)),
        )


@dataclass(frozen=True)
class PositionalEncodingConfig:
    type: str = "learned"
    max_length: int = 2048
    checkpoint_extension: str = "interpolate_then_random_init"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | str | None) -> "PositionalEncodingConfig":
        if isinstance(config, str):
            return cls(type=config)
        config = config or {}
        return cls(
            type=str(_get(config, "type", "learned")),
            max_length=int(_get(config, "max_length", 2048)),
            checkpoint_extension=str(
                _get(config, "checkpoint_extension", "interpolate_then_random_init")
            ),
        )


@dataclass(frozen=True)
class ReconstructionHeadConfig:
    enabled: bool = True
    target: str = "continuous"
    xy_distribution: str = "gaussian_mixture"
    num_mixtures: int = 20
    predict_pen_state: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "ReconstructionHeadConfig":
        config = config or {}
        return cls(
            enabled=bool(_get(config, "enabled", True)),
            target=str(_get(config, "target", _get(config, "type", "continuous"))),
            xy_distribution=str(_get(config, "xy_distribution", "gaussian_mixture")),
            num_mixtures=int(_get(config, "num_mixtures", 20)),
            predict_pen_state=bool(_get(config, "predict_pen_state", True)),
        )


@dataclass(frozen=True)
class ClassificationHeadConfig:
    enabled: bool = False
    num_classes: int = 345
    dropout: float = 0.1

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "ClassificationHeadConfig":
        config = config or {}
        return cls(
            enabled=bool(_get(config, "enabled", False)),
            num_classes=int(_get(config, "num_classes", 345)),
            dropout=float(_get(config, "dropout", 0.1)),
        )


@dataclass(frozen=True)
class TokenDictionaryConfig:
    codebook_size: int = 1000
    motion_token_offset: int = 1
    pad_token_id: int = 0
    sep_token_id: int | None = 1001
    sos_token_id: int = 1002
    eos_token_id: int = 1003
    vocab_size: int = 1004
    x_token_offset: int | None = None
    y_token_offset: int | None = None
    coordinate_bins: int | None = None
    stroke_start_token_id: int | None = None
    stroke_end_token_id: int | None = None
    mask_token_id: int | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "TokenDictionaryConfig":
        config = config or {}
        codebook_size = int(_get(config, "codebook_size", _get(config, "K", 1000)))
        motion_token_offset = int(_get(config, "motion_token_offset", 1))
        pad_token_id = int(_get(config, "pad_token_id", 0))
        anchored_layout = any(
            key in config
            for key in (
                "x_token_offset",
                "absolute_x_token_offset",
                "y_token_offset",
                "absolute_y_token_offset",
                "stroke_start_token_id",
                "stroke_end_token_id",
            )
        )
        default_sep = None if anchored_layout else motion_token_offset + codebook_size
        sep_value = config.get("sep_token_id", default_sep)
        sep_token_id = int(sep_value) if sep_value is not None else None
        default_sos = (
            int(config["stroke_end_token_id"]) + 1
            if sep_token_id is None and config.get("stroke_end_token_id") is not None
            else int(sep_token_id or 0) + 1
        )
        sos_token_id = int(_get(config, "sos_token_id", default_sos))
        eos_token_id = int(_get(config, "eos_token_id", sos_token_id + 1))
        x_token_offset = config.get(
            "x_token_offset",
            config.get("absolute_x_token_offset"),
        )
        y_token_offset = config.get(
            "y_token_offset",
            config.get("absolute_y_token_offset"),
        )
        coordinate_bins = config.get(
            "coordinate_bins",
            config.get("num_coordinate_bins"),
        )
        stroke_start_token_id = config.get("stroke_start_token_id")
        stroke_end_token_id = config.get("stroke_end_token_id")
        mask_token_id = config.get("mask_token_id")
        optional_ids = [
            sep_token_id,
            (
                int(x_token_offset) + int(coordinate_bins) - 1
                if x_token_offset is not None and coordinate_bins is not None
                else None
            ),
            (
                int(y_token_offset) + int(coordinate_bins) - 1
                if y_token_offset is not None and coordinate_bins is not None
                else None
            ),
            int(stroke_start_token_id) if stroke_start_token_id is not None else None,
            int(stroke_end_token_id) if stroke_end_token_id is not None else None,
            int(mask_token_id) if mask_token_id is not None else None,
        ]
        vocab_size = int(
            _get(
                config,
                "vocab_size",
                max(
                    motion_token_offset + codebook_size - 1,
                    sos_token_id,
                    eos_token_id,
                    pad_token_id,
                    *(token_id for token_id in optional_ids if token_id is not None),
                )
                + 1,
            )
        )
        return cls(
            codebook_size=codebook_size,
            motion_token_offset=motion_token_offset,
            pad_token_id=pad_token_id,
            sep_token_id=sep_token_id,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            x_token_offset=(
                int(x_token_offset) if x_token_offset is not None else None
            ),
            y_token_offset=(
                int(y_token_offset) if y_token_offset is not None else None
            ),
            coordinate_bins=(
                int(coordinate_bins) if coordinate_bins is not None else None
            ),
            stroke_start_token_id=(
                int(stroke_start_token_id)
                if stroke_start_token_id is not None
                else None
            ),
            stroke_end_token_id=(
                int(stroke_end_token_id)
                if stroke_end_token_id is not None
                else None
            ),
            mask_token_id=(
                int(mask_token_id) if mask_token_id is not None else None
            ),
        )

    @property
    def has_anchored_layout(self) -> bool:
        return all(
            value is not None
            for value in (
                self.x_token_offset,
                self.y_token_offset,
                self.coordinate_bins,
                self.stroke_start_token_id,
                self.stroke_end_token_id,
                self.mask_token_id,
            )
        )


@dataclass(frozen=True)
class SketchformerConfig:
    name: str = "sketchformer_continuous"
    input_mode: str = "stroke3"
    stroke_dim: int = 3
    pen_classes: int = 3
    max_seq_len: int = 2048
    token_dictionary: TokenDictionaryConfig = field(
        default_factory=TokenDictionaryConfig
    )
    d_model: int = 128
    latent_dim: int = 256
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = True
    use_final_norm: bool = True
    layer_norm_eps: float = 1e-5
    gradient_checkpointing: bool = True
    pooling_mode: str = "projected"
    pool_hidden_dim: int = 256
    latent_expander_mode: str = "projected_position"
    latent_expander_base_length: int | None = None
    pen_embedding_dim: int = 32
    combine_method: str = "add"
    positional_encoding: PositionalEncodingConfig = field(
        default_factory=PositionalEncodingConfig
    )
    encoder_attention: AttentionConfig = field(default_factory=AttentionConfig)
    decoder_attention: AttentionConfig = field(default_factory=AttentionConfig)
    decoder_autoregressive: bool = False
    blind_decoder_mask: bool = True
    decoder_memory_source: str = "latent_expander"
    tie_token_weights: bool = False
    generation_grammar: str = "auto"
    reconstruction: ReconstructionHeadConfig = field(
        default_factory=ReconstructionHeadConfig
    )
    classification: ClassificationHeadConfig = field(
        default_factory=ClassificationHeadConfig
    )
    compile_enabled: bool = False
    compile_mode: str = "default"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SketchformerConfig":
        input_cfg = config.get("input", {})
        architecture = config.get("architecture", {})
        embedding = config.get("embedding", {})
        encoder = config.get("encoder", {})
        decoder = config.get("decoder", {})
        heads = config.get("heads", {})
        compile_cfg = config.get("compile", {})

        return cls(
            name=str(_get(config, "name", "sketchformer_continuous")),
            input_mode=str(_get(input_cfg, "type", _get(input_cfg, "mode", "stroke3"))),
            stroke_dim=int(_get(input_cfg, "stroke_dim", 3)),
            pen_classes=int(_get(input_cfg, "pen_classes", 3)),
            max_seq_len=int(_get(input_cfg, "max_seq_len", 2048)),
            token_dictionary=TokenDictionaryConfig.from_mapping(
                input_cfg.get("token_dictionary", {})
            ),
            d_model=int(_get(architecture, "d_model", 128)),
            latent_dim=int(_get(architecture, "latent_dim", 256)),
            num_encoder_layers=int(_get(architecture, "num_encoder_layers", 4)),
            num_decoder_layers=int(_get(architecture, "num_decoder_layers", 4)),
            num_heads=int(_get(architecture, "num_heads", 8)),
            dim_feedforward=int(_get(architecture, "dim_feedforward", 512)),
            dropout=float(_get(architecture, "dropout", 0.1)),
            activation=str(_get(architecture, "activation", "gelu")),
            norm_first=bool(_get(architecture, "norm_first", True)),
            use_final_norm=bool(_get(architecture, "use_final_norm", True)),
            layer_norm_eps=float(_get(architecture, "layer_norm_eps", 1e-5)),
            gradient_checkpointing=bool(
                _get(architecture, "gradient_checkpointing", True)
            ),
            pooling_mode=str(_get(architecture, "pooling_mode", "projected")),
            pool_hidden_dim=int(
                _get(architecture, "pool_hidden_dim", _get(architecture, "latent_dim", 256))
            ),
            latent_expander_mode=str(
                _get(architecture, "latent_expander_mode", "projected_position")
            ),
            latent_expander_base_length=(
                int(architecture["latent_expander_base_length"])
                if architecture.get("latent_expander_base_length") is not None
                else None
            ),
            pen_embedding_dim=int(_get(embedding, "pen_embedding_dim", 32)),
            combine_method=str(_get(embedding, "combine_method", "add")),
            positional_encoding=PositionalEncodingConfig.from_mapping(
                embedding.get("positional_encoding", {})
            ),
            encoder_attention=AttentionConfig.from_mapping(encoder.get("attention", {})),
            decoder_attention=AttentionConfig.from_mapping(decoder.get("attention", {})),
            decoder_autoregressive=bool(_get(decoder, "autoregressive", False)),
            blind_decoder_mask=bool(_get(decoder, "blind_decoder_mask", True)),
            decoder_memory_source=str(
                _get(decoder, "memory_source", "latent_expander")
            ),
            tie_token_weights=bool(_get(decoder, "tie_token_weights", False)),
            generation_grammar=str(_get(decoder, "generation_grammar", "auto")),
            reconstruction=ReconstructionHeadConfig.from_mapping(
                heads.get("reconstruction", {})
            ),
            classification=ClassificationHeadConfig.from_mapping(
                heads.get("classification", {})
            ),
            compile_enabled=bool(_get(compile_cfg, "enabled", False)),
            compile_mode=str(_get(compile_cfg, "mode", "default")),
        )

    def validate(self) -> None:
        if self.input_mode == "stroke3":
            if self.stroke_dim != 3:
                raise ValueError("stroke3 input requires input.stroke_dim=3")
            if self.reconstruction.target not in {"continuous", "stroke3"}:
                raise ValueError("stroke3 input requires a continuous reconstruction target")
        elif self.input_mode in {"tok_dict", "token", "tokens"}:
            special_ids = tuple(
                token_id
                for token_id in (
                    self.token_dictionary.pad_token_id,
                    self.token_dictionary.sep_token_id,
                    self.token_dictionary.sos_token_id,
                    self.token_dictionary.eos_token_id,
                    self.token_dictionary.stroke_start_token_id,
                    self.token_dictionary.stroke_end_token_id,
                    self.token_dictionary.mask_token_id,
                )
                if token_id is not None
            )
            if len(set(special_ids)) != len(special_ids):
                raise ValueError("tok_dict special token IDs must be distinct")
            motion_start = self.token_dictionary.motion_token_offset
            motion_end = motion_start + self.token_dictionary.codebook_size
            if motion_start < 0:
                raise ValueError("tok_dict motion_token_offset must be non-negative")
            for token_id in special_ids:
                if motion_start <= token_id < motion_end:
                    raise ValueError("tok_dict special token IDs must not overlap motion tokens")
            max_special = max(special_ids)
            max_motion = self.token_dictionary.motion_token_offset + self.token_dictionary.codebook_size - 1
            if self.token_dictionary.vocab_size <= max_special:
                raise ValueError("token dictionary vocab_size must include special tokens")
            if self.token_dictionary.vocab_size <= max_motion:
                raise ValueError("token dictionary vocab_size must include motion tokens")
            anchored_fields = (
                self.token_dictionary.x_token_offset,
                self.token_dictionary.y_token_offset,
                self.token_dictionary.coordinate_bins,
                self.token_dictionary.stroke_start_token_id,
                self.token_dictionary.stroke_end_token_id,
                self.token_dictionary.mask_token_id,
            )
            if any(value is not None for value in anchored_fields) and not all(
                value is not None for value in anchored_fields
            ):
                raise ValueError("anchored token dictionary layout is incomplete")
            if self.token_dictionary.has_anchored_layout:
                self._validate_anchored_token_layout()
            elif self.generation_grammar == "anchored_v3":
                raise ValueError(
                    "anchored_v3 generation requires coordinate ranges and stroke tokens"
                )
            if self.reconstruction.target not in {"tok_dict", "token", "tokens"}:
                raise ValueError("tok_dict input requires a token reconstruction target")
        else:
            raise ValueError("input.type must be one of: stroke3, tok_dict")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.combine_method != "add":
            raise ValueError("Only embedding.combine_method=add is currently supported")
        if self.positional_encoding.type not in {"learned", "sinusoidal"}:
            raise ValueError("positional encoding must be one of: learned, sinusoidal")
        if self.max_seq_len > self.positional_encoding.max_length:
            raise ValueError("max_seq_len exceeds positional encoding max_length")
        if self.pooling_mode not in {"projected", "tf_self_attn_v1"}:
            raise ValueError("pooling_mode must be one of: projected, tf_self_attn_v1")
        if self.latent_expander_mode not in {"projected_position", "tf_dense"}:
            raise ValueError("latent_expander_mode must be one of: projected_position, tf_dense")
        if self.decoder_memory_source not in {"latent_expander", "encoder"}:
            raise ValueError(
                "decoder.memory_source must be one of: latent_expander, encoder"
            )
        if self.generation_grammar not in {"auto", "legacy", "anchored_v3"}:
            raise ValueError(
                "decoder.generation_grammar must be one of: auto, legacy, anchored_v3"
            )
        if self.tie_token_weights and self.input_mode not in {
            "tok_dict",
            "token",
            "tokens",
        }:
            raise ValueError("decoder.tie_token_weights requires token input")
        if self.tie_token_weights and not self.decoder_autoregressive:
            raise ValueError(
                "decoder.tie_token_weights requires decoder.autoregressive=true"
            )
        if self.latent_expander_base_length is not None:
            if self.latent_expander_base_length <= 0:
                raise ValueError("latent_expander_base_length must be positive")
            if self.latent_expander_base_length > self.max_seq_len:
                raise ValueError("latent_expander_base_length exceeds max_seq_len")

    @property
    def pool_output_dim(self) -> int:
        return self.d_model if self.pooling_mode == "tf_self_attn_v1" else self.latent_dim

    @property
    def resolved_generation_grammar(self) -> str:
        if self.generation_grammar != "auto":
            return self.generation_grammar
        return (
            "anchored_v3"
            if self.token_dictionary.has_anchored_layout
            else "legacy"
        )

    def _validate_anchored_token_layout(self) -> None:
        tokens = self.token_dictionary
        assert tokens.x_token_offset is not None
        assert tokens.y_token_offset is not None
        assert tokens.coordinate_bins is not None
        assert tokens.stroke_start_token_id is not None
        assert tokens.stroke_end_token_id is not None
        assert tokens.mask_token_id is not None
        if tokens.coordinate_bins <= 0:
            raise ValueError("token dictionary coordinate_bins must be positive")

        ranges = {
            "motion": set(
                range(
                    tokens.motion_token_offset,
                    tokens.motion_token_offset + tokens.codebook_size,
                )
            ),
            "x": set(
                range(
                    tokens.x_token_offset,
                    tokens.x_token_offset + tokens.coordinate_bins,
                )
            ),
            "y": set(
                range(
                    tokens.y_token_offset,
                    tokens.y_token_offset + tokens.coordinate_bins,
                )
            ),
        }
        names = tuple(ranges)
        for index, name in enumerate(names):
            for other in names[index + 1 :]:
                if ranges[name] & ranges[other]:
                    raise ValueError(
                        f"token dictionary {name} and {other} ranges overlap"
                    )

        structural_ids = {
            tokens.pad_token_id,
            tokens.sos_token_id,
            tokens.eos_token_id,
            tokens.stroke_start_token_id,
            tokens.stroke_end_token_id,
            tokens.mask_token_id,
        }
        if tokens.sep_token_id is not None:
            structural_ids.add(tokens.sep_token_id)
        expected_structural_count = 7 if tokens.sep_token_id is not None else 6
        if len(structural_ids) != expected_structural_count:
            raise ValueError("anchored token dictionary structural IDs must be distinct")
        if any(structural_ids & token_range for token_range in ranges.values()):
            raise ValueError(
                "anchored token dictionary structural IDs must not overlap token ranges"
            )
        max_token_id = max(
            max(structural_ids),
            *(max(token_range) for token_range in ranges.values()),
        )
        if tokens.vocab_size <= max_token_id:
            raise ValueError("token dictionary vocab_size does not include anchored tokens")
