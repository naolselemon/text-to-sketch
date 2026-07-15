"""Length curriculum contracts for long-sequence Sketchformer fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    max_length: int
    epochs: int
    learning_rate: float
    trainable: str = "all"
    learning_rates: dict[str, float] = field(default_factory=dict)


def parse_curriculum(
    trainer_config: Mapping[str, Any],
    *,
    default_max_length: int,
) -> list[CurriculumStage]:
    """Parse configured stages or return one conventional all-model stage."""

    curriculum = trainer_config.get("curriculum", {})
    if not bool(curriculum.get("enabled", False)):
        training = trainer_config.get("training", {})
        return [
            CurriculumStage(
                name="full",
                max_length=int(default_max_length),
                epochs=int(training.get("max_epochs", 1)),
                learning_rate=float("nan"),
                trainable="all",
            )
        ]

    raw_stages = curriculum.get("stages", [])
    if not raw_stages:
        raise ValueError("trainer.curriculum.enabled requires at least one stage")
    stages = [
        CurriculumStage(
            name=str(stage.get("name", f"stage-{index + 1}")),
            max_length=int(stage["max_length"]),
            epochs=int(stage["epochs"]),
            learning_rate=float(stage["learning_rate"]),
            trainable=str(stage.get("trainable", "all")),
            learning_rates={
                str(name): float(value)
                for name, value in stage.get("learning_rates", {}).items()
            },
        )
        for index, stage in enumerate(raw_stages)
    ]
    previous_length = 0
    for stage in stages:
        if stage.max_length <= previous_length:
            raise ValueError("curriculum max_length values must increase strictly")
        if stage.max_length > default_max_length:
            raise ValueError("curriculum stage exceeds model/data max_length")
        if stage.epochs <= 0 or stage.learning_rate <= 0:
            raise ValueError("curriculum epochs and learning_rate must be positive")
        if stage.trainable not in {
            "expander",
            "new_modules",
            "decoder",
            "decoder_top_encoder",
            "all",
        }:
            raise ValueError(
                "curriculum trainable must be expander, new_modules, decoder, "
                "decoder_top_encoder, or all"
            )
        unknown_rates = set(stage.learning_rates) - {"new_modules", "decoder", "encoder"}
        if unknown_rates:
            raise ValueError(
                "curriculum learning_rates contains unsupported groups: "
                + ", ".join(sorted(unknown_rates))
            )
        if any(value <= 0 for value in stage.learning_rates.values()):
            raise ValueError("curriculum group learning rates must be positive")
        previous_length = stage.max_length
    if stages[-1].max_length != default_max_length:
        raise ValueError("final curriculum stage must reach configured max_length")
    return stages


def set_trainable_scope(model: torch.nn.Module, scope: str) -> int:
    """Freeze model parameters outside the configured curriculum scope."""

    if scope not in {
        "expander",
        "new_modules",
        "decoder",
        "decoder_top_encoder",
        "all",
    }:
        raise ValueError(
            "scope must be expander, new_modules, decoder, decoder_top_encoder, or all"
        )
    for parameter in model.parameters():
        parameter.requires_grad = scope == "all"
    if scope == "expander":
        _set_module_trainable(model.latent_expander)
    elif scope == "new_modules":
        for module_name in (
            "input_embedding",
            "target_embedding",
            "decoder",
            "reconstruction_head",
        ):
            module = getattr(model, module_name, None)
            if module is not None:
                _set_module_trainable(module)
    elif scope == "decoder":
        for module_name in (
            "target_embedding",
            "latent_expander",
            "decoder",
            "reconstruction_head",
        ):
            module = getattr(model, module_name, None)
            if module is not None:
                _set_module_trainable(module)
    elif scope == "decoder_top_encoder":
        for module_name in ("input_embedding", "target_embedding", "decoder", "reconstruction_head"):
            module = getattr(model, module_name, None)
            if module is not None:
                _set_module_trainable(module)
        encoder = getattr(model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None or len(layers) < 2:
            raise ValueError("decoder_top_encoder requires at least two encoder layers")
        for layer in layers[-2:]:
            _set_module_trainable(layer)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count == 0:
        raise ValueError(f"curriculum scope {scope!r} selected no trainable parameters")
    return count


def build_stage_parameter_groups(
    model: torch.nn.Module,
    stage: CurriculumStage,
) -> list[dict[str, Any]]:
    """Return deduplicated optimizer groups for a curriculum stage.

    Shared V3 embedding/head parameters appear in exactly one group. Legacy
    stages without ``learning_rates`` retain their single configured rate.
    """

    set_trainable_scope(model, stage.trainable)
    rates = {
        "new_modules": stage.learning_rates.get("new_modules", stage.learning_rate),
        "decoder": stage.learning_rates.get("decoder", stage.learning_rate),
        "encoder": stage.learning_rates.get("encoder", stage.learning_rate),
    }
    assigned: set[int] = set()
    groups: list[dict[str, Any]] = []

    def add_group(name: str, modules: list[torch.nn.Module | None], lr: float) -> None:
        parameters: list[torch.nn.Parameter] = []
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                identity = id(parameter)
                if parameter.requires_grad and identity not in assigned:
                    assigned.add(identity)
                    parameters.append(parameter)
        if parameters:
            groups.append({"params": parameters, "lr": float(lr), "name": name})

    add_group(
        "new_modules",
        [
            getattr(model, "input_embedding", None),
            getattr(model, "target_embedding", None),
            getattr(model, "reconstruction_head", None),
        ],
        rates["new_modules"],
    )
    add_group("decoder", [getattr(model, "decoder", None)], rates["decoder"])
    add_group(
        "encoder",
        [getattr(model, "encoder", None), getattr(model, "pool", None)],
        rates["encoder"],
    )
    add_group(
        "other",
        [getattr(model, "latent_expander", None), getattr(model, "classification_head", None)],
        rates["encoder"],
    )

    remaining = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in assigned
    ]
    if remaining:
        groups.append({"params": remaining, "lr": rates["encoder"], "name": "other"})
    if not groups:
        raise ValueError(f"curriculum stage {stage.name!r} selected no optimizer parameters")
    return groups


def resume_epoch_for_stage(
    stages: list[CurriculumStage],
    stage_index: int,
    completed_epochs: int,
) -> int:
    """Return the first local epoch to run for a resumed curriculum stage."""

    if stage_index < 0 or stage_index >= len(stages):
        raise IndexError("stage_index is outside the curriculum")
    before = sum(stage.epochs for stage in stages[:stage_index])
    return min(stages[stage_index].epochs, max(0, int(completed_epochs) - before))


def _set_module_trainable(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True
