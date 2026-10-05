"""Initialize anchored-V3 transformer blocks from a converted checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from builders import build_model
from scripts.sketchformer.config import compose_training_config


ALLOWED_PREFIXES = ("encoder.layers.", "decoder.layers.")


@dataclass(frozen=True)
class InitializationReport:
    loaded: tuple[str, ...]
    reinitialized: tuple[str, ...]
    skipped_shape: tuple[str, ...]
    ignored_source: tuple[str, ...]


def _state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("pretrained checkpoint must be a mapping")
    state = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state, Mapping):
        raise TypeError("pretrained checkpoint model state must be a mapping")
    return state


def load_source_state(path: str | Path) -> Mapping[str, torch.Tensor]:
    """Load a converted ``.safetensors`` file or a PyTorch checkpoint."""

    source = Path(path)
    if source.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors initialization weights requires safetensors"
            ) from exc
        return load_file(str(source), device="cpu")
    return _state_dict(torch.load(source, map_location="cpu"))


def initialize_transformer_blocks(
    model: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
    *,
    allowed_prefixes: tuple[str, ...] = ALLOWED_PREFIXES,
) -> InitializationReport:
    """Load exact-shape transformer tensors and classify every other tensor."""

    target = model.state_dict()
    selected: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    ignored_source: list[str] = []
    for name, value in source_state.items():
        if not any(name.startswith(prefix) for prefix in allowed_prefixes):
            ignored_source.append(name)
            continue
        if name not in target or tuple(target[name].shape) != tuple(value.shape):
            skipped_shape.append(name)
            continue
        selected[name] = value

    incompatible = model.load_state_dict(selected, strict=False)
    unexpected = set(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"initializer produced unexpected model keys: {sorted(unexpected)}")
    reinitialized = tuple(sorted(set(target) - set(selected)))
    return InitializationReport(
        loaded=tuple(sorted(selected)),
        reinitialized=reinitialized,
        skipped_shape=tuple(sorted(skipped_shape)),
        ignored_source=tuple(sorted(ignored_source)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="anime_anchored_v3_direct")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def _write_initialization_atomic(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(dict(payload), temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_report_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    project_root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )
    config = compose_training_config(
        project_root / "configs/train.yaml",
        experiment=args.experiment,
    )
    model = build_model(config["model"])
    report = initialize_transformer_blocks(
        model,
        load_source_state(project_root / args.source),
    )

    output = project_root / args.output
    _write_initialization_atomic(
        output,
        {"model": model.state_dict(), "initialization": asdict(report)},
    )
    report_path = project_root / args.report
    _write_report_atomic(report_path, asdict(report))
    print(
        f"initialized={output} loaded={len(report.loaded)} "
        f"reinitialized={len(report.reinitialized)} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
