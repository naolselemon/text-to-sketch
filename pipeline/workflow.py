"""Core workflows for centerline and legacy contour preprocessing."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import random
import stat
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from pipeline.kinematics import generate_kinematics
from pipeline.ordering import (
    order_continuity_greedy,
    order_continuity_topology,
    order_directional_bias,
    order_greedy_nearest_neighbor,
    order_outer_to_inner,
    order_tsp,
)
from pipeline.stroke5 import stroke5_to_canvas_strokes, strokes_to_stroke5, to_stroke5
from pipeline.vectorization import (
    DEFAULT_MAX_GEOMETRY_ERROR,
    DEFAULT_RDP_EPSILON,
    centerline_metrics,
    rasterize_strokes,
    read_grayscale_image,
    source_centerline,
    vectorize_image,
    vectorize_image_with_stats,
)
from prep_data.sketch_token.create_token_dict import build_codebook, save_codebook
from utils.io import save_stroke5, save_token_sequence
from utils.tokenizer import (
    ErrorFeedbackQuantizer,
    decode_tokens,
    encode_stroke5,
    quantization_metrics,
)

_NORMALIZATION_EXTENTS = (2.5, 2.0, 1.5, 1.0)
_TARGET_TOKEN_GEOMETRY_F1 = 0.95

_ORDER_FN_MAP = {
    "continuity": order_continuity_greedy,
    "directional": order_directional_bias,
    "greedy": order_greedy_nearest_neighbor,
    "tsp": order_tsp,
    "continuity-topology": order_continuity_topology,
    "outer-to-inner": order_outer_to_inner,
}


@dataclass(frozen=True)
class _CenterlineWorkerConfig:
    """Pickle-safe configuration shared by centerline worker processes."""

    sketches_dir: str
    stroke5_dir: str
    token_dir: str
    codebook_path: str
    codebook_sha256: str
    codebook_size: int
    ordering: str
    rdp_epsilon: float
    threshold_profile: str
    max_token_length: int
    max_geometry_error: float
    extractor_name: str
    limit_library_threads: bool


@dataclass
class _CenterlineWorkerState:
    """Large process-local objects that must be initialized only once."""

    config: _CenterlineWorkerConfig
    codebook: np.ndarray
    quantizer: ErrorFeedbackQuantizer
    order_fn: Any
    thread_limiter: Any = None


_CENTERLINE_WORKER_STATE: _CenterlineWorkerState | None = None


def run_pipeline(
    sketches_dir: Path,
    stroke5_dir: Path,
    sketch_token_dir: Path,
    n_sketches: int,
    ordering: str,
    rdp_epsilon: float = DEFAULT_RDP_EPSILON,
    codebook_K: int = 1000,
    seed: int = 42,
    *,
    vectorizer: str = "centerline",
    threshold_profile: str = "hysteresis",
    max_token_length: int | None = None,
    max_geometry_error: float = DEFAULT_MAX_GEOMETRY_ERROR,
    token_dict_dir: Path | None = None,
    manifest_path: Path | None = None,
    fail_on_overlength: bool = False,
    extractor_name: str | None = None,
    num_workers: int = 1,
) -> None:
    """Run the default centerline preprocessing path or legacy contour path."""

    all_sketches = sorted(Path(sketches_dir).rglob("*.png"))
    if not all_sketches:
        raise FileNotFoundError(
            f"No sketches found in {sketches_dir}. Run the filter-sketches command first."
        )
    if ordering not in _ORDER_FN_MAP:
        valid = ", ".join(sorted(_ORDER_FN_MAP))
        raise ValueError(f"Unknown ordering {ordering!r}. Expected one of: {valid}.")
    if vectorizer not in {"contour", "centerline"}:
        raise ValueError("vectorizer must be one of: contour, centerline")
    if int(n_sketches) < 1:
        raise ValueError("n_sketches must be at least 1")
    if int(num_workers) < 1:
        raise ValueError("num_workers must be at least 1")
    if vectorizer != "centerline" and int(num_workers) != 1:
        raise ValueError("num_workers greater than 1 is only supported for centerline")

    count = min(int(n_sketches), len(all_sketches))
    samples = random.Random(seed).sample(all_sketches, count)
    if vectorizer == "centerline":
        _run_centerline_pipeline(
            samples=samples,
            sketches_dir=Path(sketches_dir),
            stroke5_dir=Path(stroke5_dir),
            token_dict_dir=Path(token_dict_dir or sketch_token_dir),
            ordering=ordering,
            rdp_epsilon=float(rdp_epsilon),
            threshold_profile=threshold_profile,
            max_token_length=int(max_token_length or 4096),
            max_geometry_error=float(max_geometry_error),
            manifest_path=Path(manifest_path) if manifest_path else None,
            fail_on_overlength=fail_on_overlength,
            extractor_name=extractor_name or Path(sketches_dir).name,
            num_workers=int(num_workers),
        )
        return

    _run_legacy_pipeline(
        samples=samples,
        stroke5_dir=Path(stroke5_dir),
        sketch_token_dir=Path(sketch_token_dir),
        order_fn=_ORDER_FN_MAP[ordering],
        ordering=ordering,
        rdp_epsilon=float(rdp_epsilon),
        codebook_K=int(codebook_K),
    )


def _run_centerline_pipeline(
    *,
    samples: list[Path],
    sketches_dir: Path,
    stroke5_dir: Path,
    token_dict_dir: Path,
    ordering: str,
    rdp_epsilon: float,
    threshold_profile: str,
    max_token_length: int,
    max_geometry_error: float,
    manifest_path: Path | None,
    fail_on_overlength: bool,
    extractor_name: str,
    num_workers: int,
) -> None:
    if max_token_length < 4:
        raise ValueError("max_token_length must be at least 4")
    if max_geometry_error < rdp_epsilon:
        raise ValueError("max_geometry_error must be >= rdp_epsilon")

    codebook_path = token_dict_dir / "codebook.npy"
    if not codebook_path.exists():
        raise FileNotFoundError(
            f"Released token dictionary not found: {codebook_path}. "
            "Run tts-create-sketch-token-dict first."
        )
    codebook = np.asarray(np.load(codebook_path), dtype=np.float32)
    if codebook.ndim != 2 or codebook.shape[1] != 2:
        raise ValueError(f"Expected codebook shape (K, 2), got {codebook.shape}")
    codebook_sha256 = hashlib.sha256(codebook_path.read_bytes()).hexdigest()

    token_dir = stroke5_dir.parent / "tokens"
    resolved_manifest = manifest_path or stroke5_dir.parent / "preprocessing_manifest.jsonl"
    records: list[dict[str, Any]] = []
    accepted = overlength = failed = 0
    effective_workers = min(
        int(num_workers),
        len(samples),
        _available_cpu_count(),
    )
    if effective_workers < int(num_workers):
        print(
            f"[pipeline] requested_workers={num_workers} capped_workers="
            f"{effective_workers}"
        )
    worker_config = _CenterlineWorkerConfig(
        sketches_dir=str(sketches_dir),
        stroke5_dir=str(stroke5_dir),
        token_dir=str(token_dir),
        codebook_path=str(codebook_path),
        codebook_sha256=codebook_sha256,
        codebook_size=len(codebook),
        ordering=ordering,
        rdp_epsilon=rdp_epsilon,
        threshold_profile=threshold_profile,
        max_token_length=max_token_length,
        max_geometry_error=max_geometry_error,
        extractor_name=extractor_name,
        limit_library_threads=effective_workers > 1,
    )
    del codebook

    print(
        f"[pipeline] sketches={len(samples)} vectorizer=centerline "
        f"ordering={ordering} max_tokens={max_token_length} "
        f"workers={effective_workers}"
    )
    print(f"[pipeline] token_dictionary={codebook_path}")

    started_at = time.perf_counter()
    pool: ProcessPoolExecutor | None = None
    try:
        if effective_workers == 1:
            _initialize_centerline_worker(worker_config)
            record_iterator = map(_process_centerline_sample, samples)
        else:
            pool = ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_centerline_worker,
                initargs=(worker_config,),
            )
            record_iterator = pool.map(
                _process_centerline_sample,
                samples,
                chunksize=1,
            )
        for record in tqdm(
            record_iterator,
            total=len(samples),
            desc="Centerline",
            unit="sketch",
        ):
            records.append(record)
            status = record.get("status")
            if status == "accepted":
                accepted += 1
            elif status == "rejected":
                overlength += 1
            else:
                failed += 1
                tqdm.write(
                    f"[pipeline] error {Path(str(record['source_path'])).name}: "
                    f"{record.get('rejection_reason', 'unknown error')}"
                )
    except BaseException:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
            pool = None
        raise
    finally:
        if pool is not None:
            pool.shutdown()

    _write_manifest(resolved_manifest, records)
    elapsed_seconds = max(time.perf_counter() - started_at, 1.0e-9)
    print(
        f"[pipeline] accepted={accepted} overlength={overlength} errors={failed} "
        f"manifest={resolved_manifest}"
    )
    print(
        f"[pipeline] elapsed_seconds={elapsed_seconds:.2f} "
        f"throughput={len(records) / elapsed_seconds:.2f}_sketches_per_second "
        f"workers={effective_workers}"
    )
    if fail_on_overlength and overlength:
        raise RuntimeError(
            f"{overlength} sketches exceeded {max_token_length} tokens within the "
            f"{max_geometry_error:g}px geometry limit; see {resolved_manifest}"
        )
    if accepted == 0:
        raise RuntimeError(f"No sketches were accepted; see {resolved_manifest}")


def _initialize_centerline_worker(config: _CenterlineWorkerConfig) -> None:
    """Build the expensive codebook indexes once in each worker process."""

    global _CENTERLINE_WORKER_STATE

    thread_limiter = None
    if config.limit_library_threads:
        import cv2

        cv2.setNumThreads(1)
        try:
            from threadpoolctl import threadpool_limits

            thread_limiter = threadpool_limits(limits=1)
        except ImportError:
            thread_limiter = None

    codebook_path = Path(config.codebook_path)
    worker_sha256 = hashlib.sha256(codebook_path.read_bytes()).hexdigest()
    if worker_sha256 != config.codebook_sha256:
        raise ValueError("Token dictionary changed while workers were starting")
    codebook = np.asarray(np.load(codebook_path), dtype=np.float32)
    if codebook.shape != (config.codebook_size, 2):
        raise ValueError(
            "Worker codebook shape changed after validation: "
            f"expected {(config.codebook_size, 2)}, got {codebook.shape}"
        )
    _CENTERLINE_WORKER_STATE = _CenterlineWorkerState(
        config=config,
        codebook=codebook,
        quantizer=ErrorFeedbackQuantizer(codebook),
        order_fn=_ORDER_FN_MAP[config.ordering],
        thread_limiter=thread_limiter,
    )


def _process_centerline_sample(image_path: Path) -> dict[str, Any]:
    """Process and persist one sketch using process-local worker state."""

    state = _CENTERLINE_WORKER_STATE
    if state is None:
        raise RuntimeError("Centerline worker was not initialized")

    config = state.config
    image_path = Path(image_path)
    sketches_dir = Path(config.sketches_dir)
    relative = image_path.relative_to(sketches_dir)
    base_record: dict[str, Any] = {
        "schema_version": 2,
        "sample_id": relative.with_suffix("").as_posix(),
        "source_path": str(image_path),
        "source_relative_path": relative.as_posix(),
        "extractor": config.extractor_name,
        "vectorizer": "centerline",
        "threshold_profile": config.threshold_profile,
        "ordering": config.ordering,
        "max_token_length": config.max_token_length,
        "token_dictionary_path": config.codebook_path,
        "token_dictionary_size": config.codebook_size,
        "token_dictionary_sha256": config.codebook_sha256,
        "quantizer": "error_feedback_pair_search",
    }
    try:
        result = fit_centerline_sequence(
            image_path=image_path,
            codebook=state.codebook,
            quantizer=state.quantizer,
            order_fn=state.order_fn,
            ordering=config.ordering,
            initial_epsilon=config.rdp_epsilon,
            max_epsilon=config.max_geometry_error,
            threshold_profile=config.threshold_profile,
            max_token_length=config.max_token_length,
        )
        base_record.update(result["record"])
        if not result["accepted"]:
            return base_record

        output_relative = relative.with_suffix(".npz")
        stroke_path = Path(config.stroke5_dir) / output_relative
        token_path = Path(config.token_dir) / output_relative
        save_stroke5(result["stroke5"], stroke_path)
        save_token_sequence(result["tokens"], token_path)
        base_record.update(
            {
                "status": "accepted",
                "stroke5_path": str(stroke_path),
                "tokens_path": str(token_path),
            }
        )
    except Exception as exc:
        base_record.update({"status": "error", "rejection_reason": str(exc)})
    return base_record


def fit_centerline_sequence(
    *,
    image_path: Path,
    codebook: np.ndarray,
    quantizer: ErrorFeedbackQuantizer,
    order_fn,
    ordering: str = "continuity",
    initial_epsilon: float,
    max_epsilon: float,
    threshold_profile: str,
    max_token_length: int,
) -> dict[str, Any]:
    image = read_grayscale_image(image_path)
    reference = source_centerline(image, threshold_profile=threshold_profile)
    epsilon_values = _epsilon_schedule(initial_epsilon, max_epsilon)
    best_feasible: dict[str, Any] | None = None
    best_overlength: dict[str, Any] | None = None
    use_structured = ordering in ("continuity-topology", "outer-to-inner")

    for epsilon in epsilon_values:
        strokes, stats = vectorize_image_with_stats(
            image_path,
            epsilon=epsilon,
            method="centerline",
            threshold_profile=threshold_profile,
            structured=use_structured,
        )
        if not strokes:
            raise ValueError("centerline vectorization produced no strokes")
        ordered = order_fn(strokes)
        rendered = rasterize_strokes(ordered, image.shape)
        vector_geometry = centerline_metrics(reference, rendered, tolerance_px=2.0)
        candidates = [
            _encode_geometry_candidate(
                ordered=ordered,
                image_shape=image.shape,
                reference=reference,
                codebook=codebook,
                quantizer=quantizer,
                normalization_extent=extent,
                epsilon=epsilon,
                stats=stats,
                vector_geometry=vector_geometry,
            )
            for extent in _NORMALIZATION_EXTENTS
        ]
        feasible = [item for item in candidates if len(item["tokens"]) <= max_token_length]
        if feasible:
            selected = max(
                feasible,
                key=lambda item: (
                    float(item["record"]["geometry_f1_2px"]),
                    -float(item["record"]["geometry_chamfer_px"]),
                    -len(item["tokens"]),
                ),
            )
            if (
                best_feasible is None
                or float(selected["record"]["geometry_f1_2px"])
                > float(best_feasible["record"]["geometry_f1_2px"])
            ):
                best_feasible = selected
            if float(selected["record"]["geometry_f1_2px"]) >= _TARGET_TOKEN_GEOMETRY_F1:
                selected["accepted"] = True
                return selected

        shortest = min(
            candidates,
            key=lambda item: (
                len(item["tokens"]),
                -float(item["record"]["geometry_f1_2px"]),
            ),
        )
        if best_overlength is None or len(shortest["tokens"]) < len(best_overlength["tokens"]):
            best_overlength = shortest

    if best_feasible is not None:
        best_feasible["accepted"] = True
        best_feasible["record"]["quality_warning"] = "below_target_token_geometry_f1"
        return best_feasible

    assert best_overlength is not None
    best_overlength["accepted"] = False
    best_overlength["record"].update(
        {
            "status": "rejected",
            "rejection_reason": "overlength_after_max_geometry_error",
        }
    )
    return best_overlength


def _encode_geometry_candidate(
    *,
    ordered: list[list[tuple[int, int]]],
    image_shape: tuple[int, int],
    reference: np.ndarray,
    codebook: np.ndarray,
    quantizer: ErrorFeedbackQuantizer,
    normalization_extent: float,
    epsilon: float,
    stats,
    vector_geometry,
) -> dict[str, Any]:
    stroke5, transform = strokes_to_stroke5(
        ordered,
        canvas_shape=image_shape,
        normalization_extent=normalization_extent,
    )
    tokens = quantizer.encode(stroke5)
    decoded_strokes = stroke5_to_canvas_strokes(decode_tokens(tokens, codebook), transform)
    decoded_render = rasterize_strokes(decoded_strokes, image_shape)
    token_geometry = centerline_metrics(reference, decoded_render, tolerance_px=2.0)
    quantization = quantization_metrics(stroke5, tokens, codebook)
    return {
        "stroke5": stroke5,
        "tokens": tokens,
        "record": {
            "rdp_epsilon": float(epsilon),
            "normalization_extent": float(normalization_extent),
            "token_length": int(len(tokens)),
            "raw_stroke_count": stats.raw_stroke_count,
            "raw_point_count": stats.raw_point_count,
            "pre_order_stroke_count": stats.simplified_stroke_count,
            "pre_order_point_count": stats.simplified_point_count,
            "stroke_count": len(ordered),
            "point_count": sum(len(stroke) for stroke in ordered),
            "geometry_f1_2px": token_geometry.f1,
            "geometry_chamfer_px": token_geometry.symmetric_chamfer,
            "vector_geometry_f1_2px": vector_geometry.f1,
            "vector_geometry_chamfer_px": vector_geometry.symmetric_chamfer,
            "quantization_mean_error": quantization.mean_point_error,
            "quantization_endpoint_error": quantization.endpoint_error,
            "transform": asdict(transform),
        },
    }


def _epsilon_schedule(initial: float, maximum: float) -> list[float]:
    values = [float(initial)]
    current = max(initial, 0.25)
    while current < maximum:
        current = min(maximum, current + 0.25)
        if current > values[-1]:
            values.append(float(current))
    return values


def _available_cpu_count() -> int:
    """Return CPUs available to this process, respecting Linux affinity."""

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, _replacement_mode(path))
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary_path, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def _replacement_mode(path: Path) -> int:
    """Match an existing target or the process's normal new-file mode."""

    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        current_umask = os.umask(0)
        os.umask(current_umask)
        return 0o666 & ~current_umask


def _run_legacy_pipeline(
    *,
    samples: list[Path],
    stroke5_dir: Path,
    sketch_token_dir: Path,
    order_fn,
    ordering: str,
    rdp_epsilon: float,
    codebook_K: int,
) -> None:
    """Retain the original contour/kinematics workflow for reproducibility."""

    print(
        f"[pipeline-legacy] sketches={len(samples)} ordering={ordering} "
        f"RDP={rdp_epsilon:g} K={codebook_K}"
    )
    successful: list[tuple[np.ndarray, Path]] = []
    skipped = 0
    for image_path in tqdm(samples, desc="Legacy contour", unit="sketch"):
        try:
            strokes = vectorize_image(image_path, epsilon=rdp_epsilon, method="contour")
            ordered = order_fn(strokes)
            timed = generate_kinematics(ordered)
            if not timed:
                skipped += 1
                continue
            stroke5 = to_stroke5(timed)
            successful.append((stroke5, image_path))
            save_stroke5(stroke5, stroke5_dir / f"{image_path.stem}.npz")
        except Exception as exc:
            tqdm.write(f"[pipeline-legacy] skip {image_path.name}: {exc}")
            skipped += 1

    if not successful:
        raise RuntimeError("No stroke-5 data was generated")
    arrays = [stroke5 for stroke5, _ in successful]
    drawing_points = int(sum(int((stroke5[:, 2] == 1.0).sum()) for stroke5 in arrays))
    codebook = build_codebook(arrays, K=codebook_K)
    save_codebook(
        codebook,
        sketch_token_dir,
        K=len(codebook),
        n_samples=drawing_points,
    )
    token_dir = stroke5_dir.parent / "tokens"
    for stroke5, image_path in successful:
        tokens = encode_stroke5(stroke5, codebook)
        save_token_sequence(tokens, token_dir / f"{image_path.stem}.npz")
    print(f"[pipeline-legacy] accepted={len(successful)} skipped={skipped}")
