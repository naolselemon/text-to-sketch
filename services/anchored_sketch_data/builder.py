"""End-to-end image-to-anchored-V3 dataset preparation."""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeVar

import cv2
import numpy as np
from tqdm.auto import tqdm

from pipeline.vectorization import centerline_metrics, source_centerline, vectorize_image

from .artifacts import EncodedSample, RejectedSample, sha256_file, write_dataset_atomic
from .codebook import TrainingStrokeSample, fit_training_codebook
from .contract import MAX_SEQUENCE_LENGTH
from .preprocessing import (
    CanvasTransform,
    augment_strokes,
    denormalize_from_canvas,
    geometry_f1,
    normalize_to_canvas_with_transform,
    perceptual_hash,
    preprocess_strokes,
    rasterize_strokes,
    simplify_strokes,
    stable_seed,
)
from .splitting import SplitCandidate, deterministic_group_split
from .tokenizer import AnchoredTokenizer


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_ProgressItem = TypeVar("_ProgressItem")


@dataclass(frozen=True)
class BuilderConfig:
    seed: int = 42
    epsilon_candidates: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
    calibration_size: int = 256
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    train_augmentation_copies: int = 1
    minimum_accepted_source_sketches: int = 1
    shard_size: int = 1024
    vectorizer_threshold_profile: str = "hysteresis"
    source_vector_f1_gate: float = 0.98
    roundtrip_median_f1_gate: float = 0.99
    roundtrip_p10_f1_gate: float = 0.97
    preprocessing_workers: int = 0
    show_progress: bool = True


@dataclass(frozen=True)
class _SourceGeometry:
    sample_id: str
    source_path: str
    source_sha256: str
    perceptual_hash: int
    strokes: list[list[tuple[float, float]]]
    point_count: int
    canvas_transform: CanvasTransform
    group_id: str = ""
    split: str = ""


@dataclass(frozen=True)
class _PreprocessingTask:
    path: str
    sample_id: str
    threshold_profile: str


@dataclass(frozen=True)
class EpsilonResult:
    epsilon: float
    p99_token_length: float
    source_vector_median_f1: float
    roundtrip_median_f1: float
    roundtrip_p10_f1: float
    passed: bool
    failure: str = ""


def build_dataset(
    source_dir: str | Path,
    output_root: str | Path,
    *,
    config: BuilderConfig = BuilderConfig(),
) -> Path:
    """Prepare raw line-art images and publish a validated V3 dataset."""

    build_started = time.perf_counter()
    source_root = Path(source_dir)
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No sketch images found in {source_root}")
    _validate_config(config)

    worker_count = _resolve_worker_count(config.preprocessing_workers, len(paths))
    _log(
        f"discovered {len(paths):,} images; preprocessing with "
        f"{worker_count} worker{'s' if worker_count != 1 else ''}",
        enabled=config.show_progress,
    )
    preprocessing_started = time.perf_counter()
    sources, rejected = _preprocess_sources(
        paths,
        source_root=source_root,
        threshold_profile=config.vectorizer_threshold_profile,
        workers=worker_count,
        show_progress=config.show_progress,
    )
    _log_rate(
        "preprocessing complete",
        completed=len(paths),
        started_at=preprocessing_started,
        detail=f"{len(sources):,} accepted, {len(rejected):,} rejected",
        enabled=config.show_progress,
    )
    if not sources:
        raise ValueError("No source sketches survived preprocessing")
    _enforce_minimum_source_count(
        len(sources),
        config.minimum_accepted_source_sketches,
        stage="preprocessing",
    )

    assignments = deterministic_group_split(
        [
            SplitCandidate(
                sample_id=sample.sample_id,
                source_sha256=sample.source_sha256,
                perceptual_hash=sample.perceptual_hash,
                point_count=sample.point_count,
            )
            for sample in sources
        ],
        seed=config.seed,
    )
    sources = [
        replace(
            sample,
            group_id=assignments[sample.sample_id].group_id,
            split=assignments[sample.sample_id].split,
        )
        for sample in sources
    ]
    training = [sample for sample in sources if sample.split == "train"]
    if not training:
        raise ValueError("Group split produced no training samples")
    calibration = _stratified_calibration(training, config.calibration_size)
    split_counts = {
        split: sum(sample.split == split for sample in sources)
        for split in ("train", "valid", "test")
    }
    _log(
        "group split complete: "
        + ", ".join(f"{split}={count:,}" for split, count in split_counts.items())
        + f"; calibration={len(calibration):,}",
        enabled=config.show_progress,
    )

    sweep: list[EpsilonResult] = []
    chosen_epsilon: float | None = None
    chosen_codebook: np.ndarray | None = None
    epsilon_candidates = sorted(set(config.epsilon_candidates))
    for epsilon_index, epsilon in enumerate(epsilon_candidates, start=1):
        epsilon_started = time.perf_counter()
        _log(
            f"calibrating RDP epsilon {epsilon:g} "
            f"({epsilon_index}/{len(epsilon_candidates)}): simplifying strokes",
            enabled=config.show_progress,
        )
        try:
            simplified = {
                sample.sample_id: simplify_strokes(sample.strokes, epsilon)
                for sample in _progress(
                    sources,
                    total=len(sources),
                    description=f"RDP epsilon {epsilon:g}",
                    unit="image",
                    enabled=config.show_progress,
                )
            }
            codebook_inputs = _training_codebook_samples(
                training,
                simplified,
                config=config,
            )
            _log(
                f"epsilon {epsilon:g}: fitting the 2,048-center codebook from "
                f"{len(codebook_inputs):,} training variants",
                enabled=config.show_progress,
            )
            codebook = fit_training_codebook(codebook_inputs, seed=config.seed)
            tokenizer = AnchoredTokenizer(codebook)
            source_scores: list[float] = []
            roundtrip_scores: list[float] = []
            lengths: list[int] = []
            for sample in _progress(
                calibration,
                total=len(calibration),
                description=f"Calibration epsilon {epsilon:g}",
                unit="image",
                enabled=config.show_progress,
            ):
                candidate = simplified[sample.sample_id]
                tokens = tokenizer.encode(candidate)
                restored = tokenizer.decode(tokens)
                source_scores.append(
                    _source_vector_geometry_f1(
                        sample,
                        candidate,
                        threshold_profile=config.vectorizer_threshold_profile,
                    )
                )
                roundtrip_scores.append(geometry_f1(candidate, restored))
                lengths.append(len(tokens))
            result = EpsilonResult(
                epsilon=float(epsilon),
                p99_token_length=float(np.quantile(lengths, 0.99, method="higher")),
                source_vector_median_f1=float(np.median(source_scores)),
                roundtrip_median_f1=float(np.median(roundtrip_scores)),
                roundtrip_p10_f1=float(np.quantile(roundtrip_scores, 0.10)),
                passed=False,
            )
            passed = (
                result.p99_token_length <= config.max_sequence_length
                and result.source_vector_median_f1 >= config.source_vector_f1_gate
                and result.roundtrip_median_f1 >= config.roundtrip_median_f1_gate
                and result.roundtrip_p10_f1 >= config.roundtrip_p10_f1_gate
            )
            result = EpsilonResult(**{**asdict(result), "passed": passed})
            sweep.append(result)
            _log(
                f"epsilon {epsilon:g} {'passed' if passed else 'failed'} in "
                f"{_format_duration(time.perf_counter() - epsilon_started)}: "
                f"p99_tokens={result.p99_token_length:g}, "
                f"source_f1={result.source_vector_median_f1:.4f}, "
                f"roundtrip_median_f1={result.roundtrip_median_f1:.4f}, "
                f"roundtrip_p10_f1={result.roundtrip_p10_f1:.4f}",
                enabled=config.show_progress,
            )
            if passed:
                chosen_epsilon = float(epsilon)
                chosen_codebook = codebook
                break
        except Exception as exc:
            sweep.append(
                EpsilonResult(
                    epsilon=float(epsilon),
                    p99_token_length=-1.0,
                    source_vector_median_f1=0.0,
                    roundtrip_median_f1=0.0,
                    roundtrip_p10_f1=0.0,
                    passed=False,
                    failure=f"{type(exc).__name__}:{exc}",
                )
            )
            _log(
                f"epsilon {epsilon:g} errored after "
                f"{_format_duration(time.perf_counter() - epsilon_started)}: "
                f"{type(exc).__name__}: {exc}",
                enabled=config.show_progress,
            )
    if chosen_epsilon is None or chosen_codebook is None:
        details = "; ".join(
            f"epsilon={result.epsilon}: {result.failure or asdict(result)}" for result in sweep
        )
        raise ValueError(f"No RDP epsilon passed anchored V3 calibration gates: {details}")

    tokenizer = AnchoredTokenizer(chosen_codebook)
    accepted: list[EncodedSample] = []
    encoding_started = time.perf_counter()
    _log(
        f"encoding {len(sources):,} accepted source images at epsilon "
        f"{chosen_epsilon:g}",
        enabled=config.show_progress,
    )
    for sample in _progress(
        sources,
        total=len(sources),
        description="Encoding",
        unit="image",
        enabled=config.show_progress,
    ):
        simplified = simplify_strokes(sample.strokes, chosen_epsilon)
        variants: list[tuple[str, list[list[tuple[float, float]]], bool]] = [
            (sample.sample_id, simplified, False)
        ]
        if sample.split == "train":
            for copy_index in range(config.train_augmentation_copies):
                namespace = f"{sample.sample_id}:augmentation:{copy_index + 1}"
                variants.append(
                    (
                        f"{sample.sample_id}#aug-{copy_index + 1:02d}",
                        augment_strokes(
                            simplified,
                            seed=stable_seed(namespace, config.seed),
                        ),
                        True,
                    )
                )
        for variant_id, strokes, augmented in variants:
            tokens = tokenizer.encode(strokes)
            preprocessing = {
                "rdp_epsilon": chosen_epsilon,
                "canvas_size": 256,
                "canvas_margin": 8,
                "augmented": augmented,
                "source_sample_id": sample.sample_id,
            }
            if len(tokens) > config.max_sequence_length:
                rejected.append(
                    RejectedSample(
                        sample_id=variant_id,
                        source_path=sample.source_path,
                        source_sha256=sample.source_sha256,
                        perceptual_hash=sample.perceptual_hash,
                        rejection_reason=(
                            f"overlength:{len(tokens)}>{config.max_sequence_length}"
                        ),
                        preprocessing=preprocessing,
                        group_id=sample.group_id,
                        split=sample.split,
                        point_count=sum(len(stroke) for stroke in strokes),
                        stroke_count=len(strokes),
                        token_length=len(tokens),
                    )
                )
                continue
            accepted.append(
                EncodedSample(
                    sample_id=variant_id,
                    source_path=sample.source_path,
                    source_sha256=sample.source_sha256,
                    perceptual_hash=sample.perceptual_hash,
                    group_id=sample.group_id,
                    split=sample.split,
                    point_count=sum(len(stroke) for stroke in strokes),
                    stroke_count=len(strokes),
                    tokens=tokens,
                    preprocessing=preprocessing,
                )
            )
    _log_rate(
        "encoding complete",
        completed=len(sources),
        started_at=encoding_started,
        detail=f"{len(accepted):,} variants accepted, {len(rejected):,} total rejected",
        enabled=config.show_progress,
    )
    if not accepted:
        raise ValueError("All encoded samples exceeded the maximum sequence length")
    accepted_source_count = sum(
        not bool(sample.preprocessing.get("augmented", False)) for sample in accepted
    )
    _enforce_minimum_source_count(
        accepted_source_count,
        config.minimum_accepted_source_sketches,
        stage="encoding",
    )

    preparation: dict[str, Any] = {
        "builder": "anchored_v3",
        "source_dir": str(source_root.resolve()),
        "seed": config.seed,
        "minimum_accepted_source_sketches": config.minimum_accepted_source_sketches,
        "accepted_source_sketches": accepted_source_count,
        "selected_rdp_epsilon": chosen_epsilon,
        "epsilon_sweep": [asdict(result) for result in sweep],
        "augmentation": {
            "copies_per_training_sample": config.train_augmentation_copies,
            "horizontal_flip_probability": 0.5,
            "rotation_degrees": [-5.0, 5.0],
            "scale": [0.9, 1.1],
            "translation_pixels": [-4.0, 4.0],
            "point_jitter_sigma": 0.25,
        },
        "preprocessing": {
            "vectorizer_method": "centerline",
            "vectorizer_threshold_profile": config.vectorizer_threshold_profile,
            "minimum_component_pixels": 8,
            "minimum_stroke_arc_length_pixels": 3.0,
            "border_frame_margin_pixels": 2.0,
            "branch_merge_distance_pixels": 1.5,
            "deterministic_orientation_and_ordering": True,
            "canvas_size": 256,
            "canvas_margin": 8,
            "maximum_token_length": config.max_sequence_length,
            "rdp_epsilon_candidates": list(config.epsilon_candidates),
            "calibration_size": config.calibration_size,
            "source_vector_f1_gate": config.source_vector_f1_gate,
            "roundtrip_median_f1_gate": config.roundtrip_median_f1_gate,
            "roundtrip_p10_f1_gate": config.roundtrip_p10_f1_gate,
        },
        "split": {
            "fractions": [0.8, 0.1, 0.1],
            "grouped_by": ["source_sha256", "perceptual_hash_hamming_lte_4"],
            "point_count_buckets": [512, 1024, 2048, "inf"],
        },
    }
    _log(
        f"publishing {len(accepted):,} encoded samples in shards of "
        f"{config.shard_size:,}",
        enabled=config.show_progress,
    )
    destination = write_dataset_atomic(
        output_root,
        accepted,
        chosen_codebook,
        preparation=preparation,
        rejected=rejected,
        shard_size=config.shard_size,
    )
    _log(
        f"build complete in {_format_duration(time.perf_counter() - build_started)}: "
        f"{destination}",
        enabled=config.show_progress,
    )
    return destination


def _preprocess_sources(
    paths: Sequence[Path],
    *,
    source_root: Path,
    threshold_profile: str,
    workers: int,
    show_progress: bool,
) -> tuple[list[_SourceGeometry], list[RejectedSample]]:
    """Vectorize independent source images in worker processes."""

    tasks = [
        _PreprocessingTask(
            path=str(path),
            sample_id=path.relative_to(source_root).as_posix(),
            threshold_profile=threshold_profile,
        )
        for path in paths
    ]
    if workers == 1:
        results = _progress(
            map(_preprocess_source, tasks),
            total=len(tasks),
            description="Preprocessing",
            unit="image",
            enabled=show_progress,
        )
        materialized = list(results)
    else:
        chunksize = min(16, max(1, len(tasks) // (workers * 8)))
        context = _multiprocessing_context()
        with context.Pool(
            processes=workers,
            initializer=_initialize_preprocessing_worker,
        ) as pool:
            results = _progress(
                pool.imap_unordered(
                    _preprocess_source,
                    tasks,
                    chunksize=chunksize,
                ),
                total=len(tasks),
                description="Preprocessing",
                unit="image",
                enabled=show_progress,
            )
            materialized = list(results)

    sources = sorted(
        (result for result in materialized if isinstance(result, _SourceGeometry)),
        key=lambda sample: sample.sample_id,
    )
    rejected = sorted(
        (result for result in materialized if isinstance(result, RejectedSample)),
        key=lambda sample: sample.sample_id,
    )
    return sources, rejected


def _preprocess_source(task: _PreprocessingTask) -> _SourceGeometry | RejectedSample:
    path = Path(task.path)
    source_digest = sha256_file(path)
    try:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("image could not be decoded")
        raw = vectorize_image(
            path,
            epsilon=0.0,
            method="centerline",
            threshold_profile=task.threshold_profile,
            min_object_size=1,
        )
        cleaned = preprocess_strokes(raw, source_shape=image.shape)
        normalized, canvas_transform = normalize_to_canvas_with_transform(cleaned)
        if not normalized:
            raise ValueError("no drawable strokes remained after cleaning")
        return _SourceGeometry(
            sample_id=task.sample_id,
            source_path=str(path),
            source_sha256=source_digest,
            perceptual_hash=perceptual_hash(image),
            strokes=normalized,
            point_count=sum(len(stroke) for stroke in normalized),
            canvas_transform=canvas_transform,
        )
    except Exception as exc:
        return RejectedSample(
            sample_id=task.sample_id,
            source_path=str(path),
            source_sha256=source_digest,
            perceptual_hash=None,
            rejection_reason=f"preprocessing_error:{type(exc).__name__}:{exc}",
        )


def _initialize_preprocessing_worker() -> None:
    """Prevent nested OpenCV pools from oversubscribing each process."""

    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)


def _multiprocessing_context() -> multiprocessing.context.BaseContext:
    """Avoid forking native OpenCV/NumPy threads into preprocessing workers."""

    if "forkserver" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("forkserver")
    return multiprocessing.get_context("spawn")


def _resolve_worker_count(requested: int, image_count: int) -> int:
    if image_count <= 0:
        raise ValueError("image_count must be positive")
    if requested < 0:
        raise ValueError("preprocessing_workers must be non-negative")
    if requested == 0:
        try:
            available = len(os.sched_getaffinity(0))
        except AttributeError:
            available = os.cpu_count() or 1
    else:
        available = requested
    return max(1, min(int(available), int(image_count)))


def _progress(
    values: Iterable[_ProgressItem],
    *,
    total: int,
    description: str,
    unit: str,
    enabled: bool,
) -> Iterable[_ProgressItem]:
    return tqdm(
        values,
        total=total,
        desc=description,
        unit=unit,
        disable=not enabled,
        dynamic_ncols=True,
        mininterval=0.25,
    )


def _log(message: str, *, enabled: bool) -> None:
    if enabled:
        print(f"[anchored-v3] {message}", file=sys.stderr, flush=True)


def _log_rate(
    label: str,
    *,
    completed: int,
    started_at: float,
    detail: str,
    enabled: bool,
) -> None:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    _log(
        f"{label} in {_format_duration(elapsed)} "
        f"({completed / elapsed:,.2f} images/s): {detail}",
        enabled=enabled,
    )


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {remaining_seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {remaining_seconds:02d}s"
    return f"{remaining_seconds:d}s"


def _training_codebook_samples(
    training: Sequence[_SourceGeometry],
    simplified: dict[str, list[list[tuple[float, float]]]],
    *,
    config: BuilderConfig,
) -> list[TrainingStrokeSample]:
    output: list[TrainingStrokeSample] = []
    for sample in training:
        strokes = simplified[sample.sample_id]
        output.append(TrainingStrokeSample(sample.sample_id, "train", strokes))
        for copy_index in range(config.train_augmentation_copies):
            namespace = f"{sample.sample_id}:augmentation:{copy_index + 1}"
            output.append(
                TrainingStrokeSample(
                    f"{sample.sample_id}#aug-{copy_index + 1:02d}",
                    "train",
                    augment_strokes(
                        strokes,
                        seed=stable_seed(namespace, config.seed),
                    ),
                )
            )
    return output


def _stratified_calibration(
    training: Sequence[_SourceGeometry], maximum_size: int
) -> list[_SourceGeometry]:
    if maximum_size <= 0:
        raise ValueError("calibration_size must be positive")
    ordered = sorted(training, key=lambda sample: (sample.point_count, sample.sample_id))
    if len(ordered) <= maximum_size:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, num=maximum_size, dtype=np.int64)
    return [ordered[int(index)] for index in indices]


def _validate_config(config: BuilderConfig) -> None:
    if not config.epsilon_candidates or any(value < 0 for value in config.epsilon_candidates):
        raise ValueError("epsilon_candidates must contain non-negative values")
    if config.max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if config.train_augmentation_copies < 0:
        raise ValueError("train_augmentation_copies must be non-negative")
    if config.minimum_accepted_source_sketches <= 0:
        raise ValueError("minimum_accepted_source_sketches must be positive")
    if config.shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if (
        isinstance(config.preprocessing_workers, bool)
        or not isinstance(config.preprocessing_workers, int)
        or config.preprocessing_workers < 0
    ):
        raise ValueError("preprocessing_workers must be a non-negative integer")
    if not isinstance(config.show_progress, bool):
        raise ValueError("show_progress must be a boolean")


def _source_vector_geometry_f1(
    sample: _SourceGeometry,
    candidate: Sequence[Sequence[Sequence[float]]],
    *,
    threshold_profile: str,
) -> float:
    """Compare a simplified vector directly with its source raster centerline."""

    image = cv2.imread(sample.source_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"source image could not be decoded: {sample.source_path}")
    reference = source_centerline(image, threshold_profile=threshold_profile)
    source_coordinates = denormalize_from_canvas(candidate, sample.canvas_transform)
    rendered = rasterize_strokes(source_coordinates, image.shape)
    return float(centerline_metrics(reference, rendered, tolerance_px=2.0).f1)


def _enforce_minimum_source_count(count: int, minimum: int, *, stage: str) -> None:
    if int(count) < int(minimum):
        raise ValueError(
            f"Anchored V3 requires at least {minimum} accepted source sketches; "
            f"only {count} remained after {stage}."
        )
