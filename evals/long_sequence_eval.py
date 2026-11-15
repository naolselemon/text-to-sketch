"""Audit a real benchmark run across the pretrained and extended length ranges.

First run benchmark_eval.py with --modes teacher-forced and a labelled
converted checkpoint; use --prepare-comparison-manifest to freeze one
sample per populated five-bin range. Rerun benchmark_eval.py with both
modes and --comparison-manifest <prepared.json>, then run this tool
without --prepare-comparison-manifest on that second benchmark report.
The final report references its saved images; it does not score visual
quality or claim TensorFlow parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image
from safetensors import SafetensorError, safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LENGTH_DEFINITION = "non-padding model input tokens, including SOS and EOS"
BINS = ((1, 200), (201, 512), (513, 1024), (1025, 2048), (2049, 4096))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(
        type(value) is int and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return value


def _score(value: Any, label: str) -> float:
    _require(
        type(value) in (float, int)
        and math.isfinite(value)
        and 0 <= value <= 1,
        f"{label} must be a finite number in [0, 1]",
    )
    return float(value)


def _flag(value: Any, label: str) -> int:
    score = _score(value, label)
    _require(score in (0, 1), f"{label} must be binary")
    return int(score)


def _sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None,
        f"{label} must be a SHA-256 digest",
    )
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_png(path: Path) -> None:
    """Require a complete PNG plot rather than an empty or renamed artifact."""
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"Not a PNG comparison image: {path}")
        image.verify()


def _read_json(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8")), str(path))


def _artifact(report_path: Path, name: Any) -> Path:
    _require(
        isinstance(name, str) and name not in ("", ".", "..")
        and Path(name).name == name,
        "Benchmark artifact paths must be filenames beside the report",
    )
    path = report_path.parent / name
    _require(path.is_file(), f"Missing benchmark artifact: {path}")
    return path


def _records(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = _object(json.loads(line), f"{path}:{line_number}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            sample_id = row.get("sample_id")
            _require(
                isinstance(sample_id, str) and bool(sample_id),
                f"{path}:{line_number}: missing sample_id",
            )
            _require(sample_id not in by_id, f"Duplicate sample_id: {sample_id}")
            by_id[sample_id] = row
    _require(bool(by_id), f"Empty per-sample artifact: {path}")
    return by_id


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    _require(
        path.suffix == ".safetensors",
        "Pretrained assessment requires a converted, identity-labelled .safetensors checkpoint",
    )
    try:
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata()
    except SafetensorError as exc:
        raise ValueError(f"Invalid safetensors checkpoint: {path}") from exc
    metadata = _object(metadata, "checkpoint identity metadata")
    _sha(metadata.get("source_tf_checkpoint_sha256"), "source_tf_checkpoint_sha256")
    _sha(metadata.get("original_token_dict_sha256"), "original_token_dict_sha256")
    _sha(metadata.get("codebook_sha256"), "checkpoint codebook_sha256")
    layout_json = metadata.get("token_layout_json")
    _require(isinstance(layout_json, str), "Checkpoint lacks token layout identity")
    layout = _object(json.loads(layout_json), "checkpoint token_layout_json")
    size = _integer(layout.get("codebook_size"), "codebook_size", minimum=1)
    _require(
        layout == {
            "codebook_size": size,
            "motion_token_offset": 1,
            "pad_token_id": 0,
            "sep_token_id": size + 1,
            "sos_token_id": size + 2,
            "eos_token_id": size + 3,
            "vocab_size": size + 4,
        },
        "Checkpoint token layout is not the released dictionary layout",
    )
    return metadata


def _verify_provenance(
    provenance: dict[str, Any],
    codebook_path: Path,
    *,
    require_codebook_record: bool = True,
) -> None:
    _require(
        provenance.get("length_definition") == LENGTH_DEFINITION,
        "Unsupported benchmark length definition",
    )
    _require(
        provenance.get("split") in ("valid", "test"),
        "Length evaluation requires a held-out valid/test split",
    )
    _require(
        provenance.get("random_initialization") is False,
        "Random initialization cannot establish pretrained length quality",
    )
    load = _object(provenance.get("checkpoint_load"), "checkpoint_load")
    _require(
        load.get("strict") is True
        and load.get("missing_keys") == []
        and load.get("unexpected_keys") == [],
        "Length evaluation requires a fully loaded checkpoint",
    )
    checkpoint = provenance.get("checkpoint")
    _require(isinstance(checkpoint, str) and bool(checkpoint), "Missing checkpoint path")
    checkpoint_path = PROJECT_ROOT / checkpoint
    _require(checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}")
    _require(
        _file_hash(checkpoint_path) == _sha(provenance.get("checkpoint_sha256"), "checkpoint_sha256"),
        "Checkpoint hash does not match benchmark provenance",
    )
    _require(codebook_path.is_file(), f"Missing codebook: {codebook_path}")
    codebook_hash = _file_hash(codebook_path)
    metadata = _checkpoint_metadata(checkpoint_path)
    _require(
        metadata["codebook_sha256"] == codebook_hash,
        "Checkpoint was converted with a different codebook",
    )
    if require_codebook_record:
        _require(
            codebook_hash == _sha(provenance.get("codebook_sha256"), "codebook_sha256"),
            "Codebook hash does not match benchmark provenance",
        )
    files = _object(provenance.get("dataset_files"), "dataset_files")
    _require(bool(files), "Missing held-out dataset file hashes")
    data_root = provenance.get("data_root")
    _require(isinstance(data_root, str) and bool(data_root), "Missing dataset root")
    root = PROJECT_ROOT / data_root
    for name, expected in files.items():
        _require(
            isinstance(name, str) and Path(name).name == name and name not in (".", ".."),
            "Invalid dataset filename",
        )
        path = root / name
        _require(path.is_file(), f"Missing dataset file: {path}")
        _require(
            _file_hash(path) == _sha(expected, f"dataset_files[{name}]"),
            f"Dataset file hash mismatch: {path}",
        )
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    _require(
        digest == _sha(provenance.get("dataset_sha256"), "dataset_sha256"),
        "Dataset manifest hash mismatch",
    )
    _integer(provenance.get("seed"), "seed")


def _bin(length: int) -> str:
    for low, high in BINS:
        if low <= length <= high:
            return f"{low}-{high}"
    raise ValueError(f"Sample length outside supported range 1-4096: {length}")


def _legacy_bucket(length: int) -> str:
    if length <= 512:
        return "short"
    if length <= 1024:
        return "medium"
    if length <= 2048:
        return "long"
    return "extended"


def prepare_comparison_manifest(
    benchmark_path: Path, codebook_path: Path, *, output_path: Path
) -> dict[str, Any]:
    """Freeze one real teacher-forced sample per populated five-bin range."""
    benchmark_path = benchmark_path.resolve()
    output_path = output_path.resolve()
    benchmark = _read_json(benchmark_path)
    _require(
        benchmark.get("schema_version") == 2
        and set(benchmark) == {"schema_version", "provenance", "teacher_forced"},
        "Manifest preparation requires a teacher-forced-only benchmark",
    )
    provenance = _object(benchmark["provenance"], "provenance")
    _verify_provenance(provenance, codebook_path.resolve(), require_codebook_record=False)
    teacher = _object(benchmark["teacher_forced"], "teacher_forced")
    _require(teacher.get("mode") == "teacher-forced", "Unsupported benchmark mode")
    rows = _records(_artifact(benchmark_path, teacher.get("per_sample_path")))
    _require(
        _integer(teacher.get("total_samples"), "teacher_forced.total_samples", minimum=1)
        == len(rows),
        "Reported sample count disagrees with teacher-forced JSONL",
    )
    groups: dict[str, list[dict[str, Any]]] = {f"{low}-{high}": [] for low, high in BINS}
    source_root = PROJECT_ROOT / provenance["data_root"]
    for row in rows.values():
        sample_id = row["sample_id"]
        length = _integer(row.get("target_length"), f"{sample_id}.target_length", minimum=1)
        bucket = _bin(length)
        _sha(row.get("target_sha256"), f"{sample_id}.target_sha256")
        _require(
            row.get("length_bucket") == _legacy_bucket(length),
            f"Incorrect benchmark length bucket: {sample_id}",
        )
        source = row.get("source_file")
        _require(
            isinstance(source, str)
            and (PROJECT_ROOT / source).resolve() == (source_root / Path(source).name).resolve()
            and Path(source).name in provenance["dataset_files"],
            f"Sample source is not in the hashed dataset: {sample_id}",
        )
        _integer(row.get("source_index"), f"{sample_id}.source_index")
        _integer(row.get("stroke_count"), f"{sample_id}.stroke_count")
        _require(
            _integer(row.get("tf_scored_tokens"), "tf_scored_tokens", minimum=1) <= length,
            f"Invalid teacher-forced token count: {sample_id}",
        )
        _score(row.get("tf_token_accuracy"), "tf_token_accuracy")
        groups[bucket].append(row)
    seed = provenance["seed"]
    selection = [
        {
            "sample_id": winner["sample_id"],
            "length_bucket": winner["length_bucket"],
            "target_length": winner["target_length"],
        }
        for group in groups.values() if group
        for winner in [min(
            group,
            key=lambda row: (
                hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest(),
                row["sample_id"],
            ),
        )]
    ]
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "length_definition": LENGTH_DEFINITION,
        "dataset_sha256": provenance["dataset_sha256"],
        "samples": selection,
    }
    _require(output_path != benchmark_path, "Output must not overwrite benchmark")
    _require(not output_path.exists(), f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def evaluate(
    benchmark_path: Path, codebook_path: Path, *, output_path: Path
) -> dict[str, Any]:
    """Validate inputs before writing an auditable report, including insufficient bins."""
    benchmark_path = benchmark_path.resolve()
    codebook_path = codebook_path.resolve()
    output_path = output_path.resolve()
    benchmark = _read_json(benchmark_path)
    _require(benchmark.get("schema_version") == 2, "Unsupported benchmark schema")
    _require(
        set(benchmark) == {"schema_version", "provenance", "teacher_forced", "free_running"},
        "Benchmark must contain both modes and no unsupported modes",
    )
    provenance = _object(benchmark["provenance"], "provenance")
    _verify_provenance(provenance, codebook_path)
    tf = _object(benchmark["teacher_forced"], "teacher_forced")
    fr = _object(benchmark["free_running"], "free_running")
    _require(
        tf.get("mode") == "teacher-forced" and fr.get("mode") == "free-running",
        "Unsupported benchmark evaluation mode",
    )
    tf_path = _artifact(benchmark_path, tf.get("per_sample_path"))
    fr_path = _artifact(benchmark_path, fr.get("per_sample_path"))
    tf_rows, fr_rows = _records(tf_path), _records(fr_path)
    _require(
        _integer(tf.get("total_samples"), "teacher_forced.total_samples", minimum=1) == len(tf_rows)
        and _integer(fr.get("total_samples"), "free_running.total_samples", minimum=1) == len(fr_rows),
        "Reported sample counts disagree with JSONL artifacts",
    )
    _require(tf_rows.keys() == fr_rows.keys(), "Teacher-forced/free-running sample identities differ")

    groups: dict[str, list[dict[str, Any]]] = {
        f"{low}-{high}": [] for low, high in BINS
    }
    source_root = PROJECT_ROOT / provenance["data_root"]
    for sample_id, row in fr_rows.items():
        teacher = tf_rows[sample_id]
        length = _integer(row.get("target_length"), f"{sample_id}.target_length", minimum=1)
        _require(
            length == _integer(teacher.get("target_length"), f"{sample_id}.teacher_length", minimum=1)
            and row.get("target_sha256") == teacher.get("target_sha256"),
            f"Target identity differs between modes: {sample_id}",
        )
        _sha(row.get("target_sha256"), f"{sample_id}.target_sha256")
        _require(
            row.get("length_bucket") == teacher.get("length_bucket") == _legacy_bucket(length),
            f"Incorrect benchmark length bucket: {sample_id}",
        )
        for key in ("source_file", "source_index", "stroke_count"):
            _require(
                key in row and row[key] == teacher.get(key),
                f"Sample metadata differs between modes: {sample_id}.{key}",
            )
        _integer(row["source_index"], f"{sample_id}.source_index")
        _integer(row["stroke_count"], f"{sample_id}.stroke_count")
        source = row["source_file"]
        _require(
            isinstance(source, str)
            and (PROJECT_ROOT / source).resolve() == (source_root / Path(source).name).resolve()
            and Path(source).name in provenance["dataset_files"],
            f"Sample source is not in the hashed dataset: {sample_id}",
        )
        _require(
            _integer(teacher.get("tf_scored_tokens"), f"{sample_id}.tf_scored_tokens", minimum=1)
            <= length,
            f"Invalid teacher-forced token count: {sample_id}",
        )
        _score(teacher.get("tf_token_accuracy"), f"{sample_id}.tf_token_accuracy")
        _score(row.get("geometry_f1_2px"), f"{sample_id}.geometry_f1_2px")
        _score(row.get("token_accuracy"), f"{sample_id}.token_accuracy")
        eos = _flag(row.get("eos_rate"), f"{sample_id}.eos_rate")
        hit = _flag(row.get("max_length_hit"), f"{sample_id}.max_length_hit")
        _flag(row.get("exact_match"), f"{sample_id}.exact_match")
        _require(not (eos and hit), f"EOS and max-length hit conflict: {sample_id}")
        groups[_bin(length)].append(row)

    manifest_path = _artifact(benchmark_path, fr.get("comparison_manifest_path"))
    manifest = _read_json(manifest_path)
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("seed") == provenance["seed"]
        and manifest.get("dataset_sha256") == provenance["dataset_sha256"]
        and manifest.get("length_definition") == LENGTH_DEFINITION,
        "Comparison manifest provenance differs from benchmark",
    )
    selected = manifest.get("samples")
    _require(isinstance(selected, list), "Missing comparison samples")
    plot_dir = benchmark_path.parent / f"{benchmark_path.stem.removesuffix('_report')}_plots"
    plot_refs: dict[str, list[dict[str, Any]]] = {key: [] for key in groups}
    seen: set[str] = set()
    for index, selected_row in enumerate(selected, start=1):
        selected_row = _object(selected_row, f"comparison sample {index}")
        sample_id = selected_row.get("sample_id")
        _require(
            isinstance(sample_id, str) and sample_id in fr_rows and sample_id not in seen,
            f"Unknown or duplicate comparison sample: {sample_id}",
        )
        seen.add(sample_id)
        row = fr_rows[sample_id]
        _require(
            selected_row.get("target_length") == row["target_length"]
            and selected_row.get("length_bucket") == row["length_bucket"],
            f"Comparison sample identity mismatch: {sample_id}",
        )
        plot = plot_dir / f"reconstruction_{index:03d}.png"
        _require(plot.is_file(), f"Missing saved comparison image: {plot}")
        _validate_png(plot)
        plot_refs[_bin(row["target_length"])].append(
            {"sample_id": sample_id, "target_length": row["target_length"], "plot": str(plot)}
        )

    bins: dict[str, dict[str, Any]] = {}
    for low, high in BINS:
        key = f"{low}-{high}"
        rows = groups[key]
        references = sorted(plot_refs[key], key=lambda ref: ref["sample_id"])
        _require(
            not rows or references,
            f"No qualitative comparison image for populated length bin {key}; rerun benchmark with more plots or a comparison manifest covering every bin",
        )
        eos_count = sum(int(row["eos_rate"]) for row in rows)
        hit_count = sum(int(row["max_length_hit"]) for row in rows)
        exact_count = sum(int(row["exact_match"]) for row in rows)
        bins[key] = {
            "range": [low, high],
            "sample_count": len(rows),
            "status": "review_required" if rows else "insufficient",
            "eos_count": eos_count,
            "eos_rate": eos_count / len(rows) if rows else None,
            "max_length_hit_count": hit_count,
            "max_length_hit_rate": hit_count / len(rows) if rows else None,
            "exact_match_count": exact_count,
            "exact_match_rate": exact_count / len(rows) if rows else None,
            "geometry_f1_2px_mean": mean(row["geometry_f1_2px"] for row in rows) if rows else None,
            "geometry_f1_2px_median": median(row["geometry_f1_2px"] for row in rows) if rows else None,
            "token_accuracy_mean": mean(row["token_accuracy"] for row in rows) if rows else None,
            "teacher_forced_token_accuracy_mean": (
                mean(tf_rows[row["sample_id"]]["tf_token_accuracy"] for row in rows)
                if rows else None
            ),
            "representative": references[0] if references else None,
            "comparison_plots": references,
            "qualitative_review": "pending" if rows else "unavailable",
        }
    result = {
        "schema_version": 1,
        "evaluation": "held_out_length_bins",
        "benchmark_report": str(benchmark_path),
        "provenance": provenance,
        "teacher_forced_per_sample": str(tf_path),
        "free_running_per_sample": str(fr_path),
        "comparison_manifest": str(manifest_path),
        "total_samples": len(fr_rows),
        "status": "insufficient" if any(not rows for rows in groups.values()) else "review_required",
        "bins": bins,
    }
    _require(output_path.resolve() != benchmark_path.resolve(), "Output must not overwrite benchmark")
    _require(not output_path.exists(), f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--codebook-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prepare-comparison-manifest",
        action="store_true",
        help="Select one stable plot sample per populated five-bin range from a teacher-forced-only benchmark.",
    )
    args = parser.parse_args(argv)
    try:
        if args.prepare_comparison_manifest:
            prepare_comparison_manifest(
                args.benchmark_report, args.codebook_path, output_path=args.output
            )
            print(f"Five-bin benchmark comparison manifest: {args.output}")
            return 0
        result = evaluate(args.benchmark_report, args.codebook_path, output_path=args.output)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Length evaluation failed: {exc}", file=sys.stderr)
        return 2
    if result["status"] == "insufficient":
        print(f"Length evaluation insufficient (empty bins): {args.output}", file=sys.stderr)
        return 1
    print(f"Length-bin review report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
