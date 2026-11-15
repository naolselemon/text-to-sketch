"""Exact duplicate image detection via SHA-256 byte hashing.

This module detects duplicate sketch and image files based on exact byte content
without silently dropping or mutating data. Duplicate occurrences are reported
and grouped with their canonical file paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".npy", ".npz")


def compute_file_sha256(path: str | Path, chunk_size: int = 65536) -> str:
    """Compute hex SHA-256 digest of a file's raw byte contents."""
    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class DuplicateGroup:
    """A collection of identical files sharing the exact same SHA-256 byte hash."""

    sha256: str
    file_size_bytes: int
    canonical_path: str
    duplicate_paths: list[str]

    @property
    def total_copies(self) -> int:
        return 1 + len(self.duplicate_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
            "canonical_path": self.canonical_path,
            "duplicate_paths": list(self.duplicate_paths),
            "total_copies": self.total_copies,
        }


@dataclass
class DuplicateReport:
    """Detailed summary of exact duplicate detection."""

    total_files_scanned: int = 0
    unique_files_count: int = 0
    duplicate_files_count: int = 0
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)

    @property
    def has_duplicates(self) -> bool:
        return len(self.duplicate_groups) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_files_scanned": self.total_files_scanned,
            "unique_files_count": self.unique_files_count,
            "duplicate_files_count": self.duplicate_files_count,
            "duplicate_groups_count": len(self.duplicate_groups),
            "has_duplicates": self.has_duplicates,
            "duplicate_groups": [g.to_dict() for g in self.duplicate_groups],
        }

    def write_json(self, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out

    def write_csv(self, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sha256", "file_size_bytes", "canonical_path", "duplicate_path"])
            for group in self.duplicate_groups:
                for dup_path in group.duplicate_paths:
                    writer.writerow([group.sha256, group.file_size_bytes, group.canonical_path, dup_path])
        return out


def find_files_to_scan(
    paths_or_directory: str | Path | Iterable[str | Path],
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> list[Path]:
    """Gather all matching files from directory or path list."""
    ext_set = {ext.lower() for ext in extensions}
    if isinstance(paths_or_directory, (str, Path)):
        p = Path(paths_or_directory)
        if p.is_file():
            return [p]
        if p.is_dir():
            return sorted(
                [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in ext_set]
            )
        return []

    collected: list[Path] = []
    for item in paths_or_directory:
        p = Path(item)
        if p.is_file() and (not ext_set or p.suffix.lower() in ext_set):
            collected.append(p)
        elif p.is_dir():
            collected.extend([f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in ext_set])
    return sorted(set(collected))


def detect_exact_duplicates(
    paths_or_directory: str | Path | Iterable[str | Path],
    *,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> DuplicateReport:
    """Detect all exact byte duplicates among files.

    Args:
        paths_or_directory: Directory path or list of file paths to scan.
        extensions: Allowed file extensions when scanning directories.

    Returns:
        DuplicateReport containing counts and DuplicateGroups without mutating any files.
    """
    files = find_files_to_scan(paths_or_directory, extensions=extensions)
    if not files:
        return DuplicateReport(total_files_scanned=0, unique_files_count=0, duplicate_files_count=0)

    # Group first by file size to avoid unnecessary hashing of uniquely sized files
    size_map: dict[int, list[Path]] = defaultdict(list)
    for f in files:
        try:
            size_map[f.stat().st_size].append(f)
        except OSError:
            continue

    hash_map: dict[str, list[Path]] = defaultdict(list)
    file_sizes: dict[str, int] = {}

    for size, candidates in size_map.items():
        if len(candidates) < 2:
            # Unique file size cannot be a duplicate
            continue
        for candidate in candidates:
            try:
                digest = compute_file_sha256(candidate)
                hash_map[digest].append(candidate)
                file_sizes[digest] = size
            except OSError:
                continue

    duplicate_groups: list[DuplicateGroup] = []
    total_duplicates = 0

    for digest, group_paths in sorted(hash_map.items()):
        if len(group_paths) > 1:
            sorted_paths = sorted(str(p.resolve()) for p in group_paths)
            canonical = sorted_paths[0]
            duplicates = sorted_paths[1:]
            total_duplicates += len(duplicates)
            duplicate_groups.append(
                DuplicateGroup(
                    sha256=digest,
                    file_size_bytes=file_sizes[digest],
                    canonical_path=canonical,
                    duplicate_paths=duplicates,
                )
            )

    unique_files = len(files) - total_duplicates

    return DuplicateReport(
        total_files_scanned=len(files),
        unique_files_count=unique_files,
        duplicate_files_count=total_duplicates,
        duplicate_groups=duplicate_groups,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect exact duplicate files via SHA-256 byte hashing without deleting data."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        type=Path,
        help="Directory to scan recursively for duplicate files.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        type=Path,
        help="Optional path to write duplicate detection JSON report.",
    )
    parser.add_argument(
        "--report-csv",
        default=None,
        type=Path,
        help="Optional path to write duplicate detection CSV report.",
    )
    parser.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        help="Exit with non-zero code if any duplicates are found.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON report to standard output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = detect_exact_duplicates(args.input_dir)

    if args.report_json:
        report.write_json(args.report_json)
        print(f"[dedup] Wrote JSON report to {args.report_json}")

    if args.report_csv:
        report.write_csv(args.report_csv)
        print(f"[dedup] Wrote CSV report to {args.report_csv}")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print("=== DUPLICATE DETECTION REPORT ===")
        print(f"Scanned files: {report.total_files_scanned}")
        print(f"Unique files: {report.unique_files_count}")
        print(f"Duplicate files: {report.duplicate_files_count}")
        print(f"Duplicate groups: {len(report.duplicate_groups)}")
        if report.duplicate_groups:
            print("\nDuplicate groups found:")
            for g in report.duplicate_groups[:10]:
                print(f"  SHA256: {g.sha256} ({g.total_copies} copies, {g.file_size_bytes} bytes)")
                print(f"    Canonical: {g.canonical_path}")
                for d in g.duplicate_paths:
                    print(f"    Duplicate: {d}")
            if len(report.duplicate_groups) > 10:
                print(f"  ... and {len(report.duplicate_groups) - 10} more groups")
        else:
            print("No exact byte duplicate files found.")

    if args.fail_on_duplicates and report.has_duplicates:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
