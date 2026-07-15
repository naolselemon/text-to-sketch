"""Duplicate clustering and deterministic group-stratified splitting."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable


SPLIT_NAMES = ("train", "valid", "test")
DEFAULT_SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
POINT_COUNT_BUCKETS = (512, 1024, 2048)


@dataclass(frozen=True)
class SplitCandidate:
    sample_id: str
    source_sha256: str
    perceptual_hash: int
    point_count: int


@dataclass(frozen=True)
class SplitAssignment:
    sample_id: str
    group_id: str
    split: str
    point_count_bucket: int


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, index: int) -> int:
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        smaller, larger = sorted((first_root, second_root))
        self.parents[larger] = smaller


@dataclass
class _BKNode:
    value: int
    indices: list[int]
    children: dict[int, "_BKNode"]

    def add(self, value: int, index: int) -> None:
        distance = (self.value ^ value).bit_count()
        if distance == 0:
            self.indices.append(index)
            return
        child = self.children.get(distance)
        if child is None:
            self.children[distance] = _BKNode(value, [index], {})
        else:
            child.add(value, index)

    def query(self, value: int, maximum_distance: int, output: list[int]) -> None:
        distance = (self.value ^ value).bit_count()
        if distance <= maximum_distance:
            output.extend(self.indices)
        lower = distance - maximum_distance
        upper = distance + maximum_distance
        for child_distance, child in self.children.items():
            if lower <= child_distance <= upper:
                child.query(value, maximum_distance, output)


def assign_duplicate_groups(
    candidates: Iterable[SplitCandidate],
    *,
    maximum_hamming_distance: int = 4,
) -> dict[str, str]:
    """Cluster exact hashes and near perceptual hashes without quadratic scans."""

    values = sorted(candidates, key=lambda sample: sample.sample_id)
    if not values:
        return {}
    if maximum_hamming_distance < 0:
        raise ValueError("maximum_hamming_distance must be non-negative")
    sample_ids = [sample.sample_id for sample in values]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique")

    groups = _UnionFind(len(values))
    exact: dict[str, int] = {}
    tree: _BKNode | None = None
    for index, candidate in enumerate(values):
        previous = exact.get(candidate.source_sha256)
        if previous is not None:
            groups.union(index, previous)
        else:
            exact[candidate.source_sha256] = index

        matches: list[int] = []
        if tree is None:
            tree = _BKNode(int(candidate.perceptual_hash), [index], {})
        else:
            tree.query(
                int(candidate.perceptual_hash), maximum_hamming_distance, matches
            )
            for match in matches:
                groups.union(index, match)
            tree.add(int(candidate.perceptual_hash), index)

    components: dict[int, list[str]] = {}
    for index, candidate in enumerate(values):
        components.setdefault(groups.find(index), []).append(candidate.sample_id)
    component_ids = {
        root: hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()
        for root, members in components.items()
    }
    return {
        candidate.sample_id: component_ids[groups.find(index)]
        for index, candidate in enumerate(values)
    }


def deterministic_group_split(
    candidates: Iterable[SplitCandidate],
    *,
    seed: int = 42,
    fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS,
    maximum_hamming_distance: int = 4,
) -> dict[str, SplitAssignment]:
    """Assign duplicate groups together while balancing point-count buckets."""

    values = sorted(candidates, key=lambda sample: sample.sample_id)
    if len(fractions) != 3 or any(value < 0 for value in fractions):
        raise ValueError("fractions must contain three non-negative values")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    group_for = assign_duplicate_groups(
        values, maximum_hamming_distance=maximum_hamming_distance
    )
    by_id = {sample.sample_id: sample for sample in values}
    grouped: dict[str, list[str]] = {}
    for sample in values:
        grouped.setdefault(group_for[sample.sample_id], []).append(sample.sample_id)

    groups_by_bucket: dict[int, list[tuple[str, list[str]]]] = {}
    for group_id, members in grouped.items():
        bucket = point_count_bucket(max(by_id[name].point_count for name in members))
        groups_by_bucket.setdefault(bucket, []).append((group_id, sorted(members)))

    assignments: dict[str, SplitAssignment] = {}
    for bucket in sorted(groups_by_bucket):
        bucket_groups = groups_by_bucket[bucket]
        namespace = hashlib.sha256(f"{int(seed)}:{bucket}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(namespace[:8], "big"))
        decorated = [(rng.random(), group_id, members) for group_id, members in bucket_groups]
        decorated.sort(key=lambda value: (-len(value[2]), value[0], value[1]))
        total = sum(len(members) for _, _, members in decorated)
        targets = [total * fraction for fraction in fractions]
        counts = [0, 0, 0]
        for _, group_id, members in decorated:
            deficits = [targets[index] - counts[index] for index in range(3)]
            chosen = max(range(3), key=lambda index: (deficits[index], -index))
            counts[chosen] += len(members)
            split = SPLIT_NAMES[chosen]
            for sample_id in members:
                assignments[sample_id] = SplitAssignment(
                    sample_id=sample_id,
                    group_id=group_id,
                    split=split,
                    point_count_bucket=bucket,
                )
    return assignments


def point_count_bucket(point_count: int) -> int:
    value = int(point_count)
    if value < 0:
        raise ValueError("point_count must be non-negative")
    for index, upper_bound in enumerate(POINT_COUNT_BUCKETS):
        if value <= upper_bound:
            return index
    return len(POINT_COUNT_BUCKETS)

