"""
Step 2 — Stroke Ordering.

Imposes a drawing order on an unordered set of vectorised stroke paths.

Three ordering strategies are provided:

    order_directional_bias(strokes)      – top-left → bottom-right (default)
    order_greedy_nearest_neighbor(strokes) – minimise pen-travel greedily
    order_tsp(strokes)                   – global TSP minimisation via python-tsp

All functions accept and return ``list[list[tuple[int, int]]]`` — a list of
strokes, where each stroke is an ordered list of (x, y) pixel coordinates.
"""

from __future__ import annotations

import math

import numpy as np
from python_tsp.heuristics import solve_tsp_local_search



# Shared geometry


def _dist(pt1: tuple, pt2: tuple) -> float:
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


# Strategy A – Directional Bias (default)

def order_directional_bias(
    strokes: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    """Order strokes from top-left to bottom-right.

    Each stroke is ranked by (min_y * 2 + min_x), weighting vertical position
    slightly more than horizontal to mimic natural handwriting convention.
    """
    if not strokes:
        return []

    def _score(stroke: list[tuple[int, int]]) -> float:
        return min(pt[1] for pt in stroke) * 2.0 + min(pt[0] for pt in stroke)

    return sorted(strokes, key=_score)


# Strategy B – Greedy Nearest-Neighbor

def order_greedy_nearest_neighbor(
    strokes: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    """Always move to the closest undrawn stroke endpoint.

    Each candidate stroke may be flipped (reversed) if its tail is closer to
    the current pen position than its head, minimising pen-lift travel.
    """
    if not strokes:
        return []

    unvisited = list(strokes)
    ordered: list[list[tuple[int, int]]] = []

    current_stroke = unvisited.pop(0)
    ordered.append(current_stroke)
    current_end = current_stroke[-1]

    while unvisited:
        best_idx = -1
        best_dist = float("inf")
        flip_best = False

        for i, stroke in enumerate(unvisited):
            d_start = _dist(current_end, stroke[0])
            d_end   = _dist(current_end, stroke[-1])

            if d_start < best_dist:
                best_dist, best_idx, flip_best = d_start, i, False
            if d_end < best_dist:
                best_dist, best_idx, flip_best = d_end, i, True

        next_stroke = unvisited.pop(best_idx)
        if flip_best:
            next_stroke = list(reversed(next_stroke))

        ordered.append(next_stroke)
        current_end = next_stroke[-1]

    return ordered


# Strategy C – TSP Approximation

def order_tsp(
    strokes: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    """Minimise total pen-up travel via TSP on stroke centres.

    Falls back to greedy nearest-neighbor when there are ≤ 2 strokes or
    when the stroke count exceeds 800 (TSP becomes prohibitively slow).
    """
    if not strokes:
        return []

    n = len(strokes)
    if n <= 2 or n > 800:
        return order_greedy_nearest_neighbor(strokes)

    centers = [
        (
            sum(pt[0] for pt in s) / len(s),
            sum(pt[1] for pt in s) / len(s),
        )
        for s in strokes
    ]

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = _dist(centers[i], centers[j])

    tour, _ = solve_tsp_local_search(dist_matrix)
    ordered = [strokes[i] for i in tour]

    # Orient each stroke to minimise pen-lift from previous stroke end.
    if len(ordered) > 1:
        s0, s1 = ordered[0], ordered[1]
        if _dist(s0[0], s1[0]) < _dist(s0[-1], s1[0]):
            ordered[0] = list(reversed(s0))

    current_end = ordered[0][-1]
    for i in range(1, len(ordered)):
        s = ordered[i]
        if _dist(current_end, s[-1]) < _dist(current_end, s[0]):
            ordered[i] = list(reversed(s))
        current_end = ordered[i][-1]

    return ordered
