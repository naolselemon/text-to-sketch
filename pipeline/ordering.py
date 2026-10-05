"""
Step 2 — Stroke Ordering.

Imposes a drawing order on an unordered set of vectorised stroke paths.

Three ordering strategies are provided:

    order_directional_bias(strokes)      – top-left → bottom-right (default)
    order_greedy_nearest_neighbor(strokes) – minimise pen-travel greedily
    order_tsp(strokes)                   – global TSP approximation via NetworkX
    order_continuity_greedy(strokes)     – join centerline branches smoothly

All functions accept and return ``list[list[tuple[int, int]]]`` — a list of
strokes, where each stroke is an ordered list of (x, y) pixel coordinates.
"""

from __future__ import annotations

import math
from collections import defaultdict

import networkx as nx
import numpy as np

from pipeline.vectorization import CenterlineBranch



# Shared geometry


def _dist(pt1: tuple, pt2: tuple) -> float:
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


def _turn_cost(previous: tuple, junction: tuple, following: tuple) -> float:
    """Return zero for straight continuation and two for a full reversal."""

    incoming = np.asarray(junction, dtype=float) - np.asarray(previous, dtype=float)
    outgoing = np.asarray(following, dtype=float) - np.asarray(junction, dtype=float)
    denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
    if denominator == 0:
        return 1.0
    cosine = float(np.clip(np.dot(incoming, outgoing) / denominator, -1.0, 1.0))
    return 1.0 - cosine


def order_continuity_greedy(
    strokes: list[list[tuple[int, int]]] | list[CenterlineBranch],
    *,
    junction_tolerance: float = 2.0,
) -> list[list[tuple[int, int]]]:
    """Join skeleton branches through junctions while preserving smooth direction.

    Skan returns graph branches between endpoints and junctions. This routine
    pairs the smoothest continuation at a shared junction, then orders the
    remaining strokes by nearest endpoint. T/Y branches remain separate once
    the main continuation has consumed the junction.
    """

    if strokes and isinstance(strokes[0], CenterlineBranch):
        return order_continuity_topology(strokes)

    remaining = [list(stroke) for stroke in strokes if len(stroke) >= 2]
    if not remaining:
        return []

    remaining.sort(
        key=lambda stroke: (
            -len(stroke),
            min(point[1] for point in stroke),
            min(point[0] for point in stroke),
        )
    )
    ordered: list[list[tuple[int, int]]] = []

    while remaining:
        active = remaining.pop(0)
        if (active[-1][1], active[-1][0]) < (active[0][1], active[0][0]):
            active.reverse()

        while remaining:
            best: tuple[float, int, bool] | None = None
            for index, candidate in enumerate(remaining):
                for reverse in (False, True):
                    oriented = candidate[::-1] if reverse else candidate
                    distance = _dist(active[-1], oriented[0])
                    if distance > junction_tolerance:
                        continue
                    turn = _turn_cost(active[-2], active[-1], oriented[1])
                    score = turn * 10.0 + distance
                    proposal = (score, index, reverse)
                    if best is None or proposal < best:
                        best = proposal

            if best is None:
                break
            _, index, reverse = best
            candidate = remaining.pop(index)
            if reverse:
                candidate.reverse()
            if candidate[0] == active[-1]:
                active.extend(candidate[1:])
            else:
                active.extend(candidate)

        ordered.append(active)
        if remaining:
            current_end = active[-1]
            remaining.sort(
                key=lambda stroke: min(
                    _dist(current_end, stroke[0]),
                    _dist(current_end, stroke[-1]),
                )
            )
            if _dist(current_end, remaining[0][-1]) < _dist(current_end, remaining[0][0]):
                remaining[0].reverse()

    return ordered


# Strategy D – Topology-Aware Continuity


def _assemble_topology_chains(
    branches: list[CenterlineBranch],
) -> list[list[tuple[int, int]]]:
    """Join skeleton branches using exact skeleton graph topology into continuous strokes."""
    if not branches:
        return []

    node_to_branches: dict[int, list[CenterlineBranch]] = defaultdict(list)
    for branch in branches:
        node_to_branches[branch.start_node_id].append(branch)
        node_to_branches[branch.end_node_id].append(branch)

    used: set[int] = set()
    strokes: list[list[tuple[int, int]]] = []

    for branch in branches:
        if branch.is_loop:
            strokes.append(list(branch.points))
            used.add(branch.branch_id)

    for branch in branches:
        if branch.branch_id in used:
            continue
        used.add(branch.branch_id)
        chain = list(branch.points)
        chain_start_node = branch.start_node_id
        chain_end_node = branch.end_node_id

        while True:
            candidates = [
                b for b in node_to_branches[chain_end_node]
                if b.branch_id not in used
            ]
            if not candidates:
                break
            best = _pick_smooth_forward(chain, candidates, chain_end_node)
            if best is None:
                break
            used.add(best.branch_id)
            if best.start_node_id == chain_end_node:
                chain.extend(best.points[1:])
                chain_end_node = best.end_node_id
            else:
                chain.extend(best.points[:-1][::-1])
                chain_end_node = best.start_node_id

        while True:
            candidates = [
                b for b in node_to_branches[chain_start_node]
                if b.branch_id not in used
            ]
            if not candidates:
                break
            best = _pick_smooth_backward(chain, candidates, chain_start_node)
            if best is None:
                break
            used.add(best.branch_id)
            if best.end_node_id == chain_start_node:
                chain = best.points[:-1] + chain
                chain_start_node = best.start_node_id
            else:
                chain = best.points[1:][::-1] + chain
                chain_start_node = best.end_node_id

        strokes.append(chain)

    return strokes



def _order_outer_to_inner_chains(
    strokes: list[list[tuple[int, int]]],
    outer_band_fraction: float = 0.20,
    local_threshold: float = 40.0,
    inner_junction_tolerance: float | None = None,
) -> list[list[tuple[int, int]]]:
    """Order continuous chains from outermost concentric perimeter inward.

    Uses layer clustering and adaptive proximity thresholds:
    1. Pre-computes sketch centroid and partitions strokes into discrete concentric
       radial shell clusters from outside in.
    2. Priority 1: If an outer-layer stroke is within `local_threshold`, stay on the
       outer shell and draw it.
    3. Priority 2: If all remaining outer strokes in the active layer are far (> `local_threshold`),
       check for nearby strokes in inner layers (distance <= `inner_threshold`). If found,
       transition smoothly to the closest inner stroke (breaking ties by outer radius)
       to maintain continuous local flow and avoid wasteful cross-canvas jumps.
    4. Priority 3: After drawing that stroke, the new pen position is checked again against
       the active outer layer: if an outer stroke is now within `local_threshold`, the pen
       snaps back to complete the outer shell before advancing.
    5. Fallback: If no stroke is nearby in either layer, fly across to the closest stroke
       in the active outer layer.
    """
    if len(strokes) <= 1:
        return strokes

    all_points = [pt for s in strokes for pt in s]
    n_pts = len(all_points)
    cx = sum(pt[0] for pt in all_points) / n_pts
    cy = sum(pt[1] for pt in all_points) / n_pts

    # 1. Precalculate stroke center distances and endpoints once
    stroke_items = []
    for s in strokes:
        r = sum(math.hypot(pt[0] - cx, pt[1] - cy) for pt in s) / len(s)
        stroke_items.append({
            "stroke": list(s),
            "radius": r,
            "start": s[0],
            "end": s[-1],
        })

    radii = [item["radius"] for item in stroke_items]
    max_r, min_r = max(radii), min(radii)
    r_span = max(1.0, max_r - min_r)

    # 2. Cluster into discrete concentric radial layers up front
    n_layers = max(1, int(round(1.0 / outer_band_fraction)))
    layer_width = r_span * outer_band_fraction
    layers: list[list[dict[str, Any]]] = [[] for _ in range(n_layers)]
    for item in stroke_items:
        idx = int((max_r - item["radius"]) / max(0.001, layer_width))
        idx = min(n_layers - 1, max(0, idx))
        item["layer"] = idx
        layers[idx].append(item)

    # Sort outermost layer by radius to pick initial outermost stroke
    layers[0].sort(key=lambda x: x["radius"], reverse=True)
    ordered: list[list[tuple[int, int]]] = []

    # Pick outermost stroke first from the first non-empty outer layer
    active_layer_idx = 0
    while active_layer_idx < n_layers and not layers[active_layer_idx]:
        active_layer_idx += 1

    first = layers[active_layer_idx].pop(0)
    active = first["stroke"]
    # Orient outside-in
    d_start_sq = (active[0][0] - cx) ** 2 + (active[0][1] - cy) ** 2
    d_end_sq = (active[-1][0] - cx) ** 2 + (active[-1][1] - cy) ** 2
    if d_end_sq > d_start_sq:
        active.reverse()
    ordered.append(active)

    local_thresh_sq = local_threshold ** 2
    inner_tol = local_threshold if inner_junction_tolerance is None else inner_junction_tolerance
    inner_tol_sq = inner_tol ** 2

    def pen_dist_sq(item: dict[str, Any], curr_x: int, curr_y: int) -> float:
        s_x, s_y = item["start"]
        e_x, e_y = item["end"]
        d1 = (curr_x - s_x) ** 2 + (curr_y - s_y) ** 2
        d2 = (curr_x - e_x) ** 2 + (curr_y - e_y) ** 2
        return d1 if d1 < d2 else d2

    remaining_count = sum(len(l) for l in layers)

    while remaining_count > 0:
        curr_x, curr_y = ordered[-1][-1]

        # Advance active layer if current outer layer is exhausted
        while active_layer_idx < n_layers and not layers[active_layer_idx]:
            active_layer_idx += 1

        if active_layer_idx >= n_layers:
            break

        current_outer_layer = layers[active_layer_idx]

        # Constrained search: search ONLY within the active outer layer
        best_outer_idx = min(
            range(len(current_outer_layer)),
            key=lambda i: pen_dist_sq(current_outer_layer[i], curr_x, curr_y),
        )
        best_outer_item = current_outer_layer[best_outer_idx]
        best_outer_d_sq = pen_dist_sq(best_outer_item, curr_x, curr_y)

        # Priority 1: If an outer-layer stroke is within local_threshold, stay on outer shell
        if best_outer_d_sq <= local_thresh_sq:
            chosen_layer_idx = active_layer_idx
            chosen_idx = best_outer_idx
        else:
            # Priority 2: Outer strokes in active layer are far (> local_threshold).
            # Check for nearby strokes in inner layers (within inner_tol)
            touching_inner: list[tuple[float, float, int, int]] = []
            for l_idx in range(active_layer_idx + 1, n_layers):
                for item_idx, inner_item in enumerate(layers[l_idx]):
                    d_sq = pen_dist_sq(inner_item, curr_x, curr_y)
                    if d_sq <= inner_tol_sq:
                        touching_inner.append((d_sq, -inner_item["radius"], l_idx, item_idx))

            if touching_inner:
                # Pick the nearby inner stroke with closest endpoint, breaking ties by outer radius
                touching_inner.sort()
                _, _, chosen_layer_idx, chosen_idx = touching_inner[0]
            else:
                # Priority 3 (Fallback): No local continuation available; fly to closest outer stroke
                chosen_layer_idx = active_layer_idx
                chosen_idx = best_outer_idx

        # Pop from chosen layer
        chosen = layers[chosen_layer_idx].pop(chosen_idx)
        remaining_count -= 1
        next_stroke = chosen["stroke"]

        # Orient next stroke smoothly from current_end
        s_x, s_y = chosen["start"]
        e_x, e_y = chosen["end"]
        d_end = (curr_x - e_x) ** 2 + (curr_y - e_y) ** 2
        d_start = (curr_x - s_x) ** 2 + (curr_y - s_y) ** 2
        if d_end < d_start:
            next_stroke.reverse()

        ordered.append(next_stroke)

    return ordered


def order_continuity_topology(
    branches: list[CenterlineBranch],
) -> list[list[tuple[int, int]]]:
    """Order strokes using skeleton graph topology for true connectivity.

    Branches are joined only when their endpoints share the same skeleton
    graph-node ID.  At junctions, the smoothest continuation is chosen via
    turn cost.  Disconnected components remain separate strokes; endpoint
    distance is used only to order those separate strokes, never to join them.
    Closed loops are preserved as complete strokes.
    """
    if not branches:
        return []

    strokes = _assemble_topology_chains(branches)
    if len(strokes) > 1:
        strokes = _order_strokes_by_distance(strokes)
    return strokes


# Strategy E – Outer-to-Inner Continuity (Radial Perimeter to Core)


def order_outer_to_inner(
    branches: list[CenterlineBranch],
    *,
    outer_band_fraction: float = 0.20,
    local_threshold: float = 40.0,
    inner_junction_tolerance: float | None = None,
) -> list[list[tuple[int, int]]]:
    """Order strokes from outermost contours inward using skeleton topology.

    1. Joins connected skeleton branches into continuous curves and closed loops
       using skeleton graph node IDs (topology-aware chain assembly).
    2. Imposes concentric radial shell outer-to-inner ordering on the assembled
       chains with adaptive thresholding: follows touching inner branches when
       outer strokes are far, snapping back to outer strokes whenever nearby.
    """
    if not branches:
        return []

    chains = _assemble_topology_chains(branches)

    return _order_outer_to_inner_chains(
        chains,
        outer_band_fraction=outer_band_fraction,
        local_threshold=local_threshold,
        inner_junction_tolerance=inner_junction_tolerance,
    )


def _pick_smooth_forward(
    chain: list[tuple[int, int]],
    candidates: list[CenterlineBranch],
    shared_node: int,
) -> CenterlineBranch | None:
    """Select the smoothest continuation from the end of a chain."""

    if len(chain) < 2:
        return candidates[0] if candidates else None

    best_branch: CenterlineBranch | None = None
    best_score = float("inf")

    for branch in candidates:
        if branch.start_node_id == shared_node:
            oriented = branch.points
        else:
            oriented = branch.points[::-1]

        if len(oriented) < 2:
            continue

        turn = _turn_cost(chain[-2], chain[-1], oriented[1])
        if turn < best_score:
            best_score = turn
            best_branch = branch

    return best_branch


def _pick_smooth_backward(
    chain: list[tuple[int, int]],
    candidates: list[CenterlineBranch],
    shared_node: int,
) -> CenterlineBranch | None:
    """Select the smoothest continuation from the start of a chain."""

    if len(chain) < 2:
        return candidates[0] if candidates else None

    best_branch: CenterlineBranch | None = None
    best_score = float("inf")

    for branch in candidates:
        if branch.end_node_id == shared_node:
            oriented = branch.points
        else:
            oriented = branch.points[::-1]

        if len(oriented) < 2:
            continue

        turn = _turn_cost(oriented[-2], chain[0], chain[1])
        if turn < best_score:
            best_score = turn
            best_branch = branch

    return best_branch


def _order_strokes_by_distance(
    strokes: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    """Order separate strokes by nearest endpoint (pen-up transitions only)."""

    remaining = list(range(len(strokes)))
    first = remaining.pop(0)
    ordered_indices = [first]

    while remaining:
        current_end = strokes[ordered_indices[-1]][-1]
        best_idx = min(
            remaining,
            key=lambda i: min(
                _dist(current_end, strokes[i][0]),
                _dist(current_end, strokes[i][-1]),
            ),
        )
        remaining.remove(best_idx)
        ordered_indices.append(best_idx)

    return [strokes[i] for i in ordered_indices]


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

    graph = nx.complete_graph(n)
    for i, j in graph.edges:
        graph[i][j]["weight"] = _dist(centers[i], centers[j])
    cycle = nx.approximation.greedy_tsp(graph, source=0, weight="weight")
    tour = cycle[:-1] if len(cycle) > 1 and cycle[0] == cycle[-1] else cycle
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
