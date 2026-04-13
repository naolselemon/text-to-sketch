"""
Step 3 — Sigma-Lognormal Kinematics.

Adds realistic human hand dynamics to raw vectorised strokes.

Public API
----------
    generate_kinematics(strokes, min_duration, delay_between_strokes, sigma)
        -> list[list[list[float]]]   # [x, y, t] per point
""" 

from __future__ import annotations

import numpy as np
from scipy.stats import lognorm


def generate_kinematics(
    strokes: list[list[tuple[int, int]]],
    min_duration: float = 0.10,
    delay_between_strokes: float = 0.15,
    sigma: float = 0.50,
) -> list[list[list[float]]]:
    """Apply Sigma-Lognormal kinematics to a list of ordered strokes.

    Each point gains a realistic timestamp derived from an inverse lognormal
    CDF mapping over cumulative arc-length — slow at stroke ends, fast in the
    middle — calibrated to QuickDraw human drawing data.

    Parameters
    ----------
    strokes : list of strokes, each a list of (x, y) tuples.
    min_duration : minimum stroke duration in seconds (default 0.10 s).
    delay_between_strokes : pen-lift time between consecutive strokes (0.15 s).
    sigma : lognormal spread (typical range 0.4–0.7; default 0.50).

    Returns
    -------
    list[list[list[float]]]
        Same structure as input but each point is ``[x, y, timestamp_seconds]``.
    """
    if not strokes:
        return []

    # Percentile bounds — avoids ±∞ at the absolute CDF tails.
    P_START, P_END = 0.01, 0.99

    timed_strokes: list[list[list[float]]] = []
    current_time = 0.0

    for stroke in strokes:
        pts = np.array(stroke, dtype=float)
        n = len(pts)

        if n < 2:
            timed_strokes.append([[pts[0, 0], pts[0, 1], current_time]])
            current_time += min_duration + delay_between_strokes
            continue

        # Cumulative arc-length along the stroke.
        diffs     = np.diff(pts, axis=0)
        seg_dists = np.linalg.norm(diffs, axis=1)
        cum_dists = np.concatenate(([0.0], np.cumsum(seg_dists)))
        D         = cum_dists[-1]

        # Duration scales sub-linearly with length (QuickDraw empirical fit).
        T_stroke = max(min_duration, 0.04 * (D ** 0.5))

        if D == 0:
            timestamps = np.full(n, current_time)
        else:
            # Map arc-fraction → CDF percentile → lognormal time → normalize.
            s_frac = cum_dists / D
            p      = P_START + s_frac * (P_END - P_START)
            tau    = lognorm.ppf(p, s=sigma)

            tau_range = tau[-1] - tau[0]
            tau_norm  = (tau - tau[0]) / tau_range if tau_range > 0 else np.zeros(n)
            timestamps = current_time + tau_norm * T_stroke

        stroke_timed = [
            [float(pts[i, 0]), float(pts[i, 1]), float(timestamps[i])]
            for i in range(n)
        ]
        timed_strokes.append(stroke_timed)
        current_time += T_stroke + delay_between_strokes

    return timed_strokes
