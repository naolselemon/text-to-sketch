"""
Sigma-Lognormal Kinematics.

Adds realistic human hand dynamics to raw vectorised strokes.  
""" 

from __future__ import annotations

import numpy as np
from scipy.stats import lognorm


def generate_kinematics(
    strokes: list[list[tuple[int, int]]],
    min_duration: float = 0.10,
    delay_between_strokes: float = 0.15,
    sigma: float = 0.50,
    sample_rate_hz: float = 60.0,
) -> list[list[list[float]]]:
    """Apply Sigma-Lognormal kinematics to a list of ordered strokes."""

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

        if D > 0 and len(timestamps) >= 2 and timestamps[-1] > timestamps[0]:
            dt = 1.0 / sample_rate_hz
            # Generate uniform time steps
            t_target = np.arange(timestamps[0], timestamps[-1], dt)
            
            # Ensure the final point is precisely included
            if len(t_target) == 0 or t_target[-1] < timestamps[-1] - 1e-6:
                t_target = np.append(t_target, timestamps[-1])
            
            # Linearly interpolate X and Y over time
            x_res = np.interp(t_target, timestamps, pts[:, 0])
            y_res = np.interp(t_target, timestamps, pts[:, 1])
            
            stroke_timed = [
                [float(x_res[j]), float(y_res[j]), float(t_target[j])]
                for j in range(len(t_target))
            ]
        else:
            stroke_timed = [
                [float(pts[i, 0]), float(pts[i, 1]), float(timestamps[i])]
                for i in range(n)
            ]
        timed_strokes.append(stroke_timed)
        current_time += T_stroke + delay_between_strokes

    return timed_strokes
