"""

Converts timed stroke data  ``[[x, y, t], ...]``  produced by Step 3 into
the stroke-5 format expected by Sketchformer:

    [Δx, Δy, p1, p2, p3]

where (Δx, Δy) are relative displacements to the previous point and the
three pen-state flags are mutually exclusive:

    p1 = 1  →  pen is drawing (mid-stroke)
    p2 = 1  →  last point of the current stroke; pen lifts next
    p3 = 1  →  end-of-sketch sentinel (appended once at the very end)

"""

from __future__ import annotations

import numpy as np


def to_stroke5(
    timed_strokes: list[list[list[float]]],
    canvas_size: int = 256,
) -> np.ndarray:
    """Convert timed strokes to stroke-5 format.

    Parameters
    ----------
    timed_strokes : list[list[list[float]]]
        Output of Step D.  Each element is a stroke; each point is
        ``[x, y, timestamp]`` in absolute pixel coordinates.
    canvas_size : int
        Unused directly — normalization is data-driven (bounding box).
        Kept as a parameter for future fixed-canvas workflows.

    Returns
    -------
    np.ndarray, shape (N + 1, 5), dtype float32
        N = total number of points across all strokes.
        The last row is always the end-of-sketch token ``[0, 0, 0, 0, 1]``.
    """
    if not timed_strokes:
        return np.zeros((1, 5), dtype=np.float32)

    # Collect all absolute (x, y) to compute a global bounding box.
    all_x: list[float] = []
    all_y: list[float] = []
    for stroke in timed_strokes:
        for pt in stroke:
            all_x.append(pt[0])
            all_y.append(pt[1])

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    x_range = max(x_max - x_min, 1.0)   # guard against zero-width sketches
    y_range = max(y_max - y_min, 1.0)
    scale   = max(x_range, y_range)      # uniform scale — preserves aspect ratio

    def _norm(x: float, y: float) -> tuple[float, float]:
        return (x - x_min) / scale, (y - y_min) / scale

    # Build stroke-5 rows.
    rows: list[list[float]] = []
    prev_x = prev_y = 0.0
    first  = True

    for stroke in timed_strokes:
        n_pts = len(stroke)
        for idx, pt in enumerate(stroke):
            nx, ny = _norm(pt[0], pt[1])

            # First point ever → delta is (0, 0).
            dx = 0.0 if first else nx - prev_x
            dy = 0.0 if first else ny - prev_y
            first = False

            is_last_in_stroke = (idx == n_pts - 1)
            if is_last_in_stroke:
                p1, p2, p3 = 0, 1, 0   # pen lifts after this point
            else:
                p1, p2, p3 = 1, 0, 0   # pen is drawing

            rows.append([dx, dy, float(p1), float(p2), float(p3)])
            prev_x, prev_y = nx, ny

    # End-of-sketch sentinel.
    rows.append([0.0, 0.0, 0.0, 0.0, 1.0])

    return np.array(rows, dtype=np.float32)
