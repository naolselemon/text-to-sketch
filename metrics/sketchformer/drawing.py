"""Complete token evidence and continuous, target-normalized drawing scoring."""

from __future__ import annotations

import numpy as np

from utils.tokenizer import decode_tokens

COMPLETENESS_RULE_VERSION = "complete-drawing-v1"
SCORING_RULE_VERSION = "geometry-f1-continuous-clipping-v1"
FAILURE_ORDER = (
    "missing_eos", "generation_limit_hit", "tokens_after_eos", "invalid_token",
    "nonfinite_geometry", "empty_output",
)


def token_layout(codebook, **overrides):
    size = len(codebook)
    return {**dict(codebook_size=size, motion_token_offset=1, pad_token_id=0,
                sep_token_id=size + 1, sos_token_id=size + 2,
                eos_token_id=size + 3, vocab_size=size + 4), **overrides}


def validation_tokens(tokens, *, eos_token_id, pad_token_id):
    """Discard only a trailing PAD suffix after EOS, never internal evidence."""
    array = np.asarray(tokens)
    end = len(array)
    if np.any(array == eos_token_id):
        while end and array[end - 1] == pad_token_id:
            end -= 1
    return array[:end]


def decode_drawing(tokens, codebook, layout):
    return decode_tokens(np.asarray(tokens), codebook, **{
        k: layout[k] for k in ("motion_token_offset", "pad_token_id", "sep_token_id",
                               "sos_token_id", "eos_token_id")
    })


def validate_drawing(tokens, codebook, *, generation_limit, layout=None):
    """Validate the full raw row independently of the generator's lengths."""
    if isinstance(generation_limit, bool) or not isinstance(generation_limit, int) or generation_limit < 2:
        raise ValueError("generation_limit must be an integer >= 2")
    layout = layout or token_layout(codebook)
    raw = np.asarray(tokens)
    if raw.ndim != 1:
        raise ValueError("Drawing tokens must be one dimensional")
    raw = validation_tokens(raw, eos_token_id=layout["eos_token_id"], pad_token_id=layout["pad_token_id"])
    bad = set()
    reasons = set()
    safe = []
    for i, value in enumerate(raw):
        if not np.isfinite(value) or int(value) != value or not 0 <= value < layout["vocab_size"]:
            bad.add(i)
            reasons.add("invalid_token")
            safe.append(-1)
        else:
            safe.append(int(value))
    eos_positions = [i for i, value in enumerate(safe) if value == layout["eos_token_id"]]
    eos = eos_positions[0] if eos_positions else None
    if eos is None:
        reasons.add("missing_eos")
    if len(raw) > generation_limit or (eos is None and len(raw) >= generation_limit):
        reasons.add("generation_limit_hit")
        bad.update(range(generation_limit, len(raw)))
    if eos is not None:
        after = {i for i in range(eos + 1, len(safe)) if safe[i] != layout["pad_token_id"]}
        if after:
            bad.update(after)
            reasons.add("tokens_after_eos")
    active_end = len(safe) if eos is None else eos + 1
    for i in range(active_end):
        if safe[i] == layout["pad_token_id"] or (safe[i] == layout["sos_token_id"] and i != 0):
            bad.add(i)
            reasons.add("invalid_token")
    if not safe or safe[0] != layout["sos_token_id"]:
        reasons.add("invalid_token")
        if safe:
            bad.add(0)
    with np.errstate(over="ignore", invalid="ignore"):
        stroke = decode_drawing(safe[:active_end], codebook, layout)
        points = np.cumsum(stroke[:, :2], axis=0)
    if not np.isfinite(points).all():
        reasons.add("nonfinite_geometry")
        # Count source positions with nonfinite motion vectors, not invented EOS positions.
        for i, value in enumerate(safe[:active_end]):
            index = value - layout["motion_token_offset"]
            if 0 <= index < len(codebook) and not np.isfinite(codebook[index]).all():
                bad.add(i)
    else:
        from metrics.sketchformer.free_running import _stroke_lines
        if not any(np.linalg.norm(np.diff(line, axis=0), axis=1).sum() > 0 for line in _stroke_lines(stroke)):
            reasons.add("empty_output")
    return {
        "complete": not reasons,
        "failure_reasons": [r for r in FAILURE_ORDER if r in reasons],
        "invalid_step_count": len(bad),
        "generation_limit": generation_limit,
        "ended_at_limit": eos == generation_limit - 1,
    }


def target_canvas_transform(target_lines):
    """The existing geometry F1 transform, shared with the clipping penalty."""
    points = np.concatenate(target_lines, axis=0)
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    scale = 220.0 / max(float(np.max(maximum - minimum)), 1e-6)
    offset = np.asarray([18.0, 18.0], dtype=np.float32) - minimum * scale
    return scale, offset


def segment_inside_fraction(start, end):
    """Liang-Barsky clipping: fraction of continuous segment inside the canvas."""
    lo, hi = 0.0, 1.0
    delta = end - start
    for axis in range(2):
        if delta[axis] == 0:
            if not 0 <= start[axis] <= 255:
                return 0.0
        else:
            a, b = (0 - start[axis]) / delta[axis], (255 - start[axis]) / delta[axis]
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
            if hi < lo:
                return 0.0
    return max(0.0, hi - lo)


def out_of_canvas_fraction(target, prediction):
    from metrics.sketchformer.free_running import _stroke_lines
    target_lines, lines = _stroke_lines(target), _stroke_lines(prediction)
    if not target_lines or not lines:
        return 0.0
    scale, offset = target_canvas_transform(target_lines)
    total, outside = 0.0, 0.0
    for line in lines:
        points = line.astype(np.float64) * scale + offset
        for start, end in zip(points[:-1], points[1:]):
            length = float(np.linalg.norm(end - start))
            total += length
            outside += length * (1 - segment_inside_fraction(start, end))
    return min(1.0, max(0.0, outside / total)) if total else 0.0


def score_drawing(tokens, target_tokens, codebook, *, generation_limit, layout=None):
    from metrics.sketchformer.free_running import _stroke_f1
    layout = layout or token_layout(codebook)
    result = validate_drawing(tokens, codebook, generation_limit=generation_limit, layout=layout)
    target = decode_drawing(target_tokens, codebook, layout)
    # Decoding remains diagnostic, including syntactically invalid but finite output.
    safe = [int(t) for t in tokens if np.isfinite(t) and int(t) == t]
    prediction = decode_drawing(safe, codebook, layout)
    finite = np.isfinite(prediction).all() and np.isfinite(np.cumsum(prediction[:, :2], axis=0)).all()
    f1 = _stroke_f1(target, prediction, tolerance_px=2.0) if finite else 0.0
    outside = out_of_canvas_fraction(target, prediction) if finite else 0.0
    result.update(geometry_f1_2px=f1, out_of_canvas_fraction=outside,
                  drawing_score=f1 * (1 - outside) if result["complete"] else 0.0,
                  scoring_rule_version=SCORING_RULE_VERSION)
    return result
