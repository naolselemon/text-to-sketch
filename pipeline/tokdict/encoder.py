"""
Tok-Dict Encoder.

Encodes a stroke-5 array into a sequence of discrete token indices using a
K-means codebook built by ``pipeline.tokdict.builder``.

Special tokens (appended after K regular motion tokens):
    K     →  pen-lift          (p2 == 1)
    K + 1 →  end-of-sketch     (p3 == 1)
"""

from __future__ import annotations

import numpy as np


# Encoder

def encode_stroke5(
    stroke5: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    """Map each stroke-5 row to a discrete token index."""
    K = len(codebook)
    tokens = np.empty(len(stroke5), dtype=np.int32)

    for i, row in enumerate(stroke5):
        p3 = row[4]
        p2 = row[3]

        if p3 == 1.0:
            tokens[i] = K + 1          # end-of-sketch
        elif p2 == 1.0:
            tokens[i] = K              # pen-lift
        else:
            # Nearest centroid by L2 distance.
            delta = codebook - row[:2]
            tokens[i] = int(np.argmin((delta * delta).sum(axis=1)))

    return tokens
