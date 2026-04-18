

from __future__ import annotations

import numpy as np



def decode_tokens(
    tokens: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    """Approximate inverse of ``encode_stroke5``."""
    K = len(codebook)
    rows = np.zeros((len(tokens), 5), dtype=np.float32)

    for i, tok in enumerate(tokens):
        if tok == K + 1:
            rows[i, 4] = 1.0          # p3
        elif tok == K:
            rows[i, 3] = 1.0          # p2
        else:
            rows[i, :2] = codebook[tok]
            rows[i, 2]  = 1.0         # p1

    return rows