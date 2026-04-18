"""
Evaluate Tok-Dict Encoder/Decoder.

Loads a stroke-5 array, encodes it to tokens, decodes it back to stroke-5,
and compares the reconstructed result against the original.

"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.tokdict.encoder import encode_stroke5
from pipeline.tokdict.decoder import decode_tokens
from pipeline.utils.io import load_codebook, load_stroke5


def main() -> None:
    stroke5_dir = PROJECT_ROOT / "data" / "processed" / "stroke5"
    codebook_path = PROJECT_ROOT / "data" / "processed" / "tokdict" / "codebook.npy"

    if not codebook_path.exists():
        print(f"Error: Codebook not found at {codebook_path}")
        print("Please run scripts/run_pipeline.py first to build the codebook.")
        sys.exit(1)

    codebook = load_codebook(codebook_path)
    
    available_sketches = list(stroke5_dir.glob("*.npz"))
    if not available_sketches:
        print(f"Error: No stroke-5 files found in {stroke5_dir}")
        sys.exit(1)

    # Pick a random sketch to evaluate
    sketch_path = random.choice(available_sketches)
    
    print("\n" + "=" * 60)
    print("          TOK-DICT ENCODER EVALUATION")
    print("=" * 60)
    print(f"Evaluating sketch: {sketch_path.name}")
    print(f"Using codebook K : {len(codebook)}")
    print("-" * 60)

    # 1. Load original
    original_s5 = load_stroke5(sketch_path)
    n_points = len(original_s5)

    # 2. Encode
    tokens = encode_stroke5(original_s5, codebook)

    # 3. Decode (reverse engineer)
    reconstructed_s5 = decode_tokens(tokens, codebook)

    # 4. Compare
    # Pen states (columns 2, 3, 4) should be exactly preserved
    pen_states_match = np.array_equal(original_s5[:, 2:], reconstructed_s5[:, 2:])
    
    # Motion (columns 0, 1) - calculate errors only for p1 == 1 (drawing points)
    drawing_mask = original_s5[:, 2] == 1.0
    
    orig_motion = original_s5[drawing_mask, :2]
    recon_motion = reconstructed_s5[drawing_mask, :2]

    # Calculate errors
    abs_errors = np.abs(orig_motion - recon_motion)
    mae_x = np.mean(abs_errors[:, 0])
    mae_y = np.mean(abs_errors[:, 1])
    max_err_x = np.max(abs_errors[:, 0])
    max_err_y = np.max(abs_errors[:, 1])

    print("SUMMARY")
    print(f"  Total sequence length : {n_points} points")
    print(f"  Drawing points (p1=1) : {drawing_mask.sum()} points")
    print(f"  Pen states preserved  : {'✅ YES' if pen_states_match else '❌ NO'}")
    
    print("\nQUANTIZATION ERROR (Motion)")
    print(f"  Mean Absolute Error   : dx={mae_x:.4f} px, dy={mae_y:.4f} px")
    print(f"  Max Absolute Error    : dx={max_err_x:.4f} px, dy={max_err_y:.4f} px")
    
    print("\nSAMPLE COMPARISON (First 5 points)")
    print("  idx |       Original (dx, dy, p1, p2, p3)      |    Reconstructed (dx, dy, p1, p2, p3)  ")
    print("-" * 90)
    
    for i in range(min(5, n_points)):
        o = original_s5[i]
        r = reconstructed_s5[i]
        orig_str = f"{o[0]:6.2f}, {o[1]:6.2f}, {int(o[2])}, {int(o[3])}, {int(o[4])}"
        recon_str = f"{r[0]:6.2f}, {r[1]:6.2f}, {int(r[2])}, {int(r[3])}, {int(r[4])}"
        print(f"  {i:3} | [{orig_str}] | [{recon_str}]")

    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
