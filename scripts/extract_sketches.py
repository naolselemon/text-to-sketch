"""
CLI entry-point: apply LineartAnimeDetector to raw anime images and save
binary line-art sketches under data/processed/sketches/.

Options
-------
    --input-dir DIR        Source directory containing raw anime images.
                           (default: $INPUT_DIR or data/raw/data/anime_images)
    --output-dir DIR       Destination for extracted sketches.
                           (default: $OUTPUT_DIR or data/processed/sketches)
    --detect-resolution N  Detector input resolution (default: 512).
    --image-resolution N   Output image resolution (default: 512).
    --max-per-folder N     Max images to process per sub-directory (default: 100).
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.steps.step_a_lineart import (
    collect_images,
    process_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract lineart sketches from anime images using LineartAnimeDetector."
    )
    parser.add_argument(
        "--input-dir",
        default=os.getenv(
            "INPUT_DIR",
            str(PROJECT_ROOT / "data" / "raw" / "data" / "anime_images"),
        ),
        help="Directory containing raw anime images.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "OUTPUT_DIR",
            str(PROJECT_ROOT / "data" / "processed" / "sketches"),
        ),
        help="Directory where extracted sketches will be saved.",
    )
    parser.add_argument(
        "--detect-resolution",
        type=int,
        default=int(os.getenv("DETECT_RES", "512")),
        help="Resolution passed to the detector (default: 512).",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=int(os.getenv("IMAGE_RES", "512")),
        help="Output image resolution (default: 512).",
    )
    parser.add_argument(
        "--max-per-folder",
        type=int,
        default=int(os.getenv("MAX_PER_FOLDER", "100")),
        help="Maximum number of images to process per sub-directory (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_root  = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    if not input_root.exists():
        print(f"[ERROR] Input directory not found: {input_root}", file=sys.stderr)
        sys.exit(1)

    # Load model once.
    print("[init] Loading LineartAnimeDetector …")
    from controlnet_aux import LineartAnimeDetector  # lazy import
    detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    print("[init] Model loaded.")

    print(f"[init] Scanning {input_root} …")
    print(f"[init] Limit: {args.max_per_folder} images per folder.")
    pairs = collect_images(input_root, output_root, max_per_folder=args.max_per_folder)
    total = len(pairs)
    print(f"[init] {total} images to process (already-done files are skipped).")

    if total == 0:
        print("[done] Nothing to do.")
        return

    ok = skipped = 0
    with tqdm(pairs, unit="img", dynamic_ncols=True) as bar:
        for src, dst in bar:
            bar.set_postfix_str(src.name)
            success = process_image(
                src, dst, detector,
                args.detect_resolution,
                args.image_resolution,
            )
            if success:
                ok += 1
            else:
                skipped += 1

    print(f"\n[done] Processed {ok}/{total} images ({skipped} skipped).")
    print(f"[done] Sketches saved to: {output_root}")


if __name__ == "__main__":
    main()
