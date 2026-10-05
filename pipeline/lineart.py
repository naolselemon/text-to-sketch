"""Stage 1: anime image to grayscale line-art extraction."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LINEART_ANIME = "lineart-anime"
ANIME2SKETCH = "anime2sketch"
SUPPORTED_EXTRACTORS = (LINEART_ANIME, ANIME2SKETCH)


class SketchExtractor(Protocol):
    name: str

    def extract(self, src: Path) -> Image.Image:
        """Extract an 8-bit grayscale sketch from *src*."""


def collect_images(
    input_root: Path,
    output_root: Path,
    max_images: int | None,
) -> list[tuple[Path, Path]]:
    """Collect source/destination image pairs for line-art extraction.

    The output path mirrors the source folder layout and always uses ``.png``.
    Existing outputs are skipped before the limit is applied, so the command can
    be resumed safely against a flat portraits folder.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    limit = None if max_images is None or max_images <= 0 else max_images

    pairs: list[tuple[Path, Path]] = []
    for src in sorted(input_root.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = src.relative_to(input_root).with_suffix(".png")
        dst = output_root / relative_path
        if dst.exists():
            continue

        pairs.append((src, dst))
        if limit is not None and len(pairs) >= limit:
            break

    return pairs


def extractor_output_name(extractor: str) -> str:
    return extractor.replace("-", "_")


def autocrop_black_borders(image: "Image.Image", threshold: float = 12.0) -> "Image.Image":
    """Detect and crop solid black pillarbox/letterbox padding on all four sides.

    Converts the image to grayscale, computes per-row and per-column mean
    brightness, and crops to the bounding box of pixels whose mean exceeds
    ``threshold``.  Images without detectable padding (margin <= 2px) are
    returned unchanged.

    Args:
        image: RGB or RGBA PIL image to crop.
        threshold: Pixel-brightness mean below which a row/column is
            considered solid-black padding (0-255 scale).

    Returns:
        Cropped PIL image, or the original image if no padding was detected.
    """
    import numpy as np

    gray = image.convert("L")
    np_img = np.asarray(gray, dtype=np.float32)

    col_means = np_img.mean(axis=0)
    valid_cols = np.where(col_means > threshold)[0]
    row_means = np_img.mean(axis=1)
    valid_rows = np.where(row_means > threshold)[0]

    if len(valid_cols) == 0 or len(valid_rows) == 0:
        return image

    left, right = int(valid_cols[0]), int(valid_cols[-1]) + 1
    top, bottom = int(valid_rows[0]), int(valid_rows[-1]) + 1

    if left > 2 or right < image.width - 2 or top > 2 or bottom < image.height - 2:
        return image.crop((left, top, right, bottom))
    return image


def create_extractor(
    extractor: str,
    detect_resolution: int,
    image_resolution: int,
    anime2sketch_dir: str | Path | None = None,
    anime2sketch_python: str | Path | None = None,
    anime2sketch_model: str = "default",
    anime2sketch_gpu_ids: str = "",
    anime2sketch_clahe_clip: float = -1.0,
) -> SketchExtractor:
    """Load a sketch extractor by name."""
    if extractor == LINEART_ANIME:
        return ControlNetLineartAnimeExtractor(
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
        )

    if extractor == ANIME2SKETCH:
        return Anime2SketchExtractor(
            repo_dir=anime2sketch_dir,
            python_executable=anime2sketch_python,
            load_size=image_resolution,
            model=anime2sketch_model,
            gpu_ids=anime2sketch_gpu_ids,
            clahe_clip=anime2sketch_clahe_clip,
        )

    supported = ", ".join(SUPPORTED_EXTRACTORS)
    raise ValueError(f"Unknown extractor '{extractor}'. Supported: {supported}")


class ControlNetLineartAnimeExtractor:
    name = LINEART_ANIME

    def __init__(self, detect_resolution: int, image_resolution: int) -> None:
        print("[init] Loading ControlNet LineartAnimeDetector ...")
        from controlnet_aux import LineartAnimeDetector  # lazy import

        self.detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        self.detect_resolution = detect_resolution
        self.image_resolution = image_resolution

    def extract(self, src: Path) -> Image.Image:
        with Image.open(src) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = autocrop_black_borders(image)
            detected = self.detector(
                image,
                detect_resolution=self.detect_resolution,
                image_resolution=self.image_resolution,
            )

        return _to_grayscale(detected)


class Anime2SketchExtractor:
    name = ANIME2SKETCH

    def __init__(
        self,
        repo_dir: str | Path | None,
        python_executable: str | Path | None,
        load_size: int,
        model: str,
        gpu_ids: str,
        clahe_clip: float,
    ) -> None:
        if not repo_dir:
            raise RuntimeError(
                "Anime2Sketch requires --anime2sketch-dir or ANIME2SKETCH_DIR "
                "pointing to a local Mukosame/Anime2Sketch checkout."
            )

        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.script_path = self.repo_dir / "test.py"
        if not self.script_path.exists():
            raise RuntimeError(f"Anime2Sketch test.py not found: {self.script_path}")

        self.python_executable = _resolve_python_executable(python_executable)
        self.load_size = load_size
        self.model = model
        self.gpu_ids = gpu_ids.strip()
        self.clahe_clip = clahe_clip

        print(f"[init] Using Anime2Sketch checkout: {self.repo_dir}")

    def extract(self, src: Path) -> Image.Image:
        with tempfile.TemporaryDirectory(prefix="anime2sketch_") as tmp_dir:
            output_dir = Path(tmp_dir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                self.python_executable,
                str(self.script_path),
                "--dataroot",
                str(src),
                "--load_size",
                str(self.load_size),
                "--output_dir",
                str(output_dir),
                "--model",
                self.model,
            ]
            if self.gpu_ids:
                cmd.extend(["--gpu_ids", self.gpu_ids])
            if self.clahe_clip > 0:
                cmd.extend(["--clahe_clip", str(self.clahe_clip)])

            try:
                subprocess.run(
                    cmd,
                    cwd=self.repo_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                details = stderr or stdout or f"exit code {exc.returncode}"
                raise RuntimeError(f"Anime2Sketch failed for {src}: {details}") from exc

            output_path = output_dir / src.name
            if not output_path.exists():
                candidates = sorted(
                    path
                    for path in output_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
                if not candidates:
                    raise RuntimeError(f"Anime2Sketch did not write an image for {src}")
                output_path = candidates[0]

            with Image.open(output_path) as image:
                return _to_grayscale(image)


def process_image(src: Path, dst: Path, extractor: SketchExtractor) -> bool:
    """Run a sketch extractor on one source image."""
    try:
        sketch = extractor.extract(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sketch.save(dst)
        return True
    except Exception as exc:
        print(f"[skip] {src}: {exc}")
        return False


def _resolve_python_executable(python_executable: str | Path | None) -> str:
    if not python_executable:
        return sys.executable

    python_text = str(python_executable)
    python_path = Path(python_text).expanduser()
    if python_path.is_absolute() or python_path.parent != Path("."):
        # Keep virtualenv executables as their venv path. Path.resolve() follows
        # the venv python symlink back to /usr/bin/python, losing site-packages.
        return str(python_path.absolute())

    return python_text


def _to_grayscale(image_like: Any) -> Image.Image:
    """Preserve line confidence for thresholding during vectorization."""

    image = _to_pil_image(image_like)
    return ImageOps.grayscale(image)


def _to_binary_grayscale(image_like: Any) -> Image.Image:
    """Compatibility helper for callers that still require legacy binary output."""

    gray = _to_grayscale(image_like)
    return gray.point(lambda pixel: 0 if pixel < 128 else 255, mode="L")


def _to_pil_image(image_like: Any) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like

    try:
        return Image.fromarray(image_like)
    except Exception as exc:
        image_type = type(image_like)
        raise TypeError(f"Detector returned unsupported image type: {image_type!r}") from exc
