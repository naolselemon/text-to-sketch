"""
Step 1 — Vectorization + RDP Simplification.

Converts a binary line-art image into a list of simplified vector strokes
using OpenCV contour detection followed by the Ramer-Douglas-Peucker (RDP)
algorithm (cv2.approxPolyDP).

Public API
----------
    vectorize_image(image_path, epsilon_factor=0.002)
        -> list[list[tuple[int, int]]]
"""

from pathlib import Path

import cv2


def vectorize_image(
    image_path: str | Path,
    epsilon_factor: float = 0.002,
) -> list[list[tuple[int, int]]]:
    """Vectorize a lineart image into a list of simplified strokes.

    Parameters
    ----------
    image_path : str or Path
        Path to the input lineart (grayscale) image.
    epsilon_factor : float
        Controls RDP simplification aggressiveness.  Higher ⇒ simpler strokes.
        Default 0.002 (≈ 0.2 % of the image's longest dimension).

    Returns
    -------
    list[list[tuple[int, int]]]
        Each element is a stroke: an ordered list of (x, y) pixel coordinates.

    Raises
    ------
    ValueError
        If the image cannot be read.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Invert so drawn lines become white (255) for findContours.
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    max_dim = max(img.shape)
    epsilon = epsilon_factor * max_dim

    strokes: list[list[tuple[int, int]]] = []
    for contour in contours:
        if len(contour) < 2:
            continue
        approx = cv2.approxPolyDP(contour, epsilon, closed=False)
        points = [tuple(pt[0]) for pt in approx]
        if len(points) > 1:
            strokes.append(points)

    return strokes
