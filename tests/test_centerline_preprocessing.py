from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from metrics.preprocessing.evaluate_centerline_preprocessing import (
    parse_extractor_dirs,
    select_common_samples,
)
from pipeline.ordering import order_outer_to_inner
from pipeline.lineart import _to_grayscale
from pipeline.stroke5 import stroke5_to_canvas_strokes, strokes_to_stroke5
from pipeline.vectorization import (
    centerline_metrics,
    rasterize_strokes,
    simplify_strokes,
    source_centerline,
    vectorize_image_with_stats,
)
from utils.tokenizer import decode_tokens, encode_stroke5, quantization_metrics


class CenterlinePreprocessingTest(unittest.TestCase):
    def test_empty_static_sketch_is_an_explicit_eos_sentinel(self) -> None:
        stroke5, _ = strokes_to_stroke5([], canvas_shape=(64, 64))

        self.assertEqual(stroke5.tolist(), [[0.0, 0.0, 0.0, 0.0, 1.0]])

    def test_lineart_preserves_grayscale_confidence(self) -> None:
        source = np.asarray([[0, 64, 128, 255]], dtype=np.uint8)

        result = np.asarray(_to_grayscale(source))

        self.assertEqual(result.tolist(), source.tolist())

    def test_thick_line_becomes_one_centerline_not_two_contours(self) -> None:
        image = np.full((64, 64), 255, dtype=np.uint8)
        cv2.line(image, (8, 32), (56, 32), 0, thickness=7)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thick.png"
            cv2.imwrite(str(path), image)
            centerline, stats = vectorize_image_with_stats(
                path,
                epsilon=0.5,
                method="centerline",
                threshold_profile="legacy",
            )
            contours, _ = vectorize_image_with_stats(
                path,
                epsilon=0.5,
                method="contour",
            )

        self.assertEqual(len(centerline), 1)
        self.assertLessEqual(stats.simplified_point_count, 4)
        self.assertGreater(sum(len(stroke) for stroke in contours), len(centerline[0]))
        self.assertLessEqual(max(y for _, y in centerline[0]) - min(y for _, y in centerline[0]), 2)

    def test_junction_and_loop_paths_preserve_source_topology(self) -> None:
        image = np.full((96, 96), 255, dtype=np.uint8)
        cv2.line(image, (10, 20), (60, 20), 0, thickness=3)
        cv2.line(image, (35, 20), (35, 70), 0, thickness=3)
        cv2.circle(image, (75, 65), 12, 0, thickness=3)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topology.png"
            cv2.imwrite(str(path), image)
            strokes, stats = vectorize_image_with_stats(
                path,
                epsilon=0.0,
                method="centerline",
                threshold_profile="legacy",
            )

        rendered = rasterize_strokes(strokes, image.shape)
        self.assertGreaterEqual(stats.raw_stroke_count, 4)
        reference = source_centerline(image, threshold_profile="legacy")
        self.assertGreater(centerline_metrics(reference, rendered).f1, 0.98)

    def test_y_junction_retains_all_three_branches(self) -> None:
        image = np.full((96, 96), 255, dtype=np.uint8)
        cv2.line(image, (48, 75), (48, 45), 0, thickness=3)
        cv2.line(image, (48, 45), (25, 20), 0, thickness=3)
        cv2.line(image, (48, 45), (71, 20), 0, thickness=3)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y-junction.png"
            cv2.imwrite(str(path), image)
            strokes, stats = vectorize_image_with_stats(
                path,
                epsilon=0.0,
                method="centerline",
                threshold_profile="legacy",
            )

        rendered = rasterize_strokes(strokes, image.shape)
        reference = source_centerline(image, threshold_profile="legacy")
        self.assertGreaterEqual(stats.raw_stroke_count, 3)
        self.assertGreaterEqual(centerline_metrics(reference, rendered).f1, 0.95)

    def test_rdp_simplification_stays_within_two_pixel_gate(self) -> None:
        stroke = [
            (x, int(round(20 + 8 * np.sin(x / 8.0))))
            for x in range(5, 90)
        ]
        simplified = simplify_strokes([stroke], epsilon=2.0)
        reference = rasterize_strokes([stroke], (64, 96))
        candidate = rasterize_strokes(simplified, (64, 96))

        metrics = centerline_metrics(reference, candidate, tolerance_px=2.0)

        self.assertGreaterEqual(metrics.f1, 0.95)
        self.assertLess(len(simplified[0]), len(stroke))

    def test_error_feedback_prevents_cumulative_quantization_drift(self) -> None:
        codebook = np.asarray(
            [[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]],
            dtype=np.float32,
        )
        rows = [[0.03, 0.0, 1.0, 0.0, 0.0] for _ in range(100)]
        rows[-1][2:4] = [0.0, 1.0]
        rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
        stroke5 = np.asarray(rows, dtype=np.float32)

        independent = encode_stroke5(stroke5, codebook, error_feedback=False)
        corrected = encode_stroke5(stroke5, codebook, error_feedback=True)
        independent_error = quantization_metrics(stroke5, independent, codebook)
        corrected_error = quantization_metrics(stroke5, corrected, codebook)

        self.assertGreater(independent_error.endpoint_error, 1.0)
        self.assertLessEqual(corrected_error.endpoint_error, 0.011)
        self.assertLess(corrected_error.mean_point_error, independent_error.mean_point_error)

    def test_token_round_trip_restores_source_canvas_coordinates(self) -> None:
        strokes = [[(12, 20), (22, 20), (22, 30)], [(40, 45), (48, 45)]]
        codebook = np.asarray(
            [
                [0.0, 0.0],
                [10.0 / 36.0, 0.0],
                [0.0, 10.0 / 36.0],
                [18.0 / 36.0, 15.0 / 36.0],
                [8.0 / 36.0, 0.0],
            ],
            dtype=np.float32,
        )
        stroke5, transform = strokes_to_stroke5(strokes, canvas_shape=(64, 64))

        tokens = encode_stroke5(stroke5, codebook, error_feedback=True)
        restored = stroke5_to_canvas_strokes(decode_tokens(tokens, codebook), transform)
        expected = rasterize_strokes(strokes, (64, 64))
        actual = rasterize_strokes(restored, (64, 64))

        self.assertGreaterEqual(centerline_metrics(expected, actual).f1, 0.99)

    def test_outer_to_inner_orders_outer_structures_before_inner(self) -> None:
        image = np.full((96, 96), 255, dtype=np.uint8)
        # Outer circle (periphery)
        cv2.circle(image, (48, 48), 35, 0, thickness=2)
        # Inner detail (center)
        cv2.line(image, (40, 48), (56, 48), 0, thickness=2)
        cv2.line(image, (48, 40), (48, 56), 0, thickness=2)

        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "outer_inner.png"
            cv2.imwrite(str(file_path), image)
            branches, _ = vectorize_image_with_stats(
                file_path,
                epsilon=0.0,
                method="centerline",
                threshold_profile="legacy",
                structured=True,
            )

        ordered = order_outer_to_inner(branches)
        self.assertGreater(len(ordered), 0)

        # Confirm the first stroke is located further from the center than the last stroke
        cx, cy = 48.0, 48.0
        r_first = sum(((pt[0] - cx)**2 + (pt[1] - cy)**2)**0.5 for pt in ordered[0]) / len(ordered[0])
        r_last = sum(((pt[0] - cx)**2 + (pt[1] - cy)**2)**0.5 for pt in ordered[-1]) / len(ordered[-1])
        self.assertGreater(r_first, r_last)

        # Confirm rasterization retains fidelity
        rendered = rasterize_strokes(ordered, image.shape)
        reference = source_centerline(image, threshold_profile="legacy")
        self.assertGreater(centerline_metrics(reference, rendered).f1, 0.95)

    def test_extractor_benchmark_uses_identical_relative_samples(self) -> None:
        image = np.full((8, 8), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "anime2sketch"
            second = root / "lineart_anime"
            for directory in (first, second):
                (directory / "nested").mkdir(parents=True)
                cv2.imwrite(str(directory / "nested" / "shared.png"), image)
            cv2.imwrite(str(first / "only-first.png"), image)

            parsed = parse_extractor_dirs(
                [f"anime2sketch={first}", f"lineart-anime={second}"],
                root,
            )
            selected = select_common_samples(parsed, samples=1, seed=42)
            with self.assertRaisesRegex(ValueError, "only 1"):
                select_common_samples(parsed, samples=2, seed=42)

        self.assertEqual([path.name for path in selected["anime2sketch"]], ["shared.png"])
        self.assertEqual([path.name for path in selected["lineart-anime"]], ["shared.png"])


if __name__ == "__main__":
    unittest.main()
