from __future__ import annotations

import unittest

import numpy as np

from services.anchored_sketch_data.codebook import (
    TrainingStrokeSample,
    fit_training_codebook,
    within_stroke_deltas,
)
from services.anchored_sketch_data.preprocessing import (
    augment_strokes,
    denormalize_from_canvas,
    deterministic_order,
    merge_compatible_strokes,
    normalize_to_canvas,
    normalize_to_canvas_with_transform,
    preprocess_strokes,
    simplify_strokes,
)
from services.anchored_sketch_data.splitting import (
    SplitCandidate,
    deterministic_group_split,
)


class AnchoredV3PreprocessingTest(unittest.TestCase):
    def test_canvas_normalization_preserves_aspect_ratio_and_margin(self) -> None:
        normalized = normalize_to_canvas([[(10, 20), (110, 20), (110, 70)]])
        points = np.asarray(normalized[0])

        self.assertAlmostEqual(float(points[:, 0].min()), 8.0, places=4)
        self.assertAlmostEqual(float(points[:, 0].max()), 247.0, places=4)
        self.assertGreaterEqual(float(points[:, 1].min()), 8.0)
        self.assertLessEqual(float(points[:, 1].max()), 247.0)
        self.assertAlmostEqual(
            (points[:, 0].max() - points[:, 0].min())
            / (points[:, 1].max() - points[:, 1].min()),
            2.0,
            places=6,
        )

    def test_canvas_normalization_transform_is_invertible(self) -> None:
        source = [[(10, 20), (110, 20), (110, 70)]]
        normalized, transform = normalize_to_canvas_with_transform(source)
        restored = denormalize_from_canvas(normalized, transform)

        np.testing.assert_allclose(restored, source, atol=1e-5)

    def test_preprocessing_removes_small_components_border_frames_and_singletons(self) -> None:
        strokes = [
            [(0, 0), (63, 0)],
            [(5, 5), (6, 5)],
            [(10, 20), (11, 20), (12, 20), (13, 20), (14, 20), (15, 20), (16, 20), (17, 20)],
            [(30, 30)],
        ]

        result = preprocess_strokes(strokes, source_shape=(64, 64))

        self.assertEqual(result, [[(10.0, 20.0), (11.0, 20.0), (12.0, 20.0), (13.0, 20.0), (14.0, 20.0), (15.0, 20.0), (16.0, 20.0), (17.0, 20.0)]])

    def test_order_orientation_merge_and_simplification_are_deterministic(self) -> None:
        first = [(3, 1), (2, 1), (1, 1)]
        second = [(3, 1), (4, 1), (5, 1)]

        merged = merge_compatible_strokes([second, first])
        reordered = deterministic_order([list(reversed(second)), list(reversed(first))])
        simplified = simplify_strokes(merged, epsilon=0.5)

        self.assertEqual(merged, [[(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0)]])
        self.assertEqual(
            reordered,
            [
                [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)],
                [(3.0, 1.0), (4.0, 1.0), (5.0, 1.0)],
            ],
        )
        self.assertEqual(simplified, [[(1.0, 1.0), (5.0, 1.0)]])

    def test_augmentation_is_seeded_and_does_not_modify_input(self) -> None:
        source = [[(10.0, 10.0), (20.0, 20.0)]]

        first = augment_strokes(source, seed=123)
        second = augment_strokes(source, seed=123)
        different = augment_strokes(source, seed=124)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(source, [[(10.0, 10.0), (20.0, 20.0)]])
        self.assertTrue(all(0.0 <= value <= 255.0 for point in first[0] for value in point))

    def test_duplicate_groups_never_cross_splits_and_assignment_is_order_invariant(self) -> None:
        candidates = [
            SplitCandidate("a", "1" * 64, 0b0000, 100),
            SplitCandidate("a-copy", "1" * 64, 0b1111, 100),
            SplitCandidate("b-near", "2" * 64, 0b10000, 100),
            SplitCandidate("c", "3" * 64, (1 << 63), 900),
            SplitCandidate("d", "4" * 64, (1 << 62), 1500),
            SplitCandidate("e", "5" * 64, (1 << 61), 2500),
        ]

        first = deterministic_group_split(candidates, seed=77)
        second = deterministic_group_split(reversed(candidates), seed=77)

        self.assertEqual(first, second)
        self.assertEqual(first["a"].group_id, first["a-copy"].group_id)
        self.assertEqual(first["a"].group_id, first["b-near"].group_id)
        self.assertEqual(first["a"].split, first["b-near"].split)

    def test_codebook_fits_only_within_stroke_training_deltas(self) -> None:
        sample = TrainingStrokeSample(
            "train-sample",
            "train",
            [[(0, 0), (1, 0), (2, 0)], [(10, 10), (10, 12), (10, 14)]],
        )

        deltas = within_stroke_deltas(sample.strokes)
        centers = fit_training_codebook([sample], n_clusters=2, seed=9, batch_size=4)

        self.assertEqual(deltas.tolist(), [[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
        np.testing.assert_allclose(centers, [[0.0, 2.0], [1.0, 0.0]], atol=1e-5)
        with self.assertRaisesRegex(ValueError, "training samples"):
            fit_training_codebook(
                [TrainingStrokeSample("leak", "valid", sample.strokes)], n_clusters=2
            )


if __name__ == "__main__":
    unittest.main()
