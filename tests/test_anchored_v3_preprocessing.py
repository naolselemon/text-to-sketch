from __future__ import annotations

import unittest
from collections.abc import Sequence

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

    def test_endpoint_indexed_merge_matches_exhaustive_tie_breaking(self) -> None:
        rng = np.random.default_rng(9182)
        for _ in range(30):
            strokes = []
            for _ in range(10):
                start = rng.integers(-4, 5, size=2)
                middle = start + rng.integers(-2, 3, size=2)
                end = middle + rng.integers(-2, 3, size=2)
                strokes.append([start.tolist(), middle.tolist(), end.tolist()])

            expected = _exhaustive_merge_reference(strokes)
            actual = merge_compatible_strokes(strokes)

            self.assertEqual(actual, expected)

    def test_long_mergeable_chain_is_collapsed_without_repeated_pair_scans(self) -> None:
        strokes = [
            [(3.0 * index, 0.0), (3.0 * index + 2.0, 0.0)]
            for index in range(250)
        ]

        merged = merge_compatible_strokes(strokes)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0][0], (0.0, 0.0))
        self.assertEqual(merged[0][-1], (749.0, 0.0))

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


def _exhaustive_merge_reference(
    strokes: Sequence[Sequence[Sequence[float]]],
    *,
    max_gap: float = 1.5,
    minimum_cosine: float = 0.5,
) -> list[list[tuple[float, float]]]:
    """Previous all-pairs implementation used as a small-fixture oracle."""

    paths = [list(stroke) for stroke in deterministic_order(strokes) if len(stroke) >= 2]
    while True:
        candidates: list[tuple[float, int, int, bool, bool]] = []
        for first_index in range(len(paths)):
            for second_index in range(first_index + 1, len(paths)):
                for reverse_first in (False, True):
                    first = (
                        list(reversed(paths[first_index]))
                        if reverse_first
                        else paths[first_index]
                    )
                    first_direction = np.asarray(first[-1]) - np.asarray(first[-2])
                    for reverse_second in (False, True):
                        second = (
                            list(reversed(paths[second_index]))
                            if reverse_second
                            else paths[second_index]
                        )
                        gap = float(
                            np.linalg.norm(np.asarray(second[0]) - np.asarray(first[-1]))
                        )
                        if gap > max_gap:
                            continue
                        second_direction = np.asarray(second[1]) - np.asarray(second[0])
                        denominator = float(
                            np.linalg.norm(first_direction) * np.linalg.norm(second_direction)
                        )
                        cosine = 1.0 if denominator == 0.0 else float(
                            np.dot(first_direction, second_direction) / denominator
                        )
                        if cosine >= minimum_cosine:
                            candidates.append(
                                (
                                    gap,
                                    first_index,
                                    second_index,
                                    reverse_first,
                                    reverse_second,
                                )
                            )
        if not candidates:
            break
        _, first_index, second_index, reverse_first, reverse_second = min(candidates)
        first = (
            list(reversed(paths[first_index])) if reverse_first else paths[first_index]
        )
        second = (
            list(reversed(paths[second_index]))
            if reverse_second
            else paths[second_index]
        )
        paths[first_index] = (
            first + second[1:] if np.allclose(first[-1], second[0]) else first + second
        )
        del paths[second_index]
    return deterministic_order(paths)


if __name__ == "__main__":
    unittest.main()
