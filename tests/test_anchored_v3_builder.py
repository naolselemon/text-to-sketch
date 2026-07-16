from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import cv2
import numpy as np

from services.anchored_sketch_data.artifacts import validate_dataset
from services.anchored_sketch_data.builder import BuilderConfig, build_dataset
from services.anchored_sketch_data.builder import _preprocess_sources
from services.anchored_sketch_data.cli import parse_args


def builder_test_codebook() -> np.ndarray:
    centers = np.column_stack(
        (
            np.arange(2048, dtype=np.float32) + 1000.0,
            np.arange(2048, dtype=np.float32) + 2000.0,
        )
    )
    centers[0] = [239.0, 0.0]
    return centers


class AnchoredV3BuilderTest(unittest.TestCase):
    def test_image_builder_calibrates_and_publishes_without_truncation(self) -> None:
        raw_stroke = [[(x, 32) for x in range(10, 21)]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            image = np.full((64, 64), 255, dtype=np.uint8)
            cv2.line(image, (10, 32), (20, 32), 0, thickness=1)
            cv2.imwrite(str(source / "sample.png"), image)

            with patch(
                "services.anchored_sketch_data.builder.vectorize_image",
                return_value=raw_stroke,
            ), patch(
                "services.anchored_sketch_data.builder.fit_training_codebook",
                return_value=builder_test_codebook(),
            ) as fit_codebook, redirect_stderr(io.StringIO()) as progress_output:
                dataset = build_dataset(
                    source,
                    output,
                    config=BuilderConfig(
                        epsilon_candidates=(0.5,),
                        calibration_size=1,
                        train_augmentation_copies=0,
                        minimum_accepted_source_sketches=1,
                        shard_size=1,
                        show_progress=True,
                    ),
                )

            metadata = validate_dataset(dataset)
            codebook_samples = fit_codebook.call_args.args[0]

            self.assertEqual(metadata["preparation"]["selected_rdp_epsilon"], 0.5)
            self.assertEqual(metadata["split_counts"], {"train": 1, "valid": 0, "test": 0})
            self.assertEqual(metadata["rejected_count"], 0)
            self.assertTrue(all(sample.split == "train" for sample in codebook_samples))
            self.assertEqual(
                metadata["preparation"]["epsilon_sweep"][0]["p99_token_length"],
                7.0,
            )
            progress = progress_output.getvalue()
            self.assertIn("discovered 1 images; preprocessing with 1 worker", progress)
            self.assertIn("preprocessing complete", progress)
            self.assertIn("images/s", progress)
            self.assertIn("epsilon 0.5 passed", progress)
            self.assertIn("build complete", progress)

    def test_minimum_source_count_is_configurable_and_prevents_publication(self) -> None:
        default_args = parse_args(
            ["build", "--source-dir", "source", "--output-root", "output"]
        )
        overfit_args = parse_args(
            [
                "build",
                "--source-dir",
                "source",
                "--output-root",
                "output",
                "--minimum-accepted-source-sketches",
                "32",
            ]
        )

        self.assertEqual(default_args.minimum_accepted_source_sketches, 1)
        self.assertEqual(overfit_args.minimum_accepted_source_sketches, 32)

        raw_stroke = [[(x, 32) for x in range(10, 21)]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            image = np.full((64, 64), 255, dtype=np.uint8)
            cv2.line(image, (10, 32), (20, 32), 0, thickness=1)
            cv2.imwrite(str(source / "sample.png"), image)

            with patch(
                "services.anchored_sketch_data.builder.vectorize_image",
                return_value=raw_stroke,
            ), self.assertRaisesRegex(ValueError, "at least 2 accepted source sketches"):
                build_dataset(
                    source,
                    output,
                    config=BuilderConfig(
                        epsilon_candidates=(0.5,),
                        calibration_size=1,
                        train_augmentation_copies=0,
                        minimum_accepted_source_sketches=2,
                        show_progress=False,
                    ),
                )

            self.assertFalse(output.exists())

    def test_source_vector_gate_compares_against_raster_not_only_simplification(self) -> None:
        horizontal_vector = [[(x, 10) for x in range(10, 31)]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            image = np.full((48, 48), 255, dtype=np.uint8)
            cv2.line(image, (24, 10), (24, 30), 0, thickness=1)
            cv2.imwrite(str(source / "mismatch.png"), image)

            with patch(
                "services.anchored_sketch_data.builder.vectorize_image",
                return_value=horizontal_vector,
            ), patch(
                "services.anchored_sketch_data.builder.fit_training_codebook",
                return_value=builder_test_codebook(),
            ), self.assertRaisesRegex(ValueError, "No RDP epsilon passed"):
                build_dataset(
                    source,
                    output,
                    config=BuilderConfig(
                        epsilon_candidates=(0.5,),
                        calibration_size=1,
                        train_augmentation_copies=0,
                        minimum_accepted_source_sketches=1,
                        shard_size=1,
                        show_progress=False,
                    ),
                )

    def test_worker_cli_controls_are_explicit(self) -> None:
        defaults = parse_args(
            ["build", "--source-dir", "source", "--output-root", "output"]
        )
        tuned = parse_args(
            [
                "build",
                "--source-dir",
                "source",
                "--output-root",
                "output",
                "--workers",
                "8",
                "--no-progress",
            ]
        )

        self.assertEqual(defaults.workers, 0)
        self.assertTrue(defaults.show_progress)
        self.assertEqual(tuned.workers, 8)
        self.assertFalse(tuned.show_progress)

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            image = np.full((32, 32), 255, dtype=np.uint8)
            cv2.line(image, (4, 16), (27, 16), 0, thickness=1)
            cv2.imwrite(str(source / "sample.png"), image)
            with self.assertRaisesRegex(
                ValueError, "preprocessing_workers must be a non-negative integer"
            ):
                build_dataset(
                    source,
                    source / "output",
                    config=BuilderConfig(
                        preprocessing_workers=-1,
                        show_progress=False,
                    ),
                )

    def test_parallel_preprocessing_matches_serial_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            paths: list[Path] = []
            for index in (3, 1, 2, 0):
                image = np.full((96, 96), 255, dtype=np.uint8)
                cv2.line(
                    image,
                    (8, 15 + index * 12),
                    (87, 15 + index * 12),
                    0,
                    thickness=1,
                )
                path = source / f"sample-{index}.png"
                cv2.imwrite(str(path), image)
                paths.append(path)

            raw_stroke = [[(x, 32) for x in range(10, 81)]]
            pool = MagicMock()
            pool.__enter__.return_value = pool
            pool.imap_unordered.side_effect = lambda function, tasks, chunksize: iter(
                reversed([function(task) for task in tasks])
            )
            context = MagicMock()
            context.Pool.return_value = pool
            with patch(
                "services.anchored_sketch_data.builder.vectorize_image",
                return_value=raw_stroke,
            ), patch(
                "services.anchored_sketch_data.builder._multiprocessing_context",
                return_value=context,
            ):
                serial = _preprocess_sources(
                    paths,
                    source_root=source,
                    threshold_profile="hysteresis",
                    workers=1,
                    show_progress=False,
                )
                parallel = _preprocess_sources(
                    paths,
                    source_root=source,
                    threshold_profile="hysteresis",
                    workers=2,
                    show_progress=False,
                )

        self.assertEqual(serial, parallel)
        context.Pool.assert_called_once_with(
            processes=2,
            initializer=ANY,
        )
        pool.imap_unordered.assert_called_once()
        self.assertEqual(
            [sample.sample_id for sample in parallel[0]],
            [f"sample-{index}.png" for index in range(4)],
        )


if __name__ == "__main__":
    unittest.main()
