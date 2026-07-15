from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from services.anchored_sketch_data.artifacts import validate_dataset
from services.anchored_sketch_data.builder import BuilderConfig, build_dataset
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
            ) as fit_codebook:
                dataset = build_dataset(
                    source,
                    output,
                    config=BuilderConfig(
                        epsilon_candidates=(0.5,),
                        calibration_size=1,
                        train_augmentation_copies=0,
                        minimum_accepted_source_sketches=1,
                        shard_size=1,
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
                    ),
                )


if __name__ == "__main__":
    unittest.main()
