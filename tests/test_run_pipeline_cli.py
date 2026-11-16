from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import run_pipeline as run_pipeline_cli


class RunPipelineCliTest(unittest.TestCase):
    def test_num_workers_defaults_to_serial(self) -> None:
        self.assertEqual(run_pipeline_cli.parse_args([]).num_workers, 1)

    def test_cli_args_bypass_prompts_and_pass_pipeline_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sketches_dir = root / "sketches"
            stroke5_dir = root / "stroke5"
            token_dir = root / "sketch_token"
            sketches_dir.mkdir()
            for index in range(2):
                (sketches_dir / f"sketch_{index}.png").write_bytes(b"")

            with (
                patch.object(run_pipeline_cli, "_prompt_int") as prompt_int,
                patch.object(run_pipeline_cli, "_prompt_ordering") as prompt_ordering,
                patch.object(run_pipeline_cli, "run_pipeline") as run_pipeline,
            ):
                run_pipeline_cli.main(
                    [
                        "--sketches-dir",
                        str(sketches_dir),
                        "--stroke5-dir",
                        str(stroke5_dir),
                        "--sketch-token-dir",
                        str(token_dir),
                        "--n-sketches",
                        "1000",
                        "--ordering",
                        "directional",
                        "--rdp-epsilon",
                        "1",
                        "--codebook-k",
                        "1000",
                        "--num-workers",
                        "8",
                        "--seed",
                        "42",
                    ]
                )

            prompt_int.assert_not_called()
            prompt_ordering.assert_not_called()
            run_pipeline.assert_called_once_with(
                sketches_dir=sketches_dir,
                stroke5_dir=stroke5_dir,
                sketch_token_dir=token_dir,
                n_sketches=2,
                ordering="directional",
                rdp_epsilon=1.0,
                codebook_K=1000,
                seed=42,
                vectorizer="centerline",
                threshold_profile="hysteresis",
                max_token_length=4096,
                max_geometry_error=2.0,
                token_dict_dir=None,
                manifest_path=None,
                fail_on_overlength=False,
                extractor_name=None,
                num_workers=8,
            )

    def test_cli_rejects_non_positive_worker_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sketches_dir = Path(tmpdir) / "sketches"
            sketches_dir.mkdir()
            (sketches_dir / "sketch.png").write_bytes(b"")

            with self.assertRaisesRegex(SystemExit, "1"):
                run_pipeline_cli.main(
                    [
                        "--sketches-dir",
                        str(sketches_dir),
                        "--n-sketches",
                        "1",
                        "--ordering",
                        "continuity",
                        "--num-workers",
                        "0",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
