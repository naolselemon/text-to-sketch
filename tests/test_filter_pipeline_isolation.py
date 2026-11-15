"""Regression coverage for immutable filter and preprocessing run outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from pipeline import workflow
from prep_data import filter_sketches as sketch_filter


class FilterPipelineIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        for name, points in (("a", 2), ("b", 5)):
            image = np.full((12, 12), 255, dtype=np.uint8)
            image[3, 3 : 3 + points] = 0
            self.assertTrue(cv2.imwrite(str(self.input_dir / f"{name}.png"), image))

        self.codebook_dir = self.root / "released_codebook"
        self.codebook_dir.mkdir()
        np.save(
            self.codebook_dir / "codebook.npy",
            np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=np.float32),
        )

    def filter_run(self, run: str, max_points: int) -> tuple[Path, Path, list[sketch_filter.FilterResult]]:
        build_dir = self.root / run
        output_dir = build_dir / "sketches_filtered"
        report_path = build_dir / "filter_report.csv"
        args = argparse.Namespace(
            input_dir=self.input_dir,
            output_dir=output_dir,
            rejected_dir=build_dir / "rejected",
            report_path=report_path,
            max_points=max_points,
            epsilon=0.5,
            count="original",
            foreground_threshold=40,
            background="light",
            action="copy-kept",
            limit=None,
        )
        results = sketch_filter.filter_sketches(args)
        sketch_filter.write_report(report_path, results)
        return output_dir, report_path, results

    def pipeline_run(self, run: str, sketches_dir: Path) -> Path:
        build_dir = self.root / run
        workflow.run_pipeline(
            sketches_dir=sketches_dir,
            stroke5_dir=build_dir / "stroke5",
            sketch_token_dir=self.codebook_dir,
            token_dict_dir=self.codebook_dir,
            n_sketches=2,
            ordering="continuity",
            build_id=run,
        )
        return build_dir / "preprocessing_manifest.jsonl"

    @staticmethod
    def accepted_sequence(**_kwargs):
        return {
            "accepted": True,
            "record": {},
            "stroke5": np.array([[0, 0, 0, 0, 1]], dtype=np.float32),
            "tokens": np.array([5, 1, 6], dtype=np.int64),
        }

    def test_old_kept_sample_does_not_enter_later_build_with_stricter_filter(self) -> None:
        first_input, first_report, first_results = self.filter_run("run_1", 5)
        self.assertEqual({result.relative_path.name for result in first_results if result.kept}, {"a.png", "b.png"})
        with patch.object(workflow, "fit_centerline_sequence", side_effect=self.accepted_sequence):
            first_manifest = self.pipeline_run("run_1", first_input)
        first_tokens = self.root / "run_1" / "tokens" / "b.npz"
        old_bytes = first_tokens.read_bytes()
        old_manifest = first_manifest.read_bytes()
        old_report = first_report.read_bytes()

        second_input, second_report, second_results = self.filter_run("run_2", 2)
        self.assertEqual({result.relative_path.name for result in second_results if not result.kept}, {"b.png"})
        self.assertEqual([path.name for path in second_input.glob("*.png")], ["a.png"])
        with patch.object(workflow, "fit_centerline_sequence", side_effect=self.accepted_sequence):
            second_manifest = self.pipeline_run("run_2", second_input)

        records = [json.loads(line) for line in second_manifest.read_text().splitlines()]
        self.assertEqual([(record["sample_id"], record["build_id"], record["status"]) for record in records], [("a", "run_2", "accepted")])
        accepted = records[0]
        token_path = Path(accepted["tokens_path"])
        self.assertEqual(accepted["schema_version"], 2)
        self.assertEqual(Path(accepted["stroke5_path"]), self.root / "run_2" / "stroke5" / "a.npz")
        self.assertEqual(token_path, self.root / "run_2" / "tokens" / "a.npz")
        self.assertEqual(accepted["tokens_sha256"], hashlib.sha256(token_path.read_bytes()).hexdigest())
        with np.load(token_path) as archive:
            self.assertEqual(str(archive["build_id"]), "run_2")
            self.assertEqual(str(archive["sample_id"]), "a")
            self.assertEqual(str(archive["token_dict_sha256"]), accepted["token_dictionary_sha256"])
        self.assertFalse((self.root / "run_2" / "tokens" / "b.npz").exists())
        self.assertEqual(first_tokens.read_bytes(), old_bytes)
        self.assertEqual(first_manifest.read_bytes(), old_manifest)
        self.assertEqual(first_report.read_bytes(), old_report)
        self.assertTrue(second_report.is_file())

    def test_pipeline_rejection_does_not_carry_prior_run_token_file(self) -> None:
        first_input, _, _ = self.filter_run("run_1", 5)
        with patch.object(workflow, "fit_centerline_sequence", side_effect=self.accepted_sequence):
            self.pipeline_run("run_1", first_input)
        old_token = self.root / "run_1" / "tokens" / "b.npz"
        old_bytes = old_token.read_bytes()

        second_input, _, _ = self.filter_run("run_2", 5)

        def reject_b(*, image_path, **kwargs):
            if image_path.name == "b.png":
                return {"accepted": False, "record": {"rejection_reason": "overlength"}}
            return self.accepted_sequence(**kwargs)

        with patch.object(workflow, "fit_centerline_sequence", side_effect=reject_b):
            second_manifest = self.pipeline_run("run_2", second_input)
        records = {record["sample_id"]: record for record in map(json.loads, second_manifest.read_text().splitlines())}
        self.assertEqual(records["b"]["status"], "rejected")
        self.assertEqual(records["b"]["build_id"], "run_2")
        self.assertNotIn("tokens_path", records["b"])
        self.assertFalse((self.root / "run_2" / "tokens" / "b.npz").exists())
        self.assertEqual(old_token.read_bytes(), old_bytes)

    def test_filter_refuses_reused_and_interrupted_outputs_without_overwriting(self) -> None:
        first_input, first_report, _ = self.filter_run("run_1", 5)
        old_image = (first_input / "b.png").read_bytes()
        old_report = first_report.read_bytes()
        for reused_path in (first_input, self.root / "interrupted" / "sketches_filtered"):
            reused_path.mkdir(parents=True, exist_ok=True)
            args = argparse.Namespace(
                input_dir=self.input_dir,
                output_dir=reused_path,
                rejected_dir=self.root / "unused",
                report_path=self.root / "new_report.csv",
                max_points=2,
                epsilon=0.5,
                count="original",
                foreground_threshold=40,
                background="light",
                action="copy-kept",
                limit=None,
            )
            with self.assertRaises(FileExistsError):
                sketch_filter.filter_sketches(args)
        args.output_dir = self.root / "fresh" / "sketches_filtered"
        args.report_path = first_report
        with self.assertRaises(FileExistsError):
            sketch_filter.filter_sketches(args)
        self.assertFalse(args.output_dir.exists())
        self.assertEqual((first_input / "b.png").read_bytes(), old_image)
        self.assertEqual(first_report.read_bytes(), old_report)

    def test_pipeline_refuses_existing_output_manifest_and_interrupted_runs(self) -> None:
        sketches, _, _ = self.filter_run("run_1", 5)
        with patch.object(workflow, "fit_centerline_sequence", side_effect=self.accepted_sequence):
            manifest = self.pipeline_run("run_1", sketches)
        old_manifest = manifest.read_bytes()
        old_token = (self.root / "run_1" / "tokens" / "b.npz").read_bytes()
        for build_id in ("run_1", "different_build_id"):
            with self.assertRaises(FileExistsError):
                workflow.run_pipeline(
                    sketches_dir=sketches,
                    stroke5_dir=self.root / "run_1" / "stroke5",
                    sketch_token_dir=self.codebook_dir,
                    token_dict_dir=self.codebook_dir,
                    n_sketches=2,
                    ordering="continuity",
                    build_id=build_id,
                )
        self.assertEqual(manifest.read_bytes(), old_manifest)
        self.assertEqual((self.root / "run_1" / "tokens" / "b.npz").read_bytes(), old_token)

        interrupted = self.root / "interrupted"
        (interrupted / "tokens").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.pipeline_run("interrupted", sketches)
        self.assertFalse((interrupted / "stroke5").exists())

        manifest_only = self.root / "manifest_only"
        manifest_only.mkdir()
        (manifest_only / "preprocessing_manifest.jsonl").write_text("old manifest\n")
        with self.assertRaises(FileExistsError):
            self.pipeline_run("manifest_only", sketches)
        self.assertFalse((manifest_only / "stroke5").exists())
        self.assertFalse((manifest_only / "tokens").exists())


if __name__ == "__main__":
    unittest.main()
