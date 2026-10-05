from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
import copy

from scripts.sketchformer.human_review import (
    REVIEW_CRITERIA,
    build_review_template,
    evaluate_human_reviews,
    sha256_file,
    validate_release_evaluation_report,
    validate_review_binding,
)
from scripts.sketchformer.config import compose_training_config
from services.anchored_sketch_data.contract import TOKEN_LAYOUT


def review_template(sample_ids: list[str], evaluation_hash: str = "a" * 64):
    return build_review_template(
        sample_ids,
        evaluation_report_sha256=evaluation_hash,
        plot_records=[
            {
                "sample_id": sample_id,
                "plot_path": f"/tmp/{sample_id}.png",
                "plot_sha256": "b" * 64,
            }
            for sample_id in sample_ids
        ],
    )


class AnchoredV3HumanReviewTest(unittest.TestCase):
    def test_95_of_100_all_criterion_reviews_pass(self) -> None:
        sample_ids = [f"sample-{index:03d}" for index in range(100)]
        payload = review_template(sample_ids)
        for index, review in enumerate(payload["reviews"]):
            for criterion in REVIEW_CRITERIA:
                review[criterion] = index < 95

        result = evaluate_human_reviews(payload, sample_ids)

        self.assertTrue(result["passed"])
        self.assertEqual(result["pass_count"], 95)
        self.assertEqual(len(result["failed_sample_ids"]), 5)

    def test_review_sample_set_must_match_fixed_evaluation_set(self) -> None:
        payload = review_template(["wrong"])
        for criterion in REVIEW_CRITERIA:
            payload["reviews"][0][criterion] = True

        with self.assertRaisesRegex(ValueError, "do not match"):
            evaluate_human_reviews(payload, ["expected"], minimum_passes=1)

    def test_unreviewed_criterion_is_rejected(self) -> None:
        payload = review_template(["sample"])

        with self.assertRaisesRegex(ValueError, "must be true or false"):
            evaluate_human_reviews(payload, ["sample"], minimum_passes=1)

    def test_release_report_and_plot_bytes_are_bound_to_reviews(self) -> None:
        runtime_config = compose_training_config(
            "configs/train.yaml",
            experiment="anime_anchored_v3_direct",
        )
        records = [
            {
                "sample_id": f"sample-{index:03d}",
                "target_length": 3000,
                "geometry_f1_2px": 1.0,
                "symmetric_chamfer_px": 0.0,
                "premature_eos": 0.0,
                "max_length_hit": 0.0,
            }
            for index in range(100)
        ]
        evaluation = {
            "metadata": {
                "format_type": "anchored_v3",
                "format_version": 3,
                "split": "test",
                "decode_mode": "free-running",
                "limit_batches": 1.0,
                "max_generation_length": None,
                "enforce_v3_gates": True,
                "allow_legacy_checkpoint": False,
                "minimum_source_sketches": 1,
                "token_layout": TOKEN_LAYOUT.to_dict(),
                "decoder_memory_source": "encoder",
                "checkpoint_contract": {
                    "token_layout_version": 3,
                    "compatibility_config": runtime_config,
                },
            },
            "metrics": {},
            "records": records,
        }
        self.assertEqual(validate_release_evaluation_report(evaluation)[0], "sample-000")
        latent_report = copy.deepcopy(evaluation)
        latent_report["metadata"]["checkpoint_contract"]["compatibility_config"][
            "model"
        ]["decoder"]["memory_source"] = "latent_expander"
        with self.assertRaisesRegex(ValueError, "not canonical anchored V3"):
            validate_release_evaluation_report(latent_report)
        mismatched_minimum = copy.deepcopy(evaluation)
        mismatched_minimum["metadata"]["minimum_source_sketches"] = 2
        with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
            validate_release_evaluation_report(mismatched_minimum)
        with tempfile.TemporaryDirectory() as tmp:
            plot = Path(tmp) / "plot.png"
            plot.write_bytes(b"plot-v1")
            payload = build_review_template(
                ["sample-000"],
                evaluation_report_sha256="c" * 64,
                plot_records=[
                    {
                        "sample_id": "sample-000",
                        "plot_path": str(plot),
                        "plot_sha256": sha256_file(plot),
                    }
                ],
            )
            validate_review_binding(
                payload,
                evaluation_report_sha256="c" * 64,
            )
            plot.write_bytes(b"plot-v2")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_review_binding(
                    payload,
                    evaluation_report_sha256="c" * 64,
                )


if __name__ == "__main__":
    unittest.main()
