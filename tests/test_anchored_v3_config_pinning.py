from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.sketchformer.config import pin_anchored_v3_artifacts


def _v3_config(*, root: str, manifest_path: str) -> dict:
    return {
        "data": {
            "dataset": {
                "root": root,
                "manifest_file": "manifest.jsonl",
                "manifest_path": manifest_path,
            },
            "format": {
                "type": "anchored_v3",
                "token_dictionary": {"codebook_path": f"{root}/codebook.npy"},
            },
        },
        "model": {
            "input": {
                "token_dictionary": {"codebook_path": f"{root}/codebook.npy"}
            }
        },
    }


def _write_minimal_artifact(root: Path) -> None:
    root.mkdir()
    (root / "manifest.jsonl").write_text("", encoding="utf-8")
    (root / "codebook.npy").write_bytes(b"fixture")


class AnchoredV3ConfigPinningTest(unittest.TestCase):
    def test_pins_root_manifest_and_codebooks_to_one_immutable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            artifact = project_root / "artifact-a"
            _write_minimal_artifact(artifact)
            (project_root / "current").symlink_to(artifact, target_is_directory=True)
            config = _v3_config(
                root="current",
                manifest_path="current/manifest.jsonl",
            )

            resolved = pin_anchored_v3_artifacts(config, project_root=project_root)

            self.assertEqual(resolved, artifact)
            self.assertEqual(config["data"]["dataset"]["root"], "artifact-a")
            self.assertEqual(
                config["data"]["dataset"]["manifest_path"],
                "artifact-a/manifest.jsonl",
            )
            self.assertEqual(
                config["data"]["format"]["token_dictionary"]["codebook_path"],
                "artifact-a/codebook.npy",
            )
            self.assertEqual(
                config["model"]["input"]["token_dictionary"]["codebook_path"],
                "artifact-a/codebook.npy",
            )

    def test_rejects_manifest_path_from_a_different_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            first = project_root / "artifact-a"
            second = project_root / "artifact-b"
            _write_minimal_artifact(first)
            _write_minimal_artifact(second)
            (project_root / "current").symlink_to(first, target_is_directory=True)
            config = _v3_config(
                root="current",
                manifest_path="artifact-b/manifest.jsonl",
            )

            with self.assertRaisesRegex(ValueError, "manifest_path"):
                pin_anchored_v3_artifacts(config, project_root=project_root)


if __name__ == "__main__":
    unittest.main()
