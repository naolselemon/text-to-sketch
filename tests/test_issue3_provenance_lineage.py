"""Comprehensive tests for Issue 3: Know Exactly Which Data Made a Model.

Validates:
1. Deterministic split hashing and dataset fingerprinting.
2. Dataset lineage extraction and cross-split data leakage detection.
3. Sample membership querying (train / valid / test / not found).
4. Exact duplicate image detection without data loss.
5. Checkpoint provenance persistence (lineage, config, step, metrics).
6. Compatibility validation (dictionary, build_id, dataset fingerprint, split hash, model config).
7. Legacy unverified checkpoint handling with --allow-unverified-checkpoint.
8. Evaluation and export compatibility gates.
9. CLI inspection tools.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from core.callbacks import CheckpointCallback
from core.checkpointing import (
    CheckpointLoadResult,
    CorruptedProvenanceError,
    IncompatibleCheckpointError,
    UnverifiedCheckpointError,
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from dataloaders.lineage import (
    DatasetLineage,
    compute_chunk_checksum,
    compute_dataset_fingerprint,
    compute_split_hash,
    load_dataset_lineage,
    query_sample_membership,
)
from prep_data.deduplication import (
    compute_file_sha256,
    detect_exact_duplicates,
)
from dataloaders.datamodule import StrokeSequenceDataModule
from dataloaders.stroke_sequence_dataset import StrokeSequenceDataset
from prep_data.prepare_sketchformer import migrate_legacy_stroke3_dataset
from prep_data.sketch_token.create_token_dict import (
    compute_codebook_artifact_checksum,
    compute_token_dictionary_fingerprint,
    get_token_dict_fingerprint,
    load_codebook_from_dir,
    save_codebook,
)


class DummyModel(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ProvenanceLineageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    # 1. Deterministic split hashing and dataset fingerprinting
    def test_compute_split_hash_is_deterministic_and_order_invariant(self) -> None:
        ids_order_a = ["sample_003", "sample_001", "sample_002"]
        ids_order_b = ["sample_001", "sample_002", "sample_003"]
        ids_with_dups = ["sample_001", "sample_002", "sample_003", "sample_001"]

        hash_a = compute_split_hash(ids_order_a)
        hash_b = compute_split_hash(ids_order_b)

        self.assertTrue(len(hash_a) == 64)
        self.assertEqual(hash_a, hash_b)
        self.assertEqual(compute_split_hash([]), "")

        # Duplicate sample IDs must be rejected (not silently deduplicated)
        with self.assertRaises(ValueError) as ctx:
            compute_split_hash(ids_with_dups)
        self.assertIn("Duplicate sample ID", str(ctx.exception))

        # Empty or whitespace-only sample IDs must be rejected
        with self.assertRaises(ValueError):
            compute_split_hash(["sample_001", ""])
        with self.assertRaises(ValueError):
            compute_split_hash(["   "])

    def test_compute_dataset_fingerprint_sensitivity(self) -> None:
        base_kwargs = {
            "build_id": "20261115_001",
            "dictionary_fingerprint": "dict_hash_123",
            "split_hashes": {"train": "thash", "valid": "vhash", "test": "thash2"},
            "split_counts": {"train": 10, "valid": 2, "test": 2},
            "train_chunks": ["train_0.npz"],
            "valid_chunks": ["valid_0.npz"],
            "test_chunks": ["test_0.npz"],
            "format_type": "tok_dict",
        }
        fp_base = compute_dataset_fingerprint(**base_kwargs)
        self.assertTrue(len(fp_base) == 64)

        # Mutate build_id
        mutated_build = dict(base_kwargs, build_id="20261115_002")
        self.assertNotEqual(fp_base, compute_dataset_fingerprint(**mutated_build))

        # Mutate dictionary fingerprint
        mutated_dict = dict(base_kwargs, dictionary_fingerprint="dict_hash_456")
        self.assertNotEqual(fp_base, compute_dataset_fingerprint(**mutated_dict))

        # Mutate split hash
        mutated_splits = dict(
            base_kwargs,
            split_hashes={"train": "thash_changed", "valid": "vhash", "test": "thash2"},
        )
        self.assertNotEqual(fp_base, compute_dataset_fingerprint(**mutated_splits))

        # Mutate split counts
        mutated_counts = dict(
            base_kwargs,
            split_counts={"train": 11, "valid": 2, "test": 2},
        )
        self.assertNotEqual(fp_base, compute_dataset_fingerprint(**mutated_counts))

    # 2. Lineage loading and cross-split leakage detection
    def test_load_dataset_lineage_detects_data_leakage_across_splits(self) -> None:
        dataset_dir = self.root / "leaky_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create chunks where sample_002 is in both train and valid
        train_chunk = dataset_dir / "train_000.npz"
        np.savez_compressed(
            train_chunk,
            x=np.zeros((2, 5)),
            sample_ids=np.array(["sample_001", "sample_002"]),
        )

        valid_chunk = dataset_dir / "valid_000.npz"
        np.savez_compressed(
            valid_chunk,
            x=np.zeros((1, 5)),
            sample_ids=np.array(["sample_002"]),  # LEAK!
        )

        test_chunk = dataset_dir / "test_000.npz"
        np.savez_compressed(
            test_chunk,
            x=np.zeros((1, 5)),
            sample_ids=np.array(["sample_003"]),
        )

        # Create meta.npz with verified=True and dataset_fingerprint
        np.savez_compressed(
            dataset_dir / "meta.npz",
            build_id=np.array("build_leak_test"),
            verified=np.array(True),
            dictionary_fingerprint=np.array("dict_fp"),
            dataset_fingerprint=np.array("ds_fp_leak_test"),
            split_counts=np.array([2, 1, 1]),
            train_chunks=np.array(["train_000.npz"]),
            valid_chunks=np.array(["valid_000.npz"]),
            test_chunks=np.array(["test_000.npz"]),
            format=np.array("tok_dict"),
        )

        with self.assertRaises(ValueError) as ctx:
            load_dataset_lineage(dataset_dir, verify_chunks=True)
        self.assertIn("Data leakage detected", str(ctx.exception))
        self.assertIn("sample_002", str(ctx.exception))

    def test_load_dataset_lineage_rejects_unverified_dataset(self) -> None:
        dataset_dir = self.root / "unverified_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # meta.npz without verified=True
        np.savez_compressed(
            dataset_dir / "meta.npz",
            build_id=np.array("build_unverified"),
            verified=np.array(False),
            dictionary_fingerprint=np.array("dict_fp"),
            dataset_fingerprint=np.array("ds_fp_unverified"),
            split_counts=np.array([1, 1, 1]),
            train_chunks=np.array(["train_000.npz"]),
            valid_chunks=np.array(["valid_000.npz"]),
            test_chunks=np.array(["test_000.npz"]),
            format=np.array("tok_dict"),
        )

        with self.assertRaises(RuntimeError) as ctx:
            load_dataset_lineage(dataset_dir)
        self.assertIn("not published or build was interrupted", str(ctx.exception))

    # 3. Query sample membership
    def test_query_sample_membership_locates_sample_and_split(self) -> None:
        dataset_dir = self.root / "valid_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            dataset_dir / "train_000.npz",
            x=np.zeros((2, 5)),
            sample_ids=np.array(["train_s1", "train_s2"]),
        )
        np.savez_compressed(
            dataset_dir / "valid_000.npz",
            x=np.zeros((1, 5)),
            sample_ids=np.array(["valid_s1"]),
        )
        np.savez_compressed(
            dataset_dir / "test_000.npz",
            x=np.zeros((1, 5)),
            sample_ids=np.array(["test_s1"]),
        )

        np.savez_compressed(
            dataset_dir / "meta.npz",
            build_id=np.array("build_valid_test"),
            verified=np.array(True),
            dictionary_fingerprint=np.array("dict_fp"),
            dataset_fingerprint=np.array("ds_fp_valid_test"),
            split_counts=np.array([2, 1, 1]),
            train_chunks=np.array(["train_000.npz"]),
            valid_chunks=np.array(["valid_000.npz"]),
            test_chunks=np.array(["test_000.npz"]),
            format=np.array("tok_dict"),
        )

        mem_train = query_sample_membership("train_s2", dataset_dir)
        self.assertEqual(mem_train["split"], "train")
        self.assertEqual(mem_train["chunk_file"], "train_000.npz")
        self.assertEqual(mem_train["chunk_index"], 1)

        mem_valid = query_sample_membership("valid_s1", dataset_dir)
        self.assertEqual(mem_valid["split"], "valid")
        self.assertEqual(mem_valid["chunk_file"], "valid_000.npz")
        self.assertEqual(mem_valid["chunk_index"], 0)

        mem_missing = query_sample_membership("non_existent_sample", dataset_dir)
        self.assertIsNone(mem_missing["split"])
        self.assertIsNone(mem_missing["chunk_file"])

    # 4. Exact duplicate image detection without data loss
    def test_detect_exact_duplicates_identifies_and_preserves_files(self) -> None:
        images_dir = self.root / "test_images"
        images_dir.mkdir(parents=True, exist_ok=True)

        content_a = b"unique_image_a_content_12345"
        content_b = b"duplicate_image_b_content_shared_bytes"
        content_c = b"duplicate_image_c_content_also_shared"

        file_a = images_dir / "img_a.png"
        file_a.write_bytes(content_a)

        file_b1 = images_dir / "img_b1.png"
        file_b1.write_bytes(content_b)
        file_b2 = images_dir / "img_b2.png"
        file_b2.write_bytes(content_b)

        file_c1 = images_dir / "img_c1.png"
        file_c1.write_bytes(content_c)
        file_c2 = images_dir / "img_c2.png"
        file_c2.write_bytes(content_c)
        file_c3 = images_dir / "img_c3.png"
        file_c3.write_bytes(content_c)

        report = detect_exact_duplicates(images_dir)

        # Total 6 files: 3 unique content sets (a, b, c). Total duplicates = 1 for b + 2 for c = 3 duplicates.
        self.assertEqual(report.total_files_scanned, 6)
        self.assertEqual(report.unique_files_count, 3)
        self.assertEqual(report.duplicate_files_count, 3)
        self.assertEqual(len(report.duplicate_groups), 2)
        self.assertTrue(report.has_duplicates)

        # Verify all original files STILL exist (no silent deletion)
        for f in (file_a, file_b1, file_b2, file_c1, file_c2, file_c3):
            self.assertTrue(f.is_file(), f"File {f} was unexpectedly deleted!")

        # Verify report serialization
        json_path = self.root / "dup_report.json"
        csv_path = self.root / "dup_report.csv"
        report.write_json(json_path)
        report.write_csv(csv_path)

        self.assertTrue(json_path.is_file())
        self.assertTrue(csv_path.is_file())

        loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded_json["duplicate_files_count"], 3)

        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            self.assertEqual(len(rows), 4)  # 1 header + 3 duplicate rows

    # 5. Checkpoint provenance persistence
    def test_save_and_load_checkpoint_preserves_lineage_and_config(self) -> None:
        model = DummyModel(dim=16)
        checkpoint_path = self.root / "ckpt.pt"

        lineage_data = {
            "build_id": "test_build_999",
            "dictionary_fingerprint": "dict_abc",
            "dataset_fingerprint": "ds_xyz",
            "split_counts": {"train": 100, "valid": 10, "test": 10},
            "split_hashes": {"train": "thash", "valid": "vhash", "test": "testhash"},
        }
        config_data = {
            "model": {
                "name": "sketchformer_tok_dict",
                "architecture": {"d_model": 16, "num_encoder_layers": 2},
            }
        }
        metrics_data = {"val/loss": 0.42}

        save_checkpoint(
            checkpoint_path,
            model,
            epoch=3,
            step=420,
            metrics=metrics_data,
            config=config_data,
            lineage=lineage_data,
        )

        load_res = load_checkpoint(checkpoint_path, model)
        self.assertEqual(load_res.epoch, 3)
        self.assertEqual(load_res.step, 420)
        self.assertEqual(load_res.metrics["val/loss"], 0.42)
        self.assertEqual(load_res.config["model"]["name"], "sketchformer_tok_dict")
        self.assertEqual(load_res.lineage["build_id"], "test_build_999")
        self.assertEqual(load_res.lineage["dataset_fingerprint"], "ds_xyz")

    def test_checkpoint_callback_saves_lineage_to_last_and_best(self) -> None:
        model = DummyModel(dim=16)
        ckpt_dir = self.root / "cb_ckpts"
        lineage_data = {
            "build_id": "cb_build",
            "dataset_fingerprint": "cb_ds_fp",
            "dictionary_fingerprint": "cb_dict_fp",
        }
        config_data = {"model": {"name": "cb_model"}}

        cb = CheckpointCallback(
            directory=ckpt_dir,
            monitor="val/loss",
            mode="min",
            lineage=lineage_data,
            config=config_data,
        )

        saved = cb.on_validation_end(
            model,
            optimizer=None,
            scheduler=None,
            epoch=1,
            step=10,
            metrics={"val/loss": 0.5},
        )
        self.assertIn("last", saved)
        self.assertIn("best", saved)

        loaded_best = load_checkpoint(saved["best"], model)
        self.assertEqual(loaded_best.lineage["build_id"], "cb_build")
        self.assertEqual(loaded_best.config["model"]["name"], "cb_model")

    # 6. Checkpoint compatibility validation
    def test_validate_checkpoint_compatibility_rules(self) -> None:
        valid_lineage = {
            "build_id": "build_A",
            "dictionary_fingerprint": "dict_A",
            "dataset_fingerprint": "ds_A",
            "split_hashes": {"train": "t_hash", "valid": "v_hash", "test": "test_hash"},
        }
        valid_config = {
            "model": {
                "name": "sketchformer_tok_dict",
                "input": {"type": "tok_dict", "token_dictionary": {"vocab_size": 1004}},
                "architecture": {"d_model": 128, "num_encoder_layers": 4},
            }
        }

        # 1. Identical lineage & config -> Valid
        self.assertTrue(
            validate_checkpoint_compatibility(
                valid_lineage,
                valid_lineage,
                checkpoint_config=valid_config,
                active_config=valid_config,
            )
        )

        # 2. Dictionary fingerprint mismatch -> Rejects
        mismatched_dict = dict(valid_lineage, dictionary_fingerprint="dict_B")
        with self.assertRaises(IncompatibleCheckpointError) as ctx:
            validate_checkpoint_compatibility(
                valid_lineage,
                mismatched_dict,
                checkpoint_config=valid_config,
                active_config=valid_config,
            )
        self.assertIn("Token dictionary fingerprint mismatch", str(ctx.exception))

        # 3. Build ID mismatch -> Rejects
        mismatched_build = dict(valid_lineage, build_id="build_B")
        with self.assertRaises(IncompatibleCheckpointError) as ctx:
            validate_checkpoint_compatibility(
                valid_lineage,
                mismatched_build,
                checkpoint_config=valid_config,
                active_config=valid_config,
            )
        self.assertIn("Dataset build_id mismatch", str(ctx.exception))

        # 4. Dataset fingerprint mismatch -> Rejects
        mismatched_ds = dict(valid_lineage, dataset_fingerprint="ds_B")
        with self.assertRaises(IncompatibleCheckpointError) as ctx:
            validate_checkpoint_compatibility(
                valid_lineage,
                mismatched_ds,
                checkpoint_config=valid_config,
                active_config=valid_config,
            )
        self.assertIn("Dataset fingerprint mismatch", str(ctx.exception))

        # 5. Split hash mismatch -> Rejects
        mismatched_split = dict(
            valid_lineage,
            split_hashes={"train": "t_hash", "valid": "v_diff", "test": "test_hash"},
        )
        with self.assertRaises(IncompatibleCheckpointError) as ctx:
            validate_checkpoint_compatibility(
                valid_lineage,
                mismatched_split,
                split="valid",
                checkpoint_config=valid_config,
                active_config=valid_config,
            )
        self.assertIn("Dataset split 'valid' hash mismatch", str(ctx.exception))

        # 6. Model architecture mismatch (d_model) -> Rejects
        mismatched_model_cfg = {
            "model": {
                "name": "sketchformer_tok_dict",
                "input": {"type": "tok_dict", "token_dictionary": {"vocab_size": 1004}},
                "architecture": {"d_model": 256, "num_encoder_layers": 4},
            }
        }
        with self.assertRaises(IncompatibleCheckpointError) as ctx:
            validate_checkpoint_compatibility(
                valid_lineage,
                valid_lineage,
                checkpoint_config=valid_config,
                active_config=mismatched_model_cfg,
            )
        self.assertIn("Model architecture mismatch: checkpoint has 'd_model=128'", str(ctx.exception))

    # 7. Legacy unverified checkpoint handling
    def test_legacy_unverified_checkpoint_requires_explicit_opt_in(self) -> None:
        empty_lineage = {}
        target_dataset_lineage = {
            "build_id": "build_A",
            "dictionary_fingerprint": "dict_A",
            "dataset_fingerprint": "ds_A",
        }

        # Default: rejects with IncompatibleCheckpointError
        with self.assertRaises(IncompatibleCheckpointError) as ctx:
            validate_checkpoint_compatibility(
                empty_lineage,
                target_dataset_lineage,
                allow_unverified=False,
            )
        self.assertIn("Checkpoint lacks dataset lineage and provenance metadata", str(ctx.exception))
        self.assertIn("--allow-unverified-checkpoint", str(ctx.exception))

        # With allow_unverified=True: warns but passes
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            result = validate_checkpoint_compatibility(
                empty_lineage,
                target_dataset_lineage,
                allow_unverified=True,
            )
            self.assertTrue(result)
            self.assertTrue(any(issubclass(w.category, UserWarning) for w in recorded_warnings))

    # 8. CLI inspection of checkpoint provenance and dataset membership
    def test_inspect_checkpoint_cli_json_and_text(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        model = DummyModel(dim=16)
        ckpt_path = self.root / "test_cli_ckpt.pt"
        lineage_data = {
            "build_id": "build_cli_001",
            "dictionary_fingerprint": "dict_cli_fp",
            "dataset_fingerprint": "ds_cli_fp",
            "split_counts": {"train": 5, "valid": 1, "test": 1},
            "split_hashes": {"train": "thash", "valid": "vhash", "test": "testhash"},
        }
        save_checkpoint(
            ckpt_path,
            model,
            epoch=2,
            step=100,
            metrics={"val/loss": 0.25},
            lineage=lineage_data,
        )

        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        # Test text output
        result_text = subprocess.run(
            [
                sys.executable,
                "-B",
                "scripts/sketchformer/inspect_checkpoint_data.py",
                "--checkpoint",
                str(ckpt_path),
            ],
            cwd=project_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result_text.returncode, 0, msg=result_text.stderr)
        self.assertIn("CHECKPOINT PROVENANCE", result_text.stdout)
        self.assertIn("build_cli_001", result_text.stdout)
        self.assertIn("ds_cli_fp", result_text.stdout)

        # Test JSON output
        result_json = subprocess.run(
            [
                sys.executable,
                "-B",
                "scripts/sketchformer/inspect_checkpoint_data.py",
                "--checkpoint",
                str(ckpt_path),
                "--json",
            ],
            cwd=project_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result_json.returncode, 0, msg=result_json.stderr)
        parsed = json.loads(result_json.stdout)
        self.assertIn("checkpoint", parsed)
        self.assertEqual(parsed["checkpoint"]["lineage"]["build_id"], "build_cli_001")

    # 9. Export script lineage preservation and safetensors metadata
    def test_export_script_lineage_and_safetensors_metadata(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        model = DummyModel(dim=16)
        ckpt_path = self.root / "export_test_ckpt.pt"
        lineage_data = {
            "build_id": "build_export_123",
            "dictionary_fingerprint": "dict_export_fp",
            "dataset_fingerprint": "ds_export_fp",
            "split_counts": {"train": 10, "valid": 2, "test": 2},
            "split_hashes": {"train": "thash", "valid": "vhash", "test": "testhash"},
        }
        save_checkpoint(
            ckpt_path,
            model,
            epoch=1,
            step=50,
            lineage=lineage_data,
        )

        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        # 1. Export without lineage override should fail if checkpoint has no lineage
        empty_ckpt = self.root / "empty_ckpt.pt"
        save_checkpoint(empty_ckpt, model, epoch=1, step=10, lineage={})
        res_fail = subprocess.run(
            [
                sys.executable,
                "-B",
                "scripts/sketchformer/export.py",
                "--checkpoint",
                str(empty_ckpt),
                "--output",
                str(self.root / "out_fail.pt"),
            ],
            cwd=project_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(res_fail.returncode, 0)
        self.assertTrue(
            "UnverifiedCheckpointError" in res_fail.stderr
            or "IncompatibleCheckpointError" in res_fail.stderr
        )

        # 2. Export with --allow-unverified-checkpoint should succeed for empty checkpoint
        res_allow = subprocess.run(
            [
                sys.executable,
                "-B",
                "scripts/sketchformer/export.py",
                "--checkpoint",
                str(empty_ckpt),
                "--output",
                str(self.root / "out_allowed.pt"),
                "--allow-unverified-checkpoint",
            ],
            cwd=project_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(res_allow.returncode, 0, msg=res_allow.stderr)
        self.assertTrue((self.root / "out_allowed.pt").is_file())

        # 3. Export to .pt with verified lineage
        out_pt = self.root / "exported.pt"
        res_pt = subprocess.run(
            [
                sys.executable,
                "-B",
                "scripts/sketchformer/export.py",
                "--checkpoint",
                str(ckpt_path),
                "--output",
                str(out_pt),
            ],
            cwd=project_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(res_pt.returncode, 0, msg=res_pt.stderr)
        loaded_exported_pt = torch.load(out_pt, map_location="cpu")
        self.assertEqual(loaded_exported_pt["lineage"]["build_id"], "build_export_123")
        self.assertEqual(loaded_exported_pt["step"], 50)

        # 4. Export to .safetensors with verified lineage metadata
        out_st = self.root / "exported.safetensors"
        res_st = subprocess.run(
            [
                sys.executable,
                "-B",
                "scripts/sketchformer/export.py",
                "--checkpoint",
                str(ckpt_path),
                "--output",
                str(out_st),
            ],
            cwd=project_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(res_st.returncode, 0, msg=res_st.stderr)
        from safetensors import safe_open

        with safe_open(str(out_st), framework="pt") as f:
            st_meta = f.metadata()
        self.assertIsNotNone(st_meta)
        self.assertEqual(st_meta.get("build_id"), "build_export_123")
        self.assertEqual(st_meta.get("dataset_fingerprint"), "ds_export_fp")
        self.assertIn("lineage", st_meta)

    # 10. Deduplication CLI
    def test_deduplication_cli_runs_and_reports(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        img_dir = self.root / "cli_imgs"
        img_dir.mkdir(parents=True, exist_ok=True)

        (img_dir / "img1.png").write_bytes(b"ident1234")
        (img_dir / "img2.png").write_bytes(b"ident1234")
        (img_dir / "img3.png").write_bytes(b"unique999")

        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        report_json = self.root / "cli_dedup.json"
        res = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "prep_data.deduplication",
                "--input-dir",
                str(img_dir),
                "--report-json",
                str(report_json),
            ],
            cwd=project_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertTrue(report_json.is_file())
        data = json.loads(report_json.read_text(encoding="utf-8"))
        self.assertEqual(data["total_files_scanned"], 3)
        self.assertEqual(data["duplicate_files_count"], 1)
        self.assertEqual(data["unique_files_count"], 2)

    # 11. Chunk checksum verification and mutation detection
    def test_chunk_checksum_mutation_detection_and_dataset_fingerprint_sensitivity(self) -> None:
        dataset_dir = self.root / "chunk_cksum_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        chunk_path = dataset_dir / "train_000.npz"
        x_init = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        sample_ids = np.array(["s1", "s2"], dtype=object)
        np.savez_compressed(chunk_path, x=x_init, sample_ids=sample_ids)

        init_cksum = compute_chunk_checksum(chunk_path)
        self.assertEqual(len(init_cksum), 64)

        valid_chunk = dataset_dir / "valid_000.npz"
        np.savez_compressed(valid_chunk, x=np.array([[5.0, 6.0]], dtype=np.float32), sample_ids=np.array(["v1"]))
        valid_cksum = compute_chunk_checksum(valid_chunk)

        test_chunk = dataset_dir / "test_000.npz"
        np.savez_compressed(test_chunk, x=np.array([[7.0, 8.0]], dtype=np.float32), sample_ids=np.array(["t1"]))
        test_cksum = compute_chunk_checksum(test_chunk)

        chunk_checksums = {
            "train_000.npz": init_cksum,
            "valid_000.npz": valid_cksum,
            "test_000.npz": test_cksum,
        }
        split_hashes = {
            "train": compute_split_hash(["s1", "s2"]),
            "valid": compute_split_hash(["v1"]),
            "test": compute_split_hash(["t1"]),
        }
        ds_fp = compute_dataset_fingerprint(
            build_id="build_cksum_001",
            dictionary_fingerprint="dict_fp_1",
            split_hashes=split_hashes,
            split_counts={"train": 2, "valid": 1, "test": 1},
            train_chunks=["train_000.npz"],
            valid_chunks=["valid_000.npz"],
            test_chunks=["test_000.npz"],
            format_type="tok_dict",
            chunk_checksums=chunk_checksums,
            schema_version=2,
        )

        np.savez_compressed(
            dataset_dir / "meta.npz",
            build_id=np.array("build_cksum_001"),
            verified=np.array(True),
            dictionary_fingerprint=np.array("dict_fp_1"),
            dataset_fingerprint=np.array(ds_fp),
            split_counts=np.array([2, 1, 1]),
            train_chunks=np.array(["train_000.npz"]),
            valid_chunks=np.array(["valid_000.npz"]),
            test_chunks=np.array(["test_000.npz"]),
            chunk_checksums=json.dumps(chunk_checksums),
            schema_version=2,
            format=np.array("tok_dict"),
        )

        lineage = load_dataset_lineage(dataset_dir, verify_chunks=True)
        self.assertTrue(lineage.is_verified)
        self.assertEqual(lineage.dataset_fingerprint, ds_fp)

        # Mutate array data in train_000.npz WITHOUT modifying sample IDs or counts
        x_mutated = np.array([[99.0, 100.0], [101.0, 102.0]], dtype=np.float32)
        np.savez_compressed(chunk_path, x=x_mutated, sample_ids=sample_ids)

        mutated_cksum = compute_chunk_checksum(chunk_path)
        self.assertNotEqual(init_cksum, mutated_cksum)

        # Mutated chunk changes dataset fingerprint
        mutated_checksums = dict(chunk_checksums, **{"train_000.npz": mutated_cksum})
        mutated_ds_fp = compute_dataset_fingerprint(
            build_id="build_cksum_001",
            dictionary_fingerprint="dict_fp_1",
            split_hashes=split_hashes,
            split_counts={"train": 2, "valid": 1, "test": 1},
            train_chunks=["train_000.npz"],
            valid_chunks=["valid_000.npz"],
            test_chunks=["test_000.npz"],
            format_type="tok_dict",
            chunk_checksums=mutated_checksums,
            schema_version=2,
        )
        self.assertNotEqual(ds_fp, mutated_ds_fp)

        # Verifying dataset now fails because actual chunk checksum mismatches meta.npz
        with self.assertRaises(ValueError) as ctx:
            load_dataset_lineage(dataset_dir, verify_chunks=True)
        self.assertIn("Chunk checksum mismatch for", str(ctx.exception))

    # 12. Strict split counts and missing chunk verification
    def test_load_dataset_lineage_strict_split_counts_and_missing_chunks(self) -> None:
        dataset_dir = self.root / "strict_counts_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create valid chunk with 2 samples
        np.savez_compressed(
            dataset_dir / "train_000.npz",
            x=np.zeros((2, 5)),
            sample_ids=np.array(["s1", "s2"]),
        )
        # Declared count is 5 (mismatch!)
        np.savez_compressed(
            dataset_dir / "meta.npz",
            build_id=np.array("build_counts_mismatch"),
            verified=np.array(True),
            dictionary_fingerprint=np.array("dict_fp"),
            dataset_fingerprint=np.array("ds_fp_count"),
            split_counts=np.array([5, 0, 0]),
            train_chunks=np.array(["train_000.npz"]),
            valid_chunks=np.array([]),
            test_chunks=np.array([]),
            format=np.array("tok_dict"),
        )

        with self.assertRaises(ValueError) as ctx:
            load_dataset_lineage(dataset_dir, verify_chunks=True)
        self.assertIn("Split sample count mismatch for train", str(ctx.exception))

        # Missing declared chunk file raises FileNotFoundError
        np.savez_compressed(
            dataset_dir / "meta.npz",
            build_id=np.array("build_missing_chunk"),
            verified=np.array(True),
            dictionary_fingerprint=np.array("dict_fp"),
            dataset_fingerprint=np.array("ds_fp_missing"),
            split_counts=np.array([2, 1, 0]),
            train_chunks=np.array(["train_000.npz"]),
            valid_chunks=np.array(["nonexistent_valid_chunk.npz"]),
            test_chunks=np.array([]),
            format=np.array("tok_dict"),
        )
        with self.assertRaises(FileNotFoundError) as ctx:
            load_dataset_lineage(dataset_dir, verify_chunks=True)
        self.assertIn("nonexistent_valid_chunk.npz", str(ctx.exception))

    # 13. Token dictionary codebook artifact parity and corruption rejection
    def test_token_dictionary_artifact_parity_and_corruption_rejection(self) -> None:
        codebook_dir = self.root / "test_codebook_dir"
        codebook_dir.mkdir(parents=True, exist_ok=True)

        codebook_arr = np.random.randn(64, 5).astype(np.float32)

        npy_path, meta_path = save_codebook(codebook_arr, codebook_dir, K=64, n_samples=100)
        self.assertTrue(npy_path.is_file())
        self.assertTrue(meta_path.is_file())

        meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
        saved_cksum = meta_json["codebook_sha256"]
        self.assertEqual(len(saved_cksum), 64)

        # Parity between file checksum and in-memory array checksum
        array_cksum = compute_codebook_artifact_checksum(codebook_arr)
        file_cksum = compute_codebook_artifact_checksum(npy_path)
        self.assertEqual(saved_cksum, file_cksum)
        self.assertEqual(array_cksum, file_cksum)

        # Successful load
        loaded_cb, loaded_meta = load_codebook_from_dir(codebook_dir)
        np.testing.assert_array_equal(loaded_cb, codebook_arr)

        # Corrupt codebook file on disk
        with npy_path.open("r+b") as f:
            f.seek(128)
            f.write(b"\xff\xff\xff\xff")

        with self.assertRaises(ValueError) as ctx:
            load_codebook_from_dir(codebook_dir)
        self.assertIn("Codebook artifact checksum mismatch", str(ctx.exception))

    # 14. Token dictionary layout changes semantic fingerprint
    def test_token_dictionary_fingerprint_layout_sensitivity(self) -> None:
        codebook = np.random.randn(32, 5).astype(np.float32)
        fp_base = compute_token_dictionary_fingerprint(
            codebook=codebook,
            codebook_size=32,
            motion_token_offset=1,
            pad_token_id=0,
            sep_token_id=33,
            sos_token_id=34,
            eos_token_id=35,
            vocab_size=36,
        )
        self.assertEqual(len(fp_base), 64)

        # Changing special token layout changes fingerprint
        fp_different_pad = compute_token_dictionary_fingerprint(
            codebook=codebook,
            codebook_size=32,
            motion_token_offset=1,
            pad_token_id=99,
            sep_token_id=33,
            sos_token_id=34,
            eos_token_id=35,
            vocab_size=100,
        )
        self.assertNotEqual(fp_base, fp_different_pad)

    # 15. Checkpoint compatibility rejects corrupted provenance even with allow_unverified
    def test_checkpoint_compatibility_rejects_corrupted_provenance(self) -> None:
        # Partial provenance missing dataset_fingerprint
        corrupted_lineage = {
            "build_id": "build_partial",
            "dictionary_fingerprint": "dict_partial",
        }
        active_dataset_lineage = {
            "build_id": "build_partial",
            "dictionary_fingerprint": "dict_partial",
            "dataset_fingerprint": "ds_full",
        }

        # Must raise CorruptedProvenanceError with allow_unverified=False
        with self.assertRaises(CorruptedProvenanceError) as ctx:
            validate_checkpoint_compatibility(
                corrupted_lineage,
                active_dataset_lineage,
                allow_unverified=False,
            )
        self.assertIn("incomplete or corrupted provenance metadata", str(ctx.exception))

        # Must STILL raise CorruptedProvenanceError even with allow_unverified=True!
        with self.assertRaises(CorruptedProvenanceError) as ctx:
            validate_checkpoint_compatibility(
                corrupted_lineage,
                active_dataset_lineage,
                allow_unverified=True,
            )
        self.assertIn("incomplete or corrupted provenance metadata", str(ctx.exception))

        # Saving a checkpoint with incomplete lineage raises CorruptedProvenanceError
        model = DummyModel(dim=16)
        with self.assertRaises(CorruptedProvenanceError):
            save_checkpoint(
                self.root / "corrupted.pt",
                model,
                lineage={"build_id": "incomplete_only"},
            )

    # 16. Architecture validation on deep architecture parameters
    def test_checkpoint_compatibility_deep_architecture_parameters(self) -> None:
        base_lineage = {
            "build_id": "b1",
            "dictionary_fingerprint": "d1",
            "dataset_fingerprint": "ds1",
        }
        base_cfg = {
            "model": {
                "name": "sketchformer_continuous",
                "architecture": {
                    "d_model": 128,
                    "dim_feedforward": 512,
                    "latent_dim": 256,
                    "pooling_mode": "projected",
                    "latent_expander_mode": "projected_position",
                    "num_encoder_layers": 4,
                    "num_decoder_layers": 4,
                    "num_heads": 8,
                },
            }
        }

        # 1. dim_feedforward mismatch
        mismatched_ff = copy.deepcopy(base_cfg)
        mismatched_ff["model"]["architecture"]["dim_feedforward"] = 1024
        with self.assertRaises(IncompatibleCheckpointError) as ctx:
            validate_checkpoint_compatibility(
                base_lineage,
                base_lineage,
                checkpoint_config=base_cfg,
                active_config=mismatched_ff,
            )
        self.assertIn("dim_feedforward=512", str(ctx.exception))

        # 2. latent_dim mismatch
        mismatched_latent = copy.deepcopy(base_cfg)
        mismatched_latent["model"]["architecture"]["latent_dim"] = 512
        with self.assertRaises(IncompatibleCheckpointError) as ctx:
            validate_checkpoint_compatibility(
                base_lineage,
                base_lineage,
                checkpoint_config=base_cfg,
                active_config=mismatched_latent,
            )
        self.assertIn("latent_dim=256", str(ctx.exception))

        # 3. pooling_mode mismatch
        mismatched_pool = copy.deepcopy(base_cfg)
        mismatched_pool["model"]["architecture"]["pooling_mode"] = "mean"
        with self.assertRaises(IncompatibleCheckpointError) as ctx:
            validate_checkpoint_compatibility(
                base_lineage,
                base_lineage,
                checkpoint_config=base_cfg,
                active_config=mismatched_pool,
            )
        self.assertIn("pooling_mode=projected", str(ctx.exception))

        # 4. latent_expander_mode mismatch
        mismatched_expander = copy.deepcopy(base_cfg)
        mismatched_expander["model"]["architecture"]["latent_expander_mode"] = "tile"
        with self.assertRaises(IncompatibleCheckpointError) as ctx:
            validate_checkpoint_compatibility(
                base_lineage,
                base_lineage,
                checkpoint_config=base_cfg,
                active_config=mismatched_expander,
            )
        self.assertIn("latent_expander_mode=projected_position", str(ctx.exception))

    # 17. Evaluate CLI requires checkpoint and fails closed
    def test_evaluate_cli_requires_checkpoint_or_random_weights(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        # Missing checkpoint without --allow-random-weights must fail
        res = subprocess.run(
            [
                sys.executable,
                "-B",
                "scripts/sketchformer/evaluate.py",
                "--experiment",
                "smoke_test",
            ],
            cwd=project_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Evaluation requires a trained model checkpoint", res.stderr)

    # 18. Dataset preparation rejects duplicate build ID
    def test_dataset_preparation_rejects_duplicate_build_id(self) -> None:
        target_dir = self.root / "existing_build_target"
        target_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            target_dir / "meta.npz",
            build_id=np.array("immutable_build_001"),
            verified=np.array(True),
        )

        from prep_data.prepare_sketchformer_tokens import main as prep_tokens_main
        test_args = [
            "prepare_sketchformer_tokens.py",
            "--tokens-dir",
            str(self.root),
            "--target-dir",
            str(target_dir),
            "--build-id",
            "immutable_build_001",
        ]
        with unittest.mock.patch("sys.argv", test_args):
            with self.assertRaises(ValueError) as ctx:
                prep_tokens_main()
            self.assertIn("already exists", str(ctx.exception))
            self.assertIn("Dataset builds are immutable", str(ctx.exception))

    # 19. Codebook Checksum Enforcement and Legacy Opt-in
    def test_codebook_checksum_enforcement_and_legacy_opt_in(self) -> None:
        codebook_dir = self.root / "codebook_test"
        codebook_dir.mkdir(parents=True, exist_ok=True)
        cb_arr = np.random.randn(64, 16).astype(np.float32)
        save_codebook(cb_arr, codebook_dir, 64, 100)

        # Normal load succeeds and is verified
        cb, info = load_codebook_from_dir(codebook_dir)
        self.assertTrue(info.get("is_verified", False))
        self.assertTrue(bool(info.get("codebook_sha256")))

        # Tamper with codebook metadata by removing codebook_sha256 (simulating unverified legacy codebook)
        meta_path = codebook_dir / "metadata.json"
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        del meta["codebook_sha256"]
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f)

        # Strict load must reject unverified codebook
        with self.assertRaises(ValueError) as ctx:
            load_codebook_from_dir(codebook_dir, allow_unverified=False)
        self.assertIn("codebook_sha256", str(ctx.exception))

        # Fingerprint retrieval must reject unverified codebook by default
        with self.assertRaises(ValueError):
            get_token_dict_fingerprint(codebook_dir, allow_unverified=False)

        # Legacy opt-in succeeds but marks is_verified=False and does NOT fabricate checksum
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            cb, legacy_info = load_codebook_from_dir(codebook_dir, allow_unverified=True)
            self.assertFalse(legacy_info.get("is_verified", True))
            self.assertIsNone(legacy_info.get("codebook_sha256"))
            self.assertTrue(any("unverified legacy mode" in str(w.message) for w in ws))

        # Fingerprint retrieval in legacy mode returns empty string without fabricating
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            legacy_fp = get_token_dict_fingerprint(codebook_dir, allow_unverified=True)
            self.assertEqual(legacy_fp, "")

        # Mismatched recorded checksum must ALWAYS fail even with allow_unverified=True
        meta["codebook_sha256"] = "0" * 64
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f)
        with self.assertRaises(ValueError) as ctx:
            load_codebook_from_dir(codebook_dir, allow_unverified=True)
        self.assertIn("Codebook artifact checksum mismatch", str(ctx.exception))

    # 20. Mid-Run Chunk Mutation Detection and Cache Invalidation (Zero TOCTOU)
    def test_stroke_sequence_dataset_chunk_integrity_and_tampering_detection(self) -> None:
        ds_dir = self.root / "tamper_test_ds"
        ds_dir.mkdir(parents=True, exist_ok=True)
        train_file = ds_dir / "train_000.npz"
        valid_file = ds_dir / "valid.npz"
        test_file = ds_dir / "test.npz"

        x_sample = np.asarray([[[0.1, 0.2, 0.0], [0.3, 0.4, 1.0]]], dtype=object)
        y_sample = np.asarray([0], dtype=np.int32)
        np.savez_compressed(train_file, x=x_sample, y=y_sample, sample_ids=np.asarray(["s1"], dtype=object))
        np.savez_compressed(valid_file, x=x_sample, y=y_sample, sample_ids=np.asarray(["s2"], dtype=object))
        np.savez_compressed(test_file, x=x_sample, y=y_sample, sample_ids=np.asarray(["s3"], dtype=object))

        train_ck = compute_chunk_checksum(train_file)
        valid_ck = compute_chunk_checksum(valid_file)
        test_ck = compute_chunk_checksum(test_file)
        ck_map = {"train_000.npz": train_ck, "valid.npz": valid_ck, "test.npz": test_ck}

        split_hashes = {
            "train": compute_split_hash(["s1"]),
            "valid": compute_split_hash(["s2"]),
            "test": compute_split_hash(["s3"]),
        }
        split_counts = {"train": 1, "valid": 1, "test": 1}
        fp = compute_dataset_fingerprint(
            build_id="tamper_bld",
            dictionary_fingerprint="",
            split_hashes=split_hashes,
            split_counts=split_counts,
            train_chunks=["train_000.npz"],
            valid_chunks=["valid.npz"],
            test_chunks=["test.npz"],
            chunk_checksums=ck_map,
            format_type="stroke3",
            schema_version=2,
        )

        np.savez(
            ds_dir / "meta.npz",
            build_id="tamper_bld",
            dataset_fingerprint=fp,
            dictionary_fingerprint="",
            split_hash_train=split_hashes["train"],
            split_hash_valid=split_hashes["valid"],
            split_hash_test=split_hashes["test"],
            chunk_checksums=json.dumps(ck_map),
            train_chunks=np.asarray(["train_000.npz"], dtype=object),
            train_chunk_counts=np.asarray([1], dtype=np.int32),
            valid_chunks=np.asarray(["valid.npz"], dtype=object),
            valid_chunk_counts=np.asarray([1], dtype=np.int32),
            test_chunks=np.asarray(["test.npz"], dtype=object),
            test_chunk_counts=np.asarray([1], dtype=np.int32),
            n_samples_train=1,
            n_samples_valid=1,
            n_samples_test=1,
            is_published=True,
            format="stroke3",
            schema_version=2,
        )

        lineage = load_dataset_lineage(ds_dir, verify_chunks=True)
        self.assertTrue(lineage.is_verified)

        # Initialize StrokeSequenceDataset with lineage and read-time verification
        dataset = StrokeSequenceDataset(
            ds_dir,
            split="train",
            lineage=lineage,
            verify_integrity=True,
        )
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(sample["sample_id"], "s1")

        # Mutate the chunk on disk behind the dataset's back (simulating mid-run tampering or disk bit-rot)
        tampered_x = np.asarray([[[9.9, 9.9, 1.0]]], dtype=object)
        np.savez_compressed(train_file, x=tampered_x, y=y_sample, sample_ids=np.asarray(["s1"], dtype=object))

        # Accessing the dataset must immediately detect mtime change and chunk SHA-256 integrity mismatch
        with self.assertRaises(ValueError) as ctx:
            _ = dataset[0]
        self.assertIn("Chunk integrity check failed", str(ctx.exception))

    # 21. DataLoader Provenance Binding (root, build, and verification checks)
    def test_dataloader_provenance_binding(self) -> None:
        ds_dir = self.root / "binding_test_ds"
        ds_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ds_dir / "train_000.npz", x=np.asarray([[[0, 0, 1]]], dtype=object), y=np.asarray([0]), sample_ids=np.asarray(["s1"]))
        np.savez_compressed(ds_dir / "valid.npz", x=np.asarray([[[0, 0, 1]]], dtype=object), y=np.asarray([0]), sample_ids=np.asarray(["s2"]))
        np.savez_compressed(ds_dir / "test.npz", x=np.asarray([[[0, 0, 1]]], dtype=object), y=np.asarray([0]), sample_ids=np.asarray(["s3"]))

        ck_map = {
            "train_000.npz": compute_chunk_checksum(ds_dir / "train_000.npz"),
            "valid.npz": compute_chunk_checksum(ds_dir / "valid.npz"),
            "test.npz": compute_chunk_checksum(ds_dir / "test.npz"),
        }
        split_hashes = {
            "train": compute_split_hash(["s1"]),
            "valid": compute_split_hash(["s2"]),
            "test": compute_split_hash(["s3"]),
        }
        fp = compute_dataset_fingerprint(
            build_id="bld_123",
            dictionary_fingerprint="",
            split_hashes=split_hashes,
            split_counts={"train": 1, "valid": 1, "test": 1},
            train_chunks=["train_000.npz"],
            valid_chunks=["valid.npz"],
            test_chunks=["test.npz"],
            chunk_checksums=ck_map,
            format_type="stroke3",
            schema_version=2,
        )
        np.savez(
            ds_dir / "meta.npz",
            build_id="bld_123",
            dataset_fingerprint=fp,
            dictionary_fingerprint="",
            split_hash_train=split_hashes["train"],
            split_hash_valid=split_hashes["valid"],
            split_hash_test=split_hashes["test"],
            chunk_checksums=json.dumps(ck_map),
            train_chunks=np.asarray(["train_000.npz"]),
            train_chunk_counts=np.asarray([1]),
            valid_chunks=np.asarray(["valid.npz"]),
            valid_chunk_counts=np.asarray([1]),
            test_chunks=np.asarray(["test.npz"]),
            test_chunk_counts=np.asarray([1]),
            n_samples_train=1,
            n_samples_valid=1,
            n_samples_test=1,
            is_published=True,
            format="stroke3",
            schema_version=2,
        )

        lineage = load_dataset_lineage(ds_dir, verify_chunks=True)

        # Mismatched root must be rejected
        other_dir = self.root / "other_dir"
        other_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(ds_dir / "meta.npz", other_dir / "meta.npz")
        shutil.copy(ds_dir / "train_000.npz", other_dir / "train_000.npz")
        shutil.copy(ds_dir / "valid.npz", other_dir / "valid.npz")
        shutil.copy(ds_dir / "test.npz", other_dir / "test.npz")
        with self.assertRaises(ValueError) as ctx:
            StrokeSequenceDataset(other_dir, split="train", lineage=lineage)
        self.assertIn("DataLoader root mismatch", str(ctx.exception))

        # Unverified lineage must be rejected unless allow_unverified=True
        unverified_lineage = copy.deepcopy(lineage)
        object.__setattr__(unverified_lineage, "is_verified", False)
        with self.assertRaises(ValueError) as ctx:
            StrokeSequenceDataset(ds_dir, split="train", lineage=unverified_lineage, allow_unverified=False)
        self.assertIn("Provided dataset lineage is unverified", str(ctx.exception))

        # StrokeSequenceDataModule forwards lineage and allow_unverified
        dm_config = {
            "dataset": {"root": str(ds_dir)},
            "sequence": {"max_length": 100},
            "format": {"type": "stroke3"},
        }
        dm = StrokeSequenceDataModule(dm_config, lineage=lineage)
        dm.setup("fit")
        self.assertIsNotNone(dm.train_dataset)
        self.assertEqual(dm.train_dataset.lineage, lineage)

    # 22. Detection and Rejection of Unexpected / Untracked Chunks
    def test_load_dataset_lineage_detects_unexpected_untracked_chunks(self) -> None:
        ds_dir = self.root / "untracked_test_ds"
        ds_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ds_dir / "train_000.npz", x=np.asarray([[[0, 0, 1]]], dtype=object), y=np.asarray([0]), sample_ids=np.asarray(["s1"]))
        np.savez_compressed(ds_dir / "valid.npz", x=np.asarray([[[0, 0, 1]]], dtype=object), y=np.asarray([0]), sample_ids=np.asarray(["s2"]))
        np.savez_compressed(ds_dir / "test.npz", x=np.asarray([[[0, 0, 1]]], dtype=object), y=np.asarray([0]), sample_ids=np.asarray(["s3"]))

        ck_map = {
            "train_000.npz": compute_chunk_checksum(ds_dir / "train_000.npz"),
            "valid.npz": compute_chunk_checksum(ds_dir / "valid.npz"),
            "test.npz": compute_chunk_checksum(ds_dir / "test.npz"),
        }
        split_hashes = {
            "train": compute_split_hash(["s1"]),
            "valid": compute_split_hash(["s2"]),
            "test": compute_split_hash(["s3"]),
        }
        fp = compute_dataset_fingerprint(
            build_id="bld_untracked",
            dictionary_fingerprint="",
            split_hashes=split_hashes,
            split_counts={"train": 1, "valid": 1, "test": 1},
            train_chunks=["train_000.npz"],
            valid_chunks=["valid.npz"],
            test_chunks=["test.npz"],
            chunk_checksums=ck_map,
            format_type="stroke3",
            schema_version=2,
        )
        np.savez(
            ds_dir / "meta.npz",
            build_id="bld_untracked",
            dataset_fingerprint=fp,
            dictionary_fingerprint="",
            split_hash_train=split_hashes["train"],
            split_hash_valid=split_hashes["valid"],
            split_hash_test=split_hashes["test"],
            chunk_checksums=json.dumps(ck_map),
            train_chunks=np.asarray(["train_000.npz"]),
            train_chunk_counts=np.asarray([1]),
            valid_chunks=np.asarray(["valid.npz"]),
            valid_chunk_counts=np.asarray([1]),
            test_chunks=np.asarray(["test.npz"]),
            test_chunk_counts=np.asarray([1]),
            n_samples_train=1,
            n_samples_valid=1,
            n_samples_test=1,
            is_published=True,
            format="stroke3",
            schema_version=2,
        )

        # Drop a rogue untracked chunk file into the dataset directory
        rogue_file = ds_dir / "rogue_extra_chunk.npz"
        np.savez_compressed(rogue_file, x=np.asarray([[[1, 1, 1]]], dtype=object), y=np.asarray([1]))

        # Strict verification must detect and reject the untracked chunk
        with self.assertRaises(ValueError) as ctx:
            load_dataset_lineage(ds_dir, verify_chunks=True, reject_unexpected_chunks=True)
        self.assertIn("Unexpected untracked dataset chunk", str(ctx.exception))
        self.assertIn("rogue_extra_chunk.npz", str(ctx.exception))

        # In allow_unverified mode, it loads with unexpected_chunks recorded in metadata
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lin = load_dataset_lineage(ds_dir, verify_chunks=True, reject_unexpected_chunks=True, allow_unverified=True)
            self.assertIn("unexpected_chunks", lin.metadata)
            self.assertIn("rogue_extra_chunk.npz", lin.metadata["unexpected_chunks"])

    # 23. Legacy Stroke-3 Dataset Handling and Migration
    def test_legacy_stroke3_dataset_rejection_and_migration(self) -> None:
        ds_dir = self.root / "legacy_stroke3_ds"
        ds_dir.mkdir(parents=True, exist_ok=True)
        train_file = ds_dir / "train_000.npz"
        valid_file = ds_dir / "valid.npz"
        test_file = ds_dir / "test.npz"

        # Legacy stroke3 chunks have x and y, but NO sample_ids
        np.savez_compressed(train_file, x=np.asarray([[[0, 0, 1]], [[0, 1, 1]]], dtype=object), y=np.asarray([0, 0]))
        np.savez_compressed(valid_file, x=np.asarray([[[1, 0, 1]]], dtype=object), y=np.asarray([0]))
        np.savez_compressed(test_file, x=np.asarray([[[1, 1, 1]]], dtype=object), y=np.asarray([0]))

        # Legacy meta.npz lacks build_id, dataset_fingerprint, chunk_checksums
        np.savez(
            ds_dir / "meta.npz",
            train_chunks=np.asarray(["train_000.npz"]),
            train_chunk_counts=np.asarray([2]),
            valid_chunks=np.asarray(["valid.npz"]),
            valid_chunk_counts=np.asarray([1]),
            test_chunks=np.asarray(["test.npz"]),
            test_chunk_counts=np.asarray([1]),
            n_samples_train=2,
            n_samples_valid=1,
            n_samples_test=1,
            is_published=True,
        )

        # 1. Strict loading must reject the legacy dataset
        with self.assertRaises(ValueError) as ctx:
            load_dataset_lineage(ds_dir, allow_unverified=False)
        self.assertIn("missing required lineage fields", str(ctx.exception))

        # 2. Explicit opt-in allows loading with is_verified=False and empty fingerprints (no fabrication)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            legacy_lin = load_dataset_lineage(ds_dir, allow_unverified=True)
            self.assertFalse(legacy_lin.is_verified)
            self.assertEqual(legacy_lin.build_id, "")
            self.assertEqual(legacy_lin.dataset_fingerprint, "")
            self.assertTrue(any("unverified legacy mode" in str(w.message) for w in ws))

        # 3. Run migration utility to upgrade legacy dataset to modern schema v2
        mig_result = migrate_legacy_stroke3_dataset(ds_dir, build_id="stroke3_migrated_v1")
        self.assertEqual(mig_result["build_id"], "stroke3_migrated_v1")
        self.assertTrue(len(mig_result["dataset_fingerprint"]) == 64)

        # 4. Strict load of migrated dataset now succeeds and is fully verified
        migrated_lin = load_dataset_lineage(ds_dir, verify_chunks=True)
        self.assertTrue(migrated_lin.is_verified)
        self.assertEqual(migrated_lin.build_id, "stroke3_migrated_v1")
        self.assertEqual(migrated_lin.schema_version, 2)
        self.assertEqual(len(migrated_lin.chunk_checksums), 3)

        # Chunks now contain valid sample_ids with legacy_generated prefix
        with np.load(train_file, allow_pickle=True) as c:
            self.assertIn("sample_ids", c)
            self.assertEqual(len(c["sample_ids"]), 2)
            gen_id = str(c["sample_ids"][0])
            self.assertTrue(gen_id.startswith("legacy_generated:"))

        # Migration result and meta.npz contain explicit legacy migration flags
        self.assertTrue(mig_result["sample_ids_are_generated"])
        self.assertFalse(mig_result["has_original_sample_ids"])
        self.assertTrue(mig_result["is_migrated_legacy"])

        with np.load(ds_dir / "meta.npz", allow_pickle=True) as m:
            self.assertTrue(bool(m["sample_ids_are_generated"]))
            self.assertFalse(bool(m["has_original_sample_ids"]))
            self.assertTrue(bool(m["is_migrated_legacy"]))

        # query_sample_membership identifies synthetic legacy ID
        membership = query_sample_membership(gen_id, ds_dir)
        self.assertEqual(membership["split"], "train")
        self.assertTrue(membership["is_synthetic_id"])

    def test_27_cache_immutability_and_defensive_copying(self):
        """Test that StrokeSequenceDataset enforces read-only cached buffers and defensive copies."""
        ds_dir = self.root / "ds_cache_test"
        ds_dir.mkdir(parents=True, exist_ok=True)

        x_sample = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        train_file = ds_dir / "train_000.npz"
        np.savez_compressed(
            train_file,
            x=np.array([x_sample.copy(), x_sample.copy()], dtype=object),
            y=np.array([0, 1], dtype=np.int64),
            sample_ids=np.array(["cache_s0", "cache_s1"], dtype=object),
        )
        valid_file = ds_dir / "valid.npz"
        test_file = ds_dir / "test.npz"
        np.savez_compressed(
            valid_file,
            x=np.array([x_sample.copy()], dtype=object),
            y=np.array([0], dtype=np.int64),
            sample_ids=np.array(["cache_v0"], dtype=object),
        )
        np.savez_compressed(
            test_file,
            x=np.array([x_sample.copy()], dtype=object),
            y=np.array([1], dtype=np.int64),
            sample_ids=np.array(["cache_t0"], dtype=object),
        )

        train_ck = compute_chunk_checksum(train_file)
        val_ck = compute_chunk_checksum(valid_file)
        test_ck = compute_chunk_checksum(test_file)
        ck_map = {"train_000.npz": train_ck, "valid.npz": val_ck, "test.npz": test_ck}

        split_hashes = {
            "train": compute_split_hash(["cache_s0", "cache_s1"]),
            "valid": compute_split_hash(["cache_v0"]),
            "test": compute_split_hash(["cache_t0"]),
        }
        split_counts = {"train": 2, "valid": 1, "test": 1}

        fp = compute_dataset_fingerprint(
            build_id="cache_immutability_v1",
            dictionary_fingerprint="",
            split_hashes=split_hashes,
            split_counts=split_counts,
            train_chunks=["train_000.npz"],
            valid_chunks=["valid.npz"],
            test_chunks=["test.npz"],
            chunk_checksums=ck_map,
            format_type="stroke3",
            schema_version=2,
        )

        np.savez(
            ds_dir / "meta.npz",
            build_id="cache_immutability_v1",
            dataset_fingerprint=fp,
            dictionary_fingerprint="",
            split_counts=np.array([2, 1, 1]),
            train_chunks=["train_000.npz"],
            valid_chunks=["valid.npz"],
            test_chunks=["test.npz"],
            chunk_checksums=json.dumps(ck_map),
            format="stroke3",
            schema_version=2,
            is_published=True,
        )

        lineage = load_dataset_lineage(ds_dir, verify_chunks=True)
        ds = StrokeSequenceDataset(
            root=ds_dir,
            split="train",
            format_type="stroke3",
            lineage=lineage,
            verify_integrity=True,
        )

        # 1. Fetch sample 0
        s0 = ds[0]
        original_x0 = s0["stroke3"][0, 0]

        # 2. Mutate sample in place
        s0["stroke3"][:, :2] += 999.0

        # 3. Fetch sample 0 again from dataset - verify cache was NOT mutated
        s0_again = ds[0]
        self.assertEqual(s0_again["stroke3"][0, 0], original_x0)
        self.assertNotEqual(s0_again["stroke3"][0, 0], 999.0)

        # 4. Check that cached chunk arrays are marked read-only
        cached_entry = ds._cache[train_file]
        cached_data = cached_entry[0]
        self.assertFalse(cached_data["y"].flags.writeable)
        with self.assertRaises(ValueError):
            cached_data["y"][0] = 99

        # Nested arrays in object array x are also read-only
        self.assertFalse(cached_data["x"][0].flags.writeable)
        with self.assertRaises(ValueError):
            cached_data["x"][0][0, 0] = 999.0

    def test_28_expanded_architecture_compatibility(self):
        """Test that validate_checkpoint_compatibility covers all structural architecture parameters."""
        valid_lineage = {
            "build_id": "b1",
            "dictionary_fingerprint": "dict_fp_1",
            "dataset_fingerprint": "ds_fp_1",
            "split_hashes": {"train": "h_tr"},
        }

        base_config = {
            "name": "sketchformer_continuous",
            "input": {
                "type": "stroke3",
                "stroke_dim": 3,
                "pen_classes": 3,
                "token_dictionary": {"vocab_size": 1004},
            },
            "architecture": {
                "d_model": 128,
                "latent_dim": 256,
                "num_encoder_layers": 4,
                "num_decoder_layers": 4,
                "num_heads": 8,
                "dim_feedforward": 512,
                "pooling_mode": "projected",
                "pool_hidden_dim": 256,
                "latent_expander_mode": "projected_position",
            },
            "embedding": {
                "pen_embedding_dim": 32,
                "combine_method": "add",
                "positional_encoding": {"type": "learned", "max_length": 2048},
            },
            "heads": {
                "reconstruction": {
                    "enabled": True,
                    "target": "continuous",
                    "xy_distribution": "gaussian_mixture",
                    "num_mixtures": 20,
                    "predict_pen_state": True,
                },
                "classification": {
                    "enabled": False,
                    "num_classes": 345,
                },
            },
        }

        # Compatible configs succeed
        self.assertTrue(
            validate_checkpoint_compatibility(
                checkpoint_lineage=valid_lineage,
                dataset_lineage=valid_lineage,
                checkpoint_config={"model": base_config},
                active_config={"model": base_config},
            )
        )

        # Mismatch in stroke_dim
        import copy
        mismatch_dim = copy.deepcopy(base_config)
        mismatch_dim["input"]["stroke_dim"] = 5
        with self.assertRaises(IncompatibleCheckpointError) as cm:
            validate_checkpoint_compatibility(
                checkpoint_lineage=valid_lineage,
                dataset_lineage=valid_lineage,
                checkpoint_config={"model": base_config},
                active_config={"model": mismatch_dim},
            )
        self.assertIn("stroke_dim", str(cm.exception))

        # Mismatch in positional_encoding.type
        mismatch_pe = copy.deepcopy(base_config)
        mismatch_pe["embedding"]["positional_encoding"]["type"] = "sinusoidal"
        with self.assertRaises(IncompatibleCheckpointError) as cm:
            validate_checkpoint_compatibility(
                checkpoint_lineage=valid_lineage,
                dataset_lineage=valid_lineage,
                checkpoint_config={"model": base_config},
                active_config={"model": mismatch_pe},
            )
        self.assertIn("positional_encoding.type", str(cm.exception))

        # Mismatch in reconstruction.xy_distribution
        mismatch_recon = copy.deepcopy(base_config)
        mismatch_recon["heads"]["reconstruction"]["xy_distribution"] = "regression"
        with self.assertRaises(IncompatibleCheckpointError) as cm:
            validate_checkpoint_compatibility(
                checkpoint_lineage=valid_lineage,
                dataset_lineage=valid_lineage,
                checkpoint_config={"model": base_config},
                active_config={"model": mismatch_recon},
            )
        self.assertIn("reconstruction.xy_distribution", str(cm.exception))

        # Mismatch in classification.enabled
        mismatch_cls = copy.deepcopy(base_config)
        mismatch_cls["heads"]["classification"]["enabled"] = True
        with self.assertRaises(IncompatibleCheckpointError) as cm:
            validate_checkpoint_compatibility(
                checkpoint_lineage=valid_lineage,
                dataset_lineage=valid_lineage,
                checkpoint_config={"model": base_config},
                active_config={"model": mismatch_cls},
            )
        self.assertIn("classification.enabled", str(cm.exception))


if __name__ == "__main__":
    unittest.main()


