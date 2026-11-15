"""Small-array tests of the real-input identity and masked cross-framework gate."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from evals.conversion_fidelity import _file_hash, evaluate_verified_parity
from scripts.sketchformer.tf_reference_export import LAYOUT, WIDTH, select_real_sequences, validate_input


class VerifiedParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dictionary = self.root / "original.pkl"
        self.dictionary.write_bytes(b"original ordered clusters")
        self.codebook = self.root / "codebook.npy"
        np.save(self.codebook, np.zeros((1000, 2), dtype=np.float32))
        self.input = self.root / "input.npz"
        self.reference = self.root / "tf.npz"
        self.candidate = self.root / "torch.npz"
        self.checkpoint = self.root / "weights.safetensors"
        self.prefix = self.root / "ckpt-12"
        self.original_config = self.root / "original-config.json"
        self.original_config.write_text('{"max_seq_len":200,"token_type":"dictionary"}', encoding="utf-8")
        self.tf_hash = "a" * 64
        self.layout = json.dumps(LAYOUT, sort_keys=True, separators=(",", ":"))
        self.dictionary_sha = _file_hash(self.dictionary)
        self.codebook_sha = _file_hash(self.codebook)
        self.patches = [
            patch("scripts.sketchformer.tf_reference_export.original_codebook_sha256",
                  return_value=(self.dictionary_sha, self.codebook_sha)),
            patch("models.sketchformer.pretrained.fingerprint_tensorflow_checkpoint",
                  return_value=self.tf_hash),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        save_file({"weight": torch.zeros(1)}, str(self.checkpoint), metadata={
            "codebook_sha256": self.codebook_sha,
            "source_tf_checkpoint_sha256": self.tf_hash,
            "token_layout_json": self.layout,
        })
        self.tokens = np.zeros((2, WIDTH), dtype=np.int64)
        self.tokens[0, :3] = [1002, 1, 1003]
        self.tokens[1, :4] = [1002, 2, 3, 1003]
        self.lengths = np.array([3, 4], dtype=np.int64)
        self.ids = np.asarray(["real-1", "real-2"])
        self.dataset = self.root / "prepared"
        self.dataset.mkdir()
        self.dataset_path = self.dataset / "valid.npz"
        raw = np.zeros((2, 8), dtype=np.int64)
        raw[0, 0] = 1
        raw[1, :2] = [2, 3]
        np.savez(self.dataset_path, x=raw, y=np.array([0, 0]), sample_ids=self.ids)
        self.input_meta = {
            "original_dictionary_sha256": self.dictionary_sha,
            "codebook_sha256": self.codebook_sha,
            "token_layout": LAYOUT,
            "preprocessing": "prepared_tok_dict_no_augmentation_sos_eos_pad200",
            "dataset_split": "valid", "dataset_source": str(self.dataset),
            "dataset_files_sha256": {"valid.npz": _file_hash(self.dataset_path)},
            "source_rows": [
                {"file": "valid.npz", "index": 0},
                {"file": "valid.npz", "index": 1},
            ],
        }
        self.save_input()
        self.logits = np.zeros((2, WIDTH - 1, 1004), dtype=np.float32)
        self.logits[:, :, 1] = 2.0
        self.save_exports()

    def save_input(self):
        np.savez(self.input, tokens=self.tokens, lengths=self.lengths,
                 sample_ids=self.ids, metadata=json.dumps(self.input_meta))

    def save_exports(self, reference=None, candidate=None, reference_meta=None,
                     candidate_meta=None, candidate_ids=None):
        common = {"input_sha256": _file_hash(self.input),
                  "source_tf_checkpoint_sha256": self.tf_hash,
                  "codebook_sha256": self.codebook_sha,
                  "original_dictionary_sha256": self.dictionary_sha,
                  "token_layout_json": self.layout,
                  "logit_layout": "tokens[:, :-1] predict tokens[:, 1:]"}
        ref = {**common, "framework": "tensorflow_original",
               "original_config_sha256": _file_hash(self.original_config)}
        cand = {**common, "framework": "pytorch_converted",
                "converted_checkpoint_sha256": _file_hash(self.checkpoint)}
        np.savez(self.reference, logits=self.logits if reference is None else reference,
                 tokens=self.tokens, lengths=self.lengths, sample_ids=self.ids,
                 metadata=json.dumps(ref if reference_meta is None else {**ref, **reference_meta}))
        np.savez(self.candidate, logits=self.logits if candidate is None else candidate,
                 tokens=self.tokens, lengths=self.lengths,
                 sample_ids=self.ids if candidate_ids is None else candidate_ids,
                 metadata=json.dumps(cand if candidate_meta is None else {**cand, **candidate_meta}))

    def gate(self):
        return evaluate_verified_parity(
            self.candidate, self.reference, self.input, candidate_checkpoint=self.checkpoint,
            original_checkpoint=self.prefix, original_config=self.original_config,
            original_dictionary=self.dictionary,
            codebook=self.codebook,
        )

    def test_selects_short_samples_even_when_storage_rows_are_padded(self):
        cfg = {"data": {
            "format": {"type": "tok_dict", "token_dictionary": {
                "pad_token_id": 0, "sep_token_id": 1001,
                "sos_token_id": 1002, "eos_token_id": 1003,
            }},
            "dataset": {"root": str(self.dataset)},
            "sequence": {"max_length": 4096, "truncate_long_sequences": False},
            "augmentation": {"enabled": False},
        }}
        selected = self.root / "selected.npz"
        with patch("scripts.sketchformer.config.compose_training_config", return_value=cfg):
            select_real_sequences(
                "configs/train.yaml", "anime_tok_dict_long", "valid", 2,
                self.dictionary, self.codebook, selected,
            )
        with np.load(selected, allow_pickle=False) as payload:
            np.testing.assert_array_equal(payload["tokens"], self.tokens)
            np.testing.assert_array_equal(payload["lengths"], self.lengths)

    def test_identical_real_positions_pass_and_padding_ignored(self):
        modified = self.logits.copy()
        modified[0, 2:, :] = -100
        modified[1, 3:, :] = 100
        self.save_exports(candidate=modified)
        result = self.gate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["num_positions_evaluated"], 5)
        self.assertEqual(result["verification"], "original_tensorflow_vs_converted_pytorch")

    def test_altered_logit_without_argmax_change_fails(self):
        modified = self.logits.copy()
        modified[0, 0, 1] += 0.1
        self.save_exports(candidate=modified)
        result = self.gate()
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["thresholds"]["absolute_difference_passed"])


    def test_altered_nonpadding_logits_fail(self):
        modified = self.logits.copy()
        modified[0, 0, 1] = -2
        modified[0, 0, 2] = 3
        self.save_exports(candidate=modified)
        self.assertEqual(self.gate()["status"], "FAIL")

    def test_non_tf_reference_and_mismatched_samples_rejected(self):
        self.save_exports(reference_meta={"framework": "pytorch_converted"})
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.gate()
        self.save_exports(reference_meta={"original_config_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.gate()
        self.save_exports(candidate_ids=np.asarray(["real-2", "real-1"]))
        with self.assertRaisesRegex(ValueError, "sample_ids"):
            self.gate()

    def test_wrong_codebook_checkpoint_and_changed_input_rejected(self):
        self.input_meta["codebook_sha256"] = "wrong"
        self.save_input()
        with self.assertRaisesRegex(ValueError, "dictionary/codebook"):
            self.gate()
        self.input_meta["codebook_sha256"] = self.codebook_sha
        self.save_input()
        save_file({"weight": torch.zeros(1)}, str(self.checkpoint), metadata={
            "codebook_sha256": self.codebook_sha, "source_tf_checkpoint_sha256": "wrong",
            "token_layout_json": self.layout,
        })
        with self.assertRaisesRegex(ValueError, "checkpoint lacks"):
            self.gate()
        save_file({"weight": torch.zeros(1)}, str(self.checkpoint), metadata={
            "codebook_sha256": self.codebook_sha, "source_tf_checkpoint_sha256": self.tf_hash,
            "token_layout_json": self.layout,
        })
        self.tokens[0, 1] = 4
        self.save_input()
        with self.assertRaisesRegex(ValueError, "differs from real prepared sample"):
            self.gate()

    def test_dataset_source_is_required_and_tampering_is_detected(self):
        rows = self.input_meta.pop("source_rows")
        self.save_input()
        with self.assertRaisesRegex(ValueError, "source rows"):
            self.gate()
        self.input_meta["source_rows"] = rows
        self.save_input()
        self.dataset_path.write_bytes(b"modified dataset")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.gate()

    def test_invalid_tokens_and_padding_rejected(self):
        self.tokens[0, 3] = 1
        self.save_input()
        with self.assertRaisesRegex(ValueError, "padding"):
            validate_input(self.input, original_dictionary=self.dictionary, codebook=self.codebook)
        self.tokens[0, 3] = 0
        self.tokens[1, 1] = LAYOUT["eos_token_id"]
        self.save_input()
        with self.assertRaisesRegex(ValueError, "Interior EOS"):
            validate_input(self.input, original_dictionary=self.dictionary, codebook=self.codebook)
        self.tokens[1, 1] = 2
        self.ids[1] = self.ids[0]
        self.save_input()
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_input(self.input, original_dictionary=self.dictionary, codebook=self.codebook)


if __name__ == "__main__":
    unittest.main()
