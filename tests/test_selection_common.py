import copy
import json
import tempfile
import unittest
from pathlib import Path
from evals.selection_common import validate_report, read_json, read_jsonl, inference_settings, json_sha256
from tests.selection_fixtures import synthetic_report, write
from evals.benchmark_common import select_comparison_samples


class ReportValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path, self.manifest, self.book, self.candidate = synthetic_report(self.root)

    def validate(self):
        return validate_report(self.path, self.manifest, self.book, self.candidate)

    def test_valid_saved_evidence(self):
        self.validate()

    def test_missing_duplicate_unexpected_and_changed_targets(self):
        path = self.root / 'a_per_sample.jsonl'
        original = path.read_text()
        row = read_jsonl(path)[0]
        for content in ['', original + original, original.replace('"sample"', '"unexpected"'),
                        original.replace('"target_length": 5', '"target_length": 6'),
                        original.replace('"target_tokens": [6, 1, 2', '"target_tokens": [6, 2, 2')]:
            path.write_text(content)
            with self.assertRaises(ValueError):
                self.validate()
        path.write_text(original)
        self.validate()

    def test_scores_totals_dictionary_and_images(self):
        report = read_json(self.path)
        for field in ('metrics', 'per_bucket', 'total_samples'):
            altered = copy.deepcopy(report)
            altered['free_running'][field] = {} if field != 'total_samples' else 2
            write(self.path, altered)
            with self.assertRaises(ValueError): self.validate()
        write(self.path, report)
        path = self.root / 'a_per_sample.jsonl'
        original = path.read_text()
        path.write_text(original.replace('"drawing_score": 1.0', '"drawing_score": 0.5'))
        with self.assertRaises(ValueError): self.validate()
        path.write_text(original)
        book_bytes = self.book.read_bytes()
        self.book.write_bytes(book_bytes + b'changed')
        with self.assertRaises(ValueError): self.validate()
        self.book.write_bytes(book_bytes)
        (self.root / 'a.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing comparison image'): self.validate()

    def test_frozen_specification_and_checkpoint(self):
        report = read_json(self.path)
        report['evaluation']['inference_settings']['generation_limit'] = 7
        write(self.path, report)
        with self.assertRaisesRegex(ValueError, 'generation_limit'): self.validate()
        report['evaluation']['inference_settings']['generation_limit'] = 8
        write(self.path, report)
        Path(self.candidate['path']).write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError, 'checkpoint hash'): self.validate()

    def test_deterministic_subset_across_buckets(self):
        rows = [dict(sample_id=str(i), length_bucket='short' if i < 8 else 'extended', target_length=20 if i < 8 else 3000) for i in range(10)]
        a = select_comparison_samples(rows, 4, 42)
        self.assertEqual(a, select_comparison_samples(rows[::-1], 4, 42))
        self.assertEqual({r['length_bucket'] for r in a}, {'short', 'extended'})

    def test_invalid_scores_and_changed_full_token_evidence(self):
        path = self.root / 'a_per_sample.jsonl'
        original = read_jsonl(path)[0]
        for value in (-.1, 1.1, float('nan'), float('inf')):
            row = copy.deepcopy(original)
            row['drawing_score'] = value
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'drawing_score'):
                self.validate()
        row = copy.deepcopy(original)
        row['validation_tokens'].append(2)
        path.write_text(json.dumps(row))
        with self.assertRaises(ValueError): self.validate()

    def test_candidate_identity_and_fingerprint_changes(self):
        report = read_json(self.path)
        for field, value in [('inference_fingerprint', 'changed'), ('dictionary', {})]:
            altered = copy.deepcopy(report)
            altered['evaluation'][field] = value
            write(self.path, altered)
            with self.assertRaises(ValueError): self.validate()
        write(self.path, report)
        path = self.root / 'a_per_sample.jsonl'
        row = read_jsonl(path)[0]
        row['candidate_id'] = 'another'
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'record candidate'): self.validate()
