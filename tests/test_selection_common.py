import copy
import json
import tempfile
import unittest
from pathlib import Path
from evals.selection_common import validate_comparison_manifest, validate_report, read_json, read_jsonl, inference_settings, json_sha256
from tests.selection_fixtures import synthetic_report, write
from evals.benchmark_common import select_comparison_samples
from evals.benchmark_eval import _fr_summary


class ReportValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path, self.manifest, self.book, self.candidate = synthetic_report(self.root)
        self.comparison = read_json(self.root / 'comparison_manifest.json')

    def validate(self):
        return validate_report(self.path, self.manifest, self.book, self.candidate,
                               comparison_manifest=self.comparison)

    def test_valid_saved_evidence(self):
        self.validate()

    def test_cleared_image_evidence_cannot_hide_missing_examples(self):
        report = read_json(self.path)
        report['evaluation']['comparison_samples'] = []
        write(self.path, report)
        write(self.root / report['free_running']['comparison_manifest_path'], {'samples': []})
        write(self.root / report['free_running']['images_index_path'], {'images': []})
        (self.root / 'a.png').unlink()
        with self.assertRaisesRegex(ValueError, 'frozen comparison population'):
            self.validate()

    def add_evaluated_sample(self):
        """Keep a second valid ID available so substitution cannot fail on coverage."""
        second = dict(read_jsonl(self.root / 'a_per_sample.jsonl')[0], sample_id='other')
        self.manifest['samples'].append({key: second[key] for key in (
            'sample_id', 'target_length', 'target_sha256', 'length_bucket')})
        write(self.root / 'evaluation_manifest.json', self.manifest)
        records = [read_jsonl(self.root / 'a_per_sample.jsonl')[0], second]
        (self.root / 'a_per_sample.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
        report = read_json(self.path)
        report['evaluation']['manifest_sha256'] = json_sha256(self.manifest)
        summary = _fr_summary(records)
        report['free_running'].update(total_samples=2,
            metrics={f'valid/free_running/{key[3:]}': value for key, value in summary.items() if key.startswith('fr_')},
            per_bucket={'short': summary})
        write(self.path, report)
        return {key: second[key] for key in ('sample_id', 'length_bucket', 'target_length')}

    def replace_candidate_comparisons(self, samples):
        report = read_json(self.path)
        report['evaluation']['comparison_samples'] = samples
        write(self.path, report)
        write(self.root / report['free_running']['comparison_manifest_path'], {'samples': samples})
        image = read_json(self.root / 'a_images_index.json')['images'][0]
        write(self.root / 'a_images_index.json', {'images': [dict(image, sample_id=row['sample_id']) for row in samples]})

    def test_substituted_evaluated_image_id_is_rejected(self):
        other = self.add_evaluated_sample()
        self.validate()
        self.replace_candidate_comparisons([other])
        with self.assertRaisesRegex(ValueError, 'frozen comparison population'):
            self.validate()

    def test_missing_duplicate_and_reordered_image_ids_are_rejected(self):
        other = self.add_evaluated_sample()
        original = self.comparison['samples'][0]
        self.comparison['samples'].append(other)
        self.replace_candidate_comparisons([original, other])
        self.validate()
        for samples in ([original], [original, original], [other, original]):
            with self.subTest(samples=samples):
                self.replace_candidate_comparisons(samples)
                with self.assertRaisesRegex(ValueError, 'frozen comparison population'):
                    self.validate()

    def test_image_index_must_cover_frozen_population_in_order(self):
        other = self.add_evaluated_sample()
        self.comparison['samples'].append(other)
        self.replace_candidate_comparisons(self.comparison['samples'])
        images = read_json(self.root / 'a_images_index.json')['images']
        self.validate()
        for entries in ([], images[:1], [images[0], images[0]], images[::-1]):
            with self.subTest(entries=entries):
                write(self.root / 'a_images_index.json', {'images': entries})
                with self.assertRaisesRegex(ValueError, 'image order'):
                    self.validate()

    def test_frozen_comparison_metadata_and_population_are_verified(self):
        cases = [('seed', 0), ('dataset_sha256', 'changed'), ('length_definition', 'changed'),
                 ('samples', [dict(self.comparison['samples'][0], sample_id='unexpected')]),
                 ('samples', [dict(self.comparison['samples'][0], target_length=6)]),
                 ('samples', self.comparison['samples'] * 2), ('samples', [])]
        for key, value in cases:
            comparison = dict(self.comparison, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_comparison_manifest(comparison, self.manifest, require_images=True)

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
