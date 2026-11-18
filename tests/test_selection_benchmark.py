"""CPU benchmark fixtures exercise identities and labels, not model quality."""
import json
import unittest
from pathlib import Path

from safetensors.torch import load_file, save_file

import tests.test_benchmark_eval as benchmark_tests
from evals.selection_common import validate_report, read_json, read_jsonl


class SelectionBenchmarkTest(unittest.TestCase):
    setUp = benchmark_tests.BenchmarkIntegrationTest.setUp
    write_data = benchmark_tests.BenchmarkIntegrationTest.write_data
    run_benchmark = benchmark_tests.BenchmarkIntegrationTest.run_benchmark

    def fixture_metadata(self):
        state = load_file(str(self.checkpoint))
        save_file(state, str(self.checkpoint), metadata={'training_origin': 'fixture', 'global_step': '0'})

    def validate(self, name):
        path = self.out / (name + '_report.json')
        report = read_json(path)
        manifest = read_json(report['evaluation']['manifest_path'])
        return validate_report(path, manifest, self.root / 'codebook.npy')

    def test_saved_fixture_origin_and_image_index_preserve_legacy_names(self):
        self.fixture_metadata()
        self.run_benchmark('explicit-fixture', '--training-origin', 'real_training', '--image-captions')
        report, records = self.validate('explicit-fixture')
        self.assertEqual(report['schema_version'], 2)
        self.assertEqual(report['report_kind'], 'not_quality')
        self.assertEqual(report['not_quality_reasons'], ['smoke_test'])
        self.assertTrue(report['origin_override_ignored'])
        self.assertEqual(report['candidate']['global_step'], 0)
        index_path = self.out / report['free_running']['images_index_path']
        self.assertEqual(index_path.name, 'images_index.json')
        images = read_json(index_path)['images']
        comparison = read_json(self.out / report['free_running']['comparison_manifest_path'])['samples']
        self.assertEqual([r['sample_id'] for r in images], [r['sample_id'] for r in comparison])
        self.assertEqual([Path(r['path']).name for r in images], ['reconstruction_001.png', 'reconstruction_002.png'])
        for row in records:
            self.assertIn('target_tokens', row)
            self.assertIn('predicted_tokens', row)
            self.assertIn('validation_tokens', row)
        report['report_kind'], report['not_quality_reasons'] = 'quality', []
        (self.out / 'explicit-fixture_report.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'report evidence'):
            self.validate('explicit-fixture')

    def test_subset_is_frozen_and_inference_restricted(self):
        self.fixture_metadata()
        self.run_benchmark('subset-a', '--eval-samples', '2')
        report, records = self.validate('subset-a')
        self.assertEqual(len(records), 2)
        path = report['evaluation']['manifest_path']
        self.run_benchmark('subset-b', '--evaluation-manifest', str(path))
        other, rows = self.validate('subset-b')
        self.assertEqual([r['sample_id'] for r in rows], [r['sample_id'] for r in records])
        self.assertEqual(other['evaluation']['manifest_sha256'], report['evaluation']['manifest_sha256'])

    def test_partial_diagnostic_cannot_pass_report_validation(self):
        self.fixture_metadata()
        self.run_benchmark('partial', '--limit-batches', '1', '--num-plots', '0')
        with self.assertRaisesRegex(ValueError, 'partial evaluation'):
            self.validate('partial')

    def test_internal_target_padding_is_not_repaired(self):
        self.write_data([[6, 1, 0, 2, 5, 7]])
        result = self.run_benchmark('bad-target', success=False)
        self.assertIn('Invalid benchmark target', result.stderr)
        self.assertFalse((self.out / 'bad-target_report.json').exists())
