"""Synthetic selection mechanics and non-quality CPU generation, never quality evidence."""
import argparse
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from safetensors.torch import load_file, save_file

from evals.benchmark_eval import _fr_summary
from evals.candidate_evaluation import (
    candidates_from_sources, evaluate_candidate, prepare_frozen_evaluation,
)
from evals.held_out_evaluation import evaluate_test, verify_selection
from evals.select_checkpoint import eligible_report, main, select
from evals.selection_common import (
    atomic_json, file_sha256, inference_settings, json_sha256, read_json,
    validate_report,
)
from metrics.sketchformer.drawing import score_drawing
from tests.selection_fixtures import synthetic_report, write
import tests.test_benchmark_eval as benchmark_tests

ROOT = benchmark_tests.ROOT


class SelectionVerificationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.report, self.manifest, self.book, self.candidate = synthetic_report(self.root)
        self.selection_path = self.root / 'selection.json'
        self.make_selection()

    def make_selection(self, materialize=False):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(select(argparse.Namespace(evaluation_manifest=self.root / 'evaluation_manifest.json',
                reports_dir=self.root, codebook_path=self.book, output=self.selection_path,
                tolerance=1e-6, materialize=materialize)), 0)

    def verify(self, checkpoint=None):
        return verify_selection(self.selection_path, self.book, checkpoint)

    def test_valid_selection_and_matching_override(self):
        selection, manifest, checkpoint, digest = self.verify()
        self.assertEqual(selection['winner']['candidate_id'], 'a')
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(digest, file_sha256(self.selection_path))
        self.assertEqual(str(checkpoint), self.candidate['path'])
        override = self.root / 'same.pt'
        override.write_bytes(checkpoint.read_bytes())
        self.assertEqual(self.verify(override)[2], override)
        override.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'test checkpoint hash'):
            self.verify(override)

    def test_changed_referenced_artifacts_and_dictionary(self):
        artifacts = read_json(self.selection_path)['winner']['artifacts']
        for path in [*map(Path, artifacts), Path(self.candidate['path']), self.book]:
            original = path.read_bytes()
            path.write_bytes(original + b'changed')
            with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                self.verify()
            path.write_bytes(original)
        self.verify()

    def test_all_ranked_artifacts_are_verified(self):
        self.selection_path.unlink()
        _, _, _, candidate = synthetic_report(self.root, 'b', step=2)
        self.make_selection()
        Path(candidate['path']).write_bytes(b'changed runner-up checkpoint')
        with self.assertRaisesRegex(ValueError, 'ranked checkpoint hash'):
            self.verify()

    def test_unsupported_rules_changed_settings_winner_ranking_and_trace(self):
        original = read_json(self.selection_path)
        changes = [('schema_version', 2), ('rule_version', 'future'),
                   ('scoring_rule_version', 'future'), ('completeness_rule_version', 'future'),
                   ('evaluation_identity', 'changed'), ('inference_fingerprint', 'changed'),
                   ('dictionary', {}), ('inference_settings', {}), ('winner', None),
                   ('ranking', []), ('tie_resolution_trace', [])]
        for key, value in changes:
            changed = copy.deepcopy(original)
            changed[key] = value
            write(self.selection_path, changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify()
        changed = copy.deepcopy(original)
        changed['ranking'][0]['mean_drawing_score'] = .5
        changed['winner'] = changed['ranking'][0]
        write(self.selection_path, changed)
        with self.assertRaisesRegex(ValueError, 'ranked candidate evidence'):
            self.verify()
        changed = copy.deepcopy(original)
        changed['winner']['artifacts'] = {}
        changed['ranking'][0]['artifacts'] = {}
        write(self.selection_path, changed)
        with self.assertRaisesRegex(ValueError, 'artifacts'):
            self.verify()
        write(self.selection_path, original)

    def test_materialized_checkpoint_is_verified_even_with_override(self):
        self.selection_path.unlink()
        self.make_selection(materialize=True)
        selection = read_json(self.selection_path)
        self.assertEqual(str(self.verify()[2]), selection['selected_checkpoint'])
        Path(selection['selected_checkpoint']).write_bytes(b'changed copy')
        with self.assertRaisesRegex(ValueError, 'materialized checkpoint hash'):
            self.verify(self.candidate['path'])

    def test_held_out_reports_cannot_reenter_selection(self):
        report = read_json(self.report)
        report.update(evaluation_role='held_out', selection_eligible=False)
        write(self.report, report)
        with self.assertRaisesRegex(ValueError, 'Held-out reports'):
            eligible_report(self.report, self.manifest, self.book)

    def test_cli_rejects_changed_selection_before_generation(self):
        selection = read_json(self.selection_path)
        selection['rule_version'] = 'future'
        write(self.selection_path, selection)
        result = subprocess.run([sys.executable, '-B', 'evals/select_checkpoint.py', 'test',
            '--selection', str(self.selection_path), '--data-root', str(self.root / 'missing-data'),
            '--codebook-path', str(self.book), '--device', 'cpu', '--output-dir', str(self.root / 'test')],
            cwd=ROOT, capture_output=True, text=True, timeout=90)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unsupported selection', result.stderr)
        self.assertFalse((self.root / 'test').exists())


class HeldOutPopulationTest(unittest.TestCase):
    setUp = benchmark_tests.BenchmarkIntegrationTest.setUp
    write_data = benchmark_tests.BenchmarkIntegrationTest.write_data

    def prepare_selection(self):
        # These report/checkpoint inputs are synthetic; no model is scored or accepted.
        self.validation_dir = self.root / 'synthetic-validation'
        path, manifest, book, candidate = synthetic_report(self.validation_dir)
        self.book = book
        config = yaml.safe_load(self.config.read_text())
        config['experiment']['name'] = 'custom-synthetic-experiment'
        config['data']['sequence'].update(truncate_long_sequences=False, add_start_token=True, add_end_token=True)
        config['data']['format']['token_dictionary']['codebook_path'] = str(book)
        config['model']['input']['token_dictionary']['codebook_path'] = str(book)
        manifest['inference_settings'] = inference_settings(config, '32-true', 8)
        manifest['inference_fingerprint'] = json_sha256(manifest['inference_settings'])
        write(self.validation_dir / 'evaluation_manifest.json', manifest)
        report = read_json(path)
        report['provenance']['experiment'] = 'custom-synthetic-experiment'
        report['evaluation'].update(inference_settings=manifest['inference_settings'],
            inference_fingerprint=manifest['inference_fingerprint'], manifest_sha256=json_sha256(manifest))
        write(path, report)
        index_path = self.validation_dir / 'a_images_index.json'
        index = read_json(index_path)
        index['images'][0]['settings'] = manifest['inference_settings']
        write(index_path, index)
        self.selection_path = self.root / 'selection.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(select(argparse.Namespace(evaluation_manifest=self.validation_dir / 'evaluation_manifest.json',
                reports_dir=self.validation_dir, codebook_path=book, output=self.selection_path,
                tolerance=1e-6, materialize=False)), 0)
        self.frozen_config = config
        self.frozen_validation = manifest
        self.write_test_data()
        self.args = argparse.Namespace(selection=self.selection_path, checkpoint=None, data_root=self.data,
            codebook_path=book, device='cpu', precision=None, max_generation_length=None,
            config=None, num_plots=2, output_dir=self.out)

    def write_test_data(self, ids=None):
        x = np.empty(len(self.sequences), dtype=object)
        x[:] = [np.array(seq, dtype=np.int64) for seq in self.sequences]
        np.savez(self.data / 'test.npz', x=x, y=np.zeros(len(x), dtype=np.int64),
                 sample_ids=np.array(ids or [f'test-{i}' for i in range(len(x))]))

    def synthetic_test_report(self, candidate, args, manifest_path, manifest, comparison_path, config_path, *, held_out):
        output = self.out / 'synthetic-test'
        output.mkdir()
        records = []
        for row, tokens in zip(manifest['samples'], self.sequences):
            records.append(dict(**row, **score_drawing(tokens, tokens, np.load(self.book), generation_limit=8),
                target_tokens=tokens, predicted_tokens=tokens, validation_tokens=tokens,
                candidate_id=candidate['candidate_id'], global_step=candidate['global_step'],
                checkpoint_sha256=candidate['sha256'], token_accuracy=1., exact_match=1., eos_rate=1., max_length_hit=0.))
        if getattr(self, 'partial_report', False):
            records = records[:1]
        (output / 'test_per_sample.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
        comparison = read_json(comparison_path)
        write(output / 'comparison.json', comparison)
        images = []
        for i, row in enumerate(comparison['samples']):
            image = output / f'{i}.png'
            image.write_bytes(b'synthetic image existence fixture')
            images.append(dict(path=image.name, sample_id=row['sample_id'], checkpoint_sha256=candidate['sha256'],
                               settings=manifest['inference_settings']))
        write(output / 'images_index.json', dict(images=images))
        summary = _fr_summary(records)
        report = dict(schema_version=2, report_kind='quality', not_quality_reasons=[],
            evaluation_role='held_out', held_out=held_out, selection_eligible=False, selection_score=None,
            candidate=dict(candidate, manifest_path=str(self.out / 'candidates.json')),
            provenance=dict(split='test', seed=manifest['seed'], checkpoint=candidate['path'], checkpoint_sha256=candidate['sha256']),
            evaluation=dict(manifest_path=str(manifest_path), manifest_sha256=json_sha256(manifest),
                dictionary=manifest['dictionary'], inference_settings=manifest['inference_settings'],
                inference_fingerprint=manifest['inference_fingerprint'], comparison_samples=comparison['samples']),
            free_running=dict(total_samples=len(records), per_sample_path='test_per_sample.jsonl',
                metrics={f'test/free_running/{k[3:]}': v for k, v in summary.items() if k.startswith('fr_')},
                per_bucket={'short': summary}, comparison_manifest_path='comparison.json', images_index_path='images_index.json'))
        path = output / 'test_report.json'
        write(path, report)
        return path, False

    def test_complete_test_population_different_dataset_hash_and_selection_unchanged(self):
        self.prepare_selection()
        before = self.selection_path.read_bytes()
        with patch('evals.held_out_evaluation.evaluate_candidate', side_effect=self.synthetic_test_report) as generate:
            command = ['select_checkpoint.py', 'test', '--selection', str(self.selection_path),
                '--data-root', str(self.data), '--codebook-path', str(self.book), '--device', 'cpu',
                '--num-plots', '2', '--output-dir', str(self.out)]
            with patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(generate.call_args.args[0]['candidate_id'], 'a')
        manifest = read_json(self.out / 'evaluation_manifest.json')
        self.assertEqual(manifest['split'], 'test')
        self.assertEqual(len(manifest['samples']), len(self.sequences))
        self.assertNotEqual(manifest['split_file_sha256'], self.frozen_validation['split_file_sha256'])
        self.assertEqual(manifest['inference_settings'], self.frozen_validation['inference_settings'])
        result = read_json(self.out / 'held_out_evaluation_attempt_0001.json')
        self.assertFalse(result['selection_eligible'])
        self.assertIsNone(result['selection_score'])
        self.assertEqual(result['selection_sha256'], file_sha256(self.selection_path))
        self.assertEqual(self.selection_path.read_bytes(), before)
        report, records = validate_report(result['report_path'], manifest, self.book)
        self.assertEqual(report['evaluation_role'], 'held_out')
        self.assertEqual(len(records), len(self.sequences))
        with self.assertRaisesRegex(ValueError, 'Held-out reports'):
            eligible_report(result['report_path'], self.frozen_validation, self.book)

    def test_synthetic_cli_handoff_ranking_materialization_and_test_isolation(self):
        # Handwritten token/report evidence exercises mechanics, not a trained model.
        self.prepare_selection()
        self.selection_path.unlink()
        for name, prediction in [('b', None), ('c', [6, 1, 2, 5])]:
            path, _, _, _ = synthetic_report(self.validation_dir, name, step=0, prediction=prediction)
            report = read_json(path)
            report['provenance']['experiment'] = 'custom-synthetic-experiment'
            report['evaluation'].update(inference_settings=self.frozen_validation['inference_settings'],
                inference_fingerprint=self.frozen_validation['inference_fingerprint'],
                manifest_sha256=json_sha256(self.frozen_validation))
            write(path, report)
            index_path = self.validation_dir / f'{name}_images_index.json'
            index = read_json(index_path)
            index['images'][0]['settings'] = self.frozen_validation['inference_settings']
            write(index_path, index)
        write(self.validation_dir / 'evaluation_manifest.json', self.frozen_validation)
        incomplete = json.loads((self.validation_dir / 'c_per_sample.jsonl').read_text())
        self.assertEqual(incomplete['geometry_f1_2px'], 1.)
        self.assertFalse(incomplete['complete'])
        self.assertEqual(incomplete['drawing_score'], 0.)
        result = subprocess.run([sys.executable, '-B', 'evals/select_checkpoint.py', 'select',
            '--evaluation-manifest', str(self.validation_dir / 'evaluation_manifest.json'),
            '--reports-dir', str(self.validation_dir), '--codebook-path', str(self.book),
            '--output', str(self.selection_path), '--materialize'],
            cwd=ROOT, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        selection = read_json(self.selection_path)
        self.assertEqual([row['candidate_id'] for row in selection['ranking']], ['b', 'a', 'c'])
        self.assertEqual(selection['winner']['candidate_id'], 'b')
        self.assertTrue(selection['tie_resolution_trace'])
        materialized = Path(selection['selected_checkpoint'])
        self.assertEqual(materialized.name, 'model.pt')
        self.assertEqual(file_sha256(materialized), selection['winner']['sha256'])
        self.assertEqual(materialized.read_bytes(), (self.validation_dir / 'b.pt').read_bytes())
        before = self.selection_path.read_bytes()
        command = ['select_checkpoint.py', 'test', '--selection', str(self.selection_path),
            '--data-root', str(self.data), '--codebook-path', str(self.book), '--device', 'cpu',
            '--num-plots', '2', '--output-dir', str(self.out)]
        # Candidate report generation is substituted; the CLI, artifact verification,
        # frozen test population and report validation execute through production code.
        with patch('evals.held_out_evaluation.evaluate_candidate', side_effect=self.synthetic_test_report) as generate:
            with patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
        generate.assert_called_once()
        self.assertEqual(generate.call_args.args[0]['candidate_id'], 'b')
        self.assertEqual(generate.call_args.args[0]['path'], str(materialized))
        test_manifest = read_json(self.out / 'evaluation_manifest.json')
        self.assertEqual(len(test_manifest['samples']), len(self.sequences))
        self.assertTrue({row['sample_id'] for row in test_manifest['samples']}.isdisjoint(
            row['sample_id'] for row in self.frozen_validation['samples']))
        attempt = read_json(self.out / 'held_out_evaluation_attempt_0001.json')
        report, records = validate_report(attempt['report_path'], test_manifest, self.book)
        self.assertEqual(len(records), len(self.sequences))
        self.assertEqual(report['held_out']['selection_sha256'], file_sha256(self.selection_path))
        self.assertFalse(report['selection_eligible'])
        self.assertIsNone(report['selection_score'])
        # A held-out report cannot select a new winner even if its drawings are perfect.
        reselect_path = self.root / 'test-reselection.json'
        result = subprocess.run([sys.executable, '-B', 'evals/select_checkpoint.py', 'select',
            '--evaluation-manifest', str(self.validation_dir / 'evaluation_manifest.json'),
            '--reports-dir', str(self.out), '--codebook-path', str(self.book),
            '--output', str(reselect_path)], cwd=ROOT, capture_output=True, text=True, timeout=90)
        self.assertNotEqual(result.returncode, 0)
        rejected = read_json(reselect_path)
        self.assertIsNone(rejected['winner'])
        self.assertIn('Held-out reports are ineligible', rejected['rejected_candidates'][0]['reasons'][0])
        self.assertEqual(self.selection_path.read_bytes(), before)
        self.assertEqual(file_sha256(materialized), selection['winner']['sha256'])

    def test_overlap_and_settings_mismatch_fail_before_generation(self):
        self.prepare_selection()
        with patch('evals.held_out_evaluation.evaluate_candidate') as generate:
            self.write_test_data(['sample', 'test-1', 'test-2'])
            with self.assertRaisesRegex(ValueError, 'overlap'):
                evaluate_test(copy.deepcopy(self.args))
            self.assertFalse(self.out.exists())
            self.write_test_data()
            for key, value in [('max_generation_length', 9), ('precision', 'bf16-mixed')]:
                args = copy.deepcopy(self.args)
                setattr(args, key, value)
                with self.assertRaisesRegex(ValueError, 'inference settings'):
                    evaluate_test(args)
            changed = copy.deepcopy(self.frozen_config)
            changed['model']['architecture']['d_model'] = 32
            self.config.write_text(yaml.safe_dump(changed))
            args = copy.deepcopy(self.args)
            args.config = str(self.config)
            with self.assertRaisesRegex(ValueError, 'inference settings'):
                evaluate_test(args)
            changed = copy.deepcopy(self.frozen_config)
            changed['trainer']['runtime']['precision'] = 'bf16-mixed'
            self.config.write_text(yaml.safe_dump(changed))
            with self.assertRaisesRegex(ValueError, 'effective_precision'):
                evaluate_test(copy.deepcopy(args))
            generate.assert_not_called()
            self.assertFalse(self.out.exists())

    def test_changed_selection_reference_requires_fresh_output(self):
        self.prepare_selection()
        with patch('evals.held_out_evaluation.evaluate_candidate', side_effect=self.synthetic_test_report):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate_test(copy.deepcopy(self.args)), 0)
        reference_path = self.out / 'selection_reference.json'
        reference = read_json(reference_path)
        reference['selection_sha256'] = 'changed'
        write(reference_path, reference)
        with patch('evals.held_out_evaluation.evaluate_candidate') as generate:
            with self.assertRaisesRegex(ValueError, 'fresh output directory'):
                evaluate_test(copy.deepcopy(self.args))
            generate.assert_not_called()

    def test_incomplete_test_report_is_rejected_and_selection_remains_unchanged(self):
        self.prepare_selection()
        self.partial_report = True
        before = self.selection_path.read_bytes()
        with patch('evals.held_out_evaluation.evaluate_candidate', side_effect=self.synthetic_test_report):
            with self.assertRaisesRegex(ValueError, 'partial evaluation'):
                evaluate_test(self.args)
        self.assertEqual(self.selection_path.read_bytes(), before)
        self.assertFalse((self.out / 'held_out_evaluation_attempt_0001.json').exists())

    def test_real_cpu_generation_helper_keeps_fixture_origin_and_safe_skipping(self):
        self.write_test_data()
        state = load_file(str(self.checkpoint))
        save_file(state, str(self.checkpoint), metadata={'training_origin': 'fixture', 'global_step': '1'})
        candidate = candidates_from_sources(checkpoints=[self.checkpoint])[0]
        config = yaml.safe_load(self.config.read_text())
        args = argparse.Namespace(config=str(self.config), experiment='smoke_test', data_root=self.data,
            codebook_path=self.root / 'codebook.npy', device='cpu', precision='32-true', max_generation_length=8,
            seed=42, eval_samples=None, num_plots=2, output_dir=self.out, smoke_test=False, training_origin=None)
        manifest_path, manifest, comparison_path, config_path = prepare_frozen_evaluation(args, split='test', config=config)
        atomic_json(self.out / 'candidates.json', [candidate])
        reference = dict(selection_path='synthetic', selection_sha256='synthetic', winner_candidate_id=candidate['candidate_id'])
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with patch.dict(os.environ, env):
            path, skipped = evaluate_candidate(candidate, args, manifest_path, manifest, comparison_path,
                                                config_path, held_out=reference)
            self.assertFalse(skipped)
            report, _ = validate_report(path, manifest, args.codebook_path, candidate)
            self.assertEqual(report['report_kind'], 'not_quality')
            self.assertEqual(report['not_quality_reasons'], ['smoke_test'])
            self.assertEqual(report['evaluation_role'], 'held_out')
            self.assertFalse(report['selection_eligible'])
            before = path.read_bytes()
            self.assertTrue(evaluate_candidate(candidate, args, manifest_path, manifest, comparison_path,
                                               config_path, held_out=reference)[1])
            self.assertEqual(path.read_bytes(), before)
            report['held_out']['selection_sha256'] = 'changed'
            write(path, report)
            retry, skipped = evaluate_candidate(candidate, args, manifest_path, manifest, comparison_path,
                                                config_path, held_out=reference)
            self.assertFalse(skipped)
            self.assertNotEqual(retry, path)
            self.assertIn('attempt-0002', str(retry))


if __name__ == '__main__':
    unittest.main()
