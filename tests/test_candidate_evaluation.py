"""CPU fixtures verify candidate orchestration; their scores are not quality evidence."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from evals.candidate_evaluation import candidates_from_sources, freeze_candidate_snapshot
from evals.selection_common import file_sha256, read_json, validate_report
import tests.test_benchmark_eval as benchmark_tests

ROOT = benchmark_tests.ROOT


class CandidateSourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'legacy.pt'
        self.manifest = self.root / 'manifest.jsonl'

    def write_manifest(self, **extra):
        self.manifest.write_text(json.dumps(dict(path='legacy.pt', **extra)) + '\n')

    def test_legacy_ids_steps_and_manifest_only_step(self):
        torch.save({'model': {}, 'step': 17}, self.path)
        candidate = candidates_from_sources(checkpoints=[self.path])[0]
        self.assertEqual(candidate['candidate_id'], 'legacy-' + file_sha256(self.path)[:12])
        self.assertEqual(candidate['global_step'], 17)
        torch.save({'model': {}}, self.path)
        with self.assertRaisesRegex(ValueError, 'candidate-manifest'):
            candidates_from_sources(checkpoints=[self.path])
        self.write_manifest(candidate_id='legacy', global_step=19, sha256=file_sha256(self.path))
        result = candidates_from_sources(self.manifest)[0]
        self.assertEqual(result['global_step'], 19)
        self.assertEqual(result['path'], str(self.path))

    def test_conflicts_invalid_steps_and_hashes(self):
        torch.save({'model': {}, 'global_step': 5, 'candidate_id': 'embedded'}, self.path)
        for row, message in [({'global_step': 6}, 'global step'),
                             ({'candidate_id': 'different'}, 'candidate ID'),
                             ({'sha256': 'wrong'}, 'manifest hash')]:
            self.write_manifest(**row)
            with self.assertRaisesRegex(ValueError, message):
                candidates_from_sources(self.manifest)
        for step in (-1, True, 1.5, '5'):
            torch.save({'model': {}, 'global_step': step}, self.path)
            with self.assertRaisesRegex(ValueError, 'invalid global step'):
                candidates_from_sources(checkpoints=[self.path])
            self.write_manifest(global_step=1)
            with self.assertRaisesRegex(ValueError, 'invalid global step'):
                candidates_from_sources(self.manifest)

    def test_duplicate_ids_reject_every_conflict_and_sources_are_exclusive(self):
        torch.save({'model': {}, 'step': 1}, self.path)
        self.manifest.write_text('\n'.join(json.dumps(dict(path='legacy.pt', candidate_id='same')) for _ in range(2)))
        with self.assertRaisesRegex(ValueError, 'Duplicate candidate ID'):
            candidates_from_sources(self.manifest)
        failures = []
        self.assertEqual(candidates_from_sources(self.manifest, failures=failures), [])
        self.assertEqual(len(failures), 2)
        for manifest, paths in ((None, None), (self.manifest, [self.path])):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                candidates_from_sources(manifest, paths)

    def test_missing_candidate_on_resume_keeps_registry_and_other_candidates(self):
        snapshot = self.root / 'candidates.json'
        candidates = [dict(candidate_id=name, path=str(self.root / (name + '.pt')),
                           sha256=name, global_step=step) for step, name in enumerate(('a', 'b'))]
        freeze_candidate_snapshot(snapshot, candidates, [])
        before = snapshot.read_bytes()
        freeze_candidate_snapshot(snapshot, candidates[1:], [dict(path=candidates[0]['path'], status='failed')])
        self.assertEqual(snapshot.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'fresh output directory'):
            freeze_candidate_snapshot(snapshot, candidates[1:], [])
        changed = [dict(candidates[0], sha256='changed'), candidates[1]]
        with self.assertRaisesRegex(ValueError, 'candidate snapshot'):
            freeze_candidate_snapshot(snapshot, changed, [])
        self.assertEqual(snapshot.read_bytes(), before)


class CandidateEvaluationCLITest(unittest.TestCase):
    setUp = benchmark_tests.BenchmarkIntegrationTest.setUp
    write_data = benchmark_tests.BenchmarkIntegrationTest.write_data

    def fixture_candidates(self):
        state = load_file(str(self.checkpoint))
        save_file(state, str(self.checkpoint), metadata={'training_origin': 'fixture', 'global_step': '1'})
        second = self.root / 'second-stage.pt'
        torch.save(dict(model=state, training_origin='fixture', global_step=2,
                        stage='stage2', stage_max_length=8), second)
        return [self.checkpoint, second]

    def run_evaluate(self, checkpoints=None, *extra, success=True):
        command = [sys.executable, '-B', 'evals/select_checkpoint.py', 'evaluate',
            '--config', str(self.config), '--experiment', 'smoke_test', '--data-root', str(self.data),
            '--codebook-path', str(self.root / 'codebook.npy'), '--device', 'cpu',
            '--max-generation-length', '8', '--seed', '42', '--num-plots', '2',
            '--output-dir', str(self.out)]
        if checkpoints is not None:
            command += ['--checkpoints', *map(str, checkpoints)]
        command += list(extra)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        if success:
            logs = '\n'.join(p.read_text() for p in self.out.rglob('*_process.log')) if result.returncode else ''
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout + logs)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def outcomes(self, attempt):
        return read_json(self.out / f'evaluation_attempt_{attempt:04d}.json')['outcomes']

    def test_consistent_population_skip_and_retry_preserve_previous_attempt(self):
        candidates = self.fixture_candidates()
        self.run_evaluate(candidates, '--eval-samples', '2', '--training-origin', 'real_training')
        manifest = read_json(self.out / 'evaluation_manifest.json')
        self.assertEqual(len(manifest['samples']), 2)
        self.assertEqual(manifest['split'], 'valid')
        first = self.outcomes(1)
        self.assertEqual([r['status'] for r in first], ['complete', 'complete'])
        paths = [Path(row['report_path']) for row in first]
        snapshots = [path.read_bytes() for path in paths]
        for path in paths:
            report, records = validate_report(path, manifest, self.root / 'codebook.npy')
            self.assertTrue(path.name.startswith('not-quality-'))
            self.assertEqual(report['report_kind'], 'not_quality')
            self.assertTrue(report['origin_override_ignored'])
            self.assertEqual(report['not_quality_reasons'], ['smoke_test'])
            self.assertTrue(report['provenance']['checkpoint_load']['strict'])
            self.assertEqual([r['sample_id'] for r in records], [r['sample_id'] for r in manifest['samples']])
            self.assertTrue(all('validation_tokens' in r for r in records))
            self.assertEqual(report['free_running']['images_index_path'], 'images_index.json')
            images = read_json(path.parent / 'images_index.json')['images']
            self.assertEqual([Path(r['path']).name for r in images], ['reconstruction_001.png', 'reconstruction_002.png'])
            self.assertEqual([r['sample_id'] for r in images],
                             [r['sample_id'] for r in read_json(self.out / 'comparison_manifest.json')['samples']])
        self.run_evaluate(candidates, '--eval-samples', '2', '--training-origin', 'real_training')
        self.assertEqual([r['status'] for r in self.outcomes(2)], ['skipped', 'skipped'])
        self.assertEqual([path.read_bytes() for path in paths], snapshots)
        image = read_json(paths[0].parent / 'images_index.json')['images'][0]['path']
        (paths[0].parent / image).unlink()
        # A reserved interrupted attempt also forces the next attempt number.
        candidate_dir = paths[0].parent.parent
        (candidate_dir / 'not-quality-attempt-0002_attempt.json').write_text('{}')
        self.run_evaluate(candidates, '--eval-samples', '2', '--training-origin', 'real_training')
        retried = self.outcomes(3)
        self.assertEqual([r['status'] for r in retried], ['complete', 'skipped'])
        self.assertIn('attempt-0003', retried[0]['report_path'])
        self.assertEqual(paths[0].read_bytes(), snapshots[0])
        self.assertTrue((candidate_dir / 'not-quality-attempt-0002_attempt.json').exists())
        validate_report(retried[0]['report_path'], manifest, self.root / 'codebook.npy')
        baseline = (self.out / 'evaluation_manifest.json').read_bytes()
        result = self.run_evaluate(candidates, '--eval-samples', '2', '--max-generation-length', '9', success=False)
        self.assertIn('fresh output directory', result.stderr)
        self.assertEqual((self.out / 'evaluation_manifest.json').read_bytes(), baseline)

    def test_candidate_failures_continue_other_candidates(self):
        candidates = self.fixture_candidates()
        bad = self.root / 'wrong-architecture.pt'
        torch.save(dict(model={'not_a_model': torch.ones(1)}, global_step=3, training_origin='fixture'), bad)
        missing_step = self.root / 'missing-step.pt'
        torch.save({'model': {}, 'training_origin': 'fixture'}, missing_step)
        corrupt = self.root / 'corrupt.pt'
        corrupt.write_bytes(b'invalid checkpoint')
        self.run_evaluate([missing_step, corrupt, bad, candidates[0]], success=False)
        outcomes = self.outcomes(1)
        self.assertEqual([r['status'] for r in outcomes], ['failed', 'failed', 'failed', 'complete'])
        self.assertIn('candidate-manifest', outcomes[0]['reason'])
        self.assertIn('Benchmark exited', outcomes[2]['reason'])
        manifest = read_json(self.out / 'evaluation_manifest.json')
        self.assertEqual(len(manifest['samples']), len(self.sequences))
        validate_report(outcomes[-1]['report_path'], manifest, self.root / 'codebook.npy')
        self.assertTrue(list(self.out.rglob('*_process.log')))

    def test_manifest_supplies_legacy_step_and_classifies_unknown_origin(self):
        state = load_file(str(self.checkpoint))
        legacy = self.root / 'legacy.pt'
        torch.save({'model': state}, legacy)
        manifest = self.root / 'candidate_manifest.jsonl'
        manifest.write_text(json.dumps(dict(candidate_id='legacy', path='legacy.pt',
                                            global_step=23, sha256=file_sha256(legacy))) + '\n')
        self.run_evaluate(None, '--candidate-manifest', str(manifest))
        path = Path(self.outcomes(1)[0]['report_path'])
        report = read_json(path)
        self.assertEqual(report['candidate']['global_step'], 23)
        self.assertEqual(report['not_quality_reasons'], ['smoke_test', 'unknown_training_origin'])
        self.assertEqual(report['report_kind'], 'not_quality')
        self.assertTrue(report['candidate']['manifest_path'].endswith('candidates.json'))
        validate_report(path, read_json(self.out / 'evaluation_manifest.json'), self.root / 'codebook.npy')

    def test_changed_candidates_dictionary_comparison_and_source_exclusivity(self):
        candidates = self.fixture_candidates()[:1]
        self.run_evaluate(candidates)
        original = sorted(str(p) for p in self.out.rglob('*_report.json'))
        result = self.run_evaluate(candidates, '--num-plots', '1', success=False)
        self.assertIn('fresh output directory', result.stderr)
        result = self.run_evaluate(candidates, '--candidate-manifest', 'unused', success=False)
        self.assertIn('not allowed with argument', result.stderr)
        save_file(load_file(str(candidates[0])), str(candidates[0]),
                  metadata={'training_origin': 'fixture', 'global_step': '4'})
        result = self.run_evaluate(candidates, success=False)
        self.assertIn('candidate snapshot', result.stderr)
        self.assertEqual(sorted(str(p) for p in self.out.rglob('*_report.json')), original)
        import numpy as np
        book = np.load(self.root / 'codebook.npy')
        np.save(self.root / 'codebook.npy', book * 2)
        result = self.run_evaluate(candidates, success=False)
        self.assertIn('dictionary.sha256', result.stderr)

    def test_training_cli_retains_two_stages_with_fixture_origin(self):
        import yaml
        config = yaml.safe_load(self.config.read_text())
        config['fixture'] = True
        config['model'].setdefault('checkpoint', {})['load_converted_weights'] = False
        config['experiment'].pop('pretrained', None)
        config['trainer']['checkpointing']['retain_candidates'] = True
        config['trainer']['curriculum'] = dict(enabled=True, early_stopping_patience=3, stages=[
            dict(name='short', max_length=8, epochs=1, learning_rate=.0001, trainable='all'),
            dict(name='long', max_length=16, epochs=1, learning_rate=.0001, trainable='all'),
        ])
        self.config.write_text(yaml.safe_dump(config))
        (self.data / 'train_000.npz').write_bytes((self.data / 'valid.npz').read_bytes())
        command = [sys.executable, '-B', 'scripts/sketchformer/train.py', '--config', str(self.config),
                   '--device', 'cpu', '--output-dir', str(self.out)]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        manifest = self.out / 'candidates/manifest.jsonl'
        candidates = candidates_from_sources(manifest)
        self.assertEqual(len(candidates), 2)
        self.assertLess(candidates[0]['global_step'], candidates[1]['global_step'])
        for candidate, stage in zip(candidates, ('short', 'long')):
            checkpoint = torch.load(candidate['path'], weights_only=False)
            self.assertEqual(checkpoint['stage'], stage)
            self.assertEqual(checkpoint['training_origin'], 'smoke_test')
            self.assertTrue(checkpoint['config']['fixture'])
            self.assertNotIn('optimizer', checkpoint)
            self.assertNotIn('scheduler', checkpoint)
            self.assertEqual(file_sha256(candidate['path']), candidate['sha256'])
        self.assertTrue((self.out / 'last.pt').is_file())
        self.assertTrue((self.out / 'best.pt').is_file())
        # Resume after the final stage must preserve all candidates without collisions.
        before = manifest.read_bytes()
        result = subprocess.run(command + ['--resume', str(self.out / 'last.pt')],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertEqual(manifest.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
