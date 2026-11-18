"""Synthetic selection mechanics only; no checkpoint quality measurements."""
import argparse
import contextlib
import copy
import io
import itertools
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from evals.benchmark_eval import _fr_summary
from evals.select_checkpoint import eligible_report, materialize, rank_candidates, select
from evals.selection_common import read_json, read_jsonl, file_sha256, json_sha256, token_sha256
from metrics.sketchformer.drawing import score_drawing
from tests.selection_fixtures import synthetic_report, write

ROOT = Path(__file__).resolve().parents[1]


def rank_row(name, score=.5, completion=.5, bucket=.5, step=1):
    return dict(candidate_id=name, mean_drawing_score=score, completion_rate=completion,
                minimum_bucket_mean_drawing_score=bucket, global_step=step)


class RankingTest(unittest.TestCase):
    def test_each_criterion(self):
        # Lower-priority criteria cannot overrule the first decisive criterion.
        for a, b in [(rank_row('a', score=.9, completion=0, bucket=0, step=99), rank_row('b')),
                     (rank_row('a', completion=.9, bucket=0, step=99), rank_row('b')),
                     (rank_row('a', bucket=.9, step=99), rank_row('b')),
                     (rank_row('z', step=0), rank_row('b')),
                     (rank_row('a'), rank_row('b'))]:
            for rows in ([a, b], [b, a]):
                ranking, trace = rank_candidates(rows)
                self.assertEqual(ranking[0]['candidate_id'], a['candidate_id'])
                self.assertTrue(trace)

    def test_nontransitive_and_tolerance_boundary(self):
        rows = [rank_row('a', score=.9, step=3), rank_row('b', score=.84, step=2), rank_row('c', score=.78, step=1)]
        expected = rank_candidates(rows, .1)
        for permutation in itertools.permutations(rows):
            ranking, trace = rank_candidates(list(permutation), .1)
            self.assertEqual([r['candidate_id'] for r in ranking], ['b', 'a', 'c'])
            self.assertEqual((ranking, trace), expected)
        rows = [rank_row('a', score=.75, step=2), rank_row('b', score=.5, step=1)]
        self.assertEqual(rank_candidates(rows, .25)[0][0]['candidate_id'], 'b')
        self.assertEqual(rank_candidates(rows, .249999)[0][0]['candidate_id'], 'a')
        self.assertEqual(rank_candidates(rows, math.nextafter(.25, 0))[0][0]['candidate_id'], 'a')

    def test_tolerance_refines_every_floating_criterion_and_exact_ties(self):
        rows = [rank_row('c', score=.75, completion=.75, bucket=.75, step=3),
                rank_row('b', score=.5, completion=.5, bucket=.5, step=2),
                rank_row('a', score=.5, completion=.5, bucket=.5, step=2)]
        expected = rank_candidates(rows, .25)
        self.assertEqual([r['candidate_id'] for r in expected[0]], ['a', 'b', 'c'])
        for permutation in itertools.permutations(rows):
            self.assertEqual(rank_candidates(permutation, .25), expected)
        self.assertEqual({r['criterion'] for r in expected[1]}, {
            'mean_drawing_score', 'completion_rate', 'minimum_bucket_mean_drawing_score',
            'global_step', 'candidate_id'})
        self.assertEqual(rank_candidates([], 0), ([], []))

    def test_invalid_tolerance_and_scores(self):
        for tolerance in (-1, float('nan'), float('inf'), True, 'bad'):
            with self.assertRaises(ValueError): rank_candidates([], tolerance)
        for field in ('mean_drawing_score', 'completion_rate', 'minimum_bucket_mean_drawing_score'):
            for score in (-.1, 1.1, float('nan'), float('inf'), True, 'bad'):
                row = rank_row('a')
                row[field] = score
                with self.assertRaises(ValueError): rank_candidates([row])
        for step in (-1, True, 1.0, None):
            with self.assertRaises(ValueError): rank_candidates([rank_row('a', step=step)])
        for name in ('', None, 1):
            with self.assertRaises(ValueError): rank_candidates([rank_row(name)])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            rank_candidates([rank_row('a'), rank_row('a')])


class SelectionCLITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path, self.manifest, self.book, self.candidate = synthetic_report(self.root)

    def run_select(self, *extra, output='selection.json'):
        return subprocess.run([sys.executable, '-B', 'evals/select_checkpoint.py', 'select',
            '--evaluation-manifest', str(self.root / 'evaluation_manifest.json'),
            '--reports-dir', str(self.root), '--codebook-path', str(self.book),
            '--output', str(self.root / output), *extra], cwd=ROOT, text=True, capture_output=True, timeout=90)

    def select_here(self, output='selection.json', **overrides):
        args = dict(evaluation_manifest=self.root / 'evaluation_manifest.json',
                    reports_dir=self.root, codebook_path=self.book, output=self.root / output,
                    tolerance=1e-6, materialize=False)
        args.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return select(argparse.Namespace(**args))

    def test_selection_materialization_and_empty_buckets(self):
        synthetic_report(self.root, 'b', step=0, prediction=[6, 1, 2, 3, 5, 7])
        result = self.run_select('--materialize')
        self.assertEqual(result.returncode, 0, result.stderr)
        selection = read_json(self.root / 'selection.json')
        self.assertEqual(selection['winner']['candidate_id'], 'a')
        self.assertEqual(selection['winner']['minimum_bucket_mean_drawing_score'], 1.)
        self.assertEqual(file_sha256(selection['selected_checkpoint']), self.candidate['sha256'])
        self.assertEqual(Path(selection['selected_checkpoint']).name, 'model.pt')
        self.assertEqual(selection['evaluation_manifest_sha256'], file_sha256(self.root / 'evaluation_manifest.json'))
        self.assertEqual(selection['evaluation_identity'], json_sha256(self.manifest))
        self.assertEqual(selection['winner']['report_sha256'], file_sha256(self.path))
        for path, digest in selection['winner']['artifacts'].items():
            self.assertEqual(file_sha256(path), digest)
        before = (self.root / 'selection.json').read_bytes()
        with self.assertRaises(FileExistsError): self.select_here()
        self.assertEqual((self.root / 'selection.json').read_bytes(), before)
        result = self.run_select('--materialize', output='another.json')
        self.assertNotEqual(result.returncode, 0)

    def test_nonquality_and_invalid_steps_no_loss_fallback(self):
        report = read_json(self.path)
        report['report_kind'] = 'not_quality'
        report['not_quality_reasons'] = ['smoke_test']
        write(self.path, report)
        self.assertNotEqual(self.run_select().returncode, 0)
        rejected = read_json(self.root / 'selection.json')['rejected_candidates']
        self.assertIn('Non-quality', rejected[0]['reasons'][0])
        report['report_kind'], report['not_quality_reasons'] = 'quality', []
        report['candidate']['global_step'] = None
        write(self.path, report)
        self.assertNotEqual(self.run_select(output='missing-step.json').returncode, 0)

    def test_test_reports_cannot_select(self):
        path, _, _, _ = synthetic_report(self.root / 'test', 'test', split='test')
        result = self.run_select()
        self.assertEqual(result.returncode, 0, result.stderr)
        selection = read_json(self.root / 'selection.json')
        self.assertEqual(len(selection['ranking']), 1)
        self.assertEqual(len(selection['rejected_candidates']), 1)
        self.assertIn('test reports are ineligible', selection['rejected_candidates'][0]['reasons'][0])

    def test_candidate_is_checked_against_frozen_specification(self):
        # This incompatible candidate sorts before the compatible one.
        path, _, _, _ = synthetic_report(self.root / 'first', '0-incompatible')
        report = read_json(path)
        report['evaluation']['inference_settings']['generation_limit'] = 7
        write(path, report)
        self.assertEqual(self.select_here(), 0)
        selection = read_json(self.root / 'selection.json')
        self.assertEqual([r['candidate_id'] for r in selection['ranking']], ['a'])
        self.assertIn('generation_limit', selection['rejected_candidates'][0]['reasons'][0])

    def test_invalid_reports_are_rejected_without_stopping_other_candidates(self):
        path, _, _, _ = synthetic_report(self.root, 'b')
        original = read_json(path)
        evidence_path = self.root / 'b_per_sample.jsonl'
        original_evidence = evidence_path.read_text()
        cases = []
        for field, value in [('candidate_id', ''), ('global_step', -1), ('global_step', True),
                             ('global_step', 1.5), ('sha256', 'changed')]:
            report = copy.deepcopy(original)
            report['candidate'][field] = value
            cases.append((report, original_evidence))
        for field, value in [('dictionary', {}), ('inference_fingerprint', 'changed'), ('manifest_sha256', 'changed')]:
            report = copy.deepcopy(original)
            report['evaluation'][field] = value
            cases.append((report, original_evidence))
        for score in (-.1, 1.1, float('nan'), float('inf'), .5):
            row = json.loads(original_evidence)
            row['drawing_score'] = score
            cases.append((original, json.dumps(row)))
        cases.append((original, ''))  # Partial population cannot enter the ranking.
        for index, (report, evidence) in enumerate(cases):
            write(path, report)
            evidence_path.write_text(evidence)
            output = f'invalid-{index}.json'
            self.assertEqual(self.select_here(output), 0)
            selection = read_json(self.root / output)
            self.assertEqual([r['candidate_id'] for r in selection['ranking']], ['a'])
            self.assertEqual(len(selection['rejected_candidates']), 1)

    def test_duplicate_candidate_ids_reject_every_conflict(self):
        synthetic_report(self.root / 'duplicate', 'a', step=2)
        self.assertEqual(self.select_here(materialize=True), 2)
        selection = read_json(self.root / 'selection.json')
        self.assertIsNone(selection['winner'])
        self.assertEqual(selection['ranking'], [])
        self.assertEqual(len(selection['rejected_candidates']), 2)
        self.assertTrue(all(r['reasons'] == ['Duplicate candidate ID'] for r in selection['rejected_candidates']))
        self.assertFalse((self.root / 'selected').exists())

    def test_forged_quality_fixture_is_ineligible(self):
        checkpoint = Path(self.candidate['path'])
        torch.save({'model': {}, 'step': 1, 'training_origin': 'fixture'}, checkpoint)
        report = read_json(self.path)
        report['candidate']['sha256'] = file_sha256(checkpoint)
        report['provenance']['checkpoint_sha256'] = file_sha256(checkpoint)
        report['provenance']['training_origin_override'] = 'real_training'
        write(self.path, report)
        self.assertEqual(self.select_here(), 2)
        selection = read_json(self.root / 'selection.json')
        self.assertIsNone(selection['winner'])
        self.assertIn('report evidence report_kind', selection['rejected_candidates'][0]['reasons'][0])

    def test_bad_frozen_inputs_fail_before_writing_outputs(self):
        for tolerance in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): self.select_here(tolerance=tolerance)
        self.assertFalse((self.root / 'selection.json').exists())
        for field, value in [('split', 'test'), ('completeness_rule_version', 'future'),
                             ('scoring_rule_version', 'future'), ('inference_fingerprint', 'changed')]:
            manifest = copy.deepcopy(self.manifest)
            manifest[field] = value
            write(self.root / 'evaluation_manifest.json', manifest)
            with self.assertRaises(ValueError): self.select_here()
        write(self.root / 'evaluation_manifest.json', self.manifest)
        self.book.write_bytes(self.book.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'dictionary file'): self.select_here()
        self.assertFalse((self.root / 'selection.json').exists())

    def test_populated_bucket_means_and_completion_use_all_records(self):
        target = [6, 1, *([2] * 510), 5, 7]
        row = read_jsonl(self.root / 'a_per_sample.jsonl')[0]
        second = dict(row, sample_id='medium', target_length=len(target),
                      target_sha256=token_sha256(target), length_bucket='medium',
                      target_tokens=target, predicted_tokens=[6, 1, 2], validation_tokens=[6, 1, 2])
        second.update(score_drawing(second['validation_tokens'], target, np.load(self.book), generation_limit=8))
        records = [row, second]
        self.manifest['samples'].append({k: second[k] for k in (
            'sample_id', 'target_length', 'target_sha256', 'length_bucket')})
        write(self.root / 'evaluation_manifest.json', self.manifest)
        report = read_json(self.path)
        report['evaluation']['manifest_sha256'] = json_sha256(self.manifest)
        report['free_running']['total_samples'] = 2
        summary = _fr_summary(records)
        report['free_running']['metrics'] = {f'valid/free_running/{k[3:]}': v for k, v in summary.items() if k.startswith('fr_')}
        report['free_running']['per_bucket'] = {r['length_bucket']: _fr_summary([r]) for r in records}
        write(self.path, report)
        (self.root / 'a_per_sample.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        eligible = eligible_report(self.path, self.manifest, self.book)
        self.assertEqual(eligible['mean_drawing_score'], .5)
        self.assertEqual(eligible['completion_rate'], .5)
        self.assertEqual(eligible['minimum_bucket_mean_drawing_score'], 0.)
        self.assertEqual(eligible['bucket_mean_drawing_scores'], {'short': 1., 'medium': 0.})

    def test_empty_reports_directory_records_no_winner(self):
        empty = self.root / 'empty'
        empty.mkdir()
        self.assertEqual(self.select_here(reports_dir=empty, materialize=True), 2)
        selection = read_json(self.root / 'selection.json')
        self.assertEqual(selection['ranking'], [])
        self.assertEqual(selection['rejected_candidates'], [])
        self.assertIsNone(selection['winner'])
        self.assertFalse((self.root / 'selected').exists())

    def test_missing_images_and_corrupt_checkpoints_are_rejected(self):
        path, _, _, identity = synthetic_report(self.root, 'b')
        (self.root / 'b.png').unlink()
        self.assertEqual(self.select_here(), 0)
        self.assertIn('Missing comparison image', read_json(self.root / 'selection.json')['rejected_candidates'][0]['reasons'][0])
        (self.root / 'b.png').write_bytes(b'synthetic')
        Path(identity['path']).write_bytes(b'corrupt')
        report = read_json(path)
        report['candidate']['sha256'] = report['provenance']['checkpoint_sha256'] = file_sha256(identity['path'])
        write(path, report)
        self.assertEqual(self.select_here('corrupt.json'), 0)
        self.assertEqual(len(read_json(self.root / 'corrupt.json')['rejected_candidates']), 1)

    def test_safetensors_materialization_through_cli(self):
        checkpoint = self.root / 'a.safetensors'
        save_file({'synthetic': torch.zeros(1)}, str(checkpoint),
                  metadata={'training_origin': 'real_training', 'global_step': '1'})
        report = read_json(self.path)
        report['candidate'].update(path=str(checkpoint), sha256=file_sha256(checkpoint))
        report['provenance'].update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint))
        write(self.path, report)
        row = read_jsonl(self.root / 'a_per_sample.jsonl')[0]
        row['checkpoint_sha256'] = file_sha256(checkpoint)
        (self.root / 'a_per_sample.jsonl').write_text(json.dumps(row) + '\n')
        index = read_json(self.root / 'a_images_index.json')
        index['images'][0]['checkpoint_sha256'] = file_sha256(checkpoint)
        write(self.root / 'a_images_index.json', index)
        result = self.run_select('--materialize')
        self.assertEqual(result.returncode, 0, result.stderr)
        selection = read_json(self.root / 'selection.json')
        self.assertEqual(Path(selection['selected_checkpoint']).name, 'model.safetensors')
        self.assertEqual(Path(selection['selected_checkpoint']).read_bytes(), checkpoint.read_bytes())

    def test_materialization_detects_actual_format_and_verifies_hash(self):
        # A misleading extension must not change the checkpoint's byte format.
        checkpoint = self.root / 'renamed.safetensors'
        checkpoint.write_bytes(Path(self.candidate['path']).read_bytes())
        winner = dict(self.candidate, path=str(checkpoint))
        destination = materialize(winner, self.root / 'selection.json')
        self.assertEqual(Path(destination).name, 'model.pt')
        self.assertEqual(file_sha256(destination), winner['sha256'])
        with self.assertRaises(FileExistsError): materialize(winner, self.root / 'another.json')
        winner['sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'winner checkpoint'): materialize(winner, self.root / 'another.json')
