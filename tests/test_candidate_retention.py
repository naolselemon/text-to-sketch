import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
import torch
from core.candidates import retain_candidate, check_unregistered, sha256
from core.callbacks import CheckpointCallback


class CandidateRetentionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = torch.nn.Linear(2, 2)

    def save(self, stage='stage1', step=1, epoch=0):
        return retain_candidate(self.root, self.model, config={'fixture': True}, training_origin='fixture',
            stage=stage, stage_max_length=512, epoch=epoch, step=step, validation_loss=.5)

    def test_multistage_contents_and_hashes(self):
        paths = [self.save(), self.save(stage='stage2', step=2, epoch=1)]
        rows = [json.loads(line) for line in (self.root / 'candidates/manifest.jsonl').read_text().splitlines()]
        for path, row in zip(paths, rows):
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint['training_origin'], 'fixture')
            self.assertEqual(checkpoint['global_step'], row['global_step'])
            self.assertEqual(checkpoint['config'], {'fixture': True})
            self.assertNotIn('optimizer', checkpoint)
            self.assertNotIn('scheduler', checkpoint)
            self.assertEqual(sha256(path), row['sha256'])
        self.assertFalse(check_unregistered(self.root))

    def test_collision_and_interrupted_registration(self):
        path = self.save()
        before = path.read_bytes()
        with self.assertRaises(FileExistsError): self.save()
        self.assertEqual(path.read_bytes(), before)
        (self.root / 'candidates/manifest.jsonl').write_text('')
        with warnings.catch_warnings(record=True) as caught:
            orphaned = check_unregistered(self.root)
        self.assertEqual(orphaned, [str(path)])
        self.assertIn('Unregistered candidate', str(caught[0].message))

    def test_interrupted_save_does_not_register(self):
        with patch('core.candidates.torch.save', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError): self.save()
        self.assertFalse((self.root / 'candidates/manifest.jsonl').exists())
        self.assertFalse(list(self.root.rglob('*.pt')))

    def test_interrupted_registration_and_resume_report_all_files(self):
        original_open = Path.open

        def interrupted_open(path, *args, **kwargs):
            if path.name == 'manifest.jsonl' and args == ('a',):
                raise OSError('registration interrupted')
            return original_open(path, *args, **kwargs)

        with patch.object(Path, 'open', interrupted_open):
            with self.assertRaisesRegex(OSError, 'registration interrupted'):
                self.save()
        checkpoint = next(self.root.rglob('*.pt'))
        temporary = checkpoint.parent / '.candidate-interrupted'
        temporary.write_bytes(b'partial checkpoint')
        with self.assertWarnsRegex(RuntimeWarning, 'Unregistered candidate'):
            callback = CheckpointCallback(self.root, retain_candidates=True)
        self.assertTrue(callback.retain_candidates)
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(check_unregistered(self.root), sorted([str(checkpoint), str(temporary)]))
        self.assertTrue(checkpoint.exists())
        self.assertTrue(temporary.exists())
        with self.assertRaises(FileExistsError):
            self.save()

    def test_callback_retains_every_validation_with_stage_local_best(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        callback = CheckpointCallback(self.root, retain_candidates=True,
                                      config={'experiment': {'name': 'memorize10_long'}})
        paths = []
        for epoch, (stage, loss) in enumerate([('stage1', .5), ('stage1', .6), ('stage2', .8), ('stage2', .9)]):
            if stage == 'stage2' and epoch == 2:
                callback.tracker.best = None
            saved = callback.on_validation_end(self.model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, step=epoch + 1, metrics={'val/token_loss': loss}, stage=stage,
                stage_max_length=512 if stage == 'stage1' else 1024)
            self.assertEqual('best' in saved, epoch in (0, 2))
            paths.append(saved['candidate'])
        rows = [json.loads(line) for line in (self.root / 'candidates/manifest.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual([r['stage_max_length'] for r in rows], [512, 512, 1024, 1024])
        for path, row in zip(paths, rows):
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload['training_origin'], 'sanity_preset')
            self.assertEqual(payload['candidate_id'], row['candidate_id'])
            self.assertEqual(payload['global_step'], row['global_step'])
            self.assertEqual(sha256(path), row['sha256'])
            self.assertNotIn('optimizer', payload)
            self.assertNotIn('scheduler', payload)
        best = torch.load(self.root / 'best.pt', weights_only=False)
        last = torch.load(self.root / 'last.pt', weights_only=False)
        self.assertEqual(best['metrics']['val/token_loss'], .8)
        self.assertEqual(last['global_step'], 4)
        self.assertIn('optimizer', last)
        self.assertIn('scheduler', last)
        self.assertEqual(last['training_origin'], 'sanity_preset')

    def test_default_callback_and_smoke_training_origin(self):
        callback = CheckpointCallback(self.root, config={'experiment': {'name': 'smoke_test'}})
        saved = callback.on_validation_end(self.model, optimizer=None, scheduler=None,
            epoch=1, step=2, metrics={'val/token_loss': .3})
        self.assertEqual(set(saved), {'last', 'best'})
        self.assertFalse((self.root / 'candidates').exists())
        self.assertEqual(torch.load(saved['last'], weights_only=False)['training_origin'], 'smoke_test')

    def test_invalid_identity_and_symlink_collision(self):
        for step in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                self.save(step=step)
        for stage in ('', '..', '../escape'):
            with self.assertRaises(ValueError):
                self.save(stage=stage)
        path = self.root / 'candidates/stage1/epoch-0000-step-000001.pt'
        path.parent.mkdir(parents=True)
        path.symlink_to(self.root / 'absent.pt')
        with self.assertRaises(FileExistsError):
            self.save()
