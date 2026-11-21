"""Append-only, atomic checkpoints retained at validation boundaries."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import warnings
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_unregistered(directory):
    root = Path(directory) / 'candidates'
    manifest = root / 'manifest.jsonl'
    registered = set()
    if manifest.exists():
        registered = {str(Path(json.loads(line)['path']).resolve()) for line in manifest.read_text().splitlines() if line.strip()}
    # Process termination may leave either a published checkpoint or its temporary file.
    files = [*root.rglob('*.pt'), *root.rglob('.candidate-*')]
    orphaned = sorted(str(p.resolve()) for p in files if str(p.resolve()) not in registered)
    if orphaned:
        warnings.warn('Unregistered candidate files left by interrupted saves: ' + ', '.join(orphaned), RuntimeWarning)
    return orphaned


def retain_candidate(directory, model, *, config, training_origin, stage, stage_max_length,
                     epoch, step, validation_loss):
    if not isinstance(stage, str) or not stage or Path(stage).name != stage or stage in {'.', '..'}:
        raise ValueError('Stage name must be a filename')
    for value in (epoch, step):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError('Candidate epoch and global step must be nonnegative integers')
    if not math.isfinite(validation_loss):
        raise ValueError('Candidate validation loss must be finite')
    if stage_max_length is not None and (
            isinstance(stage_max_length, bool) or not isinstance(stage_max_length, int) or stage_max_length < 2):
        raise ValueError('Candidate stage maximum length must be an integer of at least two')
    root = Path(directory).resolve() / 'candidates'
    path = root / stage / f'epoch-{epoch:04d}-step-{step:06d}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError('Candidate collision: ' + str(path))
    candidate_id = f'{stage}/epoch-{epoch:04d}-step-{step:06d}'
    payload = dict(model=model.state_dict(), config=dict(config), training_origin=training_origin,
                   epoch=epoch, step=step, global_step=step, candidate_id=candidate_id,
                   stage=stage, stage_max_length=stage_max_length,
                   metrics={'val/token_loss': validation_loss})
    fd, temporary = tempfile.mkstemp(prefix='.candidate-', dir=path.parent)
    os.close(fd)
    try:
        # PyTorch's path writer rejects hidden filenames; a stream retains atomic temp names.
        with open(temporary, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)
    record = dict(candidate_id=candidate_id, path=str(path), stage=stage, epoch=epoch,
                  global_step=step, stage_max_length=stage_max_length, sha256=sha256(path),
                  validation_loss=validation_loss)
    # Registration follows successful publication; an interruption leaves a reported orphan.
    with (root / 'manifest.jsonl').open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return path
