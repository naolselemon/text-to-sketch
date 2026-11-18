"""Training provenance labels. Strict weight loading is not training evidence."""
from __future__ import annotations

import json
from pathlib import Path

REASON_ORDER = ('random_init', 'smoke_test', 'sanity_preset', 'partial_load', 'unknown_training_origin')


def wrapper_run_name(name, classification):
    """The selection wrapper labels non-quality runs; explicit benchmark names stay intact."""
    return ('not-quality-' if classification['report_kind'] == 'not_quality' else '') + name


def checkpoint_metadata(path):
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors import safe_open
        with safe_open(str(path), framework='pt', device='cpu') as stream:
            metadata = stream.metadata() or {}
        result = {}
        for key, value in metadata.items():
            try:
                result[key] = json.loads(value)
            except (ValueError, TypeError):
                result[key] = value
        return result
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        return {}
    return {k: v for k, v in checkpoint.items() if k not in {'model', 'state_dict', 'optimizer', 'scheduler'}}


def training_origin_from_config(config):
    name = config.get('experiment', {}).get('name', '')
    if name == 'smoke_test' or config.get('fixture') or config.get('smoke_test'):
        return 'smoke_test'
    if name == 'memorize10_long' or config.get('sanity_preset'):
        return 'sanity_preset'
    return config.get('training_origin', 'real_training')


def classify_report(metadata, *, random_init=False, smoke_test=False, experiment='',
                    partial_load=False, training_origin_override=None):
    config = metadata.get('config', {})
    embedded = metadata.get('training_origin') or config.get('training_origin')
    origin = embedded or training_origin_override
    ignored = bool(embedded and training_origin_override and embedded != training_origin_override)
    reasons = set()
    names = {experiment, config.get('experiment', {}).get('name', '')}
    if random_init or metadata.get('random_init') or origin == 'random_init':
        reasons.add('random_init')
    if (smoke_test or 'smoke_test' in names or origin in {'smoke_test', 'fixture'}
            or metadata.get('fixture') or metadata.get('smoke_test') or config.get('fixture') or config.get('smoke_test')):
        reasons.add('smoke_test')
    if 'memorize10_long' in names or origin == 'sanity_preset' or metadata.get('sanity_preset') or config.get('sanity_preset'):
        reasons.add('sanity_preset')
    if partial_load or metadata.get('partial_load') or config.get('partial_load'):
        reasons.add('partial_load')
    if origin not in {'real_training', 'smoke_test', 'fixture', 'sanity_preset', 'random_init'}:
        reasons.add('unknown_training_origin')
    ordered = [reason for reason in REASON_ORDER if reason in reasons]
    return dict(report_kind='not_quality' if ordered else 'quality', not_quality_reasons=ordered,
                training_origin=origin, origin_override_ignored=ignored)
