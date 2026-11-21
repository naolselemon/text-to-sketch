"""Frozen populations, inference identities and verification of token evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from evals.benchmark_common import LENGTH_DEFINITION, length_bucket, records_by_id, select_comparison_samples
from metrics.sketchformer.drawing import (
    COMPLETENESS_RULE_VERSION, SCORING_RULE_VERSION, score_drawing, validate_drawing,
    validation_tokens,
)

SELECTION_RULE_VERSION = "free-running-selection-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def json_sha256(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def token_sha256(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype='<i8').tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic_json(path, value):
    """Publish a new artifact without overwriting an existing one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def dictionary_identity(path, layout):
    return {'sha256': file_sha256(path), 'token_layout': {
        k: int(layout[k]) for k in ('codebook_size', 'motion_token_offset', 'pad_token_id',
                                  'sep_token_id', 'sos_token_id', 'eos_token_id', 'vocab_size')
    }}


def inference_settings(config, precision, generation_limit):
    """Exclude training-only and location settings from the inference specification."""
    model = json.loads(json.dumps(config['model']))
    for key in ('checkpoint', 'name', 'framework'):
        model.pop(key, None)
    model['input']['token_dictionary'] = {
        k: v for k, v in model['input']['token_dictionary'].items()
        if k not in {'codebook_path', 'codebook_dir'}
    }
    runtime = config['trainer']['runtime']
    return {
        'effective_precision': str(precision), 'model': model,
        'token_processing': config['data']['sequence'],
        'token_layout': dictionary_layout(config),
        'decoding': 'greedy', 'use_cache': True,
        'generation_limit': int(generation_limit),
        'runtime': {k: runtime.get(k) for k in ('allow_tf32', 'deterministic', 'benchmark')},
    }


def dictionary_layout(config):
    return {k: int(v) for k, v in config['data']['format']['token_dictionary'].items()
            if k not in {'codebook_dir', 'codebook_path'}}


def differing_fields(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            result.extend(differing_fields(a.get(key), b.get(key), prefix + key + '.'))
        return result
    return [] if a == b else [prefix.rstrip('.')]


def require_same(a, b, label):
    fields = differing_fields(a, b)
    if fields:
        raise ValueError(f'{label} mismatch: {", ".join(fields)}')


def build_evaluation_manifest(dataset, split, dictionary, settings, seed, eval_samples=None):
    records = []
    for index in range(len(dataset)):
        sample = dataset[index]
        tokens = sample['tokens']
        records.append({'sample_id': str(sample['sample_id']), 'target_length': len(tokens),
                        'target_sha256': token_sha256(tokens), 'length_bucket': length_bucket(len(tokens))})
    by_id = records_by_id(records)
    if eval_samples is not None:
        if isinstance(eval_samples, bool) or not isinstance(eval_samples, int) or eval_samples <= 0:
            raise ValueError('eval_samples must be a positive integer')
        chosen = select_comparison_samples(records, eval_samples, seed)
        records = [by_id[row['sample_id']] for row in chosen]
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return {
        'schema_version': 1, 'split': split,
        'split_file_sha256': {p.name: file_sha256(p) for p in dataset.files},
        'samples': records, 'dictionary': dictionary,
        'inference_settings': settings, 'inference_fingerprint': json_sha256(settings),
        'seed': int(seed), 'code_revision': revision.stdout.strip() or None,
        'completeness_rule_version': COMPLETENESS_RULE_VERSION,
        'scoring_rule_version': SCORING_RULE_VERSION,
    }


def validate_manifest(manifest):
    if manifest['schema_version'] != 1 or manifest['split'] not in {'valid', 'test'}:
        raise ValueError('Unsupported evaluation manifest')
    if manifest['completeness_rule_version'] != COMPLETENESS_RULE_VERSION or manifest['scoring_rule_version'] != SCORING_RULE_VERSION:
        raise ValueError('Unsupported completeness/scoring rule')
    if not manifest['samples']:
        raise ValueError('Empty evaluation population')
    records_by_id(manifest['samples'])
    require_same(manifest['inference_fingerprint'], json_sha256(manifest['inference_settings']), 'inference fingerprint')
    require_same(manifest['dictionary']['token_layout'], manifest['inference_settings']['token_layout'], 'token layout')


def freeze_manifest(path, proposed):
    path = Path(path)
    validate_manifest(proposed)
    if path.exists():
        frozen = read_json(path)
        validate_manifest(frozen)
        # Code revision is provenance, not an inference setting.
        require_same({k: v for k, v in proposed.items() if k != 'code_revision'},
                     {k: v for k, v in frozen.items() if k != 'code_revision'}, 'evaluation manifest')
        return frozen
    atomic_json(path, proposed)
    return proposed


def restrict_loader(loader, manifest):
    """Retain the collator and batch size; yield only the frozen ordered population."""
    from torch.utils.data import DataLoader, Subset
    dataset = loader.dataset
    ids = {entry.sample_id: i for i, entry in enumerate(dataset.index)}
    if len(ids) != len(dataset):
        raise ValueError('Duplicate dataset sample IDs')
    selected = [ids[row['sample_id']] for row in manifest['samples']]
    batch_size = loader.batch_size or getattr(loader.batch_sampler, 'max_batch_size', 1)
    return DataLoader(Subset(dataset, selected), batch_size=batch_size,
                      collate_fn=loader.collate_fn, num_workers=0)


def validate_comparison_manifest(comparison, manifest, *, require_images=False):
    """Validate the shared image population independently of candidate reports."""
    if comparison.get('schema_version') != 1:
        raise ValueError('Unsupported frozen comparison manifest')
    require_same(comparison['seed'], manifest['seed'], 'comparison seed')
    require_same(comparison['length_definition'], LENGTH_DEFINITION, 'comparison length definition')
    dataset_hash = hashlib.sha256(json.dumps(manifest['split_file_sha256'], sort_keys=True).encode()).hexdigest()
    require_same(comparison['dataset_sha256'], dataset_hash, 'comparison dataset hash')
    samples = comparison['samples']
    if require_images and not samples:
        raise ValueError('Selection requires a nonempty frozen comparison population')
    expected = records_by_id(manifest['samples'])
    if not records_by_id(samples).keys() <= expected.keys():
        raise ValueError('Unexpected frozen comparison IDs')
    for row in samples:
        require_same(row, {key: expected[row['sample_id']][key] for key in (
            'sample_id', 'length_bucket', 'target_length')}, 'comparison target identity')


def validate_report(report_path, manifest, codebook_path, candidate=None, *, comparison_manifest=None):
    """Recompute all selection evidence; raise on any incomplete or changed artifact."""
    from evals.benchmark_eval import _fr_summary
    path = Path(report_path).resolve()
    report = read_json(path)
    validate_manifest(manifest)
    if report.get('schema_version') != 2:
        raise ValueError('Unsupported report schema')
    require_same(report['evaluation']['manifest_sha256'], json_sha256(manifest), 'evaluation identity')
    require_same(read_json(path.parent / report['evaluation']['manifest_path']), manifest, 'saved evaluation manifest')
    provenance = report['provenance']
    require_same(provenance['split'], manifest['split'], 'split')
    require_same(provenance['seed'], manifest['seed'], 'seed')
    require_same(report['evaluation']['dictionary'], manifest['dictionary'], 'dictionary')
    require_same(file_sha256(codebook_path), manifest['dictionary']['sha256'], 'dictionary file')
    require_same(report['evaluation']['inference_settings'], manifest['inference_settings'], 'inference settings')
    require_same(report['evaluation']['inference_fingerprint'], manifest['inference_fingerprint'], 'inference fingerprint')
    identity = report['candidate']
    if candidate is not None:
        for key in ('candidate_id', 'global_step', 'sha256', 'path'):
            require_same(identity[key], candidate[key], 'candidate ' + key)
    checkpoint = Path(identity['path'])
    require_same(file_sha256(checkpoint), identity['sha256'], 'checkpoint hash')
    require_same(provenance['checkpoint_sha256'], identity['sha256'], 'checkpoint provenance')
    require_same(str(Path(provenance['checkpoint']).resolve()), str(checkpoint.resolve()), 'checkpoint path')
    from evals.report_evidence import checkpoint_metadata, classify_report
    metadata = checkpoint_metadata(checkpoint)
    classification = classify_report(metadata,
        random_init=provenance.get('random_initialization', False),
        smoke_test=provenance.get('smoke_test', False), experiment=provenance.get('experiment', ''),
        partial_load=provenance.get('partial_loading_override', False) or not provenance.get('checkpoint_load', {'strict': True})['strict'],
        training_origin_override=provenance.get('training_origin_override'))
    for key in ('report_kind', 'not_quality_reasons'):
        require_same(report[key], classification[key], 'report evidence ' + key)
    for key in ('training_origin', 'origin_override_ignored'):
        if key in report:
            require_same(report[key], classification[key], 'report evidence ' + key)
    embedded_step = metadata.get('global_step', metadata.get('step'))
    if embedded_step is not None:
        require_same(identity['global_step'], embedded_step, 'checkpoint global step')
    if identity.get('manifest_path'):
        registry_path = Path(identity['manifest_path'])
        entries = read_json(registry_path)
        entries = [r for r in entries if r['candidate_id'] == identity['candidate_id']]
        if len(entries) != 1:
            raise ValueError('Candidate registry has missing/duplicate identity')
        for key in ('candidate_id', 'global_step', 'path', 'sha256'):
            require_same(identity[key], entries[0][key], 'candidate registry ' + key)
    elif embedded_step is None and identity['global_step'] is not None:
        raise ValueError('Missing global step metadata requires a candidate manifest')
    fr = report['free_running']
    records = read_jsonl(path.parent / fr['per_sample_path'])
    actual, expected = records_by_id(records), records_by_id(manifest['samples'])
    if actual.keys() != expected.keys():
        raise ValueError('Evaluation sample IDs missing or unexpected (partial evaluation)')
    book = np.load(codebook_path)
    layout = manifest['dictionary']['token_layout']
    limit = manifest['inference_settings']['generation_limit']
    for key, row in actual.items():
        for field, candidate_field in (('candidate_id', 'candidate_id'), ('global_step', 'global_step'), ('checkpoint_sha256', 'sha256')):
            require_same(row[field], identity[candidate_field], 'record candidate ' + field)
        target = row['target_tokens']
        if len(target) != expected[key]['target_length'] or token_sha256(target) != expected[key]['target_sha256']:
            raise ValueError('Changed target evidence: ' + key)
        for field in ('target_length', 'target_sha256', 'length_bucket'):
            require_same(row[field], expected[key][field], 'target ' + field)
        evidence = row['validation_tokens']
        checked = score_drawing(evidence, target, book, generation_limit=limit, layout=layout)
        for field, value in checked.items():
            if isinstance(value, float):
                observed = row[field]
                if not isinstance(observed, (float, int)) or not math.isfinite(observed) or not 0 <= observed <= 1 or abs(value - observed) > 1e-10:
                    raise ValueError('Inconsistent or invalid ' + field)
            else:
                require_same(row[field], value, 'drawing ' + field)
        eos = next((i + 1 for i, t in enumerate(evidence) if t == layout['eos_token_id']), len(evidence))
        require_same(row['predicted_tokens'], evidence[:eos], 'predicted token evidence')
        for field in ('token_accuracy', 'exact_match', 'eos_rate', 'max_length_hit'):
            if not math.isfinite(row[field]) or not 0 <= row[field] <= 1:
                raise ValueError('Invalid diagnostic score: ' + field)
    summary = _fr_summary(records)
    require_same(fr['total_samples'], len(records), 'aggregate count')
    expected_metrics = {f"{manifest['split']}/free_running/{k[3:]}": v for k, v in summary.items() if k.startswith('fr_')}
    require_same(fr['metrics'], expected_metrics, 'aggregate scores')
    buckets = {b: _fr_summary([r for r in records if r['length_bucket'] == b])
               for b in sorted({r['length_bucket'] for r in records})}
    require_same(fr['per_bucket'], buckets, 'bucket totals')
    comparison = read_json(path.parent / fr['comparison_manifest_path'])
    if comparison_manifest is not None:
        validate_comparison_manifest(comparison_manifest, manifest)
        require_same(comparison['samples'], comparison_manifest['samples'], 'frozen comparison population')
    if not {r['sample_id'] for r in comparison['samples']} <= expected.keys():
        raise ValueError('Unexpected comparison IDs')
    records_by_id(comparison['samples'])
    if report['evaluation']['comparison_samples'] != comparison['samples']:
        raise ValueError('Comparison population mismatch')
    index = read_json(path.parent / fr['images_index_path'])
    require_same([r['sample_id'] for r in index['images']], [r['sample_id'] for r in comparison['samples']], 'image order')
    for image in index['images']:
        if not (path.parent / image['path']).is_file():
            raise ValueError('Missing comparison image: ' + image['path'])
        require_same(image['checkpoint_sha256'], identity['sha256'], 'image checkpoint')
        require_same(image['settings'], manifest['inference_settings'], 'image settings')
    return report, records
