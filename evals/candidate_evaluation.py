"""Sequential candidate evaluation with frozen settings and preserved attempts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from evals.report_evidence import checkpoint_metadata, classify_report, wrapper_run_name
from evals.selection_common import (
    atomic_json, build_evaluation_manifest, dictionary_identity, dictionary_layout,
    file_sha256, freeze_manifest, inference_settings, read_json, read_jsonl,
    require_same, validate_report,
)


def candidates_from_sources(manifest_path=None, checkpoints=None, *, failures=None):
    """Verify source identities; optionally collect failures to continue other candidates."""
    if bool(manifest_path) == bool(checkpoints):
        raise ValueError('Choose exactly one candidate source')
    if manifest_path:
        manifest_path = Path(manifest_path).resolve()
        rows = read_jsonl(manifest_path)
    else:
        rows = [{'path': str(Path(path).resolve())} for path in checkpoints]
    if not rows:
        raise ValueError('No requested candidates')
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Candidate manifest entries must be objects')
        path = None
        try:
            path = Path(row['path'])
            if not path.is_absolute():
                path = manifest_path.parent / path
            path = path.resolve()
            digest = file_sha256(path)
            if row.get('sha256') is not None:
                require_same(row['sha256'], digest, 'candidate manifest hash')
            metadata = checkpoint_metadata(path)
            step = metadata.get('global_step', metadata.get('step'))
            if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
                raise ValueError(f'{path}: invalid global step in checkpoint metadata')
            if row.get('global_step') is not None:
                if step is not None:
                    require_same(row['global_step'], step, 'candidate global step')
                step = row['global_step']
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError(f'{path}: missing/invalid global step; supply a candidate-manifest entry')
            candidate_id = row.get('candidate_id', metadata.get('candidate_id'))
            if candidate_id is None:
                candidate_id = f'{path.stem}-{digest[:12]}'
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError('Missing candidate ID')
            if row.get('candidate_id') is not None and metadata.get('candidate_id') is not None:
                require_same(row['candidate_id'], metadata['candidate_id'], 'candidate ID')
            result.append(dict(candidate_id=candidate_id, path=str(path), sha256=digest, global_step=step))
        except Exception as exc:
            if failures is None:
                raise
            failures.append(dict(candidate_id=row.get('candidate_id'), path=str(path) if path is not None else row.get('path'),
                                 status='failed', reason=str(exc)))
    # Reject every ambiguous identity, independent of source order.
    counts = {}
    for row in result:
        counts[row['candidate_id']] = counts.get(row['candidate_id'], 0) + 1
    unique = []
    for row in result:
        if counts[row['candidate_id']] > 1:
            if failures is None:
                raise ValueError('Duplicate candidate ID: ' + row['candidate_id'])
            failures.append(dict(candidate_id=row['candidate_id'], path=row['path'],
                                 status='failed', reason='Duplicate candidate ID'))
        else:
            unique.append(row)
    return unique


def freeze_candidate_snapshot(path, candidates, failures):
    """Keep the registry immutable while letting unavailable candidates fail individually."""
    if not path.exists():
        atomic_json(path, candidates)
        return
    frozen = read_json(path)
    available = {row['candidate_id']: row for row in candidates}
    failed_paths = {row['path'] for row in failures}
    retained = []
    for row in frozen:
        if row['candidate_id'] in available:
            retained.append(available.pop(row['candidate_id']))
        elif row['path'] in failed_paths:
            retained.append(row)
    require_same(frozen, retained + list(available.values()),
                 'candidate snapshot; use a fresh output directory')


def prepare_frozen_evaluation(args):
    from evals.benchmark_eval import _prepare_loader
    from scripts.sketchformer.config import compose_training_config, resolve_device
    from scripts.sketchformer.train import _resolve_precision_runtime
    from evals.benchmark_common import LENGTH_DEFINITION, select_comparison_samples
    config = compose_training_config(args.config, experiment=args.experiment)
    # This is already composed. Do not reapply stage/training overrides in the child.
    config.get('experiment', {}).pop('overrides', None)
    config['data']['dataset']['root'] = str(Path(args.data_root).resolve())
    config.setdefault('project', {})['seed'] = args.seed
    layout = config['data']['format']['token_dictionary']
    layout.update(codebook_dir=None, codebook_path=str(Path(args.codebook_path).resolve()))
    config['model']['input']['token_dictionary'].update(layout)
    if config['data']['format']['type'] not in {'tok_dict', 'token', 'tokens'}:
        raise ValueError('Candidate evaluation requires tok-dict reconstruction data')
    capacity = int(config['model']['input']['max_seq_len'])
    if not 2 <= args.max_generation_length <= capacity:
        raise ValueError(f'max_generation_length must be between 2 and model capacity {capacity}')
    if args.num_plots < 0:
        raise ValueError('num_plots must be nonnegative')
    if args.precision:
        config['trainer']['runtime']['precision'] = args.precision
    precision = _resolve_precision_runtime(config, resolve_device(args.device))
    config['trainer']['runtime']['precision'] = precision.effective
    loader = _prepare_loader(config, 'valid', args.seed)
    settings = inference_settings(config, precision.effective, args.max_generation_length)
    proposed = build_evaluation_manifest(loader.dataset, 'valid',
        dictionary_identity(args.codebook_path, dictionary_layout(config)), settings, args.seed, args.eval_samples)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'evaluation_manifest.json'
    try:
        manifest = freeze_manifest(manifest_path, proposed)
    except ValueError as exc:
        raise ValueError(f'{exc}; use a fresh output directory') from exc
    comparison_path = output / 'comparison_manifest.json'
    dataset_hash = hashlib.sha256(json.dumps(manifest['split_file_sha256'], sort_keys=True).encode()).hexdigest()
    comparison = dict(schema_version=1, seed=args.seed, length_definition=LENGTH_DEFINITION,
                      dataset_sha256=dataset_hash,
                      samples=select_comparison_samples(manifest['samples'], args.num_plots, args.seed))
    if comparison_path.exists():
        require_same(read_json(comparison_path), comparison, 'comparison settings; use a fresh output directory')
    else:
        atomic_json(comparison_path, comparison)
    # Every child uses this exact composed configuration rather than its training stage.
    resolved_path = output / 'inference_config.json'
    if resolved_path.exists():
        require_same(read_json(resolved_path), config, 'resolved configuration; use a fresh output directory')
    else:
        atomic_json(resolved_path, config)
    return manifest_path, manifest, comparison_path, resolved_path


def evaluate_candidate(candidate, args, manifest_path, manifest, comparison_path, config_path):
    output = Path(args.output_dir).resolve()
    name = re.sub(r'[^a-zA-Z0-9._-]+', '-', candidate['candidate_id']).strip('.-')[:100] or 'candidate'
    directory = output / (name + '-' + hashlib.sha256(candidate['candidate_id'].encode()).hexdigest()[:8])
    directory.mkdir(parents=True, exist_ok=True)
    effective_experiment = read_json(config_path).get('experiment', {}).get('name', args.experiment)
    classification = classify_report(checkpoint_metadata(candidate['path']), experiment=effective_experiment,
                                    smoke_test=args.smoke_test, training_origin_override=args.training_origin)
    comparison = read_json(comparison_path)['samples']
    # Verify provenance labels and origin overrides too before skipping.
    for report_path in sorted(directory.rglob('*_report.json')):
        try:
            report, _ = validate_report(report_path, manifest, args.codebook_path, candidate)
            for key, value in classification.items():
                require_same(report[key], value, 'report classification')
            require_same(report['evaluation']['comparison_samples'], comparison, 'frozen comparison population')
            require_same(report['provenance']['checkpoint_load']['strict'], True, 'strict checkpoint loading')
            return report_path, True
        except (ValueError, OSError, KeyError, TypeError):
            pass
    attempt = 1
    while any(directory.glob(f'*attempt-{attempt:04d}*')):
        attempt += 1
    run_name = wrapper_run_name(f'attempt-{attempt:04d}', classification)
    # Reserve an attempt before starting; never reuse an interrupted attempt.
    atomic_json(directory / (run_name + '_attempt.json'), candidate)
    attempt_dir = directory / run_name
    attempt_dir.mkdir()
    command = [sys.executable, '-B', str(Path(__file__).with_name('benchmark_eval.py')),
        '--config', str(config_path), '--experiment', args.experiment,
        '--checkpoint', candidate['path'], '--candidate-id', candidate['candidate_id'],
        '--global-step', str(candidate['global_step']), '--split', 'valid',
        '--data-root', str(Path(args.data_root).resolve()), '--codebook-path', str(Path(args.codebook_path).resolve()),
        '--device', args.device, '--precision', manifest['inference_settings']['effective_precision'],
        '--max-generation-length', str(args.max_generation_length), '--seed', str(args.seed),
        '--num-plots', str(args.num_plots), '--modes', 'free-running', '--save-predictions',
        '--evaluation-manifest', str(manifest_path), '--output-dir', str(attempt_dir), '--run-name', run_name,
        '--image-captions']
    registry = output / 'candidates.json'
    if registry.exists():
        command += ['--candidate-registry', str(registry)]
    if args.num_plots:
        command += ['--comparison-manifest', str(comparison_path)]
    if args.training_origin:
        command += ['--training-origin', args.training_origin]
    if args.smoke_test:
        command += ['--smoke-test']
    log_path = directory / (run_name + '_process.log')
    with log_path.open('w') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'Benchmark exited {completed.returncode}; see {log_path}')
    report_path = attempt_dir / (run_name + '_report.json')
    report = read_json(report_path)
    index = read_json(attempt_dir / report['free_running']['images_index_path'])
    atomic_json(attempt_dir / 'images_index.json', index)
    report['free_running']['images_index_path'] = 'images_index.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    validate_report(report_path, manifest, args.codebook_path, candidate)
    require_same(report['evaluation']['comparison_samples'], comparison, 'frozen comparison population')
    for key, value in classification.items():
        require_same(report[key], value, 'report classification')
    return report_path, False


def evaluate(args):
    outcomes = []
    candidates = candidates_from_sources(args.candidate_manifest, args.checkpoints, failures=outcomes)
    manifest_path, manifest, comparison_path, config_path = prepare_frozen_evaluation(args)
    snapshot_path = Path(args.output_dir).resolve() / 'candidates.json'
    freeze_candidate_snapshot(snapshot_path, candidates, outcomes)
    for failure in outcomes:
        print(f"[evaluate] {failure['path']}: {failure['reason']}", file=sys.stderr, flush=True)
    for candidate in candidates:
        print(f"[evaluate] {candidate['candidate_id']}", flush=True)
        try:
            path, skipped = evaluate_candidate(candidate, args, manifest_path, manifest, comparison_path, config_path)
            outcomes.append(dict(candidate_id=candidate['candidate_id'], status='skipped' if skipped else 'complete', report_path=str(path)))
        except Exception as exc:
            outcomes.append(dict(candidate_id=candidate['candidate_id'], status='failed', reason=str(exc)))
            print(f'[evaluate] {exc}', file=sys.stderr, flush=True)
    attempt = 1
    output = Path(args.output_dir)
    while (output / f'evaluation_attempt_{attempt:04d}.json').exists():
        attempt += 1
    atomic_json(output / f'evaluation_attempt_{attempt:04d}.json', {'outcomes': outcomes})
    return 2 if any(r['status'] == 'failed' for r in outcomes) else 0
