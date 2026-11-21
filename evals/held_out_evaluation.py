"""Verify a frozen validation decision before evaluating its winner on test."""
from __future__ import annotations

import copy
from pathlib import Path

from evals.candidate_evaluation import (
    evaluate_candidate, freeze_candidate_snapshot, prepare_frozen_evaluation,
)
from evals.selection_common import (
    SELECTION_RULE_VERSION, atomic_json, file_sha256, json_sha256, read_json,
    require_same, validate_comparison_manifest, validate_manifest, validate_report,
)


def verify_selection(selection_path, codebook_path, checkpoint_override=None):
    """Check every ranked artifact and replay the decision against its frozen spec."""
    from evals.select_checkpoint import eligible_report, rank_candidates
    selection_path = Path(selection_path).resolve()
    digest = file_sha256(selection_path)
    selection = read_json(selection_path)
    if selection['schema_version'] != 1 or selection['rule_version'] != SELECTION_RULE_VERSION:
        raise ValueError('Unsupported selection rule/schema version')
    manifest_path = Path(selection['evaluation_manifest_path'])
    require_same(file_sha256(manifest_path), selection['evaluation_manifest_sha256'], 'validation manifest hash')
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    if manifest['split'] != 'valid':
        raise ValueError('Selection must refer to split valid')
    require_same(selection['evaluation_identity'], json_sha256(manifest), 'validation evaluation identity')
    for key in ('dictionary', 'inference_settings', 'inference_fingerprint',
                'completeness_rule_version', 'scoring_rule_version'):
        require_same(selection[key], manifest[key], 'selection ' + key)
    require_same(file_sha256(codebook_path), manifest['dictionary']['sha256'], 'dictionary file')
    if not all(key in selection for key in (
            'comparison_manifest_path', 'comparison_manifest_sha256', 'comparison_identity')):
        raise ValueError('Selection lacks a frozen comparison manifest; rerun selection')
    comparison_path = Path(selection['comparison_manifest_path'])
    require_same(file_sha256(comparison_path), selection['comparison_manifest_sha256'], 'frozen comparison manifest hash')
    comparison = read_json(comparison_path)
    validate_comparison_manifest(comparison, manifest, require_images=True)
    require_same(json_sha256(comparison), selection['comparison_identity'], 'frozen comparison identity')
    if not selection['ranking'] or selection['winner'] is None:
        raise ValueError('Selection has no eligible winner')
    require_same(selection['winner'], selection['ranking'][0], 'selection winner')
    verified = []
    for candidate in selection['ranking']:
        require_same(file_sha256(candidate['path']), candidate['sha256'], 'ranked checkpoint hash')
        require_same(file_sha256(candidate['report_path']), candidate['report_sha256'], 'ranked report hash')
        for artifact, expected_hash in candidate['artifacts'].items():
            require_same(file_sha256(artifact), expected_hash, 'selection artifact hash: ' + artifact)
        # Recomputing also checks that the saved artifact inventory is complete.
        current = eligible_report(candidate['report_path'], manifest, codebook_path, comparison)
        require_same(candidate, current, 'ranked candidate evidence')
        verified.append(current)
    ranking, trace = rank_candidates(verified, selection['tolerance'])
    require_same(selection['ranking'], ranking, 'selection ranking')
    require_same(selection['tie_resolution_trace'], trace, 'selection tie-resolution trace')
    winner = selection['winner']
    if selection.get('selected_checkpoint'):
        require_same(file_sha256(selection['selected_checkpoint']), winner['sha256'], 'materialized checkpoint hash')
    checkpoint = Path(checkpoint_override or selection.get('selected_checkpoint') or winner['path']).resolve()
    require_same(file_sha256(checkpoint), winner['sha256'], 'test checkpoint hash')
    require_same(file_sha256(comparison_path), selection['comparison_manifest_sha256'], 'frozen comparison manifest hash')
    require_same(file_sha256(selection_path), digest, 'selection changed during verification')
    return selection, manifest, checkpoint, digest


def replay_config(selection, report, config_path=None):
    """Recreate the frozen inference config without applying a training stage."""
    from scripts.sketchformer.config import compose_training_config
    settings = selection['inference_settings']
    experiment = report['provenance'].get('experiment', 'anime_tok_dict_long')
    # A standalone report may name a custom experiment with no repository preset.
    config = compose_training_config(config_path or 'configs/train.yaml',
                                     experiment=None if config_path else 'anime_tok_dict_long')
    config.get('experiment', {}).pop('overrides', None)
    if config_path is None:
        config['experiment']['name'] = experiment
        config['model'] = copy.deepcopy(settings['model'])
        config['data']['sequence'] = copy.deepcopy(settings['token_processing'])
        config['data']['format']['type'] = 'tok_dict'
        config['data']['format']['token_dictionary'] = copy.deepcopy(settings['token_layout'])
        config['trainer']['runtime'].update(settings['runtime'])
        config['trainer']['runtime']['precision'] = settings['effective_precision']
    return config, config.get('experiment', {}).get('name', experiment)


def evaluate_test(args):
    selection, validation, checkpoint, selection_hash = verify_selection(
        args.selection, args.codebook_path, args.checkpoint)
    winner = selection['winner']
    report = read_json(winner['report_path'])
    config, args.experiment = replay_config(selection, report, args.config)
    args.seed = validation['seed']
    args.eval_samples = None  # The complete requested test split is always evaluated.
    args.smoke_test = False
    args.training_origin = report['provenance'].get('training_origin_override')
    if args.precision is None and args.config is None:
        args.precision = selection['inference_settings']['effective_precision']
    if args.max_generation_length is None:
        args.max_generation_length = selection['inference_settings']['generation_limit']
    manifest_path, manifest, comparison_path, config_path = prepare_frozen_evaluation(
        args, split='test', config=config, validation_manifest=validation)
    reference = dict(selection_path=str(Path(args.selection).resolve()), selection_sha256=selection_hash,
                     validation_manifest_sha256=selection['evaluation_manifest_sha256'],
                     winner_candidate_id=winner['candidate_id'])
    output = Path(args.output_dir).resolve()
    reference_path = output / 'selection_reference.json'
    if reference_path.exists():
        require_same(read_json(reference_path), reference, 'selection reference; use a fresh output directory')
    else:
        atomic_json(reference_path, reference)
    # A materialized/overridden legacy checkpoint needs its own path in the registry.
    candidate = {key: winner[key] for key in ('candidate_id', 'global_step', 'sha256')}
    candidate['path'] = str(checkpoint)
    freeze_candidate_snapshot(output / 'candidates.json', [candidate], [])
    path, skipped = evaluate_candidate(candidate, args, manifest_path, manifest, comparison_path,
                                       config_path, held_out=reference)
    test_report, _ = validate_report(path, manifest, args.codebook_path, candidate,
                                     comparison_manifest=read_json(comparison_path))
    require_same(test_report.get('evaluation_role'), 'held_out', 'held-out report role')
    require_same(test_report.get('held_out'), reference, 'held-out selection reference')
    require_same(test_report.get('selection_eligible'), False, 'held-out selection eligibility')
    require_same(test_report.get('selection_score'), None, 'held-out selection score')
    # Ensure the validation decision and its evidence remained unchanged during generation.
    verify_selection(args.selection, args.codebook_path, args.checkpoint)
    require_same(file_sha256(args.selection), selection_hash, 'selection changed during test evaluation')
    attempt = 1
    while (output / f'held_out_evaluation_attempt_{attempt:04d}.json').exists():
        attempt += 1
    atomic_json(output / f'held_out_evaluation_attempt_{attempt:04d}.json',
                dict(evaluation_role='held_out', selection_eligible=False, selection_score=None,
                     **reference, report_path=str(path), status='skipped' if skipped else 'complete'))
    print(f'Held-out test report: {path}; selection is unchanged')
    return 0
