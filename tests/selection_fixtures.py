"""Temporary synthetic reports for selection mechanics, never model evidence."""
import json
from pathlib import Path
import numpy as np
import torch
from evals.benchmark_eval import _fr_summary
from evals.selection_common import (
    dictionary_identity, json_sha256, token_sha256, file_sha256,
)
from metrics.sketchformer.drawing import (
    token_layout, score_drawing, COMPLETENESS_RULE_VERSION, SCORING_RULE_VERSION,
)


def write(path, value):
    Path(path).write_text(json.dumps(value))


def synthetic_report(root, name='a', step=1, prediction=None, split='valid'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    book = np.array([[0., 0], [1, 0], [0, 1], [100, 100]], dtype=np.float32)
    codebook = root / 'codebook.npy'
    if not codebook.exists():
        np.save(codebook, book)
    target = [6, 1, 2, 5, 7]
    identity = dictionary_identity(codebook, token_layout(book))
    settings = {'token_layout': identity['token_layout'], 'generation_limit': 8}
    row = dict(sample_id='sample', target_length=5, target_sha256=token_sha256(target), length_bucket='short')
    manifest = dict(schema_version=1, split=split, samples=[row], dictionary=identity,
                    inference_settings=settings, inference_fingerprint=json_sha256(settings), seed=42,
                    split_file_sha256={'valid.npz': 'synthetic'}, code_revision='synthetic',
                    completeness_rule_version=COMPLETENESS_RULE_VERSION, scoring_rule_version=SCORING_RULE_VERSION)
    write(root / 'evaluation_manifest.json', manifest)
    checkpoint = root / (name + '.pt')
    torch.save({'model': {}, 'step': step, 'training_origin': 'real_training'}, checkpoint)
    candidate = dict(candidate_id=name, global_step=step, path=str(checkpoint), sha256=file_sha256(checkpoint))
    prediction = prediction or target
    checked = score_drawing(prediction, target, book, generation_limit=8)
    record = dict(**row, **checked, target_tokens=target, predicted_tokens=prediction[:prediction.index(7) + 1] if 7 in prediction else prediction,
                  validation_tokens=prediction, token_accuracy=1., exact_match=float(prediction == target),
                  eos_rate=float(7 in prediction), max_length_hit=float(7 not in prediction and len(prediction) >= 8))
    (root / (name + '_per_sample.jsonl')).write_text(json.dumps(record) + '\n')
    comparison = [{k: row[k] for k in ('sample_id', 'length_bucket', 'target_length')}]
    write(root / (name + '_comparison_manifest.json'), dict(samples=comparison))
    image = root / (name + '.png')
    image.write_bytes(b'synthetic image existence fixture')
    write(root / (name + '_images_index.json'), {'images': [dict(sample_id='sample', path=image.name,
        checkpoint_sha256=candidate['sha256'], settings=settings)]})
    summary = _fr_summary([record])
    report = dict(schema_version=2, report_kind='quality', not_quality_reasons=[],
                  provenance=dict(split=split, checkpoint=str(checkpoint), checkpoint_sha256=candidate['sha256'], seed=42),
                  candidate=candidate,
                  evaluation=dict(manifest_sha256=json_sha256(manifest), manifest_path=str(root / 'evaluation_manifest.json'),
                                  dictionary=identity, inference_settings=settings,
                                  inference_fingerprint=manifest['inference_fingerprint'], comparison_samples=comparison),
                  free_running=dict(total_samples=1, per_sample_path=name + '_per_sample.jsonl',
                                    metrics={f'{split}/free_running/{k[3:]}': v for k, v in summary.items() if k.startswith('fr_')},
                                    per_bucket={'short': summary}, comparison_manifest_path=name + '_comparison_manifest.json',
                                    images_index_path=name + '_images_index.json'))
    path = root / (name + '_report.json')
    write(path, report)
    return path, manifest, codebook, candidate
