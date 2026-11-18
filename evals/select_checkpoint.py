"""Select a checkpoint using verified complete free-running validation drawings."""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.selection_common import (
    SELECTION_RULE_VERSION, atomic_json, file_sha256, json_sha256, read_json,
    require_same, validate_manifest, validate_report,
)

FLOAT_CRITERIA = ('mean_drawing_score', 'completion_rate', 'minimum_bucket_mean_drawing_score')
CRITERIA = (*FLOAT_CRITERIA, 'global_step', 'candidate_id')


def validate_candidate_identity(candidate):
    step = candidate.get('global_step')
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError('Candidate requires a nonnegative integer global_step')
    if not isinstance(candidate.get('candidate_id'), str) or not candidate['candidate_id']:
        raise ValueError('Missing candidate ID')


def rank_candidates(candidates, tolerance=1e-6):
    """Best-anchored groups avoid non-transitive approximate comparisons."""
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Tolerance must be finite and nonnegative')
    candidates = list(candidates)
    trace = []
    ids = set()
    for candidate in candidates:
        validate_candidate_identity(candidate)
        if candidate['candidate_id'] in ids:
            raise ValueError('Duplicate candidate ID')
        ids.add(candidate['candidate_id'])
        for key in FLOAT_CRITERIA:
            value = candidate[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('Invalid ranking score: ' + key)

    def refine(group, level):
        if not group or level == len(CRITERIA):
            return group
        key = CRITERIA[level]
        floating = key in FLOAT_CRITERIA
        remaining = sorted(group, key=lambda r: (-r[key] if floating else r[key], r['candidate_id']))
        ordered = []
        while remaining:
            anchor = remaining[0][key]
            end = 1
            while end < len(remaining) and (
                    anchor - remaining[end][key] <= tolerance if floating
                    else anchor == remaining[end][key]):
                end += 1
            tied, remaining = remaining[:end], remaining[end:]
            trace.append({'criterion': key, 'anchor': anchor, 'parent_ids': sorted(r['candidate_id'] for r in group),
                          'comparison': 'best_anchored_tolerance' if floating else 'exact',
                          'group_ids': sorted(r['candidate_id'] for r in tied)})
            ordered.extend(refine(tied, level + 1))
        return ordered

    return refine(candidates, 0), trace


def artifact_hashes(report_path, report):
    path = Path(report_path).resolve()
    fr = report['free_running']
    paths = [path, path.parent / report['evaluation']['manifest_path']]
    paths += [path.parent / fr[key] for key in ('per_sample_path', 'comparison_manifest_path', 'images_index_path')]
    index = read_json(path.parent / fr['images_index_path'])
    paths += [path.parent / row['path'] for row in index['images']]
    if report['candidate'].get('manifest_path'):
        paths.append(Path(report['candidate']['manifest_path']))
    return {str(p.resolve()): file_sha256(p) for p in paths}


def eligible_report(path, manifest, codebook):
    report = read_json(path)
    if manifest['split'] != 'valid' or report['provenance']['split'] != 'valid':
        raise ValueError('Selection requires split valid; test reports are ineligible')
    if report.get('report_kind') != 'quality' or report.get('not_quality_reasons'):
        raise ValueError('Non-quality report: ' + ', '.join(report.get('not_quality_reasons', [])))
    validate_candidate_identity(report['candidate'])
    report, records = validate_report(path, manifest, codebook)
    identity = report['candidate']
    values = [r['drawing_score'] for r in records]
    bucket_means = {b: math.fsum(r['drawing_score'] for r in records if r['length_bucket'] == b) /
                   sum(r['length_bucket'] == b for r in records)
                   for b in sorted({r['length_bucket'] for r in records})}
    return dict(**identity, report_path=str(Path(path).resolve()), report_sha256=file_sha256(path),
                mean_drawing_score=math.fsum(values) / len(values),
                completion_rate=sum(r['complete'] for r in records) / len(records),
                minimum_bucket_mean_drawing_score=min(bucket_means.values()),
                bucket_mean_drawing_scores=bucket_means,
                artifacts=artifact_hashes(path, report))


def materialize(winner, output):
    source = Path(winner['path'])
    require_same(file_sha256(source), winner['sha256'], 'winner checkpoint')
    # Inspect format rather than trusting a filename extension.
    try:
        from safetensors import safe_open
        with safe_open(str(source), framework='pt', device='cpu'):
            pass
        suffix = '.safetensors'
    except Exception:
        import torch
        # A file object prevents torch.load dispatching by a misleading suffix.
        with source.open('rb') as stream:
            torch.load(stream, map_location='cpu', weights_only=False)
        suffix = '.pt'
    directory = Path(output).parent / 'selected'
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / ('model' + suffix)
    fd, temporary = tempfile.mkstemp(prefix='.model-', dir=directory)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        require_same(file_sha256(temporary), winner['sha256'], 'materialized checkpoint')
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        os.unlink(temporary)
    return str(destination.resolve())


def select(args):
    manifest_path = Path(args.evaluation_manifest).resolve()
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    if manifest['split'] != 'valid':
        raise ValueError('Selection manifest must use split valid')
    require_same(file_sha256(args.codebook_path), manifest['dictionary']['sha256'], 'dictionary file')
    # Reject a bad tolerance even when there are no reports, before publishing artifacts.
    rank_candidates([], args.tolerance)
    if os.path.lexists(args.output):
        raise FileExistsError('Selection output already exists')
    if not Path(args.reports_dir).is_dir():
        raise ValueError('Reports directory does not exist')
    eligible, rejected = [], []
    for path in sorted(Path(args.reports_dir).rglob('*_report.json')):
        try:
            eligible.append(eligible_report(path, manifest, args.codebook_path))
        except Exception as exc:
            # A corrupt candidate artifact must not prevent other candidates being checked.
            rejected.append({'report_path': str(path.resolve()), 'reasons': [str(exc)]})
    # Ambiguous IDs invalidate every conflicting report, independent of read order.
    counts = {}
    for row in eligible:
        counts[row['candidate_id']] = counts.get(row['candidate_id'], 0) + 1
    unique = []
    for row in eligible:
        if counts[row['candidate_id']] > 1:
            rejected.append({'report_path': row['report_path'], 'reasons': ['Duplicate candidate ID']})
        else:
            unique.append(row)
    ranking, trace = rank_candidates(unique, args.tolerance)
    winner = ranking[0] if ranking else None
    selection = dict(schema_version=1, rule_version=SELECTION_RULE_VERSION, tolerance=args.tolerance,
                     evaluation_manifest_path=str(manifest_path),
                     evaluation_manifest_sha256=file_sha256(manifest_path),
                     evaluation_identity=json_sha256(manifest), dictionary=manifest['dictionary'],
                     inference_settings=manifest['inference_settings'], inference_fingerprint=manifest['inference_fingerprint'],
                     completeness_rule_version=manifest['completeness_rule_version'], scoring_rule_version=manifest['scoring_rule_version'],
                     ranking=ranking, tie_resolution_trace=trace,
                     rejected_candidates=sorted(rejected, key=lambda r: r['report_path']), winner=winner)
    if winner and args.materialize:
        selection['selected_checkpoint'] = materialize(winner, args.output)
    atomic_json(args.output, selection)
    if not winner:
        print('No eligible candidates; loss fallback is disabled', file=sys.stderr)
        return 2
    print(f"Selected {winner['candidate_id']}; selection alone does not establish satisfactory quality")
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    pick = commands.add_parser('select')
    pick.add_argument('--evaluation-manifest', type=Path, required=True)
    pick.add_argument('--reports-dir', type=Path, required=True)
    pick.add_argument('--codebook-path', type=Path, required=True)
    pick.add_argument('--output', type=Path, required=True)
    pick.add_argument('--tolerance', type=float, default=1e-6)
    pick.add_argument('--materialize', action='store_true')
    return result


def main():
    args = parser().parse_args()
    try:
        return {'select': select}[args.command](args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'[selection] {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
