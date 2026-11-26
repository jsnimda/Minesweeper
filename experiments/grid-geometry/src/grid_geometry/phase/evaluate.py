"""Independent checkpoint evaluation; geometry thresholds are diagnostic."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import torch

from .dataset import PhaseDataset
from .decoder import decode_phase
from .metrics import TargetMetrics
from .model import PhaseGridUNet, load_phase_weights, phase_predictions
from .validation import validate_phase
from ..training import select_device


def geometry_record(metadata, result, validation):
    positive = metadata.get('geometry') is not None
    accepted = bool(validation and validation.accepted)
    record = {'sample_id': metadata['sample_id'], 'positive': positive, 'complete': metadata.get('complete'),
              'decoder_status': result.status, 'decoder_reasons': list(result.reasons),
              'accepted': accepted, 'validation_reasons': list(validation.reasons) if validation else ['no hypothesis'], 'correct': None}
    if not positive or not result.hypotheses:
        return record
    h, g = result.hypotheses[0], metadata['geometry']
    dx, dy = abs(h.x.pitch-g['pitch_x']), abs(h.y.pitch-g['pitch_y'])
    def alignment(origin, expected, pitch):
        return abs((origin-expected + pitch/2) % pitch - pitch/2)
    ax, ay = alignment(h.x.origin, g['origin_x'], g['pitch_x']), alignment(h.y.origin, g['origin_y'], g['pitch_y'])
    record.update(pitch_error_x=dx, pitch_error_y=dy, alignment_error_x=ax, alignment_error_y=ay)
    # Extents are assessable only for complete samples. Partial crops may not
    # expose physical endpoints or total dimensions.
    if metadata.get('complete'):
        rows, columns = metadata['board_dimensions']
        box = g['visible_box']
        boundaries = [h.x.visible_start, h.y.visible_start, h.x.visible_end, h.y.visible_end]
        errors = [abs(a-b) if a is not None else None for a,b in zip(boundaries, box)]
        record.update(rows_correct=h.rows == rows, columns_correct=h.columns == columns, boundary_errors=errors)
        record['correct'] = bool(h.rows == rows and h.columns == columns and max(dx/g['pitch_x'], dy/g['pitch_y'], ax/g['pitch_x'], ay/g['pitch_y']) <= .10 and all(e is not None and e <= 1.5 for e in errors))
    else:
        record['correct'] = bool(max(dx/g['pitch_x'], dy/g['pitch_y'], ax/g['pitch_x'], ay/g['pitch_y']) <= .10)
    return record


def summarize(records):
    n = len(records); positives = [r for r in records if r['positive']]; negatives = [r for r in records if not r['positive']]
    accepted = [r for r in positives if r['accepted']]
    def rate(a, b): return a / b if b else None
    def mean(key):
        vals = [r[key] for r in positives if key in r]
        return sum(vals)/len(vals) if vals else None
    complete = [r for r in positives if 'rows_correct' in r]
    boundary = [e for r in complete for e in r['boundary_errors'] if e is not None]
    return {'samples': n, 'positives': len(positives), 'negatives': len(negatives),
            'decoder_success_rate': rate(sum(r['decoder_status'] == 'fitted' for r in records), n),
            'acceptance_rate': rate(sum(r['accepted'] for r in records), n),
            'correct_among_accepted_positive': rate(sum(r['correct'] is True for r in accepted), len(accepted)),
            'correct_recovery_positive': rate(sum(r['accepted'] and r['correct'] is True for r in positives), len(positives)),
            'incorrect_accepted_positive_rate': rate(sum(r['accepted'] and r['correct'] is False for r in positives), len(positives)),
            'false_rejection_positive_rate': rate(sum(not r['accepted'] for r in positives), len(positives)),
            'false_acceptance_negative_rate': rate(sum(r['accepted'] for r in negatives), len(negatives)),
            'mean_pitch_error_x_px': mean('pitch_error_x'), 'mean_pitch_error_y_px': mean('pitch_error_y'),
            'mean_alignment_error_x_px': mean('alignment_error_x'), 'mean_alignment_error_y_px': mean('alignment_error_y'),
            'complete_with_hypothesis': len(complete),
            'row_accuracy_complete': rate(sum(r['rows_correct'] for r in complete), len(complete)),
            'column_accuracy_complete': rate(sum(r['columns_correct'] for r in complete), len(complete)),
            'mean_boundary_error_px': rate(sum(boundary), len(boundary)), 'boundary_count': len(boundary),
            'decoder_reasons': dict(Counter(reason for r in records for reason in r['decoder_reasons'])),
            'validation_reasons': dict(Counter(reason for r in records for reason in r['validation_reasons']))}


def evaluate(checkpoint, dataset_root, split='val', device='auto', limit=None, seed=42, output=None):
    if limit is not None and limit <= 0: raise ValueError('limit must be positive')
    if output and Path(output).exists(): raise FileExistsError(output)
    dataset = PhaseDataset(dataset_root, split, require_nonempty=True)
    indices = np.sort(np.random.default_rng(seed).choice(len(dataset), min(limit or len(dataset), len(dataset)), replace=False))
    device = select_device(device)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = PhaseGridUNet(); load_phase_weights(model, state); model.to(device).eval()
    metrics = TargetMetrics(state['config']['occupancy_weight']); records = []
    with torch.inference_mode():
        for index in indices:
            s = dataset.load_sample(int(index)); logits = model(s.image.unsqueeze(0).to(device)).cpu()
            if not torch.isfinite(logits).all(): raise FloatingPointError(f'non-finite output for {s.metadata["sample_id"]}')
            metrics.update(logits, s.targets.unsqueeze(0), s.phase_mask.unsqueeze(0), s.occupancy_mask.unsqueeze(0), torch.tensor([s.metadata.get('geometry') is not None]))
            fields = phase_predictions(logits)[0].numpy(); decoded = decode_phase(fields)
            validation = validate_phase(fields, decoded.hypotheses[0], decoder_status=decoded.status) if decoded.hypotheses else None
            records.append(geometry_record(s.metadata, decoded, validation))
    report = {'checkpoint': str(checkpoint), 'dataset_root': str(dataset_root), 'split': split, 'subset_seed': seed, 'subset_limit': limit, 'sample_ids': [r['sample_id'] for r in records], 'target_metrics': metrics.compute(), 'geometry': summarize(records), 'records': records}
    if output:
        path = Path(output)
        with path.open('x', encoding='utf-8') as f: json.dump(report, f, indent=2, allow_nan=False)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--dataset-root', required=True); p.add_argument('--split', choices=('train','val','test'), default='val'); p.add_argument('--device', default='auto'); p.add_argument('--limit', type=int); p.add_argument('--seed', type=int, default=42); p.add_argument('--output', required=True)
    a = p.parse_args(); evaluate(a.checkpoint, a.dataset_root, a.split, a.device, a.limit, a.seed, a.output)


if __name__ == '__main__': main()
