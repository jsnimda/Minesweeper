"""Independent checkpoint evaluation; geometry thresholds are diagnostic."""
import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..dataset.loader import PhaseDataset
from ..dataset.targets import PhaseGeometry
from ..dataset.equivalent_boundaries import equivalent_boundaries
from .decoder import DecodeResult, decode_phase
from .metrics import TargetMetrics
from .model import PhaseGridUNet, load_phase_weights, phase_predictions
from .validation import validate_phase
from .global_geometry import recognize_global
from .tiles import TileObservation
from ..training import select_device

REPORT_VERSION = 3
BOUNDARY_TOLERANCE_PX = 1.5
SIDES = ('left', 'top', 'right', 'bottom')


def boundary_record(metadata, hypothesis):
    """Compare local extent lines in sample coordinates, without clipping.

    The nearest ground-truth lattice line (half-pitch ties go upward) defines
    line selection; localization is measured to the nearest allowed side target.
    """
    g = PhaseGeometry(**metadata['geometry'])
    targets = equivalent_boundaries(g, metadata['board_dimensions'], metadata['known_box'])
    predicted = ((hypothesis.x.origin, hypothesis.y.origin, hypothesis.x.end, hypothesis.y.end)
                 if hypothesis else (None,) * 4)
    origins = (g.origin_x, g.origin_y, g.origin_x, g.origin_y)
    pitches = (g.pitch_x, g.pitch_y, g.pitch_x, g.pitch_y)
    sides = {}
    for name, value, origin, pitch in zip(SIDES, predicted, origins, pitches):
        choices = getattr(targets, name)
        matched = min(choices, key=lambda target: abs(value - target)) if value is not None else None
        error = abs(value - matched) if value is not None else None
        line = (math.floor((value - origin) / pitch + .5) ==
                round((matched - origin) / pitch)) if value is not None else None
        sides[name] = {'targets': list(choices), 'predicted': value, 'matched': matched,
                       'line_correct': line, 'error_px': error,
                       'error_cells': error / pitch if error is not None else None,
                       'correct': bool(line and error <= BOUNDARY_TOLERANCE_PX) if value is not None else None}
    # Individual midpoint choices can form a zero-area box. Check all valid
    # alternatives, rather than choosing a canonical box or combining sides.
    valid_box = any(all(pred is not None and
                        math.floor((pred-origin)/pitch + .5) == round((target-origin)/pitch) and
                        abs(pred-target) <= BOUNDARY_TOLERANCE_PX
                        for pred, target, origin, pitch in zip(predicted, box, origins, pitches))
                    for box in targets.valid_boxes())
    return {'sides': sides, 'valid_box_correct': valid_box}


def geometry_record(metadata, result, validation):
    positive = metadata.get('geometry') is not None
    accepted = bool(validation and validation.accepted)
    record = {'sample_id': metadata['sample_id'], 'positive': positive, 'complete': metadata.get('complete'),
              'category': metadata.get('category', 'unknown'), 'pitch_bin': metadata.get('actual_pitch_bin_x'),
              'decoder_status': result.status, 'decoder_reasons': list(result.reasons),
              'accepted': accepted, 'local_accepted': bool(validation and validation.local_accepted),
              'local_reasons': list(validation.local_reasons) if validation else ['no hypothesis'],
              'local_correct': None, 'local_geometry_correct': None,
              'accepted_local_geometry_correct': False,
              'validation_reasons': list(validation.reasons) if validation else ['no hypothesis'], 'correct': None}
    if positive:
        record['equivalent_boundaries'] = boundary_record(metadata, result.hypotheses[0] if result.hypotheses else None)
    if not positive or not result.hypotheses:
        return record
    h, g = result.hypotheses[0], metadata['geometry']
    dx, dy = abs(h.x.pitch-g['pitch_x']), abs(h.y.pitch-g['pitch_y'])
    def alignment(origin, expected, pitch):
        return abs((origin-expected + pitch/2) % pitch - pitch/2)
    ax, ay = alignment(h.x.origin, g['origin_x'], g['pitch_x']), alignment(h.y.origin, g['origin_y'], g['pitch_y'])
    record.update(pitch_error_x=dx, pitch_error_y=dy, alignment_error_x=ax, alignment_error_y=ay)
    record['local_correct'] = bool(max(dx/g['pitch_x'], dy/g['pitch_y'], ax/g['pitch_x'], ay/g['pitch_y']) <= .10)
    if 'equivalent_boundaries' in record:
        record['local_geometry_correct'] = bool(record['local_correct'] and record['equivalent_boundaries']['valid_box_correct'])
        record['accepted_local_geometry_correct'] = bool(record['local_accepted'] and record['local_geometry_correct'])
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
        record['correct'] = record['local_correct']
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
    categories = {}
    for category in ('complete', 'interior', 'side', 'corner', 'negative', 'hard_negative', 'unknown'):
        group = [r for r in records if r.get('category', 'unknown') == category]
        categories[category] = {'samples': len(group),
            'valid_local': sum(r.get('local_accepted', False) for r in group),
            'correct_local': sum(r.get('local_correct') is True for r in group),
            'incorrect_local': sum(r.get('local_correct') is False for r in group),
            'missing_local': sum(r.get('local_correct') is None for r in group),
            'correct_local_rejected': sum(r.get('local_correct') is True and not r.get('local_accepted', False) for r in group),
            'valid_complete': sum(r['accepted'] for r in group),
            'incorrect_accepted_complete': sum(bool(r['accepted'] and r.get('complete') and r['correct'] is False) for r in group),
            'correct_complete_rejected': sum(bool(r.get('complete') and r['correct'] is True and not r['accepted']) for r in group),
            'negative_false_acceptance': sum(r['accepted'] for r in group if not r['positive'])}
    return {'samples': n, 'positives': len(positives), 'negatives': len(negatives), 'by_category': categories,
            'local_geometry_recovery': summarize_local_geometry(records),
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


def summarize_local_geometry(records):
    """Correctness-aware sample and positive-side counts, before and after validation."""
    positives = [r for r in records if r['positive']]
    if any('equivalent_boundaries' not in r or 'local_geometry_correct' not in r for r in positives):
        raise ValueError('local geometry metrics require equivalent-boundary records')
    def rate(n, d): return n / d if d else None
    def sample_counts(key):
        correct_hypotheses = sum(r.get(key) is True for r in positives)
        tp = sum(r.get('local_accepted', False) and r.get(key) is True for r in positives)
        fp = sum(r.get('local_accepted', False) and (not r['positive'] or r.get(key) is not True) for r in records)
        fn = len(positives) - tp
        return {'correct_hypotheses': correct_hypotheses, 'correct_accepted': tp,
                'tp': tp, 'fp': fp, 'fn': fn,
                'tn': sum(not r.get('local_accepted', False) for r in records if not r['positive']),
                'precision': rate(tp, tp + fp), 'recall': rate(tp, tp + fn)}
    def boundaries(group, accepted, direction=None):
        items = [(r, r['equivalent_boundaries']['sides'][side]) for r in group
                 for side in ((direction,) if direction else SIDES)]
        available = [(r, s) for r, s in items if s['predicted'] is not None and (not accepted or r['local_accepted'])]
        tp = sum(s['correct'] is True for _, s in available)
        incorrect = len(available) - tp
        missing = len(items) - len(available)
        errors = [s['error_px'] for _, s in available]
        relative = [s['error_cells'] for _, s in available]
        selected = [s for _, s in available if s['line_correct']]
        return {'expected': len(items), 'correct': tp, 'incorrect': incorrect, 'missing_or_rejected': missing,
                'precision': rate(tp, tp + incorrect), 'recall': rate(tp, len(items)),
                'line_selection_rate': rate(len(selected), len(available)),
                'mean_error_px': rate(sum(errors), len(errors)),
                'mean_error_cells': rate(sum(relative), len(relative)),
                'mean_error_selected_line_px': rate(sum(s['error_px'] for s in selected), len(selected)),
                'available': len(available)}
    with_targets = [r for r in positives if 'equivalent_boundaries' in r]
    def breakdown(group):
        return {'lattice': sample_counts_for(group, 'local_correct'),
                'geometry': sample_counts_for(group, 'local_geometry_correct'),
                'hypothesis_boundaries': boundaries(group, False),
                'accepted_boundaries': boundaries(group, True)}
    def sample_counts_for(group, key):
        # Category and pitch breakdowns contain positive samples only.
        tp = sum(r.get('local_accepted', False) and r.get(key) is True for r in group if r['positive'])
        fp = sum(r.get('local_accepted', False) and (not r['positive'] or r.get(key) is not True) for r in group)
        pos = sum(r['positive'] for r in group)
        return {'correct_hypotheses': sum(r.get(key) is True for r in group if r['positive']),
                'correct_accepted': tp, 'precision': rate(tp, tp + fp), 'recall': rate(tp, pos)}
    return {'lattice': sample_counts('local_correct'), 'geometry': sample_counts('local_geometry_correct'),
            'hypothesis_boundaries': boundaries(with_targets, False),
            'accepted_boundaries': boundaries(with_targets, True),
            'by_direction': {side: {'hypothesis': boundaries(with_targets, False, side),
                                     'accepted': boundaries(with_targets, True, side)} for side in SIDES},
            'by_category': {category: breakdown([r for r in with_targets if r['category'] == category])
                            for category in ('complete', 'interior', 'side', 'corner')},
            'by_pitch_bin': {str(pitch): breakdown([r for r in with_targets if r.get('pitch_bin') == pitch])
                             for pitch in sorted({r['pitch_bin'] for r in with_targets if r.get('pitch_bin') is not None})},
            'negative_by_category': {category: {'samples': len(group),
                'accepted': sum(r['local_accepted'] for r in group),
                'false_acceptance_rate': rate(sum(r['local_accepted'] for r in group), len(group))}
                for category in ('negative', 'hard_negative')
                for group in [[r for r in records if not r['positive'] and r['category'] == category]]}}


def known_region(metadata, fields):
    """Return the pixel-center-selected known crop and its image-space box."""
    height, width = fields.shape[1:]
    box = metadata.get('known_box', (0, 0, width, height))
    x0, y0, x1, y1 = (math.ceil(float(v) - .5) for v in box)
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    return fields[:, y0:y1, x0:x1], (x0, y0, x1, y1)


def local_recognition(metadata, fields):
    """Validate in crop coordinates, then return hypotheses in image coordinates.

    Signal and occupancy profile arrays remain indexed within the known crop.
    """
    values, (x0, y0, _, _) = known_region(metadata, fields)
    if min(values.shape[1:]) < 2:
        return DecodeResult('insufficient_evidence', (), reasons=('known region too small',)), None
    decoded = decode_phase(values)
    h = decoded.hypotheses[0] if decoded.hypotheses else None
    local = validate_phase(values, h, decoder_status=decoded.status) if h else None
    if not (x0 or y0):
        return decoded, local

    def candidate(value, offset):
        return replace(value, origin_mod_pitch=(value.origin_mod_pitch + offset) % value.pitch)

    def extent(value, offset):
        return replace(value, origin=value.origin + offset,
                       visible_start=None if value.visible_start is None else value.visible_start + offset,
                       visible_end=None if value.visible_end is None else value.visible_end + offset)

    hypotheses = tuple(replace(h, x=extent(h.x, x0), y=extent(h.y, y0),
                               x_evidence=candidate(h.x_evidence, x0),
                               y_evidence=candidate(h.y_evidence, y0)) for h in decoded.hypotheses)
    candidates = {axis: tuple(candidate(c, offset) for c in decoded.frequency_candidates.get(axis, ()))
                  for axis, offset in (('x', x0), ('y', y0)) if axis in decoded.frequency_candidates}
    return replace(decoded, hypotheses=hypotheses, frequency_candidates=candidates), local


def global_record(metadata, fields):
    """Evaluate the lab's single-crop global stage on known source pixels."""
    values, (x0, y0, x1, y1) = known_region(metadata, fields)
    if min(values.shape[1:]) < 2:
        return {'accepted': False, 'local_accepted': False, 'reasons': ['known region too small'], 'correct': None}
    decoded = decode_phase(values)
    h = decoded.hypotheses[0] if decoded.hypotheses else None
    local = validate_phase(values, h, decoder_status=decoded.status) if h else None
    observation = TileObservation((0, 0, x1-x0, y1-y0), h, bool(local and local.local_accepted),
                                  local.local_reasons if local else decoded.reasons, values)
    result = recognize_global([observation], values.shape[1:])
    record = {'accepted': result.accepted, 'reasons': list(result.reasons), 'correct': None,
              'local_accepted': bool(local and local.local_accepted), 'known_region': [x0, y0, x1, y1]}
    if metadata.get('complete') and metadata.get('geometry') and result.geometry is not None:
        g, geom = metadata['geometry'], result.geometry
        rows, columns = metadata['board_dimensions']
        edges = [geom.x_edges[0]+x0, geom.y_edges[0]+y0, geom.x_edges[-1]+x0, geom.y_edges[-1]+y0]
        errors = [abs(float(a)-b) for a, b in zip(edges, g['visible_box'])]
        px, py = float(np.diff(geom.x_edges).mean()), float(np.diff(geom.y_edges).mean())
        ax = abs((edges[0]-g['origin_x']+g['pitch_x']/2) % g['pitch_x']-g['pitch_x']/2)
        ay = abs((edges[1]-g['origin_y']+g['pitch_y']/2) % g['pitch_y']-g['pitch_y']/2)
        record.update(rows_correct=geom.rows == rows, columns_correct=geom.columns == columns, boundary_errors=errors)
        record['correct'] = bool(geom.rows == rows and geom.columns == columns and max(abs(px-g['pitch_x'])/g['pitch_x'], abs(py-g['pitch_y'])/g['pitch_y'], ax/g['pitch_x'], ay/g['pitch_y']) <= .10 and max(errors) <= 1.5)
    return record


def summarize_global(records):
    complete = [r for r in records if r['positive'] and r['complete']]
    negative = [r for r in records if not r['positive']]
    partial = [r for r in records if r['positive'] and not r['complete']]
    def count(group, predicate): return sum(bool(predicate(r['global'])) for r in group)
    return {'complete_samples': len(complete), 'complete_accepted': count(complete, lambda g: g['accepted']),
            'complete_correct_accepted': count(complete, lambda g: g['accepted'] and g['correct'] is True),
            'complete_incorrect_accepted': count(complete, lambda g: g['accepted'] and g['correct'] is False),
            'complete_correct_rejected': count(complete, lambda g: not g['accepted'] and g['correct'] is True),
            'complete_with_geometry': count(complete, lambda g: g['correct'] is not None),
            'complete_rows_correct': count(complete, lambda g: g.get('rows_correct', False)),
            'complete_columns_correct': count(complete, lambda g: g.get('columns_correct', False)),
            'partial_samples': len(partial), 'partial_complete_accepted': count(partial, lambda g: g['accepted']),
            'negative_samples': len(negative), 'negative_accepted': count(negative, lambda g: g['accepted'])}


def evaluate(checkpoint, dataset_root, split='val', device='auto', limit=None, seed=42, output=None, global_geometry=False):
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
            fields = phase_predictions(logits)[0].numpy()
            decoded, validation = local_recognition(s.metadata, fields)
            record = geometry_record(s.metadata, decoded, validation)
            if global_geometry:
                record['global'] = global_record(s.metadata, fields)
            records.append(record)
    report = {'report_version': REPORT_VERSION,
              'local_geometry_criteria': {'lattice_pitch_and_alignment_max_fraction': .10,
                  'boundary_localization_max_px': BOUNDARY_TOLERANCE_PX,
                  'midpoint_tolerance_fraction': .05,
                  'line_selection': 'nearest ground-truth lattice line; half-pitch ties toward increasing coordinates',
                  'boundary_order': list(SIDES)},
              'checkpoint': str(checkpoint), 'dataset_root': str(dataset_root), 'split': split, 'subset_seed': seed, 'subset_limit': limit, 'sample_ids': [r['sample_id'] for r in records], 'target_metrics': metrics.compute(), 'geometry': summarize(records), 'records': records}
    if global_geometry:
        report['global_geometry_single_crop'] = summarize_global(records)
    if output:
        path = Path(output)
        with path.open('x', encoding='utf-8') as f: json.dump(report, f, indent=2, allow_nan=False)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--dataset-root', required=True); p.add_argument('--split', choices=('train','val','test'), default='val'); p.add_argument('--device', default='auto'); p.add_argument('--limit', type=int); p.add_argument('--seed', type=int, default=42); p.add_argument('--output', required=True)
    p.add_argument('--global-geometry', action='store_true', help='also evaluate the final single-crop global recognition stage')
    a = p.parse_args(); evaluate(a.checkpoint, a.dataset_root, a.split, a.device, a.limit, a.seed, a.output, a.global_geometry)


if __name__ == '__main__': main()
