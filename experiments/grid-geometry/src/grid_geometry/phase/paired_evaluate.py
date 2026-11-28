"""Paired tiled evaluation of D7.5 and D7.6 without detection.

Sources: --dataset-root uses prepared PhaseDataset images and known_box; or
--manifest uses a JSON object with a `samples` list. Each manifest entry has
sample_id, split, image (relative to manifest), category, complete, geometry,
board_dimensions and optionally known_box and crop_box. Geometry follows
PhaseGeometry: origin_x/y, pitch_x/y and visible_box in image coordinates.
Use category negative/hard_negative and geometry=null for annotated negatives;
use category multiple for multiple boards, whose correctness is unavailable.
Crop padding expands crop_box, clipped to known_box and image bounds. No crop
box means the entire known region. This is an annotation-based geometry-only
protocol, not a detector evaluation.
"""
import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from ..dataset.loader import PhaseDataset
from ..training import select_device
from .global_geometry import GlobalSettings, recognize_global
from .model import MODEL_ID, REPRESENTATION_ID, PhaseGridUNet, load_phase_weights, phase_predictions
from .tiles import observe_tiles
from .evaluate import boundary_record


def selection(length, limit, seed):
    if limit is not None and limit <= 0:
        raise ValueError('limit must be positive')
    return np.sort(np.random.default_rng(seed).choice(length, min(limit or length, length), replace=False)).tolist()


def crop_bounds(metadata, size, padding):
    w, h = size
    if padding < 0:
        raise ValueError('padding must be nonnegative')
    known = metadata.get('known_box', (0, 0, w, h))
    if not (len(known) == 4 and all(math.isfinite(v) for v in known)
            and 0 <= known[0] < known[2] <= w and 0 <= known[1] < known[3] <= h):
        raise ValueError('invalid known_box')
    crop = metadata.get('crop_box', known)
    if not (len(crop) == 4 and all(math.isfinite(v) for v in crop)
            and crop[0] < crop[2] and crop[1] < crop[3]):
        raise ValueError('invalid crop_box')
    # Membership follows the existing half-integer pixel-center convention.
    box = (max(known[0], crop[0]-padding), max(known[1], crop[1]-padding),
           min(known[2], crop[2]+padding), min(known[3], crop[3]+padding))
    x0, y0, x1, y1 = (math.ceil(v-.5) for v in box)
    if x1-x0 < 2 or y1-y0 < 2:
        raise ValueError('empty or too small known crop')
    return x0, y0, x1, y1


def paired_run(rgb, predict, index=0, recognize=recognize_global, observer=observe_tiles):
    """Observe once; alternate recognition order to reduce order timing bias."""
    start = perf_counter()
    observations, layout = observer(rgb, predict=predict, tile=predict.tile, overlap=predict.overlap)
    shared_ms = (perf_counter()-start)*1000
    results, timing = {}, {}
    order = ('d7.5', 'd7.6') if index % 2 == 0 else ('d7.6', 'd7.5')
    for strategy in order:
        start = perf_counter()
        results[strategy] = recognize(observations, rgb.shape[:2], GlobalSettings(strategy=strategy))
        timing[strategy] = (perf_counter()-start)*1000
    return observations, layout, results, {'shared_ms': shared_ms, 'd7.5_ms': timing['d7.5'],
                                           'd7.6_ms': timing['d7.6'], 'overhead_ms': timing['d7.6']-timing['d7.5'],
                                           'order': list(order)}


def result_record(result, metadata, crop):
    category = metadata.get('category', 'unknown')
    geometry = metadata.get('geometry')
    complete = category != 'multiple' and geometry is not None and metadata.get('complete') is True
    rec = {'accepted': bool(result.accepted), 'status': result.status, 'reasons': list(result.reasons),
           'lattice_available': result.lattice is not None, 'candidate': None, 'correct': None,
           'lattice': None, 'rows_correct': None, 'columns_correct': None, 'boundary_errors_px': None}
    lattice = result.lattice
    if lattice is not None and geometry is not None and category != 'multiple':
        px, py = geometry['pitch_x'], geometry['pitch_y']
        ax = abs((lattice.origin_x+crop[0]-geometry['origin_x']+px/2) % px-px/2)
        ay = abs((lattice.origin_y+crop[1]-geometry['origin_y']+py/2) % py-py/2)
        dx, dy = abs(lattice.pitch_x-px), abs(lattice.pitch_y-py)
        rec['lattice'] = {'pitch_error_x': dx, 'pitch_error_y': dy, 'alignment_error_x': ax, 'alignment_error_y': ay,
                          'correct': max(dx/px, dy/py, ax/px, ay/py) <= .10}
    g = result.geometry
    if g is not None:
        edges = [float(g.x_edges[0]+crop[0]), float(g.y_edges[0]+crop[1]),
                 float(g.x_edges[-1]+crop[0]), float(g.y_edges[-1]+crop[1])]
        rec['candidate'] = {'edges': edges, 'rows': g.rows, 'columns': g.columns}
        if geometry is not None and category != 'multiple':
            hypothesis = SimpleNamespace(x=SimpleNamespace(origin=edges[0], end=edges[2]),
                                         y=SimpleNamespace(origin=edges[1], end=edges[3]))
            rec['equivalent_boundaries'] = boundary_record(dict(metadata, known_box=crop), hypothesis)
        if complete:
            target = geometry['visible_box']
            errors = [abs(a-b) for a, b in zip(edges, target)]
            rows, cols = metadata['board_dimensions']
            rec.update(rows_correct=g.rows == rows, columns_correct=g.columns == cols, boundary_errors_px=errors,
                       correct=bool(g.rows == rows and g.columns == cols and rec['lattice'] is not None
                                    and rec['lattice']['correct'] and max(errors) <= 1.5))
    experimental = result.experimental
    if experimental:
        measurement = experimental.get('rectangle')
        rec['experimental'] = {'rectangle': _jsonable(measurement), 'comparison': _jsonable(experimental.get('comparison')),
                               'boundaries': _jsonable(result.boundaries)}
    return rec


def _jsonable(value):
    if is_dataclass(value): return _jsonable(asdict(value))
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.generic): return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def summarize(records):
    def rate(a, b): return a/b if b else None
    summaries = {}
    for strategy in ('d7.5', 'd7.6'):
        pairs = [(r, r['results'][strategy]) for r in records]
        complete = [(r, v) for r, v in pairs if r['kind'] == 'complete']
        negative = [(r, v) for r, v in pairs if r['kind'] == 'negative']
        incomplete = [(r, v) for r, v in pairs if r['kind'] == 'incomplete']
        accepted = [(r, v) for r, v in pairs if v['accepted']]
        assessed = [(r, v) for r, v in accepted if r['kind'] in ('complete', 'negative', 'incomplete')]
        correct = sum(v['accepted'] and v['correct'] is True for _, v in complete)
        candidates = [(r, v) for r, v in complete if v['candidate'] is not None]
        ac = [(r, v) for r, v in candidates if v['accepted']]
        boundary_errors = [e for _, v in candidates for e in v['boundary_errors_px']]
        lattice_values = [v['lattice'] for _, v in pairs if v['lattice'] is not None]
        summaries[strategy] = {'samples': len(pairs), 'complete': len(complete), 'negative': len(negative),
            'incomplete': len(incomplete), 'unassessable': len(pairs)-len(complete)-len(negative)-len(incomplete),
            'usable_lattice': sum(v['lattice_available'] for _, v in pairs), 'candidates': sum(v['candidate'] is not None for _, v in pairs),
            'complete_accepted': sum(v['accepted'] for _, v in complete), 'correct_accepted': correct,
            'incorrect_accepted_complete': sum(v['accepted'] and v['correct'] is False for _, v in complete),
            'recall': rate(correct, len(complete)), 'precision': rate(correct, len(assessed)),
            'precision_denominator': len(assessed), 'negative_false_acceptance': sum(v['accepted'] for _, v in negative),
            'negative_false_acceptance_rate': rate(sum(v['accepted'] for _, v in negative), len(negative)),
            'incomplete_false_acceptance': sum(v['accepted'] for _, v in incomplete),
            'incomplete_false_acceptance_rate': rate(sum(v['accepted'] for _, v in incomplete), len(incomplete)),
            'candidate_denominator': len(candidates), 'candidate_rows_accuracy': rate(sum(v['rows_correct'] for _, v in candidates), len(candidates)),
            'candidate_columns_accuracy': rate(sum(v['columns_correct'] for _, v in candidates), len(candidates)),
            'accepted_candidate_denominator': len(ac), 'accepted_rows_accuracy': rate(sum(v['rows_correct'] for _, v in ac), len(ac)),
            'accepted_columns_accuracy': rate(sum(v['columns_correct'] for _, v in ac), len(ac)),
            'candidate_correct_rejected': sum(v['correct'] is True and not v['accepted'] for _, v in complete),
            'candidate_incorrect_rejected': sum(v['correct'] is False and not v['accepted'] for _, v in complete),
            'candidate_missing': sum(v['candidate'] is None for _, v in complete),
            'boundary_error_count': len(boundary_errors),
            'mean_boundary_error_px': rate(sum(boundary_errors), len(boundary_errors)),
            'lattice_assessed': len(lattice_values),
            'correct_lattice': sum(v['correct'] for v in lattice_values),
            'mean_pitch_error_x': rate(sum(v['pitch_error_x'] for v in lattice_values), len(lattice_values)),
            'mean_pitch_error_y': rate(sum(v['pitch_error_y'] for v in lattice_values), len(lattice_values)),
            'mean_alignment_error_x': rate(sum(v['alignment_error_x'] for v in lattice_values), len(lattice_values)),
            'mean_alignment_error_y': rate(sum(v['alignment_error_y'] for v in lattice_values), len(lattice_values)),
            'reasons': dict(Counter(reason for _, v in pairs for reason in v['reasons']))}
    groups = {k: [] for k in ('both_accepted', 'd7.5_only', 'd7.6_only', 'both_rejected')}
    changes = Counter()
    acceptance_transitions, candidate_transitions = Counter(), Counter()
    reasons_added, reasons_removed = Counter(), Counter()
    def acceptance_state(kind, value):
        if kind not in ('complete', 'negative', 'incomplete'):
            return 'unassessable'
        if not value['accepted']:
            return 'rejected'
        if kind in ('negative', 'incomplete') or value['correct'] is False:
            return 'incorrect'
        return 'correct' if value['correct'] is True else 'unassessable'
    def candidate_state(value):
        if value['candidate'] is None:
            return 'missing'
        if value['correct'] is True:
            return 'correct'
        if value['correct'] is False:
            return 'incorrect'
        return 'unassessable'
    for r in records:
        a, b = (r['results'][s] for s in ('d7.5', 'd7.6'))
        group = ('both_accepted' if a['accepted'] and b['accepted'] else 'd7.5_only' if a['accepted'] else 'd7.6_only' if b['accepted'] else 'both_rejected')
        groups[group].append({'sample_id': r['sample_id'], 'kind': r['kind'],
                              'd7.5_correct': a['correct'], 'd7.6_correct': b['correct']})
        reasons_added.update(set(b['reasons']) - set(a['reasons']))
        reasons_removed.update(set(a['reasons']) - set(b['reasons']))
        old, new = acceptance_state(r['kind'], a), acceptance_state(r['kind'], b)
        acceptance_transitions[f'{old}_to_{new}'] += 1
        assessable = old != 'unassessable' and new != 'unassessable'
        changes['new_correct_acceptances'] += bool(assessable and new == 'correct' and old != 'correct')
        changes['new_incorrect_acceptances'] += bool(assessable and new == 'incorrect' and old != 'incorrect')
        changes['lost_correct_acceptances'] += bool(assessable and old == 'correct' and new != 'correct')
        changes['corrected_incorrect_acceptances'] += bool(assessable and old == 'incorrect' and new != 'incorrect')
        if r['kind'] == 'complete':
            old_candidate, new_candidate = candidate_state(a), candidate_state(b)
            candidate_transitions[f'{old_candidate}_to_{new_candidate}'] += 1
            changes['candidate_improvements'] += bool(old_candidate in ('missing', 'incorrect') and new_candidate == 'correct')
            changes['candidate_regressions'] += bool(old_candidate == 'correct' and new_candidate in ('missing', 'incorrect'))
    timing = {}
    for key in ('shared_ms', 'inference_ms', 'local_and_placement_ms', 'd7.5_ms', 'd7.6_ms', 'overhead_ms'):
        vals = [r['timing'][key] for r in records]
        timing[key] = {'count': len(vals), 'median': float(np.median(vals)) if vals else None,
                       'p95': float(np.quantile(vals, .95)) if vals else None}
    return {'strategies': summaries, 'paired_groups': groups, 'changes': dict(changes), 'timing': timing,
            'acceptance_transitions': dict(acceptance_transitions), 'candidate_transitions': dict(candidate_transitions),
            'reasons_added': dict(reasons_added), 'reasons_removed': dict(reasons_removed)}


def evaluate(*, checkpoint, output, split='val', dataset_root=None, manifest=None, device='auto', tile=384,
             overlap=96, padding=0, seed=42, limit=None):
    if (dataset_root is None) == (manifest is None):
        raise ValueError('provide exactly one source')
    if tile <= 0 or tile % 8 or not 0 <= overlap < tile:
        raise ValueError('tile must be a positive multiple of eight and overlap smaller than tile')
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    if dataset_root:
        ds = PhaseDataset(dataset_root, split, require_nonempty=True)
        entries = [(Path(dataset_root)/split/f'{name}.png', Path(dataset_root)/split/f'{name}.json') for name in ds.names]
    else:
        path = Path(manifest)
        data = json.loads(path.read_text(encoding='utf-8'))
        entries = [(path.parent/item['image'], item) for item in data['samples'] if item['split'] == split]
    if not entries: raise ValueError('empty evaluation split')
    indices = selection(len(entries), limit, seed)
    target_device = select_device(device)
    checkpoint_path = Path(checkpoint)
    digest = hashlib.sha256()
    with checkpoint_path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): digest.update(block)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = PhaseGridUNet(); load_phase_weights(model, state); model.to(target_device).eval()
    class Predict:
        def __init__(self): self.tile, self.overlap, self.inference_ms = tile, overlap, 0.
        def __call__(self, image):
            tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255).unsqueeze(0).to(target_device)
            def sync():
                if target_device.type == 'cuda': torch.cuda.synchronize(target_device)
                elif target_device.type == 'mps': torch.mps.synchronize()
            sync(); start = perf_counter()
            with torch.inference_mode():
                result = phase_predictions(model(tensor)).cpu().numpy()[0]
            sync(); self.inference_ms += (perf_counter()-start)*1000
            if not np.isfinite(result).all(): raise FloatingPointError('nonfinite model prediction')
            return result
    records = []
    for order_index, i in enumerate(indices):
        path, meta = entries[i]
        metadata = json.loads(meta.read_text(encoding='utf-8')) if isinstance(meta, Path) else meta
        if not metadata.get('sample_id') or metadata.get('split') != split:
            raise ValueError('invalid sample identity')
        category = metadata.get('category', 'unknown')
        g = metadata.get('geometry')
        if category == 'multiple': kind = 'unassessable_multiple'
        elif g is None and category in ('negative', 'hard_negative'): kind = 'negative'
        elif g is not None and metadata.get('complete') is True: kind = 'complete'
        elif g is not None and metadata.get('complete') is False: kind = 'incomplete'
        else: kind = 'unassessable'
        if g is not None:
            if not all(math.isfinite(g[k]) for k in ('pitch_x','pitch_y','origin_x','origin_y')) or min(g['pitch_x'],g['pitch_y']) <= 0:
                raise ValueError('invalid ground truth lattice')
            if kind == 'complete' and (not metadata.get('board_dimensions') or len(g.get('visible_box', ())) != 4):
                raise ValueError('complete sample lacks physical extent or dimensions')
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB'))
        crop = crop_bounds(metadata, (rgb.shape[1], rgb.shape[0]), padding)
        x0,y0,x1,y1 = crop
        # A crop that excludes a physical board edge cannot be treated as complete.
        if kind == 'complete':
            l,t,r,b = g['visible_box']
            if not (x0 <= l and y0 <= t and r <= x1 and b <= y1): kind = 'incomplete'
        predictor = Predict()
        observations, layout, results, timing = paired_run(rgb[y0:y1, x0:x1], predictor, order_index)
        timing['inference_ms'] = predictor.inference_ms
        timing['local_and_placement_ms'] = timing['shared_ms'] - predictor.inference_ms
        # Prevent candidate evaluation against physical extent after a censoring crop.
        evaluation_metadata = dict(metadata, complete=(kind == 'complete'))
        records.append({'sample_id': metadata['sample_id'], 'kind': kind, 'category': category,
                        'image': str(path), 'crop': list(crop), 'image_size': [rgb.shape[1], rgb.shape[0]],
                        'ground_truth': {'geometry': g, 'board_dimensions': metadata.get('board_dimensions')},
                        'observation_count': len(observations), 'tile_layout': list(layout), 'timing': timing,
                        'results': {s: result_record(results[s], evaluation_metadata, crop) for s in ('d7.5', 'd7.6')}})
    summary = summarize(records)
    summary['by_tile_count'] = {name: summarize([r for r in records if (r['observation_count'] == 1) == single])
                                for name, single in (('single', True), ('multiple', False))}
    summary['by_category'] = {category: summarize([r for r in records if r['category'] == category])
                              for category in sorted({r['category'] for r in records})}
    for name, key in (('board_dimensions', lambda r: r['ground_truth']['board_dimensions']),
                      ('pitch', lambda r: ([r['ground_truth']['geometry']['pitch_x'], r['ground_truth']['geometry']['pitch_y']]
                                           if r['ground_truth']['geometry'] else None)),
                      ('crop_dimensions', lambda r: [r['crop'][2]-r['crop'][0], r['crop'][3]-r['crop'][1]])):
        grouped = {}
        for record in records:
            label = json.dumps(key(record))
            grouped.setdefault(label, []).append(record)
        summary['by_' + name] = {label: summarize(group) for label, group in grouped.items()}
    multi_complete = sum(r['kind'] == 'complete' and r['observation_count'] > 1 for r in records)
    config = {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': digest.hexdigest(), 'model_id': MODEL_ID,
              'representation_id': REPRESENTATION_ID, 'dataset_root': str(dataset_root) if dataset_root else None,
              'manifest': str(manifest) if manifest else None, 'split': split, 'device_requested': device,
              'device_resolved': str(target_device), 'tile': tile, 'overlap': overlap, 'padding': padding,
              'seed': seed, 'limit': limit, 'sample_indices': indices,
              'global_settings': {s: asdict(GlobalSettings(strategy=s)) for s in ('d7.5', 'd7.6')},
              'criteria': 'complete extent: rows and columns exact, pitch and phase errors <= 0.10 pitch, each physical edge <= 1.5 px',
              'timing': 'shared includes placement, inference and local decoding; inference includes synchronized model and CPU transfer, not input preparation; order alternates by sample'}
    report = {'version': 1, 'config': config, 'summary': summary, 'records': records,
              'limitations': {'multi_tile_complete_samples': multi_complete,
                              'note': 'Prepared phase samples are resized crops; inspect multi-tile counts before drawing conclusions. Multiple boards are unassessable.'}}
    out.mkdir(parents=True, exist_ok=False)
    (out/'paired_results.json').write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False), encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(_jsonable({'config': config, 'summary': summary, 'limitations': report['limitations']}), indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset-root'); source.add_argument('--manifest')
    p.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    p.add_argument('--device', default='auto'); p.add_argument('--tile', type=int, default=384)
    p.add_argument('--overlap', type=int, default=96); p.add_argument('--padding', type=int, default=0)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--limit', type=int)
    p.add_argument('--output', required=True)
    evaluate(**vars(p.parse_args()))


if __name__ == '__main__': main()
