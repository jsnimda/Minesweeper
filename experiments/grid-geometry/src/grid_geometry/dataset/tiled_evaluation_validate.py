"""Validate generated tiled evaluation datasets without model inference."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from PIL import Image

from .targets import PhaseGeometry
from .tiled_evaluation_generate import read_excluded_ids
from .tiled_evaluation_plan import CATEGORIES, DEFAULT_COUNTS, PlanConfig, plan
from ..phase.paired_evaluate import crop_bounds
from ..phase.tiles import tile_starts


def validate(root, *, tile=384, overlap=96, padding=0, identity_sources=()):
    """Return errors, warnings and coverage statistics; do not modify the dataset."""
    tile_starts(1, tile, overlap)
    root = Path(root)
    report = {'errors': [], 'warnings': [], 'samples': [], 'identity_sources': []}
    errors, warnings = report['errors'], report['warnings']
    warnings.append('Historical training identities are not guaranteed to be available.')
    try:
        manifest = json.loads((root/'manifest.json').read_text())
        metadata = json.loads((root/'generation.json').read_text())
        entries = manifest['samples']
        config = PlanConfig(**metadata['planner_config'])
        expected = plan(config)
        provenance = metadata['samples']
        if not isinstance(entries, list) or not isinstance(provenance, list): raise ValueError('samples must be lists')
        if metadata['planning_split'] != config.split or metadata['evaluator_split'] != ('test' if config.split == 'test' else 'val'):
            raise ValueError('inconsistent split metadata')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f'dataset metadata: {exc}')
        report['status'] = 'invalid'
        return report
    counts = Counter(e.get('category') for e in entries)
    expected_counts = Counter(p['category'] for p in expected)
    if counts != expected_counts: errors.append('category counts differ from planner configuration')
    expected_by_id = {p['sample_id']: p for p in expected}
    prov_by_id = {p['sample_id']: p for p in provenance}
    if len(prov_by_id) != len(provenance) or set(prov_by_id) != set(expected_by_id):
        errors.append('generation metadata identities differ from plan')
    ids, boards = set(), {}
    margins = {side: [] for side in ('left', 'top', 'right', 'bottom')}
    sizes, dimensions, pitches, tile_counts = [], [], [], []
    for e in entries:
        sample_id = e.get('sample_id', '<missing>')
        try:
            if not isinstance(sample_id, str) or not sample_id or sample_id in ids: raise ValueError('missing or duplicate sample ID')
            ids.add(sample_id)
            planned = expected_by_id[sample_id]
            prov = prov_by_id[sample_id]
            if prov['plan'] != planned or prov['board_id'] != e['board_id'] or planned['category'] != e['category']:
                raise ValueError('manifest and generation metadata disagree')
            if e['split'] != metadata['evaluator_split'] or e['category'] not in CATEGORIES:
                raise ValueError('invalid split or category')
            image_path = (root/e['image']).resolve()
            if not image_path.is_relative_to(root.resolve()): raise ValueError('image path escapes dataset')
            with Image.open(image_path) as im:
                im.verify()
            with Image.open(image_path) as im:
                size = im.size
            if list(size) != e['image_size'] or min(size) < 2: raise ValueError('incorrect image dimensions')
            crop = crop_bounds(e, size, padding)
            x0,y0,x1,y1 = crop
            xs, ys = tile_starts(x1-x0, tile, overlap), tile_starts(y1-y0, tile, overlap)
            tile_count = len(xs)*len(ys)
            sizes.append(size); tile_counts.append(tile_count)
            item = {'sample_id': sample_id, 'category': e['category'], 'crop': crop, 'tile_count': tile_count}
            negative = e['category'] in ('negative', 'hard_negative')
            if negative:
                if e['geometry'] is not None or e['board_dimensions'] is not None or e['board_id'] is not None or e['complete'] is not None:
                    raise ValueError('invalid negative annotation')
            else:
                g = PhaseGeometry(**e['geometry'])
                rows, cols = e['board_dimensions']
                if type(rows) is not int or type(cols) is not int or min(rows, cols) <= 0: raise ValueError('invalid board dimensions')
                board_id = e['board_id']
                if not isinstance(board_id, str) or not board_id or board_id in boards: raise ValueError('missing or duplicate board identity')
                boards[board_id] = sample_id
                l,t,r,b = g.visible_box
                if not (0 <= l < r <= size[0] and 0 <= t < b <= size[1]): raise ValueError('board outside image')
                gap_x, gap_y = 2*(l-g.origin_x), 2*(t-g.origin_y)
                if (min(gap_x, gap_y) < -1e-5 or not math.isclose(gap_x, gap_y, abs_tol=1e-5)
                        or not math.isclose(r-l, cols*g.pitch_x-gap_x, rel_tol=1e-7, abs_tol=1e-5)
                        or not math.isclose(b-t, rows*g.pitch_y-gap_y, rel_tol=1e-7, abs_tol=1e-5)):
                    raise ValueError('board extent and lattice dimensions disagree')
                if not (x0 < r and l < x1 and y0 < b and t < y1): raise ValueError('crop does not intersect board')
                excluded = [side for side, flag in zip(('left','top','right','bottom'), (x0>l, y0>t, x1<r, y1<b)) if flag]
                if type(e['complete']) is not bool or e['complete'] != (not excluded) or e['complete'] != (e['category'] != 'partial'):
                    raise ValueError('incorrect completeness label')
                item['excluded_sides'] = excluded
                dimensions.append((rows, cols)); pitches.append((g.pitch_x, g.pitch_y))
                intersects = sum(x0+x < r and x0+min(x+tile, x1-x0) > l and y0+y < b and y0+min(y+tile, y1-y0) > t for y in ys for x in xs)
                item['board_intersecting_tiles'] = intersects
                if e['category'] == 'large_complete' and (intersects < 2 or (r-l <= tile and b-t <= tile)):
                    raise ValueError('large board does not require multiple tiles')
                if e['complete']:
                    vals = (l-x0, t-y0, x1-r, y1-b)
                    item['exterior_margins'] = {side: {'pixels': v, 'cells': v/p} for side,v,p in zip(margins, vals, (g.pitch_x,g.pitch_y,g.pitch_x,g.pitch_y))}
                    for side, v in item['exterior_margins'].items(): margins[side].append(v)
            report['samples'].append(item)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append(f'{sample_id}: {exc}')
    if ids != set(expected_by_id): errors.append('manifest identities differ from plan')
    for source in identity_sources:
        try:
            found = read_excluded_ids([source])
            overlap_ids = sorted(set(boards) & found)
            report['identity_sources'].append({'path': str(source), 'identities': len(found), 'overlap': len(overlap_ids),
                                               'sample_ids': [boards[x] for x in overlap_ids]})
            if not found: warnings.append(f'identity source has no usable IDs: {source}')
            if overlap_ids: errors.append(f'identity overlap with {source}')
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f'identity source unavailable: {source}: {exc}')
    def stats(values):
        return {'min': min(values), 'median': statistics.median(values), 'max': max(values)} if values else None
    report.update(status='invalid' if errors else 'valid', category_counts=dict(counts),
                  image_width=stats([x for x,y in sizes]), image_height=stats([y for x,y in sizes]),
                  rows=stats([x for x,y in dimensions]), columns=stats([y for x,y in dimensions]),
                  pitch_x=stats([x for x,y in pitches]), pitch_y=stats([y for x,y in pitches]),
                  tile_counts=stats(tile_counts), single_tile=sum(n == 1 for n in tile_counts),
                  multi_tile=sum(n > 1 for n in tile_counts),
                  tile_distribution_by_category={cat: dict(Counter(str(s['tile_count']) for s in report['samples'] if s['category'] == cat)) for cat in CATEGORIES},
                  exterior_margins={side: {unit: stats([v[unit] for v in values]) for unit in ('pixels', 'cells')} for side,values in margins.items()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset'); parser.add_argument('--tile', type=int, default=384)
    parser.add_argument('--overlap', type=int, default=96); parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--identity-source', action='append', default=[]); parser.add_argument('--report')
    args = parser.parse_args()
    report = validate(args.dataset, tile=args.tile, overlap=args.overlap, padding=args.padding, identity_sources=args.identity_source)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.report:
        with Path(args.report).open('x', encoding='utf-8') as handle: handle.write(text + '\n')
    print(text)
    raise SystemExit(1 if report['errors'] else 0)


if __name__ == '__main__': main()
