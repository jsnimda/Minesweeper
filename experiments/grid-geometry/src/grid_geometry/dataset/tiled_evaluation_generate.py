"""Render tiled evaluation plans and write paired-evaluator manifests.

Smoke and development plans map to evaluator split ``val``; test maps to
``test``. Generation metadata retains the original plan and split. Identity
exclusions accept prior paired manifests, phase dataset manifest.jsonl files,
or JSON lists of board IDs. Historical provenance may be incomplete.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import re

from .tiled_evaluation_plan import PlanConfig, plan


def crop_for_board(geometry, size, intent, *, partial):
    """Return an integer half-open crop in native image coordinates.

    Complete margins are clipped to real pixels. Partial intentions exclude
    one or more sides at one-quarter of the actual physical board dimensions.
    Pixel boundaries are integral, so evaluator pixel-center selection is exact.
    """
    width, height = size
    l, t, r, b = geometry.visible_box
    if not (0 <= l < r <= width and 0 <= t < b <= height):
        raise ValueError('physical board is not fully inside the rendered image')
    margin = intent['margin_pitches']
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('invalid crop margin')
    box = [max(0, math.floor(l-margin*geometry.pitch_x)),
           max(0, math.floor(t-margin*geometry.pitch_y)),
           min(width, math.ceil(r+margin*geometry.pitch_x)),
           min(height, math.ceil(b+margin*geometry.pitch_y))]
    clipping = intent.get('clipping')
    if partial:
        # Fixed orientation makes the intention deterministic; the planner's
        # category determines which sides are censored.
        if clipping in ('one_side', 'adjacent_sides', 'opposite_sides', 'interior'):
            box[0] = math.ceil(l + .25*(r-l))
        else:
            raise ValueError('invalid partial clipping intention')
        if clipping in ('adjacent_sides', 'interior'):
            box[1] = math.ceil(t + .25*(b-t))
        if clipping in ('opposite_sides', 'interior'):
            box[2] = math.floor(r - .25*(r-l))
        if clipping == 'interior':
            box[3] = math.floor(b - .25*(b-t))
    elif clipping is not None:
        raise ValueError('complete sample has a clipping intention')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
            and x0 < r and l < x1 and y0 < b and t < y1):
        raise ValueError('invalid or nonintersecting board crop')
    contains = x0 <= l and y0 <= t and x1 >= r and y1 >= b
    if contains == partial:
        raise ValueError('crop completeness does not match the plan')
    return box


def read_excluded_ids(paths):
    """Read board IDs from JSON manifests, JSON ID lists or JSONL records."""
    result = set()
    for path in paths:
        path = Path(path)
        text = path.read_text(encoding='utf-8')
        if path.suffix == '.jsonl':
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            if isinstance(data, dict) and 'samples' not in data:
                raise ValueError(f'identity source has no samples: {path}')
            records = data['samples'] if isinstance(data, dict) else data
        if not isinstance(records, list):
            raise ValueError(f'invalid identity source: {path}')
        for record in records:
            if isinstance(record, str):
                result.add(record)
            elif isinstance(record, dict):
                if record.get('board_id') is not None:
                    result.add(record['board_id'])
            else:
                raise ValueError(f'invalid identity record in {path}')
    return result


def generate(config, output, *, exclude_identity_paths=(), renderer=None):
    """Render plans into a new directory; publish manifest only on success.

    Failed runs retain their directory for inspection but never publish a
    manifest. Existing directories are never reused. No inference is performed.
    """
    from .generator import _geometry
    if renderer is None:
        from synthetic_scenes.render import render_scene
        renderer = render_scene
    excluded = read_excluded_ids(exclude_identity_paths)
    plans = plan(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    images = output / 'images'
    images.mkdir()
    entries, provenance, seen_ids, seen_boards = [], [], set(), set()
    evaluation_split = 'test' if config.split == 'test' else 'val'
    for sample in plans:
        sample_id = sample['sample_id']
        try:
            if not re.fullmatch(r'[A-Za-z0-9_-]+', sample_id) or sample_id in seen_ids:
                raise ValueError('invalid or duplicate sample identity')
            seen_ids.add(sample_id)
            image, resolved, annotations = renderer(sample['scene'], details=('scene',), wallpaper_assets={})
            width, height = image.size
            if min(width, height) < 2 or tuple(annotations['size']) != (width, height):
                raise ValueError('invalid image dimensions or annotation size')
            boards = annotations['boards']
            negative = sample['category'] in ('negative', 'hard_negative')
            if len(boards) != (0 if negative else 1):
                raise ValueError('unexpected number of target boards')
            board_id = None
            geometry = None
            dimensions = None
            if negative:
                crop = [0, 0, width, height]
                complete = None
            else:
                board = boards[0]
                board_id = board['board_id']
                if not isinstance(board_id, str) or not board_id:
                    raise ValueError('missing rendered board identity')
                if board_id in seen_boards or board_id in excluded:
                    raise ValueError(f'duplicate or excluded board identity: {board_id}')
                seen_boards.add(board_id)
                dimensions = [board['rows'], board['columns']]
                if any(type(v) is not int or v <= 0 for v in dimensions):
                    raise ValueError('invalid rendered board dimensions')
                geometry = _geometry(board)
                left, top, right, bottom = geometry.visible_box
                gap = board['style_parameters']['gap']
                if (not math.isfinite(gap) or gap < 0
                        or not math.isclose(right-left, dimensions[1]*geometry.pitch_x-gap, rel_tol=1e-7, abs_tol=1e-5)
                        or not math.isclose(bottom-top, dimensions[0]*geometry.pitch_y-gap, rel_tol=1e-7, abs_tol=1e-5)):
                    raise ValueError('rendered board dimensions disagree with pitch and grid box')
                complete = sample['category'] != 'partial'
                crop = crop_for_board(geometry, (width, height), sample['crop_intent'], partial=not complete)
            image_path = f'images/{sample_id}.png'
            image.save(output / image_path, format='PNG')
            entries.append({'sample_id': sample_id, 'split': evaluation_split, 'image': image_path,
                            'category': sample['category'], 'complete': complete,
                            'geometry': asdict(geometry) if geometry else None,
                            'board_dimensions': dimensions, 'board_id': board_id,
                            'image_size': [width, height], 'known_box': [0, 0, width, height], 'crop_box': crop})
            provenance.append({'sample_id': sample_id, 'plan': sample, 'resolved_scene': resolved,
                               'board_id': board_id})
        except Exception as exc:
            raise RuntimeError(f'failed to generate {sample_id}: {exc}') from exc
    metadata = {'version': 1, 'planner_config': asdict(config), 'planning_split': config.split,
                'evaluator_split': evaluation_split, 'excluded_identity_sources': [str(p) for p in exclude_identity_paths],
                'samples': provenance}
    (output / 'generation.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    # Publish only a completely written manifest; a failed write leaves the
    # temporary name rather than a completed-looking manifest.
    serialized = json.dumps({'samples': entries}, indent=2, allow_nan=False)
    temporary = output / 'manifest.json.tmp'
    temporary.write_text(serialized, encoding='utf-8')
    temporary.replace(output / 'manifest.json')
    return output / 'manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('smoke', 'development', 'test'), required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--counts', help='JSON object with all five planner category counts')
    parser.add_argument('--output', required=True)
    parser.add_argument('--exclude-identities', action='append', default=[], metavar='PATH')
    args = parser.parse_args()
    counts = json.loads(args.counts) if args.counts else None
    generate(PlanConfig(args.split, args.seed, counts), args.output, exclude_identity_paths=args.exclude_identities)


if __name__ == '__main__':
    main()
