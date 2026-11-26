"""Generate compact scale-controlled samples using synthetic-scenes render_scene.

Run with synthetic-scenes available in the generation environment. Rendering
imports are lazy; loading exported samples does not depend on synthetic-scenes.
"""
from collections import Counter, deque
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import multiprocessing
from pathlib import Path
import random
import tempfile
from time import perf_counter

import numpy as np
from PIL import Image

from .targets import PhaseGeometry


from .sampling_config import CATEGORIES, GenerationConfig, SOURCE_CELL_SIZE, STYLES, THEMES
from . import sampler as source_sampler
from .sampler import _choice, _game, _desktop, _scene
from .statistics import AcceptedStatistics
from synthetic_scenes.attempt_statistics import AttemptCounters, NOT_APPLICABLE, board_factors, fixed_bucket, summaries
from synthetic_scenes.dataset_statistics import BINS

GENERATOR_REVISION = 1
STANDALONE_MINIMUM_CELL_SIZE = 12
NEGATIVE_KINDS = ('negative', 'hard_negative')
RESAMPLING_METHODS = (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)
BLUR_CHOICES = (0., 0., 0., .25, .5)
PHOTOMETRIC_RANGE = (.95, 1.05)
PADDING_TYPES = ('black', 'dark', 'light', 'random')
PADDING_WEIGHTS = (5, 2, 2, 1)
FILL_APPLICATIONS = ('browser', 'spreadsheet', 'sudoku', 'other_puzzle')
FILL_DISPLAY_SCALE = 1.0


def _crop_axis(rng, low, high, span, mode, edge=None, source_length=None, prefer_source=False):
    if mode == 'complete':
        lower, upper = high - span + 5, low - 5
    elif mode == 'inside':
        lower, upper = low + 1, high - span - 1
    elif mode == 'edge' and edge == 'low':
        lower, upper = low - .30 * span, low - .08 * span
    elif mode == 'edge' and edge == 'high':
        lower, upper = high - .92 * span, high - .70 * span
    else:
        raise ValueError('unknown crop axis mode')
    if lower > upper:
        raise ValueError('infeasible crop interval')
    if prefer_source and source_length is not None:
        # Use the intersection when possible; otherwise choose the legal point
        # closest to a crop fully contained in the rendered source.
        source_low, source_high = sorted((0., source_length - span))
        overlap_low, overlap_high = max(lower, source_low), min(upper, source_high)
        if overlap_low <= overlap_high:
            lower, upper = overlap_low, overlap_high
        elif upper < source_low:
            lower = upper
        else:
            upper = lower
    return rng.uniform(lower, upper)


def _crop_box(rng, box, spans, category, source_size=None, prefer_source=False):
    left, top, right, bottom = box
    sw, sh = spans
    if category == 'complete':
        modes = ('complete', 'complete')
    elif category == 'interior':
        modes = ('inside', 'inside')
    elif category == 'side':
        modes = ('edge', 'inside') if rng.random() < .5 else ('inside', 'edge')
    else:
        modes = ('edge', 'edge')
    source_width, source_height = source_size if source_size else (None, None)
    x = _crop_axis(rng, left, right, sw, modes[0], rng.choice(('low', 'high')), source_width, prefer_source)
    y = _crop_axis(rng, top, bottom, sh, modes[1], rng.choice(('low', 'high')), source_height, prefer_source)
    return (x, y, x + sw, y + sh)


def _geometry(board):
    left, top, right, bottom = board['boxes']['grid']
    parameters = board['style_parameters']
    pitch = parameters['pitch']
    gap = parameters['gap']
    # Renderer internal boundaries bisect gaps; extend that ideal lattice to
    # the first and last slots instead of using clipped visible outer edges.
    return PhaseGeometry(left - gap / 2, top - gap / 2, pitch, pitch, (left, top, right, bottom))


def _fit_standalone_game(game, budget):
    """Fit native game content by reducing only its sampled base cell size."""
    from synthetic_scenes.minesweeper.geometry import measure_game_region

    appearance = game['appearance']
    requested = appearance['cell_size']
    for cell_size in range(requested, min(STANDALONE_MINIMUM_CELL_SIZE, requested) - 1, -1):
        try:
            width, height = measure_game_region(
                game['rows'], game['columns'], appearance['style'], cell_size, appearance['zoom']
            )
        except OverflowError:
            raise ValueError('standalone source exceeds image budget') from None
        if max(width, height) <= budget['max_dimension'] and width * height <= budget['max_pixels']:
            appearance['cell_size'] = cell_size
            return
    raise ValueError('standalone source exceeds image budget at minimum cell size')


def _pitch_bin(pitch, bins):
    for low, high, _ in bins:
        if low <= pitch <= high:
            return f'{low:g}-{high:g}'
    return 'outside'


def sample_fractions(size, known_box, geometry, transition_ratio):
    """Exact pixel-center coverage and soft-occupancy sums using 1-D arrays.

    Fractions use all image pixels as denominator. Padding is unknown, and
    occupancy outside known pixels contributes zero to the reported sum.
    """
    width, height = size
    x = np.arange(width, dtype=np.float32) + .5
    y = np.arange(height, dtype=np.float32) + .5
    left, top, right, bottom = known_box
    x = x[(x >= left) & (x < right)]
    y = y[(y >= top) & (y < bottom)]
    total = width * height
    padding = 1. - (x.size * y.size) / total
    if geometry is None:
        return {'phase_supervision_fraction': None, 'grid_coverage_fraction': None,
                'grid_occupancy_fraction': 0., 'padding_fraction': padding}
    left, top, right, bottom = geometry.visible_box
    dx = np.minimum(x - left, right - x) / geometry.pitch_x
    dy = np.minimum(y - top, bottom - y) / geometry.pitch_y
    inside = int(np.count_nonzero(dx >= 0)) * int(np.count_nonzero(dy >= 0))
    # clip(min(dx,dy)) = min(clip(dx),clip(dy)); sorted prefix sums
    # evaluate the full 2-D occupancy sum without allocating a dense field.
    fx = np.sort(np.clip(.5 + dx / transition_ratio, 0., 1.).astype(np.float64))
    fy = np.clip(.5 + dy / transition_ratio, 0., 1.).astype(np.float64)
    prefix = np.concatenate(([0.], np.cumsum(fx)))
    positions = np.searchsorted(fx, fy, side='right')
    occupancy_sum = np.sum(prefix[positions] + (fx.size - positions) * fy)
    coverage = inside / total
    return {'phase_supervision_fraction': coverage, 'grid_coverage_fraction': coverage,
            'grid_occupancy_fraction': float(occupancy_sum / total), 'padding_fraction': padding}


def summarize_fractions(records):
    """Summarize compact per-sample fractions, including empty groups."""
    def summarize(group):
        positives = [r for r in group if r['grid_coverage_fraction'] is not None]
        result = {'samples': len(group), 'positive_samples': len(positives),
                  'samples_with_padding': sum(r['padding_fraction'] > 0 for r in group),
                  'fraction_with_padding': sum(r['padding_fraction'] > 0 for r in group) / len(group) if group else None}
        for name in ('grid_coverage_fraction', 'phase_supervision_fraction', 'grid_occupancy_fraction', 'padding_fraction'):
            values = [r[name] for r in (group if name == 'padding_fraction' else positives)]
            result[name + '_quantiles'] = dict(zip(('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'),
                (float(v) for v in np.quantile(values, [0, .05, .25, .5, .75, .95, 1])))) if values else None
        result['positive_fraction_below_coverage'] = {
            str(threshold): sum(r['grid_coverage_fraction'] < threshold for r in positives) / len(positives) if positives else None
            for threshold in (.05, .1, .25)}
        return result
    result = {'overall': summarize(records), 'by_category': {}, 'by_pitch_bin': {}}
    for key, output in (('category', 'by_category'), ('actual_pitch_bin_x', 'by_pitch_bin')):
        for value in sorted({r[key] for r in records if r[key] is not None}):
            result[output][value] = summarize([r for r in records if r[key] == value])
    return result


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _source_policy(config):
    """Describe source-defined choices alongside the configurable dataclass."""
    from synthetic_scenes.wallpaper import PROCEDURAL_SUBTYPE_WEIGHTS
    return {
        'positive_fraction': 1 - config.negative_fraction,
        'negative_kinds_uniform': NEGATIVE_KINDS,
        'source_cell_size': SOURCE_CELL_SIZE,
        'large_board_cell_threshold': source_sampler.LARGE_BOARD_CELL_THRESHOLD,
        'desktop_board_area_limit': source_sampler.DESKTOP_BOARD_AREA_LIMIT,
        'source_desktop_size': source_sampler.SOURCE_DESKTOP_SIZE,
        'standalone_minimum_cell_size': STANDALONE_MINIMUM_CELL_SIZE,
        'game_styles_uniform': STYLES, 'themes_uniform': THEMES,
        'complete_board_candidates': source_sampler.COMPLETE_BOARD_CANDIDATES,
        'board_dimension_range': source_sampler.BOARD_DIMENSION_RANGE,
        'complete_margin_pixels': source_sampler.COMPLETE_MARGIN_PIXELS,
        'complete_random_candidates': source_sampler.COMPLETE_RANDOM_CANDIDATES,
        'complete_random_weight': source_sampler.COMPLETE_RANDOM_WEIGHT,
        'partial_standard_boards': source_sampler.PARTIAL_STANDARD_BOARDS,
        'partial_standard_probability': source_sampler.PARTIAL_STANDARD_PROBABILITY,
        'partial_extra_cells': source_sampler.PARTIAL_EXTRA_CELLS,
        'partial_row_extension': source_sampler.PARTIAL_ROW_EXTENSION,
        'partial_column_extension': source_sampler.PARTIAL_COLUMN_EXTENSION,
        'board_selection_algorithm': 'grid_geometry.dataset.sampler._game',
        'mine_density': source_sampler.MINE_DENSITY, 'game_progress_weights': source_sampler.GAME_PROGRESS,
        'progress_ranges': source_sampler.PROGRESS_RANGES, 'finished_statuses_uniform': source_sampler.FINISHED_STATUSES,
        'lost_progress_range': source_sampler.LOST_PROGRESS_RANGE, 'elapsed_seconds': source_sampler.ELAPSED_SECONDS,
        'font_families_uniform': source_sampler.FONT_FAMILIES, 'font_scale': source_sampler.FONT_SCALE,
        'bold_probability': source_sampler.BOLD_PROBABILITY, 'google_boundary_choices': source_sampler.GOOGLE_BOUNDARIES,
        'windows7_mine_choices': source_sampler.WINDOWS7_MINES,
        'wallpaper': {'category': 'procedural', 'subtype_weights': PROCEDURAL_SUBTYPE_WEIGHTS},
        'source_application_probability': source_sampler.SOURCE_APPLICATION_PROBABILITY,
        'source_applications_uniform': source_sampler.SOURCE_APPLICATIONS,
        'negative_applications_uniform': source_sampler.NEGATIVE_APPLICATIONS,
        'hard_negative_applications_uniform': source_sampler.HARD_NEGATIVE_APPLICATIONS,
        'fill': {'desktop_appearance': config.desktop_appearance, 'display_scale': FILL_DISPLAY_SCALE,
                 'desktop_size': 'output size with renderer minimum 320x240', 'applications_uniform': FILL_APPLICATIONS},
        'padding_type_choices': PADDING_TYPES, 'padding_type_weights': PADDING_WEIGHTS,
        'augmentation': {'resampling_uniform': tuple(method.name for method in RESAMPLING_METHODS),
                         'blur_choices': BLUR_CHOICES, 'brightness_range': PHOTOMETRIC_RANGE,
                         'contrast_range': PHOTOMETRIC_RANGE, 'jpeg': None},
    }


def _desktop_metadata(request, resolved):
    """Compact source/fill desktop information from the actual render."""
    wallpaper = resolved.get('resolved_wallpaper', {})
    asset = wallpaper.get('asset')
    return {'screen': request['screen'], 'appearance': request['desktop'],
            'applications': [window['type'] for window in request['windows'] if window['type'] != 'minesweeper'],
            'wallpaper': {'category': wallpaper.get('category'), 'subtype': wallpaper.get('subtype'),
                          'asset': Path(asset).name if asset else None, 'placement': wallpaper.get('placement')}}


def _diagnostic_call(function, *args, **kwargs):
    """Keep diagnostic ValueErrors out of the existing retryable source path."""
    try:
        return function(*args, **kwargs)
    except ValueError as exc:
        raise RuntimeError('attempt diagnostics failed') from exc


def _generate_one(task):
    """Spawn-safe worker: render, crop and write locally; return compact metadata."""
    from pydantic import ValidationError
    from synthetic_scenes.render import PlacementError, render_scene

    config = task['config']
    split, index = task['split'], task['index']
    sample_id = f'{index:06d}'
    seed = config['seed']
    selection = random.Random(f'{seed}:{split}:{index}:selection')
    negative = selection.random() < config['negative_fraction']
    category = _choice(selection, config['categories']) if not negative else selection.choice(NEGATIVE_KINDS)
    bin_index = selection.choices(range(len(config['pitch_bins'])), weights=[b[2] for b in config['pitch_bins']], k=1)[0]
    low, high, _ = config['pitch_bins'][bin_index]
    desired_pitch = selection.uniform(low, high)
    anisotropy_selected = selection.random() < config['anisotropic_fraction']
    aspect = selection.uniform(1 - config['anisotropic_max'], 1 + config['anisotropic_max']) if anisotropy_selected else 1.
    excluded = set(json.loads(Path(task['excluded_path']).read_text(encoding='utf-8')))
    error = None
    retries = Counter()
    attempts = AttemptCounters()
    objective = {'polarity': 'negative' if negative else 'positive',
                 'objective_type': category if negative else 'positive',
                 'negative_kind': category if negative else NOT_APPLICABLE,
                 'category': NOT_APPLICABLE if negative else category,
                 'pitch_bin': NOT_APPLICABLE if negative else f'{low:g}-{high:g}',
                 'desired_pitch': NOT_APPLICABLE if negative else desired_pitch,
                 'anisotropy_selected': NOT_APPLICABLE if negative else anisotropy_selected,
                 'anisotropy_ratio': NOT_APPLICABLE if negative else aspect}
    for attempt in range(config['max_attempts']):
        rng = random.Random(f'{seed}:{split}:{index}:{attempt}:scene')
        fields = {'objective_type': objective['objective_type'], 'objective_polarity': objective['polarity'],
                  'crop_category': objective['category'],
                  'pitch_bin': objective['pitch_bin'], 'source_mode': None, 'display_scale': None,
                  'shell': None, 'desktop_style': None, 'color_scheme': None,
                  'screen_width': None, 'screen_height': None, 'game_zoom': None,
                  'board_area': None, 'sampled_cell_size': None, 'game_style': None}
        attempted_boards = []
        fitting = {}
        stage = 'sampling'
        try:
            game, progress_kind = (None, None) if negative else _game(rng, category, desired_pitch, config['input_size'],
                coverage_exploration_fraction=config['coverage_exploration_fraction'],
                coverage_preference_exponent=config['coverage_preference_exponent'], game_zoom=config['game_zoom'])
            sampled_cell_size = game['appearance']['cell_size'] if game else None
            if game:
                factors = _diagnostic_call(board_factors, game)
                factors['effective_native_scale'] = None
                attempted_boards = [factors]
                for name in ('game_zoom', 'board_area', 'sampled_cell_size', 'game_style'):
                    fields[name] = factors[name]
            else:
                for name in ('game_zoom', 'board_area', 'sampled_cell_size', 'game_style'):
                    fields[name] = NOT_APPLICABLE
            request = _scene(rng, game, category if negative else None, config['desktop_fraction'],
                             display_scale=config['display_scale'], desktop_appearance=config['desktop_appearance'])
            fields['source_mode'] = request['mode']
            if request['mode'] == 'desktop':
                screen, desktop = request['screen'], request['desktop']
                fields.update(display_scale=screen.get('scale', 1.0), shell=desktop['shell'],
                              desktop_style=desktop['style'], color_scheme=desktop['color_scheme'],
                              screen_width=_diagnostic_call(fixed_bucket, screen['width'], BINS['dimension']),
                              screen_height=_diagnostic_call(fixed_bucket, screen['height'], BINS['dimension']))
            else:
                for name in ('display_scale', 'shell', 'desktop_style', 'color_scheme', 'screen_width', 'screen_height'):
                    fields[name] = NOT_APPLICABLE
            if game:
                scale = request['screen'].get('scale', 1.0) if request['mode'] == 'desktop' else 1.0
                zoom = game['appearance'].get('zoom')
                factors['effective_native_scale'] = zoom * scale if zoom is not None else None
            if request['mode'] == 'game' and config['standalone_source_budget'] is not None:
                stage = 'standalone_fitting'
                _fit_standalone_game(game, config['standalone_source_budget'])
                fitting['standalone_budget_fitted'] = sampled_cell_size > game['appearance']['cell_size']
            stage = 'render'
            image, resolved, annotations = render_scene(request, details=('scene',), wallpaper_assets={})
            boards = annotations['boards']
            if game and request['mode'] == 'desktop' and boards:
                fitting['desktop_placement_fitted'] = boards[0]['requested_cell_size'] > boards[0]['resolved_cell_size']
            if negative:
                if boards:
                    raise ValueError('negative scene unexpectedly contains a grid')
                # Center on the application window to make hard negatives useful.
                layout = resolved['layout'][0]
                px, py = layout['position']
                ww, wh = layout['window_size']
                crop_w, crop_h = config['input_size']
                x = px + ww / 2 - crop_w / 2 + rng.uniform(-30, 30)
                y = py + wh / 2 - crop_h / 2 + rng.uniform(-30, 30)
                crop = (x, y, x + crop_w, y + crop_h)
                source_geometry = target_geometry = None
                board = None
            else:
                if len(boards) != 1:
                    raise ValueError('expected exactly one board')
                board = boards[0]
                if board['board_id'] in excluded:
                    retries['identity_overlap'] += 1
                    _diagnostic_call(attempts.observe, fields, attempted_boards, 'rejected', 'identity_overlap', fitting)
                    continue
                stage = 'crop'
                source_geometry = _geometry(board)
                sx = desired_pitch / source_geometry.pitch_x
                sy = sx * aspect
                spans = (config['input_size'][0] / sx, config['input_size'][1] / sy)
                left, top, right, bottom = source_geometry.visible_box
                if category == 'complete':
                    feasible = spans[0] >= right - left + 10 and spans[1] >= bottom - top + 10
                else:
                    feasible = right - left > spans[0] + 2 and bottom - top > spans[1] + 2
                if not feasible:
                    raise ValueError('requested category cannot fit pitch and board')
                crop = _crop_box(rng, source_geometry.visible_box, spans, category,
                    annotations['size'], True)
                target_geometry = source_geometry.transformed(sx, sy, -crop[0] * sx, -crop[1] * sy)
                side_count = sum((crop[0] < left < crop[2], crop[0] < right < crop[2],
                                  crop[1] < top < crop[3], crop[1] < bottom < crop[3]))
                if side_count != {'complete': 4, 'interior': 0, 'side': 1, 'corner': 2}[category]:
                    raise ValueError('actual crop category differs from requested category')
            _diagnostic_call(attempts.observe, fields, attempted_boards, 'accepted', fitting=fitting)
            break
        except ValidationError:
            raise
        except (PlacementError, ValueError) as exc:
            error = exc
            retries['sampling_or_placement'] += 1
            if stage == 'standalone_fitting': fitting['standalone_fitting_stage_rejected'] = True
            elif stage == 'crop': fitting['crop_stage_rejected'] = True
            elif stage == 'render' and fields['source_mode'] == 'desktop' and isinstance(exc, PlacementError):
                fitting['desktop_render_placement_rejected'] = True
            attempts.observe(fields, attempted_boards, 'rejected', 'sampling_or_placement', fitting)
    else:
        raise RuntimeError(f'could not generate {split}/{sample_id} after {config["max_attempts"]} attempts: {error}') from error

    attempts.finish_sample(attempt + 1, objective)
    source_size = image.size
    source_desktop = _desktop_metadata(request, resolved) if request['mode'] == 'desktop' else None
    width, height = config['input_size']
    sx = width / (crop[2] - crop[0]); sy = height / (crop[3] - crop[1])
    augmentation_rng = random.Random(f'{seed}:{split}:{index}:{attempt}:augmentation')
    resampling = augmentation_rng.choice(RESAMPLING_METHODS)
    # Render only the output-sized surroundings. The transformed source is
    # overlaid with transparency outside its extent, retaining fractional alignment.
    context_rng = random.Random(f'{seed}:{split}:{index}:{attempt}:context')
    requested_context = _choice(context_rng, config['context_weights'])
    context_type = requested_context
    context_subtype = None
    fill_desktop = None
    layer = image.convert('RGBA').transform((width, height), Image.Transform.EXTENT, crop,
                                             resample=resampling, fillcolor=(0, 0, 0, 0))
    background_visible = layer.getchannel('A').getextrema()[0] < 255
    if background_visible:
        if context_type == 'procedural':
            from synthetic_scenes.wallpaper import render_procedural_wallpaper, sample_procedural_subtype
            context_subtype = sample_procedural_subtype(context_rng)
            background = render_procedural_wallpaper((width, height), context_rng, context_subtype).convert('RGBA')
        else:
            context_spec = {'mode': 'desktop', 'seed': context_rng.getrandbits(64),
                'layout_seed': context_rng.getrandbits(64), 'screen': {'width': max(320, width), 'height': max(240, height), 'scale': FILL_DISPLAY_SCALE},
                'desktop': _desktop(context_rng, config['desktop_appearance']), 'wallpaper': {'category': 'procedural', 'seed': context_rng.getrandbits(64)},
                'windows': [{'type': context_rng.choice(FILL_APPLICATIONS),
                             'seed': context_rng.getrandbits(64)}]}
            background, fill_resolved, _ = render_scene(context_spec, details=('scene',), wallpaper_assets={})
            fill_desktop = _desktop_metadata(context_spec, fill_resolved)
            background = background.crop((0, 0, width, height)).convert('RGBA')
        image = Image.alpha_composite(background, layer).convert('RGB')
    else:
        image = layer.convert('RGB')
    blur = augmentation_rng.choice(BLUR_CHOICES)
    brightness = augmentation_rng.uniform(*PHOTOMETRIC_RANGE)
    contrast = augmentation_rng.uniform(*PHOTOMETRIC_RANGE)
    from synthetic_scenes.augmentation import adjust_image
    image = adjust_image(image, blur=blur, brightness=brightness, contrast=contrast)
    # Rendered surroundings are known background. Intentional padding is
    # applied after all image transforms and excluded by the existing mask.
    padding_rng = random.Random(f'{seed}:{split}:{index}:{attempt}:padding')
    known_box = (0, 0, width, height)
    padding_type = None
    padding_side = None
    padding_color = None
    padding_extent = 0
    if padding_rng.random() < config['padding_example_fraction']:
        padding_side = padding_rng.choice(('left', 'right', 'top', 'bottom'))
        padding_type = padding_rng.choices(PADDING_TYPES, PADDING_WEIGHTS)[0]
        color = {'black': (0, 0, 0), 'dark': (36, 36, 36), 'light': (220, 220, 220),
                 'random': tuple(padding_rng.randrange(256) for _ in range(3))}[padding_type]
        padding_color = color
        from PIL import ImageDraw
        if padding_side in ('left', 'right'):
            amount = padding_rng.randint(1, max(1, int(width * config['padding_max_fraction'])))
            rect = (0, 0, amount, height) if padding_side == 'left' else (width - amount, 0, width, height)
            known_box = (amount, 0, width, height) if padding_side == 'left' else (0, 0, width - amount, height)
        else:
            amount = padding_rng.randint(1, max(1, int(height * config['padding_max_fraction'])))
            rect = (0, 0, width, amount) if padding_side == 'top' else (0, height - amount, width, height)
            known_box = (0, amount, width, height) if padding_side == 'top' else (0, 0, width, height - amount)
        padding_extent = amount
        ImageDraw.Draw(image).rectangle((rect[0], rect[1], rect[2] - 1, rect[3] - 1), fill=color)
    geometry_dict = asdict(target_geometry) if target_geometry is not None else None
    requested_bin = f'{low:g}-{high:g}'
    if board:
        box = target_geometry.visible_box
        truncated_sides = {'left': box[0] < 0, 'top': box[1] < 0,
                           'right': box[2] > width, 'bottom': box[3] > height}
        original_ideal_end = [source_geometry.origin_x + board['columns'] * source_geometry.pitch_x,
                              source_geometry.origin_y + board['rows'] * source_geometry.pitch_y]
        transformed_ideal_end = [target_geometry.origin_x + board['columns'] * target_geometry.pitch_x,
                                 target_geometry.origin_y + board['rows'] * target_geometry.pitch_y]
    else:
        truncated_sides = original_ideal_end = transformed_ideal_end = None
    fractions = sample_fractions((width, height), known_box, target_geometry, config['transition_ratio'])
    metadata = {**fractions, 'sample_id': sample_id, 'index': index, 'split': split, 'seed': seed, 'attempt': attempt,
                'scene_seed': request['seed'], 'board_id': board['board_id'] if board else None,
                'game_state_id': board['game_state_id'] if board else None,
                'style': board['style'] if board else None, 'appearance': board['style_parameters'] if board else None,
                'theme': board['theme'] if board else None, 'game_progress_category': progress_kind,
                'game_specification': game,
                'display_scale': request['screen']['scale'] if request['mode'] == 'desktop' else 1.0,
                'sampled_cell_size': sampled_cell_size,
                'requested_cell_size': board['requested_cell_size'] if board else None,
                'resolved_cell_size': board['resolved_cell_size'] if board else None,
                'requested_zoom': board['requested_zoom'] if board else None,
                'requested_display_scale': board['requested_display_scale'] if board else None,
                'requested_effective_scale': board['requested_effective_scale'] if board else None,
                'board_dimensions': [board['rows'], board['columns']] if board else None,
                'original_geometry': asdict(source_geometry) if source_geometry else None,
                'geometry': geometry_dict, 'original_ideal_end': original_ideal_end,
                'transformed_ideal_end': transformed_ideal_end, 'crop': crop,
                'transform': {'scale_x': sx, 'scale_y': sy, 'offset_x': -crop[0] * sx, 'offset_y': -crop[1] * sy},
                'known_box': known_box, 'requested_category': category, 'category': category,
                'anisotropy_ratio': aspect if board else None, 'anisotropy_selected': anisotropy_selected if board else None,
                'source_size': source_size, 'final_size': (width, height),
                'source_desktop': source_desktop, 'fill_desktop': fill_desktop,
                'requested_pitch_bin': requested_bin if board else None,
                'requested_pitch': desired_pitch if board else None,
                'effective_pitch_x': target_geometry.pitch_x if board else None,
                'effective_pitch_y': target_geometry.pitch_y if board else None,
                'actual_pitch_bin_x': _pitch_bin(target_geometry.pitch_x, config['pitch_bins']) if board else None,
                'actual_pitch_bin_y': _pitch_bin(target_geometry.pitch_y, config['pitch_bins']) if board else None,
                'complete': category == 'complete' if board else None,
                'visibility': 'unoccluded' if board else 'no_grid', 'truncated_sides': truncated_sides,
                'context_type': context_type if background_visible else ('source_desktop' if request['mode'] == 'desktop' else 'source_game'),
                'requested_context': requested_context,
                'background_visible': background_visible, 'context_subtype': context_subtype,
                'intentional_padding_requested': padding_type is not None,
                'accidental_padding_fraction': 0.,
                'padding_type': padding_type, 'padding_side': padding_side,
                'padding_extent': padding_extent, 'padding_color': padding_color,
                'scene_mode': request['mode'],
                'negative_source': request['windows'][0]['type'] if negative else None,
                'augmentation': {'resampling': resampling.name, 'blur': blur, 'brightness': brightness, 'contrast': contrast},
                'retries': dict(retries)}
    folder = Path(task['root']) / split
    image.save(folder / f'{sample_id}.png', compress_level=config['png_compression'])
    _write_json(folder / f'{sample_id}.json', metadata)
    if task.get('collect_attempt_diagnostics', False):
        return {'record': metadata, 'attempt_diagnostics': attempts}
    return metadata


def _ordered_results(tasks, executor, workers):
    if executor is None:
        for task in tasks:
            yield _generate_one(task)
        return
    iterator = iter(tasks)
    pending = deque()
    for _ in range(workers * 2):
        task = next(iterator, None)
        if task is not None:
            pending.append(executor.submit(_generate_one, task))
    while pending:
        yield pending.popleft().result()
        task = next(iterator, None)
        if task is not None:
            pending.append(executor.submit(_generate_one, task))


def generate_dataset(destination, config=GenerationConfig()):
    """Generate train/val/test with deterministic split-safe board identities.

    The destination must be empty. A failed run keeps an incomplete marker and
    cannot be silently overwritten. Procedural wallpapers need no asset download.
    """
    root = Path(destination)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f'destination is not empty: {root}')
    root.mkdir(parents=True, exist_ok=True)
    settings = asdict(config)
    from synthetic_scenes.provenance import run_provenance
    provenance = run_provenance(Path(__file__).resolve().parent, 'grid-geometry')
    provenance['wallpaper_assets'] = {'status': 'not_applicable', 'reason': 'procedural wallpaper only'}
    metadata = {'generator': 'phase-grid-v1', 'generator_revision': GENERATOR_REVISION,
                'experiment': 'grid-geometry', 'dataset_name': root.name,
                **settings, 'source_policy': deepcopy(_source_policy(config)), 'provenance': provenance,
                'status': 'incomplete'}
    _write_json(root / 'config.json', metadata)
    for split in ('train', 'val', 'test'):
        (root / split).mkdir()
    if config.progress and (config.train or config.val or config.test):
        try:
            from tqdm import tqdm
        except ImportError as exc:
            raise ImportError('generation progress requires tqdm; use progress=False or install synthetic-scenes') from exc
    started = perf_counter()
    all_ids = set()
    ids_by_split = {}
    statistics = {}
    accepted = AcceptedStatistics(config.standalone_source_budget)
    attempt_overall = AttemptCounters()
    attempt_splits = {}
    pool = (ProcessPoolExecutor(max_workers=config.workers, mp_context=multiprocessing.get_context('spawn'))
            if config.workers > 1 and (config.train or config.val or config.test) else nullcontext(None))
    with tempfile.TemporaryDirectory(prefix='phase-identities-') as temporary, pool as executor:
        with (root / 'manifest.jsonl').open('w', encoding='utf-8') as manifest:
            for split, count in (('train', config.train), ('val', config.val), ('test', config.test)):
                exclusion = Path(temporary) / f'{split}.json'
                _write_json(exclusion, sorted(all_ids))
                base = {'root': str(root), 'config': settings, 'split': split, 'excluded_path': str(exclusion),
                        'collect_attempt_diagnostics': True}
                tasks = ({**base, 'index': index} for index in range(count))
                group = {key: Counter() for key in ('category', 'style', 'requested_pitch_bin', 'actual_pitch_bin_x', 'actual_pitch_bin_y', 'board_dimensions', 'visibility', 'negative_source', 'game_progress_category', 'context_type', 'requested_context', 'padding_type')}
                identities = set(); retries = Counter(); samples = 0
                padding_counts = Counter()
                fraction_records = []
                start_split = perf_counter()
                bar_context = tqdm(total=count, desc=split, unit='sample') if config.progress and count else nullcontext(None)
                with bar_context as bar:
                    for worker_result in _ordered_results(tasks, executor, config.workers):
                        record = worker_result['record']
                        accepted.add(record)
                        fraction_records.append({key: record[key] for key in ('category', 'actual_pitch_bin_x',
                            'grid_coverage_fraction', 'phase_supervision_fraction', 'grid_occupancy_fraction', 'padding_fraction')})
                        padding_counts['requested'] += record['intentional_padding_requested']
                        padding_counts['realized'] += record['padding_type'] is not None
                        kind = 'positive' if record['board_id'] else 'negative'
                        padding_counts[kind + '_samples'] += 1
                        padding_counts[kind + '_padded'] += record['padding_type'] is not None
                        board_id = record['board_id']
                        if board_id is not None:
                            if board_id in all_ids:
                                raise RuntimeError('cross-split minefield identity overlap')
                            identities.add(board_id)
                        attempt_overall.merge(worker_result['attempt_diagnostics'])
                        attempt_splits.setdefault(split, AttemptCounters()).merge(worker_result['attempt_diagnostics'])
                        for key in group:
                            if record[key] is not None:
                                group[key][str(record[key])] += 1
                        retries.update(record['retries'])
                        samples += 1
                        manifest.write(json.dumps(record, sort_keys=True) + '\n')
                        if bar is not None:
                            bar.update(1)
                elapsed = perf_counter() - start_split
                ids_by_split[split] = identities
                all_ids.update(identities)
                statistics[split] = {'samples': samples, 'positive': sum(group['category'][k] for k in CATEGORIES),
                    'negative': group['category']['negative'] + group['category']['hard_negative'],
                    'complete': group['category']['complete'], 'partial': sum(group['category'][k] for k in ('interior', 'side', 'corner')),
                    **{key: dict(value) for key, value in group.items()}, 'unique_board_ids': len(identities),
                    'padding': {**dict(padding_counts), 'requested_probability': config.padding_example_fraction,
                        'realized_rate': padding_counts['realized'] / samples if samples else None,
                        'positive_rate': padding_counts['positive_padded'] / padding_counts['positive_samples'] if padding_counts['positive_samples'] else None,
                        'negative_rate': padding_counts['negative_padded'] / padding_counts['negative_samples'] if padding_counts['negative_samples'] else None,
                        'accidental_samples': 0},
                    'retries': dict(retries), 'sampling_fractions': summarize_fractions(fraction_records), 'duration_seconds': elapsed, 'samples_per_second': samples / elapsed if elapsed else 0.}
    overlap = {f'{a}:{b}': len(ids_by_split[a] & ids_by_split[b]) for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    result = {'splits': statistics, 'board_identity_overlap': overlap, 'duration_seconds': perf_counter() - started,
              'accepted': accepted.result({'train': config.train, 'val': config.val, 'test': config.test}),
              'attempts': summaries(attempt_overall, attempt_splits, ('train', 'val', 'test'),
                                    unit='one geometry source attempt for a fixed objective')}
    _write_json(root / 'statistics.json', result)
    metadata['status'] = 'complete'
    _write_json(root / 'config.json', metadata)
    return result
