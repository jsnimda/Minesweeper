"""Deterministic, split-safe streaming YOLO dataset generation."""
from collections import Counter, deque
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from io import BytesIO
import json
import inspect
import multiprocessing
from pathlib import Path
import random
import tempfile

from tqdm import tqdm

from synthetic_scenes.assets.wallpapers.getters import get_wallpapers
from synthetic_scenes.augmentation import augment_scene
from synthetic_scenes.render import PlacementError, render_scene
from synthetic_scenes.minesweeper.styles import THEMES
from synthetic_scenes.provenance import asset_inventory, run_provenance
from synthetic_scenes.wallpaper import PROCEDURAL_SUBTYPE_WEIGHTS
from synthetic_scenes.attempt_statistics import AttemptCounters, NOT_APPLICABLE, board_factors, fixed_bucket, summaries
from synthetic_scenes.dataset_statistics import BINS
from .sampler import sample_scene
from .sampling_config import SAMPLING_CONFIG
from .yolo import YOLO_CLASSES, write_yolo_config, yolo_lines
from .statistics import AcceptedStatistics

GENERATOR_REVISION = 7  # Resolution-conditioned native scale and desktop sampling.
CAPTURE_POLICY = {'resize_range': (.65, 1.25), 'blur_choices': (0., 0., .25, .5),
                  'jpeg_quality_choices': (None, None, 85, 95)}
_worker_assets = None
_worker_exclusion_path = None
_worker_excluded = frozenset()


def prepare_assets():
    """Resolve assets in the parent, before starting spawned workers."""
    weights = SAMPLING_CONFIG['wallpaper_type']
    if SAMPLING_CONFIG['scene_mode'].get('desktop', 0) <= 0:
        return {}
    return {category: tuple(str(path) for path in get_wallpapers(category))
            for category in ('artwork', 'photographs') if weights.get(category, 0) > 0}


def _initialize_worker(assets):
    global _worker_assets, _worker_exclusion_path, _worker_excluded
    _worker_assets = assets
    _worker_exclusion_path = None
    _worker_excluded = frozenset()


def _excluded_for_split(path):
    global _worker_exclusion_path, _worker_excluded
    if path != _worker_exclusion_path:
        _worker_excluded = frozenset(json.loads(Path(path).read_text(encoding='utf-8')))
        _worker_exclusion_path = path
    return _worker_excluded


def _attempt_observations(metadata, request):
    data = dict(metadata.get('attempt', {}))
    if 'resolution_group' in metadata:
        data.setdefault('resolution_group', metadata['resolution_group'])
    if request is not None:
        data.setdefault('scene_mode', request['mode'])
        games = [w['content'] for w in request['windows'] if w['type'] == 'minesweeper']
        data.setdefault('game_count', len(games))
        if 'boards' not in data:
            sampled = metadata.get('sampled_cell_sizes', ())
            data['boards'] = [
                {'rows': game.get('rows'), 'columns': game.get('columns'),
                 'appearance': {'style': game.get('appearance', {}).get('style'),
                                'zoom': game.get('appearance', {}).get('zoom'),
                                'cell_size': sampled[index] if index < len(sampled) else None}}
                for index, game in enumerate(games)]
        if request['mode'] == 'desktop':
            data.setdefault('display_scale', request['screen'].get('scale', 1.0))
            data.setdefault('screen_width', request['screen']['width'])
            data.setdefault('screen_height', request['screen']['height'])
            for field, key in (('shell', 'shell'), ('desktop_style', 'style'), ('color_scheme', 'color_scheme')):
                data.setdefault(field, request['desktop'].get(key))
            data.setdefault('cleanliness', request.get('cleanliness'))
            data.setdefault('application_count', sum(w['type'] != 'minesweeper' for w in request['windows']))
    mode = data.get('scene_mode')
    desktop_fields = ('resolution_group', 'display_scale', 'shell', 'desktop_style', 'color_scheme', 'cleanliness', 'application_count')
    fields = {'scene_mode': mode, 'game_count': data.get('game_count'),
              'composition': ('negative' if data['game_count'] == 0 else 'positive') if data.get('game_count') is not None else None}
    for name in desktop_fields:
        fields[name] = NOT_APPLICABLE if mode in ('game', 'grid') else data.get(name)
    for axis in ('screen_width', 'screen_height'):
        fields[axis] = NOT_APPLICABLE if mode in ('game', 'grid') else fixed_bucket(data.get(axis), BINS['dimension'])
    boards = []
    for game in data.get('boards', []):
        factors = board_factors(game)
        zoom = game.get('appearance', {}).get('zoom')
        scale = 1.0 if mode in ('game', 'grid') else data.get('display_scale')
        factors['effective_native_scale'] = zoom * scale if zoom is not None and scale is not None else None
        boards.append(factors)
    return fields, boards


def _generate_one(task):
    """Render one accepted scene and optionally encode/export in this process."""
    split, index, seed = task['split'], task['index'], task['seed']
    excluded = _excluded_for_split(task['excluded_path'])
    retries = Counter()
    attempts = AttemptCounters()
    for attempt in range(task['max_attempts']):
        scene_seed = f'{seed}:{split}:{index}:{attempt}:scene'
        sampling_metadata = {}
        request = None
        fitting = {}
        try:
            request = sample_scene(scene_seed, sampling_metadata=sampling_metadata)
            image, resolved, annotations = render_scene(
                request, details=('scene',), wallpaper_assets=_worker_assets)
        except PlacementError as exc:
            retries['placement_retries'] += 1
            last_error = exc
            fields, boards = _attempt_observations(sampling_metadata, request)
            if request is None and fields['scene_mode'] in ('game', 'grid'):
                fitting['standalone_budget_infeasible'] = True
            elif request is None and fields['scene_mode'] == 'desktop':
                fitting['desktop_precheck_rejected'] = True
            attempts.observe(fields, boards, 'rejected', 'placement_retries', fitting)
            continue
        if request['mode'] in ('game', 'grid') and 'sampled_cell_sizes' in sampling_metadata:
            games = [w['content'] for w in request['windows'] if w['type'] == 'minesweeper']
            fitting['standalone_budget_fitted'] = any(
                size > game['appearance']['cell_size'] for size, game in zip(
                    sampling_metadata['sampled_cell_sizes'], games, strict=True))
        board_ids = [board['board_id'] for board in annotations['boards']]
        fields, boards = _attempt_observations(sampling_metadata, request)
        if request['mode'] == 'desktop' and boards:
            fitting['desktop_placement_fitted'] = any(board['requested_cell_size'] > board['resolved_cell_size'] for board in annotations['boards'])
        if not excluded.isdisjoint(board_ids):
            retries['minefield_overlap_retries'] += 1
            last_error = ValueError('Minefield overlaps an earlier split')
            attempts.observe(fields, boards, 'rejected', 'minefield_overlap_retries', fitting)
            continue
        attempts.observe(fields, boards, 'accepted', fitting=fitting)
        break
    else:
        raise RuntimeError(f'Failed to generate {split} scene {index}') from last_error

    attempts.finish_sample(attempt + 1)
    native_size = image.size
    native_boxes = [board['boxes']['grid'] for board in annotations['boards']]
    if task['augment']:
        rng = random.Random(f'{seed}:{split}:{index}:{attempt}:augmentation')
        image, annotations = augment_scene(image, annotations,
            resize_factor=rng.uniform(*CAPTURE_POLICY['resize_range']), blur=rng.choice(CAPTURE_POLICY['blur_choices']),
            jpeg_quality=rng.choice(CAPTURE_POLICY['jpeg_quality_choices']))
    name = f'{index:06d}'
    mode = task['mode']
    encoded_bytes = 0
    if mode == 'encode':
        with BytesIO() as buffer:
            image.save(buffer, format='PNG', compress_level=task['png_compression'])
            encoded_bytes = buffer.tell()
    elif mode == 'export':
        root = Path(task['root'])
        image_path = root / 'images' / split / f'{name}.png'
        label_path = root / 'labels' / split / f'{name}.txt'
        lines = yolo_lines(annotations)
        image.save(image_path, format='PNG', compress_level=task['png_compression'])
        encoded_bytes = image_path.stat().st_size
        label_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    elif mode != 'render':
        raise ValueError(f'Unknown generation mode: {mode}')

    desktop = None
    if request['mode'] == 'desktop':
        wallpaper = resolved.get('resolved_wallpaper', {})
        asset = wallpaper.get('asset')
        desktop = {'screen': request['screen'], 'resolution_group': sampling_metadata.get('resolution_group'),
                   'appearance': request['desktop'], 'cleanliness': request.get('cleanliness'),
                   'applications': [w['type'] for w in request['windows'] if w['type'] != 'minesweeper'],
                   'wallpaper': {'category': wallpaper.get('category'), 'subtype': wallpaper.get('subtype'),
                                 'asset': Path(asset).name if asset else None, 'placement': wallpaper.get('placement')}}
    record = {'split': split, 'index': index, 'image': f'images/{split}/{name}.png',
              'label': f'labels/{split}/{name}.txt', 'scene_seed': scene_seed,
              'board_ids': board_ids, 'attempt': attempt,
              'scene_mode': request['mode'], 'board_count': len(board_ids), 'negative': not board_ids,
              'augmentation_enabled': task['augment'],
              'native_size': native_size, 'final_size': image.size, 'desktop': desktop,
              'display_scale': request['screen']['scale'] if request['mode'] == 'desktop' else 1.0,
              'boards': []}
    game_specs = [w['content'] for w in request['windows'] if w['type'] == 'minesweeper']
    for board, native_box, game, sampled_cell_size in zip(
        annotations['boards'], native_boxes, game_specs, sampling_metadata['sampled_cell_sizes'], strict=True
    ):
        parameters = board['style_parameters']
        record['boards'].append({
            'board_id': board['board_id'],
            'game_state_id': board['game_state_id'], 'rows': board['rows'], 'columns': board['columns'],
            'mine_count': game['mine_count'], 'status': board['status'], 'style': board['style'], 'theme': board['theme'],
            'native_grid_box': native_box, 'final_grid_box': board['boxes']['grid'],
            'sampled_cell_size': sampled_cell_size,
            'requested_cell_size': board['requested_cell_size'],
            'resolved_cell_size': board['resolved_cell_size'],
            'requested_zoom': board['requested_zoom'],
            'requested_display_scale': board['requested_display_scale'],
            'requested_effective_scale': board['requested_effective_scale'],
            'cell_width': parameters['cell_width'], 'cell_height': parameters['cell_height'],
            'pitch_x': parameters['pitch_x'], 'pitch_y': parameters['pitch_y'],
        })
    if task['augment']:
        record['augmentation'] = annotations['augmentations'][-1]
    if task['scene_details']:
        record['resolved_scene'] = resolved
    pixels = image.width * image.height
    return {'record': record, 'pixels': pixels, 'encoded_bytes': encoded_bytes, 'attempt_diagnostics': attempts,
            'retries': dict(retries), 'mode': request['mode'],
            'styles': [board['style'] for board in annotations['boards']],
            'wallpaper': request['wallpaper']['category'] if request['mode'] == 'desktop' else None,
            'applications': sum(w['type'] != 'minesweeper' for w in request['windows'])}


def _ordered_results(tasks, executor, workers):
    """Keep at most twice the worker count in flight, and yield in index order."""
    if executor is None:
        for task in tasks:
            yield _generate_one(task)
        return
    tasks = iter(tasks)
    pending = deque()
    for _ in range(workers * 2):
        task = next(tasks, None)
        if task is None:
            break
        pending.append(executor.submit(_generate_one, task))
    while pending:
        yield pending.popleft().result()
        task = next(tasks, None)
        if task is not None:
            pending.append(executor.submit(_generate_one, task))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def run_splits(counts, *, seed, workers, png_compression, augment, max_attempts,
               scene_details, progress, mode, root=None, assets=None):
    """Shared production/benchmark loop. Images never leave a worker process."""
    if assets is None:
        assets = prepare_assets() if any(count for _, count in counts) else {}
    _initialize_worker(assets)
    excluded = set()
    statistics = {}
    accepted = AcceptedStatistics()
    attempt_overall = AttemptCounters()
    attempt_splits = {}
    pool = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                initializer=_initialize_worker, initargs=(assets,))
            if workers > 1 and any(count for _, count in counts) else nullcontext(None))
    with tempfile.TemporaryDirectory(prefix='yolo-identities-') as temp_dir, pool as executor:
        manifest_context = ((Path(root) / 'manifest.jsonl').open('w', encoding='utf-8')
                            if mode == 'export' else nullcontext(None))
        with manifest_context as manifest:
            for split, count in counts:
                totals = Counter({key: 0 for key in (
                    'scenes', 'boards', 'negatives', 'pixels', 'encoded_bytes',
                    'placement_retries', 'minefield_overlap_retries', 'retries',
                )})
                modes, styles, wallpapers, applications, games = (Counter() for _ in range(5))
                if count == 0:
                    statistics[split] = {'counts': dict(totals), 'scene_modes': {},
                        'board_styles': {}, 'wallpaper_categories': {},
                        'application_counts': {}, 'game_counts': {}}
                    continue
                exclusion_path = Path(temp_dir) / f'{split}.json'
                _write_json(exclusion_path, sorted(excluded))
                task_base = {'split': split, 'seed': seed, 'max_attempts': max_attempts,
                    'augment': augment, 'scene_details': scene_details, 'mode': mode,
                    'root': str(root) if root is not None else None,
                    'png_compression': png_compression, 'excluded_path': str(exclusion_path)}
                tasks = ({**task_base, 'index': index} for index in range(count))
                current_ids = set()
                bar_context = tqdm(total=count, desc=split, unit='scene') if progress else nullcontext(None)
                with bar_context as bar:
                    for result in _ordered_results(tasks, executor, workers):
                        record = result['record']
                        if mode == 'export':
                            accepted.add(record)
                        ids = record['board_ids']
                        if not excluded.isdisjoint(ids):
                            raise RuntimeError('Worker accepted a minefield from an earlier split')
                        if mode == 'export':
                            attempt_overall.merge(result['attempt_diagnostics'])
                            attempt_splits.setdefault(split, AttemptCounters()).merge(result['attempt_diagnostics'])
                        current_ids.update(ids)
                        totals.update(result['retries'])
                        totals['scenes'] += 1
                        totals['boards'] += len(ids)
                        totals['negatives'] += not ids
                        totals['pixels'] += result['pixels']
                        totals['encoded_bytes'] += result['encoded_bytes']
                        modes[result['mode']] += 1
                        styles.update(result['styles'])
                        games[str(len(ids))] += 1
                        if result['wallpaper'] is not None:
                            wallpapers[result['wallpaper']] += 1
                            applications[str(result['applications'])] += 1
                        if manifest is not None:
                            manifest.write(json.dumps(record) + '\n')
                        if bar is not None:
                            bar.update(1)
                excluded.update(current_ids)
                totals['retries'] = totals['placement_retries'] + totals['minefield_overlap_retries']
                statistics[split] = {'counts': dict(totals), 'scene_modes': dict(modes),
                    'board_styles': dict(styles), 'wallpaper_categories': dict(wallpapers),
                    'application_counts': dict(applications), 'game_counts': dict(games)}
    if mode == 'export':
        statistics['accepted'] = accepted.result({'train': dict(counts).get('train', 0),
                                                   'val': dict(counts).get('val', 0),
                                                   'test': dict(counts).get('test', 0)})
        statistics['attempts'] = summaries(attempt_overall, attempt_splits, ('train', 'val', 'test'),
                                          unit='one complete detection scene sampling attempt')
    return statistics


def _validate_parameters(train, val, test, seed, workers, png_compression, max_attempts):
    """Validate parameters shared by dataset generation and benchmarking."""
    if any(not isinstance(n, int) or n < 0 for n in (train, val, test, seed)):
        raise ValueError('Counts and seed must be nonnegative integers')
    if not isinstance(workers, int) or workers < 1 or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError('workers and max_attempts must be positive integers')
    if not isinstance(png_compression, int) or not 0 <= png_compression <= 9:
        raise ValueError('PNG compression must be an integer from 0 to 9')


def generate_dataset(destination, train=100, val=20, test=0, seed=42, workers=1,
                     png_compression=1, *, augment=True, max_attempts=20,
                     scene_details=False, progress=True):
    """Generate accepted scenes into an empty destination, with disjoint splits."""
    _validate_parameters(train, val, test, seed, workers, png_compression, max_attempts)
    root = Path(destination)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f'Destination is not empty: {root}')
    root.mkdir(parents=True, exist_ok=True)
    counts = [('train', train), ('val', val)] + ([('test', test)] if test else [])
    scale_policy = SAMPLING_CONFIG['display_scale']
    display_kind = 'conditional_by_resolution' if any(isinstance(value, dict) for value in scale_policy.values()) else 'global'
    augmentation_defaults = inspect.signature(augment_scene).parameters
    capture_policy = {**deepcopy(CAPTURE_POLICY),
                      'resampling': augmentation_defaults['resampling'].default.name,
                      'brightness': augmentation_defaults['brightness'].default,
                      'contrast': augmentation_defaults['contrast'].default}
    config = {'experiment': 'grid-detection', 'dataset_name': root.name,
              'generator_revision': GENERATOR_REVISION, 'train': train, 'val': val, 'test': test,
              'seed': seed, 'workers': workers, 'png_compression': png_compression,
              'augment': augment, 'max_attempts': max_attempts, 'scene_details': scene_details, 'progress': progress,
              'sampling_scope': deepcopy(SAMPLING_CONFIG), 'classes': YOLO_CLASSES,
              'source_policy': {'display_scale_policy': display_kind, 'themes': deepcopy(THEMES),
                  'procedural_wallpaper_subtype_weights': deepcopy(PROCEDURAL_SUBTYPE_WEIGHTS),
                  'capture_augmentation': capture_policy,
                  'standalone_minimum_cell_size': SAMPLING_CONFIG['cell_size']['small']['range'][0],
                  'negative_scene_policy': 'desktop game_count zero'},
              'provenance': run_provenance(Path(__file__).resolve().parent, 'grid-detection'),
              'status': 'incomplete'}
    config['provenance']['wallpaper_assets'] = {'status': 'not_collected'}
    _write_json(root / 'config.json', config)
    for split, _ in counts:
        for directory in ('images', 'labels'):
            (root / directory / split).mkdir(parents=True, exist_ok=True)
    write_yolo_config(root, test=bool(test))
    assets = prepare_assets() if any(count for _, count in counts) else {}
    if assets:
        config['provenance']['wallpaper_assets'] = asset_inventory(assets)
    else:
        active = SAMPLING_CONFIG['scene_mode'].get('desktop', 0) > 0 and any(
            SAMPLING_CONFIG['wallpaper_type'].get(category, 0) > 0 for category in ('artwork', 'photographs'))
        config['provenance']['wallpaper_assets'] = {'status': 'not_collected_no_samples' if active else 'not_applicable'}
    _write_json(root / 'config.json', config)
    stats = run_splits(counts, seed=seed, workers=workers, png_compression=png_compression,
        augment=augment, max_attempts=max_attempts, scene_details=scene_details,
        progress=progress, mode='export', root=root, assets=assets)
    _write_json(root / 'statistics.json', stats)
    config['status'] = 'complete'
    _write_json(root / 'config.json', config)
    return stats
