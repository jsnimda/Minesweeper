"""Deterministic, split-safe streaming YOLO dataset generation."""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from io import BytesIO
import json
import multiprocessing
from pathlib import Path
import random
import tempfile

from tqdm import tqdm

from synthetic_scenes.assets.wallpapers.getters import get_wallpapers
from synthetic_scenes.augmentation import augment_scene
from synthetic_scenes.render import PlacementError, render_scene
from .sampler import sample_scene
from .scope import SCOPE
from .yolo import YOLO_CLASSES, write_yolo_config, yolo_lines

GENERATOR_REVISION = 4  # Procedural game-region rendering changed.
_worker_assets = None
_worker_exclusion_path = None
_worker_excluded = frozenset()


def prepare_assets():
    """Resolve assets in the parent, before starting spawned workers."""
    return {category: tuple(str(path) for path in get_wallpapers(category))
            for category in ('artwork', 'photographs')}


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


def _generate_one(task):
    """Render one accepted scene and optionally encode/export in this process."""
    split, index, seed = task['split'], task['index'], task['seed']
    excluded = _excluded_for_split(task['excluded_path'])
    retries = Counter()
    for attempt in range(task['max_attempts']):
        scene_seed = f'{seed}:{split}:{index}:{attempt}:scene'
        try:
            request = sample_scene(scene_seed)
            image, resolved, annotations = render_scene(
                request, scene_details=task['scene_details'], wallpaper_assets=_worker_assets)
        except PlacementError as exc:
            retries['placement_retries'] += 1
            last_error = exc
            continue
        board_ids = [board['board_id'] for board in annotations['boards']]
        if not excluded.isdisjoint(board_ids):
            retries['minefield_overlap_retries'] += 1
            last_error = ValueError('Minefield overlaps an earlier split')
            continue
        break
    else:
        raise RuntimeError(f'Failed to generate {split} scene {index}') from last_error

    if task['augment']:
        rng = random.Random(f'{seed}:{split}:{index}:{attempt}:augmentation')
        image, annotations = augment_scene(image, annotations,
            scale=rng.uniform(.65, 1.25), blur=rng.choice((0., 0., .25, .5)),
            jpeg_quality=rng.choice((None, None, 85, 95)))
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

    record = {'split': split, 'image': f'images/{split}/{name}.png',
              'label': f'labels/{split}/{name}.txt', 'scene_seed': scene_seed,
              'board_ids': board_ids, 'attempt': attempt}
    if task['augment']:
        record['augmentation'] = annotations['augmentations'][-1]
    if task['scene_details']:
        record['resolved_scene'] = resolved
    pixels = image.width * image.height
    return {'record': record, 'pixels': pixels, 'encoded_bytes': encoded_bytes,
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
                        ids = record['board_ids']
                        if not excluded.isdisjoint(ids):
                            raise RuntimeError('Worker accepted a minefield from an earlier split')
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
    config = {'generator_revision': GENERATOR_REVISION, 'train': train, 'val': val, 'test': test,
              'seed': seed, 'workers': workers, 'png_compression': png_compression,
              'augment': augment, 'max_attempts': max_attempts, 'scene_details': scene_details,
              'sampling_scope': SCOPE, 'classes': YOLO_CLASSES, 'status': 'incomplete'}
    _write_json(root / 'config.json', config)
    for split, _ in counts:
        for directory in ('images', 'labels'):
            (root / directory / split).mkdir(parents=True, exist_ok=True)
    write_yolo_config(root, test=bool(test))
    stats = run_splits(counts, seed=seed, workers=workers, png_compression=png_compression,
        augment=augment, max_attempts=max_attempts, scene_details=scene_details,
        progress=progress, mode='export', root=root)
    _write_json(root / 'statistics.json', stats)
    config['status'] = 'complete'
    _write_json(root / 'config.json', config)
    return stats
