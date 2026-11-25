"""Generate split-safe synthetic datasets. Run with python -m synthetic_scenes.dataset."""
import argparse
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import random

from .assets.wallpapers.getters import get_wallpapers
from .augmentation import augment_scene
from .exporters import (CELL_CLASSES, YOLO_CLASSES, export_cells, export_geometry,
                        export_yolo, validate_heatmap_sigma, write_json, write_yolo_config)
from .sampling import sample_scene
from .scene import render_scene


def generate_dataset(root, *, train_count=100, val_count=20, seed=42,
                     desktop_size=(1280,720), tasks=('yolo','geometry','cells'),
                     cell_size=32, geometry_size=None, heatmap_sigma=None,
                     targeted_probability=.25, negative_probability=.08,
                     augment=True, max_attempts=20, wallpaper_paths=(), progress=None,
                     scene_sampler=sample_scene, wallpaper_spec=None):
    """Stream samples to disk. Counts refer to scenes, not board/cell counts.

    Train and validation use separate scene seed namespaces and minefields.
    Existing nonempty destinations are refused.
    Pass tqdm as progress to display a bar per split, or None to stay silent.
    Supply a scene sampler with the same keyword arguments as sample_scene
    to customize scene distributions. A wallpaper_spec selects a shared
    asset collection.
    """
    tasks = tuple(tasks)
    wallpaper_paths = tuple(str(Path(path)) for path in wallpaper_paths)
    if wallpaper_paths and wallpaper_spec is not None:
        raise ValueError('Choose wallpaper_paths or wallpaper_spec')
    for path in wallpaper_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if not tasks or set(tasks)-{'yolo','geometry','cells'}:
        raise ValueError('Unknown or empty task selection')
    if min(train_count,val_count,seed) < 0 or max_attempts < 1:
        raise ValueError('Invalid dataset counts or seed')
    if max(train_count,val_count)*max_attempts >= 10**9:
        raise ValueError('Dataset exceeds seed namespace capacity')
    if heatmap_sigma is not None:
        validate_heatmap_sigma(heatmap_sigma)
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Destination is not empty: {root}')
    if wallpaper_spec is not None and (train_count or val_count):
        category = wallpaper_spec["type"]

        if category == "photograph":
            category = "photographs"

        wallpaper_paths = tuple(
            str(path) for path in get_wallpapers(category)
        )
    root.mkdir(parents=True,exist_ok=True)
    config = {'seed':seed,'train_count':train_count,'val_count':val_count,'tasks':tasks,
              'desktop_size':desktop_size,'cell_size':cell_size,'geometry_size':geometry_size,
              'heatmap_sigma':heatmap_sigma,'targeted_probability':targeted_probability,
              'negative_probability':negative_probability,'augment':augment,'max_attempts':max_attempts,
              'wallpaper_paths':wallpaper_paths,'yolo_classes':YOLO_CLASSES,'cell_classes':CELL_CLASSES,'status':'incomplete'}
    if wallpaper_spec is not None:
        config.update(wallpaper_spec=wallpaper_spec)
    write_json(root/'config.json',config)
    counts = {}
    train_board_ids = set()
    for split_index,(split,count) in enumerate((('train',train_count),('val',val_count))):
        folder = root/'scenes'/split
        folder.mkdir(parents=True,exist_ok=True)
        if 'yolo' in tasks:
            for sub in ('images','labels'):
                (root/'yolo'/sub/split).mkdir(parents=True,exist_ok=True)
        stats = Counter()
        classes = Counter()
        progress_bar = nullcontext() if progress is None else progress(
            total=count, desc=split, unit='scene', miniters=1, mininterval=0,
        )
        with progress_bar as bar, (folder/'manifest.jsonl').open('w',encoding='utf-8') as manifest:
            for index in range(count):
                for attempt in range(max_attempts):
                    scene_seed = seed*2*10**9 + split_index*10**9 + index*max_attempts + attempt
                    scene_rng = random.Random(scene_seed ^ 0xD47A)
                    specs,mode = scene_sampler(
                        scene_seed,
                        targeted_probability=targeted_probability,
                        negative_probability=negative_probability,
                    )
                    wallpaper_path = None

                    if mode == 'desktop':
                        if wallpaper_spec is not None:
                            category = wallpaper_spec["type"]
                            wallpaper_rng = random.Random(f"{scene_seed}:{category}")
                            wallpaper_path = str(wallpaper_rng.choice(wallpaper_paths))
                        elif wallpaper_paths:
                            wallpaper_path = scene_rng.choice(wallpaper_paths)
                    try:
                        image,meta = render_scene(specs, size=desktop_size, seed=scene_seed, mode=mode,
                                                  wallpaper_path=wallpaper_path)
                        meta['wallpaper_path'] = wallpaper_path
                    except ValueError as exc:
                        stats['retries'] += 1
                        last_error = exc
                        continue
                    board_ids = [board['board_id'] for board in meta['boards']]
                    if split == 'val' and not train_board_ids.isdisjoint(board_ids):
                        stats['retries'] += 1
                        stats['minefield_overlap_retries'] += 1
                        last_error = ValueError('Validation scene shares a minefield with training')
                        continue
                    break
                else:
                    raise RuntimeError(f'Failed {split} scene {index}: {last_error}') from last_error
                if augment:
                    image, meta = augment_scene(
                        image, meta, scale=scene_rng.uniform(.65, 1.25),
                        blur=scene_rng.choice((0., 0., .25, .5)),
                        jpeg_quality=scene_rng.choice((None, None, 85, 95)),
                    )
                geometry_rng = random.Random(f'{scene_seed}:geometry') if 'geometry' in tasks else None
                cell_rng = random.Random(f'{scene_seed}:cells') if 'cells' in tasks else None
                name = f'{index:06d}'
                # Canonical screenshots always kept. YOLO uses identical pixels.
                image.save(folder/f'{name}.png')
                write_json(folder/f'{name}.json',meta)
                if 'yolo' in tasks:
                    export_yolo(root/'yolo',split,name,image,meta)
                for board_index,board in enumerate(meta['boards']):
                    board_name = f'{name}_{board_index:02d}'
                    if geometry_rng is not None:
                        export_geometry(root/'geometry',split,board_name,image,board,rng=geometry_rng,
                                        output_size=geometry_size,sigma=heatmap_sigma)
                    if cell_rng is not None:
                        classes.update(export_cells(
                            root/'cells', split, board_name, image, board, rng=cell_rng, cell_size=cell_size,
                        ))
                    stats['boards'] += 1
                    stats[f"style_{board['style']}"] += 1
                stats['scenes'] += 1
                stats[f'mode_{mode}'] += 1
                if not meta['boards']:
                    stats['negatives'] += 1
                manifest.write(json.dumps({'image':f'{name}.png','metadata':f'{name}.json',
                    'scene_seed':scene_seed,'board_ids':board_ids})+'\n')
                if split == 'train':
                    train_board_ids.update(board_ids)
                if bar is not None:
                    bar.set_postfix(
                        boards=stats['boards'],
                        render_retries=stats['retries'] - stats['minefield_overlap_retries'],
                        refresh=False,
                    )
                    bar.update(1)
        counts[split] = {'counts':dict(stats),'cell_counts':{name:classes[i] for i,name in enumerate(CELL_CLASSES)}}
    if 'yolo' in tasks:
        write_yolo_config(root/'yolo')
    write_json(root/'statistics.json',counts)
    config['status'] = 'complete'
    write_json(root/'config.json',config)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--train',type=int,default=100)
    parser.add_argument('--val',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--tasks',nargs='+',choices=('yolo','geometry','cells'),default=('yolo','geometry','cells'))
    parser.add_argument('--desktop-size',nargs=2,type=int,default=(1280,720),metavar=('W','H'))
    parser.add_argument('--geometry-size',nargs=2,type=int,metavar=('W','H'))
    parser.add_argument('--heatmap-sigma',type=float)
    parser.add_argument('--cell-size',type=int,default=32)
    parser.add_argument('--no-augment',action='store_true')
    parser.add_argument('--targeted-probability',type=float,default=.25)
    parser.add_argument('--negative-probability',type=float,default=.08)
    parser.add_argument('--wallpapers',nargs='*',default=())
    args = parser.parse_args()
    stats = generate_dataset(args.root,train_count=args.train,val_count=args.val,seed=args.seed,
        tasks=args.tasks,desktop_size=tuple(args.desktop_size),geometry_size=args.geometry_size,
        heatmap_sigma=args.heatmap_sigma,cell_size=args.cell_size,augment=not args.no_augment,
        targeted_probability=args.targeted_probability,negative_probability=args.negative_probability,
        wallpaper_paths=args.wallpapers)
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
