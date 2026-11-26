"""Render descriptive scene dictionaries without applying dataset sampling policy.

The old :mod:`synthetic_scenes.scene` API remains available for existing users.
Coordinates in returned annotations are half-open image-space xyxy boxes.
"""
from collections.abc import Collection
from copy import deepcopy
from hashlib import sha256
import importlib
from itertools import chain
import json
import random
import sys

from PIL import Image, ImageOps

from .assets.wallpapers.getters import get_wallpapers
from .coordinates import intersects, transform_board
from .desktop_rendering.generic_window import window_geometry, window_size
from .image_conversion import image_to_pillow, pillow_to_image
from .desktop_rendering.screenshot import WindowLayer, _work_area, compose_screenshot
from .minesweeper.board import generate_game, get_visible_cells
from .minesweeper.geometry import measure_game_region
from .minesweeper.renderer import render_board, render_game_region
from .wallpaper import render_procedural_wallpaper, sample_procedural_subtype
from .schema import SceneAnnotations, SceneSpec, ResolvedScene

APPLICATIONS = {
    'browser', 'file_manager', 'spreadsheet', 'document_editor', 'presentation',
    'code_editor', 'terminal', 'calculator', 'settings', 'sudoku', 'other_puzzle',
    'media_viewer',
}


class PlacementError(ValueError):
    """A valid scene could not be packed into the desktop work area."""


def _drawing_scale(zoom, display_scale):
    # Underflow still represents a positive scale and reaches both drawing minima.
    return max(sys.float_info.min, zoom * display_scale)


def _wallpaper(size, spec, assets=None):
    rng = random.Random(spec['seed'])
    category = spec['category']
    if category == 'procedural':
        subtype = spec.get('subtype') or sample_procedural_subtype(rng)
        return render_procedural_wallpaper(size, rng, subtype, native=True), {'category': category, 'subtype': subtype, 'seed': spec['seed']}
    paths = get_wallpapers(category) if assets is None else assets[category]
    if not paths:
        raise FileNotFoundError(f'No wallpaper assets for {category}')
    path = rng.choice(paths)
    placement = spec.get('placement', 'cover')
    with Image.open(path) as source:
        source = source.convert('RGB')
        if placement == 'cover':
            image = ImageOps.fit(source, size, method=Image.Resampling.LANCZOS)
            actual = {'size': size}
        elif placement == 'fit':
            fitted = ImageOps.contain(source, size, method=Image.Resampling.LANCZOS)
            color = spec.get('background_color', '#202020')
            image = Image.new('RGB', size, color)
            position = ((size[0]-fitted.width)//2, (size[1]-fitted.height)//2)
            image.paste(fitted, position)
            actual = {'size': fitted.size, 'position': position, 'background_color': color}
        elif placement == 'repeat':
            tile_size = tuple(spec.get('tile_size') or (max(1, size[0]//3), max(1, round(source.height * (size[0]//3) / source.width))))
            if min(tile_size) < 1:
                raise ValueError('Tile size must be positive')
            # Fit within the requested tile while preserving the source aspect ratio.
            tile = ImageOps.contain(source, tile_size, method=Image.Resampling.LANCZOS)
            image = Image.new('RGB', size)
            for y in range(0, size[1], tile.height):
                for x in range(0, size[0], tile.width):
                    image.paste(tile, (x, y))
            actual = {'tile_size': tile.size}
        else:
            raise ValueError(f'Unknown wallpaper placement: {placement}')
    return pillow_to_image(image), {'category': category, 'seed': spec['seed'], 'asset': str(path), 'placement': placement, **actual}


def plan_game_windows(games, size, desktop, seed, scale=1.0):
    """Measure and pack complete game windows within the desktop work area.

    Return per-window positions and effective cell sizes, or raise PlacementError.
    """
    try:
        left, top, right, bottom = _work_area(size, desktop, scale)
    except OverflowError:
        raise PlacementError('Display scale leaves no usable work area') from None
    if right <= left or bottom <= top:
        raise PlacementError('Display scale leaves no usable work area')
    rng = random.Random(seed)
    for attempt in range(60):
        items = []
        for index, spec in games:
            requested = spec.get('appearance', {}).get('cell_size', 32)
            if requested < 8:
                raise ValueError('cell_size must be at least 8')
            cell = round(requested + (min(12, requested)-requested)*attempt/59)
            try:
                content_size = measure_game_region(
                    spec['rows'], spec['columns'],
                    spec['appearance'].get('style', 'classic'), cell,
                    _drawing_scale(spec['appearance'].get('zoom', 1.0), scale),
                )
                w, h = window_size(content_size, style=desktop['style'], scale=scale)
            except OverflowError:
                raise PlacementError('Scaled game cannot fit the work area') from None
            except ValueError as exc:
                # A product of finite positive scales may overflow to infinity.
                if str(exc) == 'zoom must be a positive finite number':
                    raise PlacementError('Scaled game cannot fit the work area') from None
                raise
            items.append((index, w, h, cell))
        placed = []
        for index, w, h, cell in sorted(items, key=lambda item: item[1]*item[2], reverse=True):
            if w > right-left or h > bottom-top:
                break
            positions = chain(((rng.randint(left, right-w), rng.randint(top, bottom-h))
                               for _ in range(180)),
                              ((left, top), (right-w, top), (left, bottom-h), (right-w, bottom-h)))
            for x, y in positions:
                box = (x, y, x+w, y+h)
                if not any(intersects(box, entry['box']) for entry in placed):
                    placed.append({'index': index, 'box': box, 'cell_size': cell})
                    break
            else:
                break
        if len(placed) == len(items):
            return {entry['index']: entry for entry in placed}
    raise PlacementError('Could not fit Minesweeper windows in the work area')


def render_scene(specification: dict, *, details=(), wallpaper_assets=None) -> tuple[Image.Image, ResolvedScene, SceneAnnotations]:
    """Return ``image, resolved_scene, annotations`` for a scene dictionary.

    Required keys are mode, seed and windows. Desktop scenes also require screen
    (width/height), desktop (shell/style/color_scheme) and wallpaper. Each game
    window has type='minesweeper' and a content dictionary containing rows,
    columns, mine_count, status, progress, seed and appearance. Other windows
    have a supported application type and an independent seed.

    `details` independently enables requested scene layout, cell metadata,
    and mine positions. Invalid requests raise Pydantic ValidationError; a
    valid desktop that cannot be packed raises PlacementError. The requested
    scene is copied into the resolved result only when 'scene' is selected.
    """
    if isinstance(details, (str, bytes)) or not isinstance(details, Collection):
        raise ValueError('details must be a collection of detail names')
    if any(not isinstance(name, str) for name in details):
        raise ValueError('detail names must be strings')
    details = frozenset(details)
    unknown = details - {'scene', 'cell', 'mine'}
    if unknown:
        raise ValueError(f'Unsupported details: {sorted(unknown)}')
    scene_details = 'scene' in details
    cell_details = 'cell' in details
    mine_details = 'mine' in details
    # Validate without serializing: omitted keys and explicit None retain their
    # original rendering meaning, and the caller's dictionary is not modified.
    SceneSpec.model_validate(specification, context={'wallpaper_assets': wallpaper_assets})
    mode = specification['mode']
    windows = specification['windows']
    if mode not in ('desktop', 'game', 'grid'):
        raise ValueError(f'Unknown scene mode: {mode}')
    games = [(i, w['content']) for i, w in enumerate(windows) if w['type'] == 'minesweeper']
    if mode != 'desktop' and (len(games) != 1 or len(windows) != 1):
        raise ValueError('Game and grid modes require exactly one game')
    seed = specification['seed']
    resolved = deepcopy(specification) if scene_details else {'mode': mode, 'seed': seed}
    display_scale = specification['screen'].get('scale', 1.0) if mode == 'desktop' else 1.0
    if mode == 'desktop':
        size = (specification['screen']['width'], specification['screen']['height'])
        desktop = {'accent_color': None, **specification['desktop']}
        plan = plan_game_windows(games, size, desktop, specification.get('layout_seed', seed), display_scale)
        wallpaper, wallpaper_info = _wallpaper(size, specification['wallpaper'], wallpaper_assets)
        if scene_details:
            resolved['resolved_wallpaper'] = wallpaper_info
            resolved['layout'] = []
        left, top, right, bottom = _work_area(size, desktop, display_scale)
        layers, board_info = [], []
        # Applications are always behind protected games.
        for index, window in enumerate(windows):
            if window['type'] == 'minesweeper':
                continue
            kind = window['type']
            if kind not in APPLICATIONS:
                raise ValueError(f'Unknown application: {kind}')
            module = importlib.import_module(f'.desktop_rendering.applications.{kind}', __package__)
            render_application = getattr(module, f'render_{kind}')
            # Check unusually large intermediate surfaces before allocating them.
            # The fixed logical defaults are owned by each application renderer.
            if display_scale > 4:
                logical, _ = render_application(seed=window['seed'], scale=1.0)
                try:
                    physical_width = round(logical.width() * display_scale)
                    physical_height = round(logical.height() * display_scale)
                except OverflowError:
                    raise PlacementError('Application surface is too large') from None
                if (max(physical_width, physical_height) > 32767 or
                        physical_width * physical_height > 64_000_000):
                    raise PlacementError('Application surface is too large')
            content, _ = render_application(seed=window['seed'], scale=display_scale)
            content_size = (content.width(), content.height())
            w, h = window_size(content_size, style=desktop['style'], scale=display_scale)
            rng = random.Random(window['seed'])
            position = (rng.randint(left, right-w) if w <= right-left else left,
                        rng.randint(top, bottom-h) if h <= bottom-top else top)
            layers.append(WindowLayer(content, {}, position, title=kind.replace('_', ' ').title(),
                                      allow_clipping=True))
            if scene_details:
                resolved['layout'].append({'index': index, 'position': position, 'window_size': (w, h)})
    else:
        plan = {games[0][0]: {'cell_size': games[0][1].get('appearance', {}).get('cell_size', 32)}}
        board_info = []

    for index, game_spec in games:
        game = generate_game(rows=game_spec['rows'], columns=game_spec['columns'],
            mine_count=game_spec['mine_count'], status=game_spec['status'], progress=game_spec['progress'],
            seed=game_spec['seed'], elapsed_seconds=game_spec.get('elapsed_seconds', 0), mode='normal')
        options = dict(game_spec['appearance'])
        options['cell_size'] = plan[index]['cell_size']
        options['zoom'] = _drawing_scale(game_spec['appearance'].get('zoom', 1.0), display_scale)
        options['cell_details'] = cell_details
        visible = get_visible_cells(game)
        content, metadata = (render_board if mode == 'grid' else render_game_region)(
            game, native=True, visible_cells=visible, **options)
        metadata['requested_cell_size'] = game_spec['appearance'].get('cell_size', 32)
        metadata['resolved_cell_size'] = plan[index]['cell_size']
        metadata['requested_zoom'] = game_spec['appearance'].get('zoom', 1.0)
        metadata['requested_display_scale'] = display_scale
        metadata['requested_effective_scale'] = metadata['requested_zoom'] * display_scale
        board = game['board']
        metadata['board_id'] = 'board-' + sha256(json.dumps(board.shape).encode('ascii') + board.tobytes()).hexdigest()
        metadata['game_state_id'] = 'state-' + sha256(metadata['board_id'].encode() + game['status'].encode() + str(game['elapsed_seconds']).encode() + visible.tobytes()).hexdigest()
        if mine_details:
            metadata['mine_positions'] = [(int(r), int(c)) for r, c in zip(*(board == -1).nonzero())]
        if mode == 'desktop':
            x, y, _, _ = plan[index]['box']
            planned_size = (plan[index]['box'][2]-x, plan[index]['box'][3]-y)
            if window_size((content.width(), content.height()), style=desktop['style'], scale=display_scale) != planned_size:
                raise RuntimeError('Measurement and rendered dimensions disagree')
            layer = WindowLayer(content, metadata['boxes'], (x, y), title='Minesweeper')
            layers.append(layer)
            _, (offset_x, offset_y) = window_geometry(
                (content.width(), content.height()), style=desktop['style'], scale=display_scale)
            metadata = transform_board(metadata, dx=x+offset_x, dy=y+offset_y)
            metadata['window_box'] = plan[index]['box']
            if scene_details:
                resolved['layout'].append({'index': index, 'box': plan[index]['box'], 'cell_size': plan[index]['cell_size']})
        elif scene_details:
            resolved['layout'] = [{'index': index, 'cell_size': plan[index]['cell_size'], 'content_size': (content.width(), content.height())}]
        board_info.append(metadata)
    if mode == 'desktop':
        image, _ = compose_screenshot(wallpaper, size, desktop, layers, scale=display_scale)
    else:
        image = content
    image = image_to_pillow(image)
    annotations = {'source': 'synthetic', 'mode': mode, 'seed': seed, 'size': image.size, 'boards': board_info}
    return image, resolved, annotations
