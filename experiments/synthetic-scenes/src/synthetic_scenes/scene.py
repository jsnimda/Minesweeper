"""Plan complete visible windows before rendering; keep boards independent."""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import random

from PIL import Image

from .coordinates import intersects, transform_board
from .desktop import wallpaper, draw_taskbar, background_window, chrome_size, content_origin, wrap_window
from .minesweeper.geometry import measure_game_region, measure_grid
from .minesweeper.board import generate_game


@dataclass
class GameSpec:
    rows: int = 9
    columns: int = 9
    mine_count: int = 10
    status: str = 'active'
    progress: float = .4
    seed: int = 42
    elapsed_seconds: int = 0
    mode: str = 'normal'
    target_number: int | None = None
    appearance: dict = field(default_factory=dict)
    chrome: str | None = None

    def generate(self):
        return generate_game(
            rows=self.rows, columns=self.columns, mine_count=self.mine_count,
            status=self.status, progress=self.progress, seed=self.seed,
            elapsed_seconds=self.elapsed_seconds, mode=self.mode,
            target_number=self.target_number,
        )


DEFAULT_CHROME = {'classic': 'classic', 'windows7': 'aero', 'microsoft': 'modern', 'google': 'browser'}


def _measure(spec, mode, cell_size):
    style = spec.appearance.get('style', 'classic')
    zoom = spec.appearance.get('zoom', 1.0)
    if mode == 'grid':
        return measure_grid(spec.rows, spec.columns, style, cell_size, zoom), 'none'
    content = measure_game_region(spec.rows, spec.columns, style, cell_size, zoom)
    chrome = (spec.chrome or DEFAULT_CHROME[style]) if mode == 'desktop' else 'none'
    return chrome_size(content, chrome), chrome


def plan_windows(specs, size, rng, *, min_cell_size=12, attempts=60):
    """Randomized packing with bounded retries; no rasterization in this phase."""
    width, height = size
    work_height = height - 40
    if width < 1 or work_height < 1 or min_cell_size < 8 or attempts < 1:
        raise ValueError('Invalid desktop or planning parameters')
    if not specs:
        return []

    for attempt in range(attempts):
        # Gradually shrink requested native sizes rather than repeatedly resize images.
        fraction = attempt / (attempts - 1) if attempts > 1 else 1.0
        candidates = []
        for index, spec in enumerate(specs):
            requested = spec.appearance.get('cell_size', 32)
            if requested < 8:
                raise ValueError('Requested cell size must be >= 8')
            smallest = min(min_cell_size, requested)
            tile = round(requested + (smallest - requested) * fraction)
            (w, h), chrome = _measure(spec, 'desktop', tile)
            candidates.append((index, w, h, tile, chrome))
        candidates.sort(key=lambda item: item[1]*item[2], reverse=True)

        placed = []
        for index, w, h, tile, chrome in candidates:
            if w > width or h > work_height:
                break
            corners = [(0, 0), (width-w, 0), (0, work_height-h), (width-w, work_height-h)]
            positions = [(rng.randint(0,width-w), rng.randint(0,work_height-h)) for _ in range(180)]
            positions.extend(corners)
            for x, y in positions:
                box = (x, y, x+w, y+h)
                if not any(intersects(box, item['box']) for item in placed):
                    placed.append({'index': index, 'box': box, 'cell_size': tile, 'chrome': chrome})
                    break
            else:
                break
        if len(placed) == len(specs):
            return sorted(placed, key=lambda item: item['index'])
    raise ValueError('Could not fit all games; reduce game count/size or enlarge desktop')


def render_scene(specs, *, size=(1280, 720), seed=42, mode='desktop',
                 background_count=2, wallpaper_path=None, min_cell_size=12):
    """Return image and metadata. Modes: desktop, game, grid; empty desktop is negative."""
    from .minesweeper.renderer import render_board, render_game_region

    specs = list(specs)
    rng = random.Random(seed)
    if mode not in ('desktop', 'game', 'grid'):
        raise ValueError('Unknown scene mode')
    if mode != 'desktop' and len(specs) != 1:
        raise ValueError('Game/grid modes require exactly one game')
    if mode == 'desktop' and (size[0] < 240 or size[1] < 200):
        raise ValueError('Desktop size must be at least 240 x 200')

    if mode == 'desktop':
        plan = plan_windows(specs, size, rng, min_cell_size=min_cell_size)
        image = wallpaper(size, rng, wallpaper_path)
        for _ in range(background_count):
            w = rng.randint(160, min(550, size[0]-2))
            h = rng.randint(90, min(400, size[1]-74))
            app = background_window((w, h), rng)
            x = rng.randint(0, size[0]-app.width)
            y = rng.randint(0, max(0, size[1]-40-app.height))
            image.paste(app, (x, y))
        draw_taskbar(image)
    else:
        spec = specs[0]
        tile = spec.appearance.get('cell_size', 32)
        (w, h), chrome = _measure(spec, mode, tile)
        plan = [{'index': 0, 'box': (0,0,w,h), 'cell_size': tile, 'chrome': chrome}]
        image = Image.new('RGB', (w,h))

    boards = []
    for item in plan:
        spec = specs[item['index']]
        options = dict(spec.appearance)
        options['cell_size'] = item['cell_size']
        render = render_board if mode == 'grid' else render_game_region
        game = spec.generate()
        content, meta = render(game, **options)
        chrome = item['chrome']
        window = wrap_window(content, chrome)
        x, y, right, bottom = item['box']
        if window.size != (right-x, bottom-y):
            raise RuntimeError('Measurement and rendered dimensions disagree')
        image.paste(window, (x,y))
        ox, oy = content_origin(chrome)
        meta = transform_board(meta, dx=x+ox, dy=y+oy)
        meta['window_box'] = item['box']
        meta['chrome'] = chrome
        minefield = game['board']
        minefield_data = json.dumps(minefield.shape).encode('ascii') + minefield.tobytes()
        meta['board_id'] = f'board-{sha256(minefield_data).hexdigest()}'
        meta['elapsed_seconds'] = game['elapsed_seconds']
        visible_state = (meta['board_id'], game['status'], game['elapsed_seconds'], meta['cell_states'])
        state_data = json.dumps(visible_state, separators=(',', ':')).encode('ascii')
        meta['game_state_id'] = f'state-{sha256(state_data).hexdigest()}'
        meta['requested_cell_size'] = spec.appearance.get('cell_size', 32)
        boards.append(meta)

    return image, {'source': 'synthetic', 'scene_id': f'scene-{seed}', 'seed': seed,
                   'mode': mode, 'size': image.size, 'boards': boards}
