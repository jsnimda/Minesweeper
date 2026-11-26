"""Sample scenes and games for the phase dataset."""
import math

from .sampling_config import DESKTOP_APPEARANCE, NEUTRAL_SCALE, SOURCE_CELL_SIZE, STYLES, THEMES, scale_choices

LARGE_BOARD_CELL_THRESHOLD = 900
DESKTOP_BOARD_AREA_LIMIT = 700
SOURCE_DESKTOP_SIZE = (1600, 1000)
GAME_PROGRESS = {'hidden': 20, 'early': 25, 'mid': 30, 'late': 20, 'finished': 5}
PROGRESS_RANGES = {'early': (.05, .25), 'mid': (.25, .65), 'late': (.65, .95)}
MINE_DENSITY = (.12, .20)
FONT_FAMILIES = ('Arial', 'Tahoma', 'Verdana')
FONT_SCALE = (.65, .77)
BOLD_PROBABILITY = .9
GOOGLE_BOUNDARIES = (0., .06, .08, .10)
WINDOWS7_MINES = ('classic', 'flower')
SOURCE_APPLICATION_PROBABILITY = .45
SOURCE_APPLICATIONS = ('spreadsheet', 'sudoku', 'browser', 'other_puzzle')
NEGATIVE_APPLICATIONS = ('browser', 'file_manager', 'document_editor', 'settings')
HARD_NEGATIVE_APPLICATIONS = ('spreadsheet', 'sudoku', 'other_puzzle')
COMPLETE_BOARD_CANDIDATES = ((9, 9, 5.), (16, 16, 5.), (16, 30, 5.), (5, 5, .5), (6, 8, .5))
PARTIAL_STANDARD_BOARDS = ((9, 9), (16, 16), (16, 30))
PARTIAL_STANDARD_PROBABILITY = .4
BOARD_DIMENSION_RANGE = (5, 100)
COMPLETE_MARGIN_PIXELS = 30
COMPLETE_RANDOM_CANDIDATES = 5
COMPLETE_RANDOM_WEIGHT = 1.
PARTIAL_EXTRA_CELLS = 2
PARTIAL_ROW_EXTENSION = (4, 25)
PARTIAL_COLUMN_EXTENSION = (4, 35)
FINISHED_STATUSES = ('won', 'lost')
LOST_PROGRESS_RANGE = (.2, .95)
ELAPSED_SECONDS = (1, 999)

def _choice(rng, weights):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _sample_scale(rng, weights):
    choices = scale_choices(weights)
    if len(choices) == 1:
        return next(iter(choices))
    return _choice(rng, choices)


def _game(rng, category, pitch, size, *, coverage_exploration_fraction=.12,
          coverage_preference_exponent=1.5, game_zoom=None):
    # Conditioning board size on category prevents infeasible samples from
    # silently changing the requested pitch. Typical sizes are preferred.
    cw, ch = size[0] / pitch, size[1] / pitch
    if category == 'complete':
        max_cols, max_rows = int((size[0] - COMPLETE_MARGIN_PIXELS) / pitch), int((size[1] - COMPLETE_MARGIN_PIXELS) / pitch)
        candidates = [(r, c, weight) for r, c, weight in COMPLETE_BOARD_CANDIDATES
            if r <= max_rows and c <= max_cols]
        minimum, maximum = BOARD_DIMENSION_RANGE
        if max_rows >= minimum and max_cols >= minimum:
            for _ in range(COMPLETE_RANDOM_CANDIDATES):
                candidates.append((rng.randint(minimum, min(maximum, max_rows)), rng.randint(minimum, min(maximum, max_cols)), COMPLETE_RANDOM_WEIGHT))
        if not candidates:
            raise ValueError('complete board does not fit requested pitch')
        explore = rng.random() < coverage_exploration_fraction
        weights = [weight * (1. if explore else ((r * c * pitch * pitch) / (size[0] * size[1])) ** coverage_preference_exponent)
                   for r, c, weight in candidates]
        rows, columns, _ = rng.choices(candidates, weights=weights, k=1)[0]
    else:
        minimum, maximum = BOARD_DIMENSION_RANGE
        min_rows, min_cols = max(minimum, math.ceil(ch) + PARTIAL_EXTRA_CELLS), max(minimum, math.ceil(cw) + PARTIAL_EXTRA_CELLS)
        if max(min_rows, min_cols) > maximum:
            raise ValueError('partial crop requires board larger than renderer policy')
        standards = [(r, c) for r, c in PARTIAL_STANDARD_BOARDS if r >= min_rows and c >= min_cols]
        if standards and rng.random() < PARTIAL_STANDARD_PROBABILITY:
            rows, columns = rng.choice(standards)
        else:
            rows = rng.randint(min_rows, min(maximum, min_rows + rng.randint(*PARTIAL_ROW_EXTENSION)))
            columns = rng.randint(min_cols, min(maximum, min_cols + rng.randint(*PARTIAL_COLUMN_EXTENSION)))
    style = rng.choice(STYLES)
    # Large boards use compact native tiles before scaling to the desired pitch.
    cell_size = rng.randint(*SOURCE_CELL_SIZE['large' if rows * columns > LARGE_BOARD_CELL_THRESHOLD else 'normal'])
    progress_kind = _choice(rng, GAME_PROGRESS)
    if progress_kind == 'hidden':
        status, progress = 'new', 0.
    elif progress_kind == 'finished':
        status = rng.choice(FINISHED_STATUSES)
        progress = 1. if status == 'won' else rng.uniform(*LOST_PROGRESS_RANGE)
    else:
        status = 'active'
        progress = rng.uniform(*PROGRESS_RANGES[progress_kind])
    appearance = {'style': style, 'theme': rng.choice(THEMES[style]), 'cell_size': cell_size,
                  'font_family': rng.choice(FONT_FAMILIES), 'font_scale': rng.uniform(*FONT_SCALE),
                  'bold': rng.random() < BOLD_PROBABILITY}
    appearance['zoom'] = _sample_scale(rng, NEUTRAL_SCALE if game_zoom is None else game_zoom)
    if style == 'google':
        appearance['boundary_ratio'] = rng.choice(GOOGLE_BOUNDARIES)
    if style == 'windows7':
        appearance['mine_variant'] = rng.choice(WINDOWS7_MINES)
    return {'rows': rows, 'columns': columns, 'mine_count': max(1, min(rows * columns - 1, round(rows * columns * rng.uniform(*MINE_DENSITY)))),
            'status': status, 'progress': progress, 'seed': rng.getrandbits(64),
            'elapsed_seconds': 0 if status == 'new' else rng.randint(*ELAPSED_SECONDS), 'appearance': appearance}, progress_kind


def _desktop(rng, desktop_appearance=None):
    config = DESKTOP_APPEARANCE if desktop_appearance is None else desktop_appearance
    shell = _choice(rng, config['shell'])
    style = _choice(rng, config['shell_styles'][shell])
    desktop = {'shell': shell, 'style': style, 'color_scheme': _choice(rng, config['color_scheme'][style]),
               'accent_color': None, 'clock_time': '10:08', 'clock_date': '2025/06/12'}
    relevant = {'taskbar': ('show_taskbar',), 'dock': ('show_dock', 'show_status_bar'),
                'top_panel': ('show_top_panel',)}[shell]
    for name in relevant:
        desktop[name] = rng.random() < config['visibility'][name]
    if shell == 'dock' and desktop['show_dock']:
        groups = config['dock_height']
        group = _choice(rng, {name: value['weight'] for name, value in groups.items()})
        desktop['dock_height'] = rng.randint(*groups[group]['range'])
    return desktop


def _scene(rng, game, negative_kind=None, desktop_fraction=.8, *, display_scale=None, desktop_appearance=None):
    seed = rng.getrandbits(64)
    if game is not None and (game['rows'] * game['columns'] > DESKTOP_BOARD_AREA_LIMIT or rng.random() >= desktop_fraction):
        return {'mode': 'game', 'seed': seed, 'windows': [{'type': 'minesweeper', 'content': game}]}
    windows = []
    if negative_kind is not None:
        kind = rng.choice(HARD_NEGATIVE_APPLICATIONS) if negative_kind == 'hard_negative' else rng.choice(NEGATIVE_APPLICATIONS)
        windows.append({'type': kind, 'seed': rng.getrandbits(64)})
    else:
        windows.append({'type': 'minesweeper', 'content': game})
        if rng.random() < SOURCE_APPLICATION_PROBABILITY:
            windows.append({'type': rng.choice(SOURCE_APPLICATIONS), 'seed': rng.getrandbits(64)})
    scale = _sample_scale(rng, NEUTRAL_SCALE if display_scale is None else display_scale)
    return {'mode': 'desktop', 'seed': seed, 'layout_seed': rng.getrandbits(64),
            'screen': {'width': SOURCE_DESKTOP_SIZE[0], 'height': SOURCE_DESKTOP_SIZE[1], 'scale': scale}, 'desktop': _desktop(rng, desktop_appearance),
            'wallpaper': {'category': 'procedural', 'seed': rng.getrandbits(64)}, 'windows': windows}


