"""Sample scenes and games for the phase dataset."""
import math

from .sampling_config import DESKTOP_APPEARANCE, NEUTRAL_SCALE, SOURCE_CELL_SIZE, STYLES, THEMES, scale_choices

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
        max_cols, max_rows = int((size[0] - 30) / pitch), int((size[1] - 30) / pitch)
        candidates = [(r, c, weight) for r, c, weight in (
            (9, 9, 5.), (16, 16, 5.), (16, 30, 5.), (5, 5, .5), (6, 8, .5))
            if r <= max_rows and c <= max_cols]
        if max_rows >= 5 and max_cols >= 5:
            for _ in range(5):
                candidates.append((rng.randint(5, min(100, max_rows)), rng.randint(5, min(100, max_cols)), 1.))
        if not candidates:
            raise ValueError('complete board does not fit requested pitch')
        explore = rng.random() < coverage_exploration_fraction
        weights = [weight * (1. if explore else ((r * c * pitch * pitch) / (size[0] * size[1])) ** coverage_preference_exponent)
                   for r, c, weight in candidates]
        rows, columns, _ = rng.choices(candidates, weights=weights, k=1)[0]
    else:
        min_rows, min_cols = max(5, math.ceil(ch) + 2), max(5, math.ceil(cw) + 2)
        if max(min_rows, min_cols) > 100:
            raise ValueError('partial crop requires board larger than renderer policy')
        standards = [(r, c) for r, c in ((9, 9), (16, 16), (16, 30)) if r >= min_rows and c >= min_cols]
        if standards and rng.random() < .4:
            rows, columns = rng.choice(standards)
        else:
            rows = rng.randint(min_rows, min(100, min_rows + rng.randint(4, 25)))
            columns = rng.randint(min_cols, min(100, min_cols + rng.randint(4, 35)))
    style = rng.choice(STYLES)
    # Large boards use compact native tiles before scaling to the desired pitch.
    cell_size = rng.randint(*SOURCE_CELL_SIZE['large' if rows * columns > 900 else 'normal'])
    progress_kind = _choice(rng, {'hidden': 20, 'early': 25, 'mid': 30, 'late': 20, 'finished': 5})
    if progress_kind == 'hidden':
        status, progress = 'new', 0.
    elif progress_kind == 'finished':
        status = rng.choice(('won', 'lost'))
        progress = 1. if status == 'won' else rng.uniform(.2, .95)
    else:
        status = 'active'
        progress = rng.uniform(*{'early': (.05, .25), 'mid': (.25, .65), 'late': (.65, .95)}[progress_kind])
    appearance = {'style': style, 'theme': rng.choice(THEMES[style]), 'cell_size': cell_size,
                  'font_family': rng.choice(('Arial', 'Tahoma', 'Verdana')), 'font_scale': rng.uniform(.65, .77),
                  'bold': rng.random() < .9}
    appearance['zoom'] = _sample_scale(rng, NEUTRAL_SCALE if game_zoom is None else game_zoom)
    if style == 'google':
        appearance['boundary_ratio'] = rng.choice((0., .06, .08, .10))
    if style == 'windows7':
        appearance['mine_variant'] = rng.choice(('classic', 'flower'))
    return {'rows': rows, 'columns': columns, 'mine_count': max(1, min(rows * columns - 1, round(rows * columns * rng.uniform(.12, .20)))),
            'status': status, 'progress': progress, 'seed': rng.getrandbits(64),
            'elapsed_seconds': 0 if status == 'new' else rng.randint(1, 999), 'appearance': appearance}, progress_kind


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
    if game is not None and (game['rows'] * game['columns'] > 700 or rng.random() >= desktop_fraction):
        return {'mode': 'game', 'seed': seed, 'windows': [{'type': 'minesweeper', 'content': game}]}
    windows = []
    if negative_kind is not None:
        kind = rng.choice(('spreadsheet', 'sudoku', 'other_puzzle')) if negative_kind == 'hard_negative' else rng.choice(('browser', 'file_manager', 'document_editor', 'settings'))
        windows.append({'type': kind, 'seed': rng.getrandbits(64)})
    else:
        windows.append({'type': 'minesweeper', 'content': game})
        if rng.random() < .45:
            windows.append({'type': rng.choice(('spreadsheet', 'sudoku', 'browser', 'other_puzzle')), 'seed': rng.getrandbits(64)})
    scale = _sample_scale(rng, NEUTRAL_SCALE if display_scale is None else display_scale)
    return {'mode': 'desktop', 'seed': seed, 'layout_seed': rng.getrandbits(64),
            'screen': {'width': 1600, 'height': 1000, 'scale': scale}, 'desktop': _desktop(rng, desktop_appearance),
            'wallpaper': {'category': 'procedural', 'seed': rng.getrandbits(64)}, 'windows': windows}


