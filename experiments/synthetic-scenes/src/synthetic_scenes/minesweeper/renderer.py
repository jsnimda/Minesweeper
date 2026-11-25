"""Render complete grids with grid-local, half-open xyxy annotations."""

import skia

from ..image_conversion import image_to_pillow
from .board import CellState, get_visible_cells
from .geometry import cell_gap, measure_grid
from .drawing import font_for, line, rect, text
from .symbols import draw_flag, draw_mine
from .styles import classic, google, microsoft, windows7


STYLES = {
    'classic': classic,
    'windows7': windows7,
    'microsoft': microsoft,
    'google': google,
}
RENDERER_IDS = {
    'classic': 'classic_hd',
    'windows7': 'windows7_hd',
    'microsoft': 'microsoft_hd',
    'google': 'google_hd',
}
DIGIT_COLORS = (
    '#0000ff', '#008000', '#ff0000', '#000080',
    '#800000', '#008080', '#000000', '#808080',
)
MICROSOFT_COLORS = (
    '#009db8', '#698323', '#b82759', '#2858a9',
    '#9e352b', '#007e80', '#303030', '#777777',
)
GOOGLE_COLORS = (
    '#1976d2', '#388e3c', '#d32f2f', '#7b1fa2',
    '#ff8f00', '#00838f', '#424242', '#616161',
)
STYLE_PALETTES = {
    'google': GOOGLE_COLORS,
    'microsoft': MICROSOFT_COLORS,
}


def _opened_surfaces(game, cells, style):
    """Resolve style-dependent visual surfaces without changing cell labels."""
    opened = (cells != CellState.HIDDEN) & (cells != CellState.FLAG)
    if style == 'windows7' and game['status'] == 'lost':
        opened |= game['board'] == -1
    return opened


def _neighbors(opened, row, col):
    rows, columns = opened.shape
    return {
        'top': row > 0 and bool(opened[row - 1, col]),
        'bottom': row + 1 < rows and bool(opened[row + 1, col]),
        'left': col > 0 and bool(opened[row, col - 1]),
        'right': col + 1 < columns and bool(opened[row, col + 1]),
        'top_covered': row > 0 and not bool(opened[row - 1, col]),
        'left_covered': col > 0 and not bool(opened[row, col - 1]),
    }


def _draw_content(canvas, x, y, size, state, font, palette, mine_under_flag,
                  style, mine_variant):
    """Draw symbols independently of the cell surface."""
    if state == CellState.EXPLODED_MINE:
        inset = size * .04
        rect(canvas, x + inset, y + inset, size - 2*inset, size - 2*inset, '#e45b50')

    if 1 <= state <= 8:
        text(canvas, str(state), x + size/2, y + size/2, font, palette[state - 1])
    elif state == CellState.FLAG:
        if mine_under_flag:
            draw_mine(canvas, x, y, size, style, mine_variant)
        draw_flag(canvas, x, y, size, style)
    elif state in (CellState.MINE, CellState.EXPLODED_MINE, CellState.WRONG_FLAG):
        draw_mine(canvas, x, y, size, style, mine_variant)
        if state == CellState.WRONG_FLAG:
            _draw_wrong_flag(canvas, x, y, size)


def _draw_wrong_flag(canvas, x, y, size):
    left, right = x + .2*size, x + .8*size
    top, bottom = y + .2*size, y + .8*size
    line(canvas, (left, top), (right, bottom), '#ee2222', size*.065)
    line(canvas, (right, top), (left, bottom), '#ee2222', size*.065)


def _axis_edges(count, pitch, gap):
    """Split internal gaps at their midpoint; keep outer bounds exact."""
    return [0.0] + [float(i*pitch - gap/2) for i in range(1, count)] + [float(count*pitch - gap)]


def _cell_boxes(rows, columns, size, pitch):
    return [
        [(c*pitch, r*pitch, c*pitch + size, r*pitch + size) for c in range(columns)]
        for r in range(rows)
    ]


def render_board(
    game,
    *,
    style='classic',
    theme='default',
    cell_size=32,
    font_family='Arial',
    font_scale=.72,
    bold=True,
    digit_palette=None,
    boundary_ratio=.08,
    mine_variant='classic',
    cell_details=True,
    native=False,
    visible_cells=None,
):
    """Render a board and its annotations.

    boundary_ratio controls Google's covered/revealed boundary as a fraction
    of tile size. Zero disables it. Other styles ignore this parameter.
    """
    if not isinstance(cell_size, int) or cell_size < 8:
        raise ValueError('cell_size must be an integer >= 8')
    if not .3 <= font_scale <= .9:
        raise ValueError('font_scale must be between .3 and .9')
    if not 0 <= boundary_ratio <= .25:
        raise ValueError('boundary_ratio must be between 0 and .25')

    if mine_variant not in ('classic', 'flower'):
        raise ValueError('Unknown mine variant')
    if mine_variant == 'flower' and style != 'windows7':
        raise ValueError('Flower variant is only supported by windows7')

    module = STYLES[style]
    if theme not in module.THEMES:
        raise ValueError(f'Unknown theme {theme!r} for {style}')
    palette = STYLE_PALETTES.get(style, DIGIT_COLORS)
    if digit_palette is not None:
        palette = tuple(digit_palette)
    if len(palette) != 8:
        raise ValueError('Expected eight digit colors')

    # Resolve the board's visible state and geometry before drawing.
    cells = get_visible_cells(game) if visible_cells is None else visible_cells
    opened = _opened_surfaces(game, cells, style)
    rows, columns = cells.shape
    size = cell_size
    gap = cell_gap(style, size)
    pitch = size + gap
    width, height = measure_grid(rows, columns, style, size)

    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    background = module.background_color() if style == 'microsoft' else skia.ColorWHITE
    canvas.clear(background)
    font = font_for(font_family, size*font_scale, bold)

    # Each tile is drawn in its own lattice slot.
    for row in range(rows):
        for col in range(columns):
            x = col*pitch
            y = row*pitch
            state = int(cells[row, col])
            is_open = bool(opened[row, col])
            neighbors = _neighbors(opened, row, col)

            module.draw_surface(canvas, x, y, size, is_open, row, col, theme, neighbors)
            if style == 'google' and not is_open:
                google.draw_boundary(canvas, x, y, size, theme, neighbors, boundary_ratio)
            if style == 'microsoft' and state == CellState.FLAG:
                microsoft.draw_flag_surface(canvas, x, y, size)

            mine_under_flag = (
                style == 'windows7'
                and game['status'] == 'lost'
                and game['board'][row, col] == -1
            )
            _draw_content(canvas, x, y, size, state, font, palette, mine_under_flag,
                          style, mine_variant)

    # Keep annotations independent of the CNN's eventual label scheme.
    parameters = {
        'cell_size': size,
        'gap': gap,
        'pitch': pitch,
        'font_family': font_family,
        'font_scale': font_scale,
        'bold': bold,
        'digit_palette': list(palette),
        'mine_variant': mine_variant,
    }
    if style == 'google':
        parameters['boundary_ratio'] = boundary_ratio

    metadata = {
        'boxes': {'grid': (0, 0, width, height)},
        'rows': rows,
        'columns': columns,
        'style': style,
        'renderer': RENDERER_IDS[style],
        'source': 'synthetic',
        'theme': theme,
        'style_parameters': parameters,
        'generation_mode': game.get('generation_mode', 'normal'),
        'target_number': game.get('target_number'),
        'target_position': game.get('target_position'),
        'board_seed': game.get('seed'),
        'status': game['status'],
    }
    if cell_details:
        metadata.update(x_edges=_axis_edges(columns, pitch, gap),
                        y_edges=_axis_edges(rows, pitch, gap),
                        cell_boxes=_cell_boxes(rows, columns, size, pitch),
                        cell_states=cells.tolist(), opened_surfaces=opened.tolist())
    image = surface.makeImageSnapshot()
    return (image if native else image_to_pillow(image)), metadata


def render_game_region(game, **board_options):
    """Render game chrome and grid; see game_region.render_game_region."""
    from .game_region import render_game_region as render_region

    return render_region(game, **board_options)
