"""Native zoom uses the same geometry for rendering, placement and labels."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from synthetic_scenes.desktop_rendering.generic_window import window_geometry
from synthetic_scenes.minesweeper.board import generate_game
from synthetic_scenes.minesweeper.geometry import (
    cell_gap, effective_cell_size, measure_game_region, measure_grid, region_layout,
)
from synthetic_scenes.minesweeper.renderer import render_board, render_game_region
from synthetic_scenes.render import PlacementError, render_scene
from synthetic_scenes.schema import GameAppearance


STYLES = ('classic', 'windows7', 'microsoft', 'google')


def scene(mode, style='classic', zoom=1.0, cell=20):
    result = {'mode': mode, 'seed': 7, 'windows': [{'type': 'minesweeper', 'content': {
        'rows': 4, 'columns': 5, 'mine_count': 3, 'status': 'new', 'progress': 0.,
        'seed': 4, 'appearance': {'style': style, 'cell_size': cell, 'zoom': zoom},
    }}]}
    if mode == 'desktop':
        result.update(screen={'width': 640, 'height': 480},
                      desktop={'shell': 'taskbar', 'style': 'modern', 'color_scheme': 'light'},
                      wallpaper={'category': 'procedural', 'seed': 2})
    return result


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), -float('inf'), True, False, '1.5', None, [], {}])
def test_invalid_zoom(value):
    with pytest.raises(ValidationError):
        GameAppearance(zoom=value)
    with pytest.raises(ValueError):
        effective_cell_size(20, value)


def test_zoom_default_and_types():
    assert GameAppearance().zoom == 1.0
    assert GameAppearance(zoom=2).zoom == 2
    assert GameAppearance(zoom=.75).zoom == .75
    assert effective_cell_size(20, .001) == 8
    assert effective_cell_size(18, 1.25) == 22  # ties to even


@pytest.mark.parametrize('style', STYLES)
@pytest.mark.parametrize('zoom', [.75, 1.0, 1.25, 1.5, 2.0])
def test_zoomed_grid_and_panel_geometry(style, zoom):
    game = generate_game(rows=2, columns=3, mine_count=0, status='new', seed=4)
    size = effective_cell_size(20, zoom)
    gap = cell_gap(style, size)
    pitch = size + gap
    grid, grid_meta = render_board(game, style=style, cell_size=20, zoom=zoom)
    panel, meta = render_game_region(game, style=style, cell_size=20, zoom=zoom)
    assert grid.size == measure_grid(2, 3, style, 20, zoom) == (3*pitch-gap, 2*pitch-gap)
    assert panel.size == measure_game_region(2, 3, style, 20, zoom)
    layout = region_layout(grid.size, style, size, zoom)
    gx, gy = layout['grid_origin']
    assert gx == (panel.width-grid.width)//2
    assert panel.crop(meta['boxes']['grid']).tobytes() == grid.tobytes()
    assert meta['cell_boxes'][1][1] == (gx+pitch, gy+pitch, gx+pitch+size, gy+pitch+size)
    params = grid_meta['style_parameters']
    assert (params['cell_width'], params['cell_height'], params['pitch_x'], params['pitch_y']) == (size, size, pitch, pitch)
    assert params['base_cell_size'] == 20 and params['zoom'] == zoom
    assert meta['boxes']['mine_counter'] == layout['mine_counter']
    assert meta['boxes']['timer'] == layout['timer']
    if zoom == 1.0:
        original, original_meta = render_game_region(game, style=style, cell_size=20)
        assert original.tobytes() == panel.tobytes()
        assert original_meta['boxes'] == meta['boxes']


@pytest.mark.parametrize('style', STYLES)
def test_zoom_scales_controls_independently_of_base_cell(style):
    layouts = []
    for zoom in (.75, 1.0, 2.0):
        effective = effective_cell_size(20, zoom)
        grid = measure_grid(2, 3, style, 20, zoom)
        layouts.append(region_layout(grid, style, effective, zoom))
    heights = [layout['timer'][3]-layout['timer'][1] for layout in layouts]
    assert heights[0] < heights[1] < heights[2]
    assert all(layout['size'][0] >= layout['minimum_width'] for layout in layouts)


@pytest.mark.parametrize('style', STYLES)
@pytest.mark.parametrize('mode', ['grid', 'game', 'desktop'])
def test_scene_zoom_annotations_and_requested_spec(style, mode):
    request = scene(mode, style, 1.5)
    original = deepcopy(request)
    image, resolved, annotations = render_scene(request, details=['scene', 'cell', 'mine'])
    assert request == original and resolved['windows'] == original['windows']
    board = annotations['boards'][0]
    params = board['style_parameters']
    pitch = params['pitch']
    assert board['requested_cell_size'] == board['resolved_cell_size'] == 20
    assert board['requested_zoom'] == 1.5 and params['cell_size'] == 30
    gx, gy, right, bottom = board['boxes']['grid']
    assert board['grid_origin'] == (gx, gy)
    assert board['cell_boxes'][1][1] == (gx+pitch, gy+pitch, gx+pitch+30, gy+pitch+30)
    assert right-gx == 5*pitch-params['gap'] and bottom-gy == 4*pitch-params['gap']
    assert len(board['mine_positions']) == 3
    assert all(0 <= row < 4 and 0 <= col < 5 for row, col in board['mine_positions'])
    if mode == 'desktop':
        layout = resolved['layout'][0]
        x, y, r, b = layout['box']
        content = measure_game_region(4, 5, style, 20, 1.5)
        window, offset = window_geometry(content, style='modern')
        assert (r-x, b-y) == window
        assert board['boxes']['game_region'] == (x+offset[0], y+offset[1], x+offset[0]+content[0], y+offset[1]+content[1])
        assert image.size == (640, 480)


def test_desktop_fits_by_reducing_base_cell_and_keeps_zoom():
    request = scene('desktop', zoom=2.0, cell=64)
    minimum_content = measure_game_region(4, 5, 'classic', 12, 2.0)
    minimum_window, _ = window_geometry(minimum_content, style='modern')
    request['screen'] = {'width': minimum_window[0], 'height': minimum_window[1] + 48}
    original = deepcopy(request)
    _, resolved, annotations = render_scene(request, details=['scene'])
    board = annotations['boards'][0]
    placement = resolved['layout'][0]
    assert request == original and resolved['windows'] == original['windows']
    assert resolved['windows'][0]['content']['appearance']['cell_size'] == 64
    assert 12 <= placement['cell_size'] < 64
    assert board['requested_cell_size'] == 64
    assert board['resolved_cell_size'] == placement['cell_size']
    params = board['style_parameters']
    assert params['cell_size'] == params['cell_width'] == params['cell_height'] == 2*placement['cell_size']
    assert board['requested_zoom'] == params['zoom'] == 2.0
    left, top, right, bottom = placement['box']
    assert 0 <= left < right <= request['screen']['width']
    assert 0 <= top < bottom <= request['screen']['height'] - 48


def test_impossible_zoomed_placement():
    request = scene('desktop', zoom=20, cell=20)
    with pytest.raises(PlacementError):
        render_scene(request)


@pytest.mark.parametrize('style', STYLES)
def test_very_small_zoom_retains_drawable_layout(style):
    game = generate_game(rows=2, columns=3, mine_count=0, status='new', seed=4)
    image, meta = render_game_region(game, style=style, cell_size=20, zoom=.001)
    assert meta['style_parameters']['cell_size'] == 8
    for x, y, right, bottom in meta['boxes'].values():
        assert 0 <= x < right <= image.width
        assert 0 <= y < bottom <= image.height


def test_zoom_does_not_scale_external_frame_or_desktop():
    chrome = []
    for zoom in (1.0, 2.0):
        request = scene('desktop', zoom=zoom)
        image, resolved, annotations = render_scene(request, details=['scene'])
        board = annotations['boards'][0]
        x, y, right, bottom = resolved['layout'][0]['box']
        left, top, cr, cb = board['boxes']['game_region']
        chrome.append((left-x, top-y, right-cr, bottom-cb))
        assert image.size == (640, 480)
    assert chrome[0] == chrome[1]


def test_extreme_zoom_cannot_overflow_desktop_planning():
    request = scene('desktop', zoom=1e308)
    with pytest.raises(PlacementError):
        render_scene(request)
