"""Integration contracts for physical desktop scaling."""
from copy import deepcopy
import random

import pytest

from synthetic_scenes.desktop_rendering.desktop_components.geometry import component_geometry
from synthetic_scenes.desktop_rendering.generic_window import window_geometry
from synthetic_scenes.desktop_rendering.scaling import pixels
from synthetic_scenes.desktop_rendering.screenshot import _work_area
from synthetic_scenes.minesweeper.geometry import measure_game_region
from synthetic_scenes.render import PlacementError, render_scene


def scene(style='classic', scale=1.5, zoom=1.25):
    return {
        'mode': 'desktop', 'seed': 5,
        'screen': {'width': 1600, 'height': 1000, 'scale': scale},
        'desktop': {'shell': 'taskbar', 'style': 'modern', 'color_scheme': 'light'},
        'wallpaper': {'category': 'procedural', 'seed': 9},
        'windows': [{'type': 'minesweeper', 'content': {
            'rows': 4, 'columns': 5, 'mine_count': 2, 'status': 'new',
            'progress': 0., 'seed': 10,
            'appearance': {'style': style, 'cell_size': 17, 'zoom': zoom},
        }}],
    }


@pytest.mark.parametrize('shell,style', [('taskbar', 'classic'), ('taskbar', 'modern'),
                                         ('dock', 'modern'), ('top_panel', 'modern')])
def test_work_area_uses_component_boundaries(shell, style):
    desktop = {'shell': shell, 'style': style, 'color_scheme': 'light'}
    size, scale = (1600, 1000), 1.5
    geometry = component_geometry(size, desktop, scale)
    area = _work_area(size, desktop, scale)
    if shell == 'taskbar':
        assert area[3] == geometry['taskbar'][1] - (pixels(8, scale) if style == 'classic' else 0)
    elif shell == 'dock':
        assert area[1] == geometry['status_bar'][3]
        assert area[3] == geometry['dock'][1]
    else:
        assert area[1] == geometry['top_panel'][3]


@pytest.mark.parametrize('style', ['classic', 'windows7', 'microsoft', 'google'])
@pytest.mark.parametrize('scale,zoom', [(1., 1.25), (1.5, 1.), (1.5, 1.25), (1.4, 1.25), (.2, .2)])
def test_combined_scale_and_physical_annotations(style, scale, zoom):
    request = scene(style, scale, zoom)
    original = deepcopy(request)
    image, resolved, annotations = render_scene(request, details=['scene', 'cell', 'mine'])
    assert request == original and resolved['windows'] == original['windows']
    assert image.size == (1600, 1000)
    board = annotations['boards'][0]
    parameters = board['style_parameters']
    assert board['requested_cell_size'] == board['resolved_cell_size'] == 17
    assert board['requested_zoom'] == zoom
    assert board['requested_display_scale'] == scale
    assert board['requested_effective_scale'] == scale * zoom
    assert parameters['cell_size'] == max(8, round(17 * zoom * scale))
    assert parameters['pitch_x'] == parameters['pitch_y'] == parameters['cell_size'] + parameters['gap']
    content_size = measure_game_region(4, 5, style, 17, zoom * scale)
    complete, offset = window_geometry(content_size, style='modern', scale=scale)
    x, y, right, bottom = board['window_box']
    assert (right-x, bottom-y) == complete
    assert board['boxes']['game_region'] == (x+offset[0], y+offset[1], x+offset[0]+content_size[0], y+offset[1]+content_size[1])
    gx, gy = board['grid_origin']
    assert board['cell_boxes'][0][0] == (gx, gy, gx+parameters['cell_size'], gy+parameters['cell_size'])
    assert len(board['mine_positions']) == 2


def test_combination_rounds_once():
    request = scene(scale=1.4, zoom=1.25)
    _, _, annotations = render_scene(request)
    size = annotations['boards'][0]['style_parameters']['cell_size']
    assert size == round(17 * 1.25 * 1.4) == 30
    assert size != round(round(17 * 1.25) * 1.4)


def test_wallpaper_resolution_does_not_depend_on_display_scale():
    request = scene(scale=1.)
    request['windows'] = []
    _, first, _ = render_scene(request, details=['scene'])
    request['screen']['scale'] = 1.5
    _, second, _ = render_scene(request, details=['scene'])
    assert first['resolved_wallpaper'] == second['resolved_wallpaper']


def test_fitting_preserves_scales_and_nonoverlap():
    request = scene(scale=1.5, zoom=1.25)
    request['screen'].update(width=1000, height=700)
    game = request['windows'][0]['content']
    game['appearance']['cell_size'] = 90
    request['windows'].append(deepcopy(request['windows'][0]))
    _, resolved, annotations = render_scene(request, details=['scene'])
    area = _work_area((1000, 700), request['desktop'], 1.5)
    boxes = [board['window_box'] for board in annotations['boards']]
    assert all(board['resolved_cell_size'] < 90 for board in annotations['boards'])
    assert all(board['requested_zoom'] == 1.25 and board['requested_display_scale'] == 1.5 for board in annotations['boards'])
    assert all(area[0] <= x < r <= area[2] and area[1] <= y < b <= area[3] for x, y, r, b in boxes)
    a, b = boxes
    assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    assert resolved['windows'] == request['windows']


def test_unusable_work_area_and_impossible_game():
    request = scene(scale=30)
    with pytest.raises(PlacementError):
        render_scene(request)
    request = scene(scale=1.5)
    request['windows'][0]['content'].update(rows=1000, columns=1000)
    with pytest.raises(PlacementError):
        render_scene(request)


def test_application_logical_size_and_frame_are_scaled_once():
    request = scene(scale=1.5)
    request['windows'] = [{'type': 'calculator', 'seed': 3}]
    image, resolved, _ = render_scene(request, details=['scene'])
    expected, _ = window_geometry((pixels(340, 1.5), pixels(520, 1.5)), style='modern', scale=1.5)
    assert resolved['layout'][0]['window_size'] == expected
    assert image.size == (1600, 1000)


@pytest.mark.parametrize('style', ['modern', 'classic'])
def test_oversized_application_retains_native_size_and_is_clipped(style):
    from synthetic_scenes.desktop_rendering.applications.browser import render_browser

    request = scene(scale=2.)
    request['desktop']['style'] = style
    request['screen'].update(width=1280, height=720)
    request['windows'] = [{'type': 'browser', 'seed': 3}]
    image, resolved, _ = render_scene(request, details=['scene'])
    native, _ = render_browser(seed=3, scale=2.)
    expected, _ = window_geometry((native.width(), native.height()), style=style, scale=2.)
    layout = resolved['layout'][0]
    left, top, right, bottom = _work_area((1280, 720), request['desktop'], 2.)
    assert layout['window_size'] == expected
    assert layout['position'] == (left, top)
    assert expected[0] > right-left and expected[1] > bottom-top
    assert image.size == (1280, 720)
    assert render_scene(request, details=['scene'])[1]['layout'] == resolved['layout']


def test_fitting_application_uses_seeded_position():
    request = scene(scale=1.)
    request['windows'] = [{'type': 'calculator', 'seed': 3}]
    _, resolved, _ = render_scene(request, details=['scene'])
    layout = resolved['layout'][0]
    left, top, right, bottom = _work_area((1600, 1000), request['desktop'], 1.)
    width, height = layout['window_size']
    rng = random.Random(3)
    assert layout['position'] == (rng.randint(left, right-width), rng.randint(top, bottom-height))


def test_oversized_application_keeps_random_position_on_fitting_axis():
    request = scene(scale=1.5)
    request['screen'].update(width=1600, height=480)
    request['windows'] = [{'type': 'calculator', 'seed': 3}]
    _, resolved, _ = render_scene(request, details=['scene'])
    layout = resolved['layout'][0]
    left, top, right, bottom = _work_area((1600, 480), request['desktop'], 1.5)
    width, height = layout['window_size']
    assert width <= right-left and height > bottom-top
    assert layout['position'] == (random.Random(3).randint(left, right-width), top)


def test_application_surface_limit_still_applies(monkeypatch):
    from synthetic_scenes.desktop_rendering.applications import browser

    class LargeLogicalImage:
        def width(self): return 10000
        def height(self): return 10000

    monkeypatch.setattr(browser, 'render_browser', lambda **kwargs: (LargeLogicalImage(), {}))
    request = scene(scale=5.)
    request['windows'] = [{'type': 'browser', 'seed': 3}]
    with pytest.raises(PlacementError, match='surface is too large'):
        render_scene(request)


def test_oversized_application_does_not_change_game_placement_or_annotations():
    request = scene(scale=1.5)
    request['screen'].update(width=640, height=480)
    request['windows'][0]['content'].update(rows=4, columns=5, mine_count=2)
    request['windows'][0]['content']['appearance']['cell_size'] = 12
    _, _, alone = render_scene(request, details=['cell'])
    request['windows'].append({'type': 'browser', 'seed': 3})
    image, _, together = render_scene(request, details=['cell'])
    assert image.size == (640, 480)
    assert together['boards'] == alone['boards']


@pytest.mark.parametrize('shell', ['taskbar', 'dock', 'top_panel'])
@pytest.mark.parametrize('style', ['modern', 'classic'])
def test_shell_components_are_composed_over_oversized_application(monkeypatch, shell, style):
    from PIL import ImageChops
    from synthetic_scenes.desktop_rendering import screenshot
    from synthetic_scenes.image_conversion import image_to_pillow

    events = []
    component_draw = screenshot.draw_desktop_components
    window_draw = screenshot.draw_generic_window

    def record_window(*args, **kwargs):
        events.append(kwargs['title'])
        return window_draw(*args, **kwargs)

    def record_components(surface, size, desktop, *, scale):
        events.append('components')
        before = image_to_pillow(surface.makeImageSnapshot()).convert('RGB')
        component_draw(surface, size, desktop, scale=scale)
        after = image_to_pillow(surface.makeImageSnapshot()).convert('RGB')
        for box in component_geometry(size, desktop, scale).values():
            assert ImageChops.difference(before.crop(box), after.crop(box)).getbbox() is not None

    monkeypatch.setattr(screenshot, 'draw_generic_window', record_window)
    monkeypatch.setattr(screenshot, 'draw_desktop_components', record_components)
    request = scene(scale=2.)
    request['screen'].update(width=1280, height=720)
    request['desktop'].update(shell=shell, style=style)
    request['windows'] = [{'type': 'browser', 'seed': 3}]
    game = scene()['windows'][0]
    game['content'].update(rows=4, columns=5, mine_count=2)
    game['content']['appearance']['cell_size'] = 12
    request['windows'].append(game)
    image, resolved, annotations = render_scene(request, details=['scene', 'cell'])
    assert resolved['layout'][0]['window_size'][1] > 720
    assert image.size == (1280, 720)
    assert len(annotations['boards']) == 1
    assert events == ['Browser', 'Minesweeper', 'components']
