"""Desktop visibility, native dock geometry and placement contracts."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

skia = pytest.importorskip('skia')

from synthetic_scenes.desktop_rendering.desktop import draw_desktop
from synthetic_scenes.desktop_rendering.desktop_components.geometry import component_geometry, dock_layout
from synthetic_scenes.desktop_rendering.scaling import pixels
from synthetic_scenes.desktop_rendering.screenshot import _work_area
from synthetic_scenes.image_conversion import image_to_pillow
from synthetic_scenes.render import PlacementError, plan_game_windows, render_scene
from synthetic_scenes.schema import DesktopSpec


def desktop(shell='dock', **kwargs):
    return {'shell': shell, 'style': 'modern', 'color_scheme': 'light', **kwargs}


def scene(config, windows, scale=1.):
    return {'mode': 'desktop', 'seed': 3, 'layout_seed': 4,
            'screen': {'width': 640, 'height': 480, 'scale': scale},
            'desktop': config, 'wallpaper': {'category': 'procedural', 'seed': 6},
            'windows': windows}


def game(cell_size=20):
    return {'type': 'minesweeper', 'content': {
        'rows': 5, 'columns': 6, 'mine_count': 3, 'status': 'new', 'progress': 0.,
        'seed': 8, 'appearance': {'style': 'windows7', 'cell_size': cell_size, 'zoom': 1.25}}}


def test_schema_defaults_and_strict_visibility():
    model = DesktopSpec.model_validate(desktop())
    assert (model.show_taskbar, model.show_dock, model.show_status_bar, model.show_top_panel, model.dock_height) == (True, True, True, True, 74)
    for field in ('show_taskbar', 'show_dock', 'show_status_bar', 'show_top_panel'):
        for value in (True, False):
            assert getattr(DesktopSpec.model_validate(desktop(**{field: value})), field) is value
        for value in (0, 1, 'true', None, 1.0):
            with pytest.raises(ValidationError):
                DesktopSpec.model_validate(desktop(**{field: value}))


@pytest.mark.parametrize('height', [32, 48, 74, 96, 120])
def test_schema_accepts_dock_heights(height):
    assert DesktopSpec.model_validate(desktop(dock_height=height)).dock_height == height


@pytest.mark.parametrize('height', [31, 121, 74., True, False, '74', None])
def test_schema_rejects_invalid_dock_heights(height):
    with pytest.raises(ValidationError):
        DesktopSpec.model_validate(desktop(dock_height=height))


@pytest.mark.parametrize('style', ['modern', 'classic'])
@pytest.mark.parametrize('visible', [False, True])
def test_taskbar_visibility_and_classic_margin(style, visible):
    size, scale = (640, 480), 1.25
    config = desktop('taskbar', style=style, show_taskbar=visible)
    boxes = component_geometry(size, config, scale)
    area = _work_area(size, config, scale)
    if visible:
        assert area[3] == boxes['taskbar'][1] - (pixels(8, scale) if style == 'classic' else 0)
    else:
        assert boxes == {} and area == (0, 0, *size)


@pytest.mark.parametrize('dock', [False, True])
@pytest.mark.parametrize('status', [False, True])
def test_dock_and_status_visibility_are_independent(dock, status):
    size, scale = (640, 480), 1.5
    config = desktop(show_dock=dock, show_status_bar=status)
    geometry = component_geometry(size, config, scale)
    assert ('dock' in geometry, 'status_bar' in geometry) == (dock, status)
    assert _work_area(size, config, scale) == (
        0, pixels(32, scale) if status else 0, size[0],
        size[1] - pixels(4, scale) - pixels(74, scale) if dock else size[1])


@pytest.mark.parametrize('shell,hidden', [('taskbar', 'show_taskbar'), ('top_panel', 'show_top_panel')])
def test_hidden_shell_has_no_geometry_and_renders_no_component(shell, hidden):
    size = (320, 240)
    config = desktop(shell, **{hidden: False})
    assert component_geometry(size, config) == {}
    assert _work_area(size, config) == (0, 0, *size)
    wallpaper = skia.Surface(*size).makeImageSnapshot()
    surface = skia.Surface(*size)
    draw_desktop(surface, wallpaper, size, config)
    assert image_to_pillow(surface.makeImageSnapshot()).tobytes() == image_to_pillow(wallpaper).tobytes()


def test_irrelevant_settings_are_ignored():
    size = (640, 480)
    for shell in ('taskbar', 'dock', 'top_panel'):
        relevant = {'taskbar': 'show_taskbar', 'dock': 'show_dock', 'top_panel': 'show_top_panel'}[shell]
        options = {'show_taskbar': False, 'show_dock': False, 'show_status_bar': False,
                   'show_top_panel': False, 'dock_height': 120}
        if shell == 'dock':
            options.pop('show_status_bar')
            options.pop('dock_height')
        options.pop(relevant)
        assert component_geometry(size, desktop(shell, **options)) == component_geometry(size, desktop(shell))


@pytest.mark.parametrize('height', [32, 48, 74, 96, 120])
@pytest.mark.parametrize('scale', [.75, 1., 1.25, 1.5, 2.])
def test_native_dock_dimensions_and_reservation(height, scale):
    size = (1600, 900)
    config = desktop(dock_height=height)
    layout = dock_layout(size, config, scale)
    left, top, right, bottom = layout['box']
    ratio = height / 74
    assert layout['icon_size'] == pixels(42 * ratio, scale)
    assert layout['gap'] == pixels(12 * ratio, scale)
    assert layout['padding'] == pixels(16 * ratio, scale)
    assert len(layout['icons']) == 6
    assert right - left == 2 * layout['padding'] + 6 * layout['icon_size'] + 5 * layout['gap']
    assert bottom - top == pixels(height, scale)
    assert size[1] - bottom == pixels(4, scale)
    assert component_geometry(size, config, scale)['dock'] == layout['box']
    assert _work_area(size, config, scale)[3] == top


@pytest.mark.parametrize('width,expected', [(400, 6), (300, 5), (200, 3), (100, 1), (20, 1)])
def test_dock_fits_maximum_visible_icons(width, expected):
    layout = dock_layout((width, 480), desktop())
    left, _, right, _ = layout['box']
    assert len(layout['icons']) == expected
    assert 0 <= left < right <= width
    assert left == (width - (right - left)) // 2
    if width >= 74:
        assert all(left <= x0 < x1 <= right for x0, _, x1, _ in layout['icons'])
    else:
        assert right - left == width and layout['icons'][0][2] - layout['icons'][0][0] == 42
    assert layout == dock_layout((width, 480), desktop())


def test_dock_drawing_uses_resolved_icon_slots(monkeypatch):
    from synthetic_scenes.desktop_rendering.desktop_components import dock

    calls = []
    monkeypatch.setattr(dock, 'draw_app_icon', lambda canvas, kind, x, y, size: calls.append((kind, x, y, size)))
    size, scale = (320, 240), 2.
    config = desktop(show_status_bar=False, dock_height=96)
    layout = dock_layout(size, config, scale)
    surface = skia.Surface(*size)
    draw_desktop(surface, skia.Surface(*size).makeImageSnapshot(), size, config, scale=scale)
    assert [call[0] for call in calls] == list(dock.DOCK_ICONS[:len(layout['icons'])])
    for (_, x, y, icon_size), (left, top, right, _) in zip(calls, layout['icons']):
        assert (x * scale, y * scale, icon_size * scale) == (left, top, right - left)


@pytest.mark.parametrize('windows', [[], [{'type': 'browser', 'seed': 2}], [game()]])
def test_impossible_work_area_raises_even_for_empty_desktop(windows):
    with pytest.raises(PlacementError, match='no usable work area'):
        render_scene(scene(desktop(dock_height=120), windows, scale=5.))


def test_hidden_dock_expands_placement_area_and_keeps_annotations():
    request = scene(desktop(show_dock=False, show_status_bar=False), [game(), game(18)], scale=1.25)
    original = deepcopy(request)
    _, resolved, annotations = render_scene(request, details=['scene', 'cell'])
    assert request == original
    assert _work_area((640, 480), request['desktop'], 1.25) == (0, 0, 640, 480)
    boxes = [board['window_box'] for board in annotations['boards']]
    assert all(0 <= x < r <= 640 and 0 <= y < b <= 480 for x, y, r, b in boxes)
    a, b = boxes
    assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    assert len(resolved['layout']) == 2
    assert all(board['requested_zoom'] == 1.25 and board['requested_display_scale'] == 1.25 for board in annotations['boards'])
    assert _work_area((640, 480), desktop(), 1.25)[3] < 480


@pytest.mark.parametrize('show_dock,show_status', [(False, False), (False, True), (True, False), (True, True)])
def test_only_visible_dock_components_are_drawn(monkeypatch, show_dock, show_status):
    from synthetic_scenes.desktop_rendering.desktop_components import dock as dock_renderer

    panels, icons, status = [], [], []
    monkeypatch.setattr(dock_renderer, '_glass_panel', lambda surface, rect, **kwargs: panels.append(rect))
    monkeypatch.setattr(dock_renderer, 'draw_panel_content', lambda *args, **kwargs: status.append(True))
    monkeypatch.setattr(dock_renderer, 'draw_app_icon', lambda canvas, kind, *args: icons.append(kind))
    size = (640, 480)
    config = desktop(show_dock=show_dock, show_status_bar=show_status)
    surface = skia.Surface(*size)
    draw_desktop(surface, skia.Surface(*size).makeImageSnapshot(), size, config)
    assert len(panels) == int(show_dock) + int(show_status)
    assert bool(status) is show_status
    assert bool(icons) is show_dock


def test_hidden_dock_can_make_an_impossible_game_fit():
    from synthetic_scenes.minesweeper.geometry import measure_game_region
    from synthetic_scenes.desktop_rendering.generic_window import window_size

    window = game()
    content = window['content']
    size = measure_game_region(content['rows'], content['columns'], 'windows7', 12, 1.25)
    _, window_height = window_size(size, style='modern')
    screen = (640, max(240, window_height + 10))
    tall = desktop(dock_height=120, show_status_bar=False)
    hidden = desktop(dock_height=120, show_status_bar=False, show_dock=False)
    assert window_height > _work_area(screen, tall)[3]
    with pytest.raises(PlacementError):
        plan_game_windows([(0, content)], screen, tall, 4)
    plan = plan_game_windows([(0, content)], screen, hidden, 4)
    assert plan[0]['box'][3] <= screen[1]


def test_oversized_application_is_native_with_hidden_components():
    from synthetic_scenes.desktop_rendering.applications.browser import render_browser
    from synthetic_scenes.desktop_rendering.generic_window import window_size

    request = scene(desktop(show_dock=False, show_status_bar=False), [{'type': 'browser', 'seed': 2}], scale=1.25)
    image, resolved, _ = render_scene(request, details=['scene'])
    content, _ = render_browser(seed=2, scale=1.25)
    assert resolved['layout'][0]['window_size'] == window_size((content.width(), content.height()), style='modern', scale=1.25)
    assert resolved['layout'][0]['window_size'][0] > image.width


def test_omitted_defaults_match_explicit_geometry_and_rendering():
    size = (640, 480)
    implicit = desktop()
    explicit = desktop(show_taskbar=True, show_dock=True, show_status_bar=True,
                       show_top_panel=True, dock_height=74)
    assert component_geometry(size, implicit) == component_geometry(size, explicit)
    assert _work_area(size, implicit) == _work_area(size, explicit)
    assert dock_layout(size, implicit)['box'][2] - dock_layout(size, implicit)['box'][0] == 344
    wallpaper = skia.Surface(*size).makeImageSnapshot()
    surfaces = [skia.Surface(*size), skia.Surface(*size)]
    for surface, config in zip(surfaces, (implicit, explicit)):
        draw_desktop(surface, wallpaper, size, config)
    assert image_to_pillow(surfaces[0].makeImageSnapshot()).tobytes() == image_to_pillow(surfaces[1].makeImageSnapshot()).tobytes()


def test_extreme_scale_uses_minimal_clipped_dock():
    size, scale = (320, 480), 10.
    layout = dock_layout(size, desktop(dock_height=120), scale)
    assert layout['box'][0] == 0 and layout['box'][2] == size[0]
    assert len(layout['icons']) == 1
    assert layout['icon_size'] > size[0]
