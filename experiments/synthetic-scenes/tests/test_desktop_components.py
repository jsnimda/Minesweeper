"""Desktop component layout and palette contracts."""

import pytest

skia = pytest.importorskip('skia')

from synthetic_scenes.desktop_rendering.desktop import _component_palette, _palette, draw_desktop
from synthetic_scenes.desktop_rendering.desktop_components.geometry import component_geometry
from synthetic_scenes.desktop_rendering.desktop_components.taskbar import _taskbar_layout
from synthetic_scenes.desktop_rendering.scaling import pixels
from synthetic_scenes.desktop_rendering.screenshot import _work_area
from synthetic_scenes.image_conversion import image_to_pillow


def desktop(style, shell, scheme='light', accent=None):
    return {'style': style, 'shell': shell, 'color_scheme': scheme, 'accent_color': accent}


def test_classic_component_palette_and_frame_palette_are_independent():
    default = desktop('classic', 'taskbar', 'default')
    light = desktop('classic', 'taskbar', 'light')
    dark = desktop('classic', 'taskbar', 'dark')
    assert _component_palette(default) == _component_palette(light)
    assert _component_palette(dark)['surface'] != _component_palette(light)['surface']
    assert _component_palette(dark)['text'] != _component_palette(light)['text']
    assert _palette(dark) == _palette(light)
    dark['accent_color'] = '#AC4D72'
    assert _component_palette(dark)['accent'] == '#AC4D72'


@pytest.mark.parametrize('style', ['modern', 'classic'])
@pytest.mark.parametrize('width', [30, 80, 160, 240, 320, 800])
def test_taskbar_layout_stays_outside_tray(style, width):
    tray_left, launcher, slots = _taskbar_layout(width, style)
    assert 0 <= tray_left <= width
    if launcher:
        assert 0 <= launcher[0] < launcher[1] <= tray_left
    for left, right in slots:
        assert 0 <= left < right <= tray_left
        if style == 'classic':
            assert right - left >= 80
    assert all(a[1] <= b[0] for a, b in zip(slots, slots[1:]))
    if launcher and slots:
        assert launcher[1] <= slots[0][0]


def test_taskbar_hides_icons_before_squeezing_them():
    for style in ('modern', 'classic'):
        assert len(_taskbar_layout(160, style)[2]) < len(_taskbar_layout(800, style)[2])


@pytest.mark.parametrize('shell', ['taskbar', 'dock', 'top_panel'])
@pytest.mark.parametrize('style,scheme', [('modern', 'light'), ('modern', 'dark'),
                                          ('classic', 'default'), ('classic', 'light'),
                                          ('classic', 'dark')])
@pytest.mark.parametrize('scale', [.75, 1., 1.25, 1.5, 2.])
def test_components_keep_physical_geometry(shell, style, scheme, scale):
    size = (800, 600)
    config = desktop(style, shell, scheme, '#AC4D72')
    geometry = component_geometry(size, config, scale)
    area = _work_area(size, config, scale)
    if shell == 'taskbar':
        box = geometry['taskbar']
        assert box[3] - box[1] == pixels(36 if style == 'classic' else 48, scale)
        assert area[3] == box[1] - (pixels(8, scale) if style == 'classic' else 0)
    elif shell == 'top_panel':
        assert geometry['top_panel'][3] == pixels(36, scale) == area[1]
    else:
        left, top, right, bottom = geometry['dock']
        assert (right - left, bottom - top) == (6 * pixels(42, scale) + 5 * pixels(12, scale) + 2 * pixels(16, scale), pixels(74, scale))
        assert size[1] - bottom == pixels(4, scale)
        assert area[1] == pixels(32, scale)
        assert area[3] == top
    wallpaper = skia.Surface(*size).makeImageSnapshot()
    surface = skia.Surface(*size)
    draw_desktop(surface, wallpaper, size, config, scale=scale)
    assert (surface.width(), surface.height()) == size


@pytest.mark.parametrize('shell', ['taskbar', 'dock', 'top_panel'])
@pytest.mark.parametrize('style', ['modern', 'classic'])
@pytest.mark.parametrize('scale', [1., 2., 4., 10.])
def test_constrained_screens_render_without_invalid_geometry(shell, style, scale):
    size = (320, 240)
    config = desktop(style, shell, 'dark')
    surface = skia.Surface(*size)
    wallpaper = skia.Surface(*size).makeImageSnapshot()
    draw_desktop(surface, wallpaper, size, config, scale=scale)
    assert image_to_pillow(surface.makeImageSnapshot()).size == size


@pytest.mark.parametrize('shell', ['dock', 'top_panel'])
def test_narrow_panel_omits_labels_and_indicators(monkeypatch, shell):
    from synthetic_scenes.desktop_rendering.desktop_components import panel_content

    texts, indicators = [], []
    monkeypatch.setattr(panel_content, '_text', lambda canvas, value, *args: texts.append(value))
    for name in ('draw_wifi', 'draw_volume', 'draw_battery'):
        monkeypatch.setattr(panel_content, name, lambda *args, **kwargs: indicators.append(True))
    size = (320, 240)
    surface = skia.Surface(*size)
    draw_desktop(surface, skia.Surface(*size).makeImageSnapshot(), size,
                 desktop('classic', shell, 'dark'), scale=2.)
    assert texts == ['12:34']
    assert not indicators


def test_classic_dark_taskbar_renders_a_distinct_surface():
    images = []
    for scheme in ('default', 'light', 'dark'):
        size = (800, 600)
        surface = skia.Surface(*size)
        wallpaper = skia.Surface(*size).makeImageSnapshot()
        draw_desktop(surface, wallpaper, size, desktop('classic', 'taskbar', scheme))
        images.append(image_to_pillow(surface.makeImageSnapshot()).convert('RGB').getpixel((500, 570)))
    assert images[0] == images[1]
    assert images[2] != images[1]
