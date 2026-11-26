"""External chrome must preserve content and its physical geometry."""

import pytest
from PIL import Image

skia = pytest.importorskip('skia')

from synthetic_scenes.desktop_rendering.generic_window import (
    _caption_layout, draw_generic_window, window_geometry,
)
from synthetic_scenes.desktop_rendering.screenshot import WindowLayer, compose_screenshot
from synthetic_scenes.image_conversion import image_to_pillow, pillow_to_image


def desktop(style, shell, scheme):
    return {'style': style, 'shell': shell, 'color_scheme': scheme, 'accent_color': '#577DB2'}


def patterned(size):
    image = Image.new('RGB', size)
    image.putdata([((x * 41 + y * 7) % 256, (x * 11 + y * 31) % 256,
                    (x * 19 + y * 23) % 256)
                   for y in range(size[1]) for x in range(size[0])])
    return image


@pytest.mark.parametrize('style', ['modern', 'classic'])
@pytest.mark.parametrize('shell', ['taskbar', 'dock', 'top_panel'])
@pytest.mark.parametrize('scheme', ['light', 'dark'])
@pytest.mark.parametrize('scale', [.75, 1., 1.25, 1.5, 2.])
@pytest.mark.parametrize('menu', [(), ('File', 'An unusually long menu label', 'Help')])
@pytest.mark.parametrize('content_size', [(1, 1), (9, 7), (79, 43)])
def test_opaque_content_is_preserved_exactly(style, shell, scheme, scale, menu, content_size):
    content = patterned(content_size)
    physical_size, origin = window_geometry(content_size, menu, style=style, scale=scale)
    position = (13, 17)
    surface = skia.Surface(physical_size[0] + 30, physical_size[1] + 40)
    surface.getCanvas().clear(skia.ColorBLUE)
    boxes = draw_generic_window(surface, position, pillow_to_image(content),
                                {'all': (0, 0, *content_size)}, 'A very long title', menu,
                                desktop(style, shell, scheme), scale=scale)
    x, y = position[0] + origin[0], position[1] + origin[1]
    assert boxes['all'] == (x, y, x + content_size[0], y + content_size[1])
    output = image_to_pillow(surface.makeImageSnapshot())
    assert output.crop(boxes['all']).tobytes() == content.tobytes()


@pytest.mark.parametrize('scheme,body', [('light', '#FAFBFC'), ('dark', '#252A33')])
def test_transparent_content_composites_over_window_body(scheme, body):
    content = Image.new('RGBA', (12, 8), (240, 20, 100, 96))
    content.putpixel((0, 0), (10, 220, 20, 0))
    native = pillow_to_image(content)
    size, (x, y) = window_geometry(content.size, style='modern', scale=1.25)
    surface = skia.Surface(*size)
    surface.getCanvas().clear(skia.ColorBLUE)
    draw_generic_window(surface, (0, 0), native, {}, 'Test', desktop=desktop('modern', 'dock', scheme), scale=1.25)
    expected = skia.Surface(*content.size)
    expected.getCanvas().clear(skia.ColorSetRGB(*bytes.fromhex(body[1:])))
    expected.getCanvas().drawImage(native, 0, 0)
    actual = image_to_pillow(surface.makeImageSnapshot()).crop((x, y, x + content.width, y + content.height))
    assert actual.tobytes() == image_to_pillow(expected.makeImageSnapshot()).tobytes()


@pytest.mark.parametrize('style', ['modern', 'classic'])
@pytest.mark.parametrize('shell', ['taskbar', 'dock', 'top_panel'])
@pytest.mark.parametrize('width', [1, 12, 24, 32, 48, 80, 180])
def test_caption_layout_is_bounded_and_nonoverlapping(style, shell, width):
    height = 32 if style == 'classic' else 40
    kind, controls, (title_left, title_right) = _caption_layout(width, height, style, shell)
    assert kind == ('classic' if style == 'classic' else 'dock' if shell == 'dock' else shell)
    assert 0 <= title_left <= title_right <= width
    for _, (left, top, right, bottom) in controls:
        assert 0 <= left < right <= width and 0 <= top < bottom <= height
        if title_right > title_left:
            assert right <= title_left or left >= title_right
    rectangles = sorted(box for _, box in controls)
    assert all(a[2] <= b[0] for a, b in zip(rectangles, rectangles[1:]))
    if controls:
        assert controls[0][0] == 'close'


@pytest.mark.parametrize('style', ['modern', 'classic'])
def test_default_and_light_match_and_dark_differs(style):
    outputs = []
    for scheme in ('default', 'light', 'dark'):
        content = pillow_to_image(patterned((100, 40)))
        size, _ = window_geometry((100, 40), style=style)
        surface = skia.Surface(*size)
        surface.getCanvas().clear(skia.ColorWHITE)
        draw_generic_window(surface, (0, 0), content, {}, 'Test', desktop=desktop(style, 'dock', scheme))
        outputs.append(image_to_pillow(surface.makeImageSnapshot()).getpixel((5, 10)))
    assert outputs[0] == outputs[1] and outputs[2] != outputs[1]


@pytest.mark.parametrize('style', ['modern', 'classic'])
def test_screenshot_composition_preserves_window_content(style):
    content = patterned((80, 40))
    wallpaper = skia.Surface(320, 240).makeImageSnapshot()
    cfg = desktop(style, 'taskbar', 'light')
    layer = WindowLayer(pillow_to_image(content), {'all': (0, 0, *content.size)}, (10, 10), title='Test')
    image, annotations = compose_screenshot(wallpaper, (320, 240), cfg, [layer], scale=1.25)
    _, offset = window_geometry(content.size, style=style, scale=1.25)
    expected = (10 + offset[0], 10 + offset[1], 10 + offset[0] + content.width,
                10 + offset[1] + content.height)
    assert annotations[0]['all'] == expected
    assert image_to_pillow(image).crop(expected).tobytes() == content.tobytes()
