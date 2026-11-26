"""Lower-level desktop scaling, independent of the scene schema."""

import importlib

import pytest

skia = pytest.importorskip("skia")

from synthetic_scenes.desktop_rendering.desktop import draw_desktop
from synthetic_scenes.desktop_rendering.desktop_components.geometry import component_geometry
from synthetic_scenes.desktop_rendering.generic_window import draw_generic_window, window_geometry
from synthetic_scenes.desktop_rendering.scaling import pixels
from synthetic_scenes.image_conversion import image_to_pillow


SCALES = (1.0, 1.25, 1.5, 2.0)
STYLES = ("modern", "classic", "aero")


def desktop(style, shell, layout="centered"):
    return {"style": style, "shell": shell, "color_scheme": "light",
            "accent_color": "#4476BE", "taskbar_layout": layout}


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("shell,layout", [("taskbar", "centered"), ("taskbar", "left"),
                                          ("dock", "centered"), ("top_panel", "centered")])
def test_desktop_component_geometry_and_surface(scale, style, shell, layout):
    size = (1600, 1000)
    config = desktop(style, shell, layout)
    boxes = component_geometry(size, config, scale)
    if shell == "taskbar":
        box = boxes["taskbar"]
        assert box == (0, size[1] - pixels(36 if style == "classic" else 48, scale), *size)
    elif shell == "top_panel":
        assert boxes["top_panel"] == (0, 0, size[0], pixels(36, scale))
    else:
        assert boxes["status_bar"] == (0, 0, size[0], pixels(32, scale))
        x0, y0, x1, y1 = boxes["dock"]
        assert (x1 - x0, y1 - y0) == (pixels(344, scale), pixels(74, scale))
        assert size[1] - y1 == pixels(16, scale)
    wallpaper = skia.Surface(*size).makeImageSnapshot()
    surface = skia.Surface(*size)
    draw_desktop(surface, wallpaper, size, config, scale=scale)
    image = surface.makeImageSnapshot()
    assert (image.width(), image.height()) == size
    if scale == 1:
        default_surface = skia.Surface(*size)
        draw_desktop(default_surface, wallpaper, size, config)
        assert image_to_pillow(image).tobytes() == image_to_pillow(default_surface.makeImageSnapshot()).tobytes()


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("shell", ["taskbar", "dock", "top_panel"])
@pytest.mark.parametrize("menu", [(), ("File", "Edit")])
def test_frame_geometry_preserves_content(scale, style, shell, menu):
    content_size = (320, 240)
    window_size, offset = window_geometry(content_size, menu, style=style, scale=scale)
    base_size, base_offset = window_geometry(content_size, menu, style=style)
    if scale == 1:
        assert (window_size, offset) == (base_size, base_offset)
    else:
        assert window_size[0] >= base_size[0] and window_size[1] > base_size[1]
    surface = skia.Surface(window_size[0] + 40, window_size[1] + 40)
    content_surface = skia.Surface(*content_size)
    content_surface.getCanvas().clear(skia.ColorRED)
    position = (20, 20)
    boxes = draw_generic_window(surface, position, content_surface.makeImageSnapshot(),
                                {"content": (0, 0, *content_size)}, "Test", menu,
                                desktop(style, shell), scale=scale)
    x, y = (position[0] + offset[0], position[1] + offset[1])
    assert boxes["content"] == (x, y, x + content_size[0], y + content_size[1])
    image = image_to_pillow(surface.makeImageSnapshot()).convert("RGB")
    assert image.getpixel((x + 10, y + 10)) == (255, 0, 0)
    assert image.getpixel((x + content_size[0] - 10, y + content_size[1] - 10)) == (255, 0, 0)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("name", ["browser", "calculator", "code_editor", "document_editor",
                                   "file_manager", "media_viewer", "other_puzzle", "presentation",
                                   "settings", "spreadsheet", "sudoku", "terminal"])
def test_applications_scale_once(scale, name):
    module = importlib.import_module(f"synthetic_scenes.desktop_rendering.applications.{name}")
    render = getattr(module, f"render_{name}")
    base, _ = render(seed=12)
    result, _ = render(seed=12, scale=scale)
    assert (result.width(), result.height()) == (pixels(base.width(), scale), pixels(base.height(), scale))
    if scale == 1:
        assert image_to_pillow(result).tobytes() == image_to_pillow(base).tobytes()
    wrapped, _ = window_geometry((result.width(), result.height()), style="aero", scale=scale)
    assert wrapped[0] - result.width() == 2 * (pixels(6, scale) + pixels(1, scale))


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan"), True])
def test_invalid_scales_are_rejected(scale):
    with pytest.raises(ValueError):
        window_geometry((200, 200), scale=scale)
    with pytest.raises(ValueError):
        component_geometry((800, 600), desktop("modern", "taskbar"), scale)
