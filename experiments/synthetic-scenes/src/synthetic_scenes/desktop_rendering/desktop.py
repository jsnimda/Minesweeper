
"""Skia desktop renderer for synthetic screenshots.

Renders the sampled shell and visual style.
Application windows are handled separately.
"""

import skia
from PIL import Image, ImageOps

from ..image_conversion import image_to_pillow, pillow_to_image

from .desktop_components.dock import _draw_dock
from .desktop_components.taskbar import _draw_taskbar
from .desktop_components.top_panel import _draw_top_panel


def _palette(desktop: dict) -> dict:
    """Derive coherent surface colors from the sampled desktop."""
    style = desktop["style"]
    scheme = desktop["color_scheme"]

    dark = scheme == "dark"

    if style == "classic":
        return {
            "surface": "#D5D7DC",
            "text": "#253044",
            "subtle": "#EEF0F3",
            "accent": desktop["accent_color"] or "#4476BE",
            "alpha": 250,
            "radius": 4,
        }

    if dark:
        return {
            "surface": "#20242D",
            "text": "#F4F6FA",
            "subtle": "#404754",
            "accent": desktop["accent_color"] or "#639DE2",
            "alpha": 220 if style == "modern" else 185,
            "radius": 16,
        }

    return {
        "surface": "#F4F6F9",
        "text": "#202A38",
        "subtle": "#DCE3EB",
        "accent": desktop["accent_color"] or "#477FC0",
        "alpha": 230 if style == "modern" else 185,
        "radius": 16,
    }


def draw_desktop(
    surface: skia.Surface,
    wallpaper: skia.Image,
    size: tuple[int, int],
    desktop: dict,
    *,
    scale: float = 1.0,
) -> None:
    """Draw the desktop on an existing Skia surface."""
    w, h = size
    if w < 320 or h < 240:
        raise ValueError("Desktop resolution is too small")

    shell = desktop["shell"]
    if shell not in {"taskbar", "dock", "top_panel"}:
        raise ValueError(f"Unsupported shell: {shell}")

    canvas = surface.getCanvas()
    if (wallpaper.width(), wallpaper.height()) != size:
        raise ValueError("Wallpaper must already match desktop dimensions")
    canvas.drawImage(wallpaper, 0, 0)

    draw_desktop_components(surface, size, desktop, scale=scale)


def draw_desktop_components(
    surface: skia.Surface,
    size: tuple[int, int],
    desktop: dict,
    *,
    scale: float = 1.0,
) -> None:
    """Draw the shell over the surface's current backdrop."""
    shell = desktop["shell"]
    if shell not in {"taskbar", "dock", "top_panel"}:
        raise ValueError(f"Unsupported shell: {shell}")

    colors = _palette(desktop)

    if shell == "taskbar":
        _draw_taskbar(surface, size, desktop, colors, scale=scale)
    elif shell == "dock":
        _draw_dock(surface, size, desktop, colors, scale=scale)
    else:
        _draw_top_panel(surface, size, desktop, colors, scale=scale)


def render_desktop(
    wallpaper: Image.Image,
    size: tuple[int, int],
    desktop: dict,
    *,
    scale: float = 1.0,
) -> Image.Image:
    """Standalone convenience renderer for notebook previews."""
    surface = skia.Surface(*size)
    if wallpaper.size != size:
        wallpaper = ImageOps.fit(wallpaper, size, method=Image.Resampling.LANCZOS)
    draw_desktop(surface, pillow_to_image(wallpaper), size, desktop, scale=scale)
    return image_to_pillow(surface.makeImageSnapshot())
