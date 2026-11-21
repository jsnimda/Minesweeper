
"""Skia desktop renderer for synthetic screenshots.

Renders the sampled shell and visual style.
Application windows are handled separately.
"""

from io import BytesIO

import skia
from PIL import Image

from .desktop_components.dock import _draw_dock
from .desktop_components.taskbar import _draw_taskbar
from .desktop_components.top_panel import _draw_top_panel


def _cover(wallpaper: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize and center-crop without distorting the wallpaper."""
    w, h = size
    scale = max(w / wallpaper.width, h / wallpaper.height)

    scaled = wallpaper.convert("RGBA").resize(
        (
            max(w, round(wallpaper.width * scale)),
            max(h, round(wallpaper.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )

    left = (scaled.width - w) // 2
    top = (scaled.height - h) // 2
    return scaled.crop((left, top, left + w, top + h))


def _skia_image(image: Image.Image) -> skia.Image:
    image = image.convert("RGBA")
    return skia.Image.frombytes(
        image.tobytes(),
        image.size,
        skia.kRGBA_8888_ColorType,
    )


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
    wallpaper: Image.Image,
    size: tuple[int, int],
    desktop: dict,
) -> None:
    """Draw the desktop on an existing Skia surface."""
    w, h = size
    if w < 320 or h < 240:
        raise ValueError("Desktop resolution is too small")

    shell = desktop["shell"]
    if shell not in {"taskbar", "dock", "top_panel"}:
        raise ValueError(f"Unsupported shell: {shell}")

    canvas = surface.getCanvas()
    canvas.drawImage(_skia_image(_cover(wallpaper, size)), 0, 0)

    colors = _palette(desktop)

    if shell == "taskbar":
        _draw_taskbar(surface, size, desktop, colors)
    elif shell == "dock":
        _draw_dock(surface, size, desktop, colors)
    else:
        _draw_top_panel(surface, size, desktop, colors)


def surface_to_pillow(surface: skia.Surface) -> Image.Image:
    """Convert the final canvas into an RGB Pillow image."""
    snapshot = surface.makeImageSnapshot()
    return Image.open(
        BytesIO(bytes(snapshot.encodeToData()))
    ).convert("RGB")


def render_desktop(
    wallpaper: Image.Image,
    size: tuple[int, int],
    desktop: dict,
) -> Image.Image:
    """Standalone convenience renderer for notebook previews."""
    surface = skia.Surface(*size)
    draw_desktop(surface, wallpaper, size, desktop)
    return surface_to_pillow(surface)
