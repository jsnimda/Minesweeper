
"""Compose synthetic desktop screenshots using Skia."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import skia

from .desktop import draw_desktop_components
from .desktop_components.geometry import component_geometry
from .scaling import pixels
from .generic_window import (
    Box,
    draw_generic_window,
    window_size,
)


@dataclass(frozen=True)
class WindowLayer:
    content_image: skia.Image
    content_boxes: Mapping[str, Box]
    position: tuple[int, int]
    title: str = "Application"
    menu_items: tuple[str, ...] = ()
    allow_clipping: bool = False


def _work_area(
    size: tuple[int, int],
    desktop: dict,
    scale: float = 1.0,
) -> Box:
    """Conservative window placement area for each shell."""
    w, h = size
    geometry = component_geometry(size, desktop, scale)

    top, bottom = 0, h
    if 'taskbar' in geometry:
        extra = pixels(8, scale) if desktop['style'] == 'classic' else 0
        bottom = geometry['taskbar'][1] - extra
    if 'status_bar' in geometry:
        top = geometry['status_bar'][3]
    if 'dock' in geometry:
        bottom = geometry['dock'][1]
    if 'top_panel' in geometry:
        top = geometry['top_panel'][3]
    return (0, top, w, bottom)



def compose_screenshot(
    wallpaper: skia.Image,
    size: tuple[int, int],
    desktop: dict,
    windows: Sequence[WindowLayer],
    *,
    scale: float = 1.0,
) -> tuple[skia.Image, list[dict[str, Box]]]:
    """Return a Skia screenshot and per-window screen boxes.

    Window positions and bounding boxes use integer,
    half-open pixel coordinates.

    Layers marked allow_clipping may extend beyond the work area and surface;
    the surface clips them. Background windows may overlap.
    """
    width, height = size
    available = _work_area(size, desktop, scale)

    # Validate everything before drawing.
    for window in windows:
        x, y = window.position
        w, h = window_size(
            (window.content_image.width(), window.content_image.height()),
            window.menu_items,
            style=desktop["style"],
            scale=scale,
        )

        extent = (x, y, x + w, y + h)

        contained = (
            available[0] <= x < x + w <= available[2]
            and available[1] <= y < y + h <= available[3]
        )
        visible_origin = available[0] <= x < available[2] and available[1] <= y < available[3]
        if not contained and not (window.allow_clipping and visible_origin):
            raise ValueError(
                f"Window outside available area: {extent}"
            )

    # One surface shared by the desktop and all windows.
    surface = skia.Surface(width, height)

    if width < 320 or height < 240:
        raise ValueError("Desktop resolution is too small")
    if (wallpaper.width(), wallpaper.height()) != size:
        raise ValueError("Wallpaper must already match desktop dimensions")
    surface.getCanvas().drawImage(wallpaper, 0, 0)

    annotations = []

    for window in windows:
        boxes = draw_generic_window(
            surface=surface,
            position=window.position,
            content_image=window.content_image,
            content_boxes=window.content_boxes,
            title=window.title,
            menu_items=window.menu_items,
            desktop=desktop,
            scale=scale,
        )
        annotations.append(boxes)

    draw_desktop_components(surface, size, desktop, scale=scale)

    return surface.makeImageSnapshot(), annotations
