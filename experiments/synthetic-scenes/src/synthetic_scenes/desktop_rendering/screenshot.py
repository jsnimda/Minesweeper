
"""Compose synthetic desktop screenshots using Skia."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import skia

from .desktop import draw_desktop
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


def _work_area(
    size: tuple[int, int],
    desktop: dict,
) -> Box:
    """Conservative window placement area for each shell."""
    w, h = size
    shell = desktop["shell"]

    if shell == "taskbar":
        bottom_reserved = (
            44 if desktop["style"] == "classic" else 48
        )
        return (0, 0, w, h - bottom_reserved)

    if shell == "dock":
        # Top menu bar and floating bottom dock.
        return (0, 32, w, h - 104)

    if shell == "top_panel":
        return (0, 36, w, h)

    raise ValueError(f"Unknown shell: {shell}")


def compose_screenshot(
    wallpaper: skia.Image,
    size: tuple[int, int],
    desktop: dict,
    windows: Sequence[WindowLayer],
) -> tuple[skia.Image, list[dict[str, Box]]]:
    """Return a Skia screenshot and per-window screen boxes.

    Window positions and bounding boxes use integer,
    half-open pixel coordinates.

    Window clipping is not supported; background windows may overlap.
    """
    width, height = size
    available = _work_area(size, desktop)

    # Validate everything before drawing.
    for window in windows:
        x, y = window.position
        w, h = window_size(
            (window.content_image.width(), window.content_image.height()),
            window.menu_items,
            style=desktop["style"],
        )

        extent = (x, y, x + w, y + h)

        if not (
            available[0] <= x < x + w <= available[2]
            and available[1] <= y < y + h <= available[3]
        ):
            raise ValueError(
                f"Window outside available area: {extent}"
            )

    # One surface shared by the desktop and all windows.
    surface = skia.Surface(width, height)

    draw_desktop(
        surface=surface,
        wallpaper=wallpaper,
        size=size,
        desktop=desktop,
    )

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
        )
        annotations.append(boxes)

    return surface.makeImageSnapshot(), annotations
