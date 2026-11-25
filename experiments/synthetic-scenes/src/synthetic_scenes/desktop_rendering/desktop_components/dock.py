"""Dock and menu bar renderer for synthetic desktops."""

import skia

from .icons import draw_app_icon
from .primitives import (
    _color,
    _glass_panel,
    _paint,
    _round_rect,
    _text,
)
from .system_icons import draw_wifi, draw_battery


def _draw_dock(surface, size, desktop, colors):
    canvas = surface.getCanvas()
    w, h = size

    # Thin, edge-attached menu bar instead of a rounded capsule.
    _glass_panel(
        surface,
        (0, 0, w, 32),
        radius=0,
        blur=9,
        tint=colors["surface"],
        opacity=205 if desktop["style"] == "modern" else 145,
    )
    # Application menu on the left.
    # Use a simple original system symbol.
    system_x = 22
    system_y = 16

    canvas.drawCircle(
        system_x,
        system_y,
        7,
        _paint(colors["text"]),
    )

    # Small notch at the top of the symbol.
    canvas.drawCircle(
        system_x + 3,
        system_y - 6,
        2.2,
        _paint(colors["surface"]),
    )

    menus = [
        ("Finder", 43),
        ("File", 108),
        ("Edit", 146),
        ("View", 185),
        ("Go", 231),
        ("Window", 265),
        ("Help", 331),
    ]

    for label, x in menus:
        _text(
            canvas,
            label,
            x,
            22,
            colors["text"],
            13,
        )

    # System indicators on the right.
    indicator_color = colors["text"]

    # Shared Wi-Fi indicator.
    draw_wifi(
        canvas,
        w - 157,
        16,
        color=indicator_color,
        size=20,
    )

    # Shared battery indicator.
    draw_battery(
        canvas,
        w - 113,
        16,
        color=indicator_color,
        size=20,
        charge=0.75,
    )

    # Clock.
    _text(
        canvas,
        desktop.get("clock_time", "12:34"),
        w - 66,
        22,
        indicator_color,
        13,
    )

    count = 6
    icon_size = 42
    gap = 12
    padding = 16

    dock_width = count * icon_size + (count - 1) * gap + padding * 2
    dock_height = icon_size + padding * 2

    x0 = (w - dock_width) // 2
    y0 = h - dock_height - 16

    _glass_panel(
        surface,
        (x0, y0, dock_width, dock_height),
        radius=18,
        blur=16,
        tint=colors["surface"],
        opacity=185 if desktop["style"] == "modern" else 125,
    )

    icons = [
        "finder",
        "browser",
        "terminal",
        "code_editor",
        "settings",
        "trash",
    ]

    for i, kind in enumerate(icons):
        x = x0 + padding + i * (icon_size + gap)
        y = y0 + padding

        draw_app_icon(canvas, kind, x, y, icon_size)

        if i == 1:
            canvas.drawCircle(
                x + icon_size / 2,
                y0 + dock_height - 7,
                2.5,
                _paint(colors["accent"]),
            )