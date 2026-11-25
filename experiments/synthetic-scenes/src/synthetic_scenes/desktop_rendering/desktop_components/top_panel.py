"""Linux-style top panel renderer for synthetic desktops."""

import skia

from .primitives import _color, _edge_panel, _paint, _round_rect, _text
from .system_icons import draw_wifi, draw_volume, draw_battery


def _draw_top_panel(surface, size, desktop, colors):
    canvas = surface.getCanvas()
    w, _ = size
    height = 36

    _edge_panel(
        surface, (0, 0, w, height), desktop, colors
    )

    text_color = colors["text"]
    center_y = height / 2

    # Left: Activities and workspace.
    _round_rect(
        canvas, 14, 11, 4, 14,
        2, colors["accent"],
    )

    _text(
        canvas, "Activities",
        26, 24, text_color, 13,
    )

    _text(
        canvas, "Workspace 1",
        125, 24, text_color, 12,
    )

    # Shared stroke style for system indicators.
    stroke = skia.Paint(
        Color=_color(text_color),
        AntiAlias=True,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=1.7,
        StrokeCap=skia.Paint.kRound_Cap,
        StrokeJoin=skia.Paint.kRound_Join,
    )

    # Right: shared Wi-Fi indicator.
    draw_wifi(
        canvas,
        w - 157,
        center_y,
        color=text_color,
        size=20,
    )

    # Shared volume indicator.
    draw_volume(
        canvas,
        w - 124,
        center_y,
        color=text_color,
        size=20,
    )

    # Shared battery indicator.
    draw_battery(
        canvas,
        w - 87,
        center_y,
        color=text_color,
        size=20,
        charge=0.75,
    )

    # Clock.
    _text(
        canvas,
        desktop.get("clock_time", "12:34"),
        w - 60, 24, text_color, 13,
    )