
"""Taskbar renderer for modern, Aero, and Classic desktop styles."""

import math

import skia

from ..scaling import validate_scale
from .geometry import component_geometry

from .flags import draw_waving_flag
from .icons import draw_app_icon
from .primitives import _color, _paint, _round_rect, _edge_panel, _text
from .system_icons import draw_volume, draw_battery


def _draw_system_tray(canvas, w, y, height, desktop, colors):
    """Draw style-specific taskbar notification controls."""
    style = desktop["style"]
    layout = desktop.get("taskbar_layout") or "centered"
    text_color = colors["text"]

    clock_time = desktop.get("clock_time", "12:34")
    clock_date = desktop.get("clock_date", "11/21/2026")

    if style == "classic":
        preferred_width = 178
        background = "#A9C7E5"
        background_alpha = 255
    elif style == "aero":
        preferred_width = 180
        background = "#C4DDEC"
        background_alpha = 75
    elif layout == "left":
        preferred_width = 198
        background = colors["surface"]
        background_alpha = 0
    else:
        preferred_width = 188
        background = colors["surface"]
        background_alpha = 0

    # Use the normal width on large screens, but scale down
    # when horizontal space is limited.
    tray_width = min(preferred_width, max(116, int(w * 0.23)))
    tray_x = w - tray_width
    compact = tray_width < 150

    if background_alpha:
        canvas.drawRect(
            skia.Rect.MakeXYWH(
                tray_x, y, tray_width, height
            ),
            _paint(background, background_alpha),
        )

    # Divider separating the notification area.
    if style in ("classic", "aero"):
        canvas.drawLine(
            tray_x, y + 3,
            tray_x, y + height - 3,
            skia.Paint(
                Color=_color("#FFFFFF", 125),
                AntiAlias=True,
                StrokeWidth=1,
            ),
        )

    stroke = skia.Paint(
        Color=_color(text_color),
        AntiAlias=True,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=1.6,
        StrokeCap=skia.Paint.kRound_Cap,
        StrokeJoin=skia.Paint.kRound_Join,
    )

    center_y = y + height / 2
    network_x = tray_x + (12 if compact else 18)
    volume_x = tray_x + (35 if compact else 52)

    # Network: ascending signal bars.
    for i, bar_height in enumerate((4, 7, 10, 14)):
        bar_x = network_x + i * 4
        canvas.drawRect(
            skia.Rect.MakeXYWH(
                bar_x,
                center_y + 7 - bar_height,
                2.5,
                bar_height,
            ),
            _paint(text_color),
        )

    # Shared volume indicator.
    draw_volume(
        canvas,
        volume_x,
        center_y,
        color=text_color,
        size=20,
    )

    # Hide the optional status icon in compact trays.
    if not compact:
        status_x = tray_x + 83

        if style == "modern":
            draw_battery(
                canvas,
                status_x + 10,
                center_y,
                color=text_color,
                size=20,
                charge=0.75,
            )
        else:
            # Small notification flag.
            canvas.drawLine(
                status_x, center_y - 7,
                status_x, center_y + 7,
                stroke,
            )

            flag = skia.Path()
            flag.moveTo(status_x + 1, center_y - 7)
            flag.lineTo(status_x + 12, center_y - 5)
            flag.lineTo(status_x + 9, center_y + 2)
            flag.lineTo(status_x + 1, center_y)
            flag.close()
            canvas.drawPath(flag, _paint(text_color))

    # Clock and date.
    if style in ("modern", "aero"):
        # Keep both lines centered on the same horizontal axis.
        # Position the group farther right for battery clearance.
        clock_center_x = w - 43

        time_size = 12
        date_size = 10

        time_font = skia.Font(None, time_size)
        date_font = skia.Font(None, date_size)

        time_width = time_font.measureText(clock_time)
        date_width = date_font.measureText(clock_date)

        _text(
            canvas,
            clock_time,
            clock_center_x - time_width / 2,
            center_y - 2,
            text_color,
            time_size,
        )

        _text(
            canvas,
            clock_date,
            clock_center_x - date_width / 2,
            center_y + 13,
            text_color,
            date_size,
        )
    else:
        # Classic taskbar: single-line clock.
        _text(
            canvas,
            clock_time,
            w - 61,
            center_y + 5,
            text_color,
            13,
        )

    # Narrow Show Desktop strip at the right edge.
    if style in ("modern", "aero"):
        canvas.drawLine(
            w - 5, y + 4,
            w - 5, y + height - 4,
            skia.Paint(
                Color=_color(text_color, 105),
                AntiAlias=True,
                StrokeWidth=1,
            ),
        )


def _draw_taskbar(surface, size, desktop, colors, scale=1.0):
    scale = validate_scale(scale)
    canvas = surface.getCanvas()
    canvas.save()
    canvas.scale(scale, scale)
    geometry = component_geometry(size, desktop, scale)
    size = (size[0] / scale, size[1] / scale)
    w, h = size

    style = desktop["style"]
    classic = style == "classic"

    box = geometry["taskbar"]
    height = (box[3] - box[1]) / scale
    y = h - height

    _edge_panel(
        surface, (0, y, w, height), desktop, colors
    )

    icons = [
        "minesweeper",
        "file_manager",
        "browser",
        "terminal",
    ]

    icon_size = 28
    gap = 12

    if style == "modern":
        layout = desktop.get("taskbar_layout") or "centered"

        if layout not in {"centered", "left"}:
            raise ValueError(
                f"Unsupported taskbar layout: {layout}"
            )

        def draw_start(x):
            """Draw a four-pane Start icon."""
            icon_y = y + (height - 24) / 2
            pane_size = 10
            pane_gap = 3

            for row in range(2):
                for col in range(2):
                    _round_rect(
                        canvas,
                        x + col * (pane_size + pane_gap),
                        icon_y + row * (pane_size + pane_gap),
                        pane_size,
                        pane_size,
                        1.5,
                        colors["accent"],
                    )

        def draw_search(cx, cy):
            """Draw a magnifying glass without font glyphs."""
            outline = skia.Paint(
                Color=_color(colors["text"]),
                AntiAlias=True,
                Style=skia.Paint.kStroke_Style,
                StrokeWidth=1.7,
                StrokeCap=skia.Paint.kRound_Cap,
            )

            canvas.drawCircle(cx, cy, 5.5, outline)
            canvas.drawLine(
                cx + 4, cy + 4,
                cx + 9, cy + 9,
                outline,
            )

        app_icons = icons

        if layout == "left":
            # Flat Start area, flush with the taskbar edges.
            start_width = 48

            # Sharp four-pane Start symbol.
            pane_size = 10
            pane_gap = 2
            symbol_width = 2 * pane_size + pane_gap
            pane_x = (start_width - symbol_width) / 2
            pane_y = y + (height - symbol_width) / 2

            for row in range(2):
                for col in range(2):
                    canvas.drawRect(
                        skia.Rect.MakeXYWH(
                            pane_x + col * (pane_size + pane_gap),
                            pane_y + row * (pane_size + pane_gap),
                            pane_size,
                            pane_size,
                        ),
                        _paint(colors["accent"]),
                    )

            # Search box fills the full taskbar height.
            search_x = start_width
            search_width = (
                250 if w >= 800
                else 190 if w >= 650
                else 120 if w >= 480
                else 0
            )

            if search_width:
                search_color = (
                    "#FFFFFF"
                    if desktop["color_scheme"] != "dark"
                    else "#343B46"
                )

                canvas.drawRect(
                    skia.Rect.MakeXYWH(
                        search_x, y,
                        search_width, height,
                    ),
                    _paint(search_color),
                )

                draw_search(
                    search_x + 20,
                    y + height / 2 - 2,
                )

                # Muted placeholder text for the Search box.
                placeholder_color = (
                    "#555555"
                    if desktop["color_scheme"] != "dark"
                    else "#C0C5CE"
                )

                _text(
                    canvas,
                    "Type here to search"
                    if search_width >= 150
                    else "Search",
                    search_x + 44,
                    y + height / 2 + 6,
                    placeholder_color,
                    17,
                )

            # Application icons start after Search.
            start_x = search_x + search_width + 8
            icon_size = 32
            gap = 16

        else:
            # Centered modern taskbar.
            start_width = 24
            search_width = 28
            app_width = (
                len(app_icons) * icon_size
                + (len(app_icons) - 1) * gap
            )

            total = (
                start_width + gap
                + search_width + gap
                + app_width
            )

            group_x = (w - total) / 2
            draw_start(group_x)

            search_x = group_x + start_width + gap
            draw_search(
                search_x + 10,
                y + height / 2 - 2,
            )

            start_x = search_x + search_width + gap

        # Draw pinned applications.
        for i, kind in enumerate(app_icons):
            x = start_x + i * (icon_size + gap)
            icon_y = y + (height - icon_size) / 2

            draw_app_icon(canvas, kind, x, icon_y, icon_size)

            # First application is active.
            if i == 0:
                if layout == "left":
                    # Full-width flat active indicator.
                    canvas.drawRect(
                        skia.Rect.MakeXYWH(
                            x - 8,
                            h - 3,
                            icon_size + 16,
                            3,
                        ),
                        _paint(colors["accent"]),
                    )
                else:
                    # Short rounded active indicator.
                    _round_rect(
                        canvas,
                        x + 7, h - 4,
                        icon_size - 14, 3,
                        1.5, colors["accent"],
                    )

    else:
        if style == "aero":
            # Large glossy Start orb.
            orb_size = height - 2
            orb_x = 5
            orb_y = y + (height - orb_size) / 2

            cx = orb_x + orb_size / 2
            cy = orb_y + orb_size / 2

            # Soft shadow.
            canvas.drawCircle(
                cx, cy + 1,
                orb_size / 2,
                _paint("#18364A", 100),
            )

            # Dark outer edge.
            canvas.drawCircle(
                cx, cy,
                orb_size / 2 - 0.5,
                _paint("#254A58"),
            )

            # Main blue-green orb.
            canvas.drawCircle(
                cx, cy,
                orb_size / 2 - 2,
                _paint("#247CAB"),
            )

            # Green central glow.
            canvas.drawCircle(
                cx, cy + 3,
                orb_size * 0.34,
                _paint("#4BA98C", 150),
            )

            # Gloss over the upper portion.
            canvas.drawOval(
                skia.Rect.MakeXYWH(
                    orb_x + 7,
                    orb_y + 5,
                    orb_size - 14,
                    orb_size * 0.28,
                ),
                _paint("#FFFFFF", 65),
            )

            # Shared four-color waving flag.
            draw_waving_flag(
                canvas, cx, cy, size=24
            )

            # Wide glass application buttons.
            slot_width = 64
            slot_gap = 2
            slot_height = height
            slot_radius = 3
            aero_icon_size = 40
            start_x = orb_x + orb_size + 9

            for i, kind in enumerate(icons):
                slot_x = start_x + i * (slot_width + slot_gap)
                slot_y = y + (height - slot_height) / 2

                active = i == 0

                # Translucent glass background.
                _round_rect(
                    canvas,
                    slot_x,
                    slot_y,
                    slot_width,
                    slot_height,
                    slot_radius,
                    "#DDECF8" if active else "#A9C4D8",
                    125 if active else 55,
                )

                # Thin glass border.
                border = skia.Paint(
                    Color=_color(
                        "#FFFFFF" if active else "#DCEBF5",
                        180 if active else 80,
                    ),
                    AntiAlias=True,
                    Style=skia.Paint.kStroke_Style,
                    StrokeWidth=1,
                )

                canvas.drawRoundRect(
                    skia.Rect.MakeXYWH(
                        slot_x + 0.5,
                        slot_y + 0.5,
                        slot_width - 1,
                        slot_height - 1,
                    ),
                    slot_radius,
                    slot_radius,
                    border,
                )

                # Highlight on the active button.
                if active:
                    _round_rect(
                        canvas,
                        slot_x + 3,
                        slot_y + 2,
                        slot_width - 6,
                        3,
                        1.5,
                        "#FFFFFF",
                        95,
                    )

                draw_app_icon(
                    canvas,
                    kind,
                    slot_x + (slot_width - aero_icon_size) / 2,
                    slot_y + (slot_height - aero_icon_size) / 2,
                    aero_icon_size,
                )

        else:
            # Classic taskbar with flush Start button.
            button_x = 0
            button_y = y
            button_w = 100
            button_h = height

            # Green Start button with a shallow circular-arc right edge.
            start_path = skia.Path()

            half_height = button_h / 2
            bulge = 10.0  # Depth of the circular segment.

            # Radius of the circle from its chord and sagitta.
            radius = (half_height**2 + bulge**2) / (2 * bulge)

            # The vertical chord joins the top and bottom edges.
            shoulder_x = button_x + button_w - bulge
            center_y = button_y + half_height

            # Bézier approximation of the corresponding circular arc.
            angle = math.atan2(half_height, radius - bulge)
            k = (4 / 3) * math.tan(angle / 2)

            control_x = shoulder_x + k * half_height
            control_dy = k * (radius - bulge)

            start_path.moveTo(button_x, button_y)
            start_path.lineTo(shoulder_x, button_y)

            start_path.cubicTo(
                control_x, button_y + control_dy,
                control_x, button_y + button_h - control_dy,
                shoulder_x, button_y + button_h,
            )

            start_path.lineTo(button_x, button_y + button_h)
            start_path.close()

            # Match the Classic Start button to the desktop accent.
            canvas.drawPath(start_path, _paint(colors["accent"]))

            # Smaller version of the same waving flag.
            draw_waving_flag(
                canvas,
                button_x + 23,
                button_y + button_h / 2,
                size=21,
            )

            # XP-like lowercase, bold italic Start lettering.
            start_typeface = skia.Typeface.MakeFromName(
                "Arial",
                skia.FontStyle.BoldItalic(),
            )

            start_font = skia.Font(start_typeface, 20)
            start_blob = skia.TextBlob("start", start_font)

            start_text_x = button_x + 42
            start_text_y = button_y + button_h / 2 + 7

            # Subtle dark shadow behind the lettering.
            canvas.drawTextBlob(
                start_blob,
                start_text_x + 1,
                start_text_y + 1,
                _paint("#000000", 105),
            )

            # White foreground.
            canvas.drawTextBlob(
                start_blob,
                start_text_x,
                start_text_y,
                _paint("#FFFFFF"),
            )

            # Rectangular application buttons with labels.
            app_labels = {
                "minesweeper": "Minesweeper",
                "file_manager": "Files",
                "browser": "Browser",
                "terminal": "Terminal",
            }

            start_x = button_x + button_w + 5
            slot_gap = 4
            slot_height = height - 6
            classic_icon_size = 23

            # Reserve space for the Classic system tray.
            # At sampled resolutions (minimum 800px), its width is 178px.
            tray_left = w - 178
            right_padding = 6

            # Keep the normal 142px buttons when possible.
            # Shrink them only when the available space requires it.
            available_width = tray_left - start_x - right_padding
            slot_width = min(
                142,
                (available_width - slot_gap * (len(icons) - 1)) / len(icons),
            )

            for i, kind in enumerate(icons):
                slot_x = start_x + i * (slot_width + slot_gap)
                slot_y = y + (height - slot_height) / 2
                active = i == 0

                # Active button appears pressed.
                _round_rect(
                    canvas,
                    slot_x,
                    slot_y,
                    slot_width,
                    slot_height,
                    2,
                    "#A7B9D1" if active else "#D9E2EF",
                )

                # Subtle border.
                border = skia.Paint(
                    Color=_color(
                        "#7C91AF" if active else "#F8FAFD"
                    ),
                    AntiAlias=True,
                    Style=skia.Paint.kStroke_Style,
                    StrokeWidth=1,
                )

                canvas.drawRoundRect(
                    skia.Rect.MakeXYWH(
                        slot_x + 0.5,
                        slot_y + 0.5,
                        slot_width - 1,
                        slot_height - 1,
                    ),
                    2,
                    2,
                    border,
                )

                draw_app_icon(
                    canvas,
                    kind,
                    slot_x + 7,
                    slot_y + (slot_height - classic_icon_size) / 2,
                    classic_icon_size,
                )

                _text(
                    canvas,
                    app_labels.get(
                        kind,
                        kind.replace("_", " ").title(),
                    ),
                    slot_x + 37,
                    slot_y + slot_height / 2 + 5,
                    "#253044",
                    13,
                )

    # Style-specific system tray.
    _draw_system_tray(canvas, w, y, height, desktop, colors)

    canvas.restore()
