
"""Procedural system settings application rendered with Skia."""

import random

import skia
from PIL import Image

from ..desktop import surface_to_pillow


THEMES = {
    "light": {
        "background": "#F5F7FA",
        "sidebar": "#EEF2F7",
        "surface": "#FFFFFF",
        "text": "#263449",
        "muted": "#76869A",
        "border": "#E1E7EE",
        "selected": "#DFEAF8",
        "accent": "#397CC1",
        "switch_off": "#BCC6D1",
    },
    "dark": {
        "background": "#1F242C",
        "sidebar": "#272E38",
        "surface": "#303844",
        "text": "#F0F3F7",
        "muted": "#A2AEBD",
        "border": "#424C59",
        "selected": "#344D6A",
        "accent": "#69A9EC",
        "switch_off": "#687584",
    },
}


def _color(value):
    value = value.lstrip("#")
    return skia.ColorSetRGB(
        *(int(value[i:i + 2], 16) for i in (0, 2, 4))
    )


def _paint(color, stroke=0):
    paint = skia.Paint(Color=_color(color), AntiAlias=True)
    if stroke:
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(stroke)
    return paint


def _rect(canvas, x, y, w, h, color, radius=0):
    shape = skia.Rect.MakeXYWH(x, y, w, h)

    if radius:
        canvas.drawRoundRect(shape, radius, radius, _paint(color))
    else:
        canvas.drawRect(shape, _paint(color))


def _line(canvas, x1, y1, x2, y2, color):
    canvas.drawLine(
        x1, y1, x2, y2,
        _paint(color, stroke=1),
    )


def _font(size, bold=False):
    style = (
        skia.FontStyle.Bold()
        if bold else skia.FontStyle.Normal()
    )
    return skia.Font(
        skia.Typeface.MakeFromName(None, style),
        size,
    )


def _text(canvas, value, x, y, size, color, bold=False):
    canvas.drawString(
        str(value),
        x, y,
        _font(size, bold),
        _paint(color),
    )


def _clipped_text(
    canvas, value, x, y, max_width, size, color, bold=False
):
    canvas.save()
    canvas.clipRect(
        skia.Rect.MakeXYWH(
            x, y - size - 4,
            max(0, max_width), size + 9,
        )
    )
    _text(canvas, value, x, y, size, color, bold)
    canvas.restore()


def _switch(canvas, x, y, enabled, colors):
    track_color = (
        colors["accent"] if enabled
        else colors["switch_off"]
    )

    _rect(canvas, x, y, 43, 24, track_color, 12)

    knob_x = x + (22 if enabled else 3)
    _rect(
        canvas, knob_x, y + 3, 18, 18,
        "#FFFFFF", 9,
    )


def _dropdown(canvas, x, y, width, value, colors):
    _rect(
        canvas, x, y, width, 33,
        colors["surface"], 6,
    )
    _rect(canvas, x, y, width, 1, colors["border"])
    _clipped_text(
        canvas, value,
        x + 13, y + 22,
        width - 35, 12, colors["text"],
    )

    # Vector chevron.
    px = x + width - 18
    py = y + 15
    path = skia.Path()
    path.moveTo(px - 4, py - 2)
    path.lineTo(px, py + 2)
    path.lineTo(px + 4, py - 2)

    canvas.drawPath(path, _paint(colors["muted"], stroke=1.5))


def _setting_row(
    canvas, x, y, width, title, description,
    colors, enabled=None,
):
    """Draw a setting row and return its next y position."""
    _text(
        canvas, title,
        x + 19, y + 26,
        14, colors["text"],
    )
    _clipped_text(
        canvas, description,
        x + 19, y + 47,
        width - 110, 11, colors["muted"],
    )

    if enabled is not None:
        _switch(
            canvas,
            x + width - 67,
            y + 19,
            enabled,
            colors,
        )

    _line(
        canvas,
        x + 18, y + 65,
        x + width - 18, y + 65,
        colors["border"],
    )
    return y + 74


def _draw_system(canvas, x, y, width, rng, colors):
    _text(
        canvas, "Device information",
        x, y + 17, 17, colors["text"], True,
    )

    y += 35
    device_name = rng.choice([
        "DESKTOP-WORKSTATION",
        "DEVELOPMENT-PC",
        "MY-COMPUTER",
        "STUDIO-DESKTOP",
    ])

    details = [
        ("Device name", device_name),
        ("Processor", "8-Core CPU"),
        ("Installed RAM", rng.choice(["8 GB", "16 GB", "32 GB"])),
        ("System type", "64-bit operating system"),
    ]

    for label, value in details:
        _text(canvas, label, x + 5, y + 18, 12, colors["muted"])
        _clipped_text(
            canvas, value,
            x + width * 0.38, y + 18,
            width * 0.57, 12, colors["text"],
        )
        _line(
            canvas, x, y + 35,
            x + width, y + 35,
            colors["border"],
        )
        y += 46

    y += 22
    _text(
        canvas, "Display preferences",
        x, y, 17, colors["text"], True,
    )
    y += 17

    _setting_row(
        canvas, x, y, width,
        "Night light",
        "Use warmer colors after sunset",
        colors, rng.choice([True, False]),
    )


def _draw_personalization(canvas, x, y, width, rng, colors):
    _text(
        canvas, "Appearance",
        x, y + 17, 17, colors["text"], True,
    )

    # Preview accent swatches.
    swatches = (
        "#397CC1", "#9274C7", "#DB8675",
        "#49A58B", "#E3AD5C",
    )
    selected = rng.randrange(len(swatches))

    for i, swatch in enumerate(swatches):
        xx = x + 7 + i * 55
        yy = y + 41

        _rect(canvas, xx, yy, 37, 37, swatch, 8)

        if i == selected:
            outline = _paint(colors["text"], stroke=2)
            canvas.drawRoundRect(
                skia.Rect.MakeXYWH(
                    xx - 3, yy - 3, 43, 43
                ),
                10, 10, outline,
            )

    y += 101
    _text(
        canvas, "Choose your mode",
        x + 5, y + 10, 13, colors["text"],
    )

    _dropdown(
        canvas, x + 5, y + 24,
        min(245, width - 10),
        rng.choice(["Light", "Dark", "Use system setting"]),
        colors,
    )

    y += 85
    y = _setting_row(
        canvas, x, y, width,
        "Transparency effects",
        "Show translucent backgrounds",
        colors, rng.choice([True, False]),
    )
    _setting_row(
        canvas, x, y, width,
        "Animations",
        "Show animations in the interface",
        colors, rng.choice([True, False]),
    )


def _draw_network(canvas, x, y, width, rng, colors):
    connected = rng.choice([True, True, False])

    _text(
        canvas, "Network status",
        x, y + 17, 17, colors["text"], True,
    )
    _rect(
        canvas, x, y + 36,
        width, 100,
        colors["surface"], 9,
    )

    # Simple Wi-Fi icon.
    cx = x + 40
    cy = y + 87
    for radius in (23, 15, 7):
        rect = skia.Rect.MakeXYWH(
            cx - radius, cy - radius,
            radius * 2, radius * 2,
        )
        canvas.drawArc(
            rect, 215, 110, False,
            _paint(colors["accent"], stroke=2),
        )

    _text(
        canvas,
        "Connected" if connected else "Disconnected",
        x + 85, y + 76, 15,
        colors["text"], True,
    )
    _text(
        canvas,
        "Wi-Fi network" if connected else "No active connection",
        x + 85, y + 101, 12,
        colors["muted"],
    )

    y += 159
    y = _setting_row(
        canvas, x, y, width,
        "Wi-Fi",
        "Connect to wireless networks",
        colors, connected,
    )
    y = _setting_row(
        canvas, x, y, width,
        "Airplane mode",
        "Turn off wireless communication",
        colors, not connected,
    )
    _setting_row(
        canvas, x, y, width,
        "Metered connection",
        "Reduce background data usage",
        colors, rng.choice([True, False]),
    )


def _draw_bluetooth(canvas, x, y, width, rng, colors):
    _text(
        canvas, "Bluetooth devices",
        x, y + 17, 17, colors["text"], True,
    )
    y += 32

    y = _setting_row(
        canvas, x, y, width,
        "Bluetooth",
        "Discover and connect nearby devices",
        colors, rng.choice([True, False]),
    )

    _text(
        canvas, "Paired devices",
        x, y + 23, 16, colors["text"], True,
    )
    y += 39

    devices = [
        ("Wireless Mouse", "Connected"),
        ("Bluetooth Keyboard", "Paired"),
        ("Headphones", "Not connected"),
    ]

    for name, status in devices:
        _rect(
            canvas, x, y,
            width, 58,
            colors["surface"], 8,
        )
        _text(
            canvas, name,
            x + 17, y + 25,
            13, colors["text"],
        )
        _text(
            canvas, status,
            x + 17, y + 43,
            11, colors["muted"],
        )
        y += 70


def _draw_privacy(canvas, x, y, width, rng, colors):
    _text(
        canvas, "App permissions",
        x, y + 17, 17, colors["text"], True,
    )

    y += 37
    for title, description in [
        ("Location", "Allow apps to access location"),
        ("Camera", "Allow apps to use the camera"),
        ("Microphone", "Allow apps to record audio"),
        ("Notifications", "Allow app notifications"),
    ]:
        y = _setting_row(
            canvas, x, y, width,
            title, description,
            colors, rng.choice([True, False]),
        )


PAGES = {
    "System": _draw_system,
    "Personalization": _draw_personalization,
    "Network & internet": _draw_network,
    "Bluetooth & devices": _draw_bluetooth,
    "Privacy & security": _draw_privacy,
}


def _sidebar_icon(
    canvas,
    name: str,
    x: float,
    y: float,
    color: str,
):
    """Draw a 20 × 20 Settings navigation icon."""
    outline = _paint(color, stroke=1.7)
    fill = _paint(color)

    def line(x1, y1, x2, y2):
        canvas.drawLine(x1, y1, x2, y2, outline)

    if name == "System":
        # Monitor and stand.
        canvas.drawRoundRect(
            skia.Rect.MakeXYWH(x + 1, y + 2, 18, 13),
            2, 2, outline,
        )
        line(x + 10, y + 15, x + 10, y + 18)
        line(x + 5, y + 18, x + 15, y + 18)

    elif name == "Personalization":
        # Paint palette with three color wells.
        canvas.drawOval(
            skia.Rect.MakeXYWH(x + 1, y + 2, 18, 16),
            outline,
        )
        for dx, dy in ((5, 6), (11, 5), (6, 12)):
            canvas.drawCircle(
                x + dx, y + dy, 1.6, fill
            )

    elif name == "Network & internet":
        # Three Wi-Fi arcs and a center dot.
        for radius in (9, 6, 3):
            rect = skia.Rect.MakeXYWH(
                x + 10 - radius,
                y + 12 - radius,
                radius * 2,
                radius * 2,
            )
            canvas.drawArc(
                rect, 215, 110, False, outline
            )
        canvas.drawCircle(x + 10, y + 15, 1.7, fill)

    elif name == "Bluetooth & devices":
        # Bluetooth rune.
        line(x + 9, y + 1, x + 9, y + 19)
        line(x + 9, y + 1, x + 15, y + 6)
        line(x + 15, y + 6, x + 5, y + 15)
        line(x + 5, y + 5, x + 15, y + 14)
        line(x + 15, y + 14, x + 9, y + 19)

    elif name == "Privacy & security":
        # Shield outline.
        path = skia.Path()
        path.moveTo(x + 10, y + 1)
        path.lineTo(x + 18, y + 4)
        path.lineTo(x + 17, y + 12)
        path.quadTo(x + 15, y + 17, x + 10, y + 19)
        path.quadTo(x + 5, y + 17, x + 3, y + 12)
        path.lineTo(x + 2, y + 4)
        path.close()
        canvas.drawPath(path, outline)


def render_settings(
    seed: int = 42,
    *,
    width: int = 820,
    height: int = 570,
    page: str | None = None,
    theme: str | None = None,
) -> tuple[Image.Image, dict]:
    """Render deterministic desktop settings content.

    Returns a Pillow image and no Minesweeper target boxes.
    """
    if width < 620 or height < 430:
        raise ValueError("Settings canvas is too small")

    rng = random.Random(seed)

    if page is None:
        page = rng.choice(tuple(PAGES))

    if page not in PAGES:
        raise ValueError(f"Unknown settings page: {page}")

    if theme is None:
        theme = rng.choices(
            ("light", "dark"),
            weights=(70, 30),
            k=1,
        )[0]

    if theme not in THEMES:
        raise ValueError(f"Unknown settings theme: {theme}")

    colors = THEMES[theme]

    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    canvas.clear(_color(colors["background"]))

    sidebar_w = 231

    # Sidebar.
    _rect(
        canvas, 0, 0,
        sidebar_w, height,
        colors["sidebar"],
    )
    _rect(
        canvas,
        sidebar_w - 1, 0,
        1, height,
        colors["border"],
    )

    # Synthetic account summary.
    canvas.drawCircle(
        37, 37, 20,
        _paint(colors["accent"]),
    )
    _text(
        canvas, "U",
        30, 44, 20, "#FFFFFF", True,
    )
    _text(
        canvas, "Local User",
        67, 32, 14, colors["text"], True,
    )
    _text(
        canvas, "Local account",
        67, 50, 11, colors["muted"],
    )

    # Search control.
    _rect(
        canvas, 15, 83,
        sidebar_w - 30, 36,
        colors["surface"], 7,
    )
    _text(
        canvas, "Find a setting",
        29, 106, 12, colors["muted"],
    )

    sidebar_items = list(PAGES)
    for index, name in enumerate(sidebar_items):
        yy = 150 + index * 53

        if name == page:
            _rect(
                canvas, 9, yy - 23,
                sidebar_w - 18, 43,
                colors["selected"], 8,
            )
            _rect(
                canvas, 12, yy - 15,
                3, 27,
                colors["accent"], 2,
            )

        # Vector navigation icon.
        _sidebar_icon(
            canvas,
            name,
            x=24,
            y=yy - 12,
            color=(
                colors["accent"]
                if name == page
                else colors["muted"]
            ),
        )
        _clipped_text(
            canvas, name,
            53, yy + 2,
            sidebar_w - 65,
            12, colors["text"],
        )

    # Main settings page.
    content_x = sidebar_w + 34
    content_width = width - content_x - 36

    canvas.save()
    canvas.clipRect(
        skia.Rect.MakeXYWH(
            sidebar_w,
            0,
            width - sidebar_w,
            height,
        )
    )

    _text(
        canvas, page,
        content_x, 56,
        25, colors["text"], True,
    )

    _text(
        canvas, "Manage your preferences",
        content_x, 83,
        12, colors["muted"],
    )

    _line(
        canvas,
        content_x, 102,
        content_x + content_width, 102,
        colors["border"],
    )

    PAGES[page](
        canvas,
        content_x,
        123,
        content_width,
        rng,
        colors,
    )

    canvas.restore()

    return surface_to_pillow(surface), {}
