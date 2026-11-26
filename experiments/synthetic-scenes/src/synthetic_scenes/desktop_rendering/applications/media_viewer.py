
"""Procedural media viewer content rendered with Skia."""

import random

import skia

from ..scaling import pixels, validate_scale



THEMES = {
    "dark": {
        "background": "#191D24",
        "toolbar": "#252A33",
        "panel": "#20252E",
        "text": "#F0F2F6",
        "muted": "#9BA6B5",
        "border": "#363E49",
        "accent": "#75A9ED",
        "selection": "#344B6C",
    },
    "light": {
        "background": "#E8EDF3",
        "toolbar": "#FAFBFD",
        "panel": "#F4F6F9",
        "text": "#273449",
        "muted": "#7C8999",
        "border": "#D3DBE5",
        "accent": "#397CC1",
        "selection": "#DCEAF9",
    },
}


def _color(value: str) -> int:
    value = value.lstrip("#")
    return skia.ColorSetRGB(
        *(int(value[i:i + 2], 16) for i in (0, 2, 4))
    )


def _paint(color: str, *, stroke: float = 0) -> skia.Paint:
    paint = skia.Paint(
        Color=_color(color),
        AntiAlias=True,
    )
    if stroke:
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(stroke)
    return paint


def _rect(canvas, x, y, w, h, color, radius=0):
    shape = skia.Rect.MakeXYWH(x, y, w, h)
    if radius:
        canvas.drawRoundRect(
            shape, radius, radius, _paint(color)
        )
    else:
        canvas.drawRect(shape, _paint(color))


def _line(canvas, x1, y1, x2, y2, color, stroke=1):
    canvas.drawLine(
        x1, y1, x2, y2,
        _paint(color, stroke=stroke),
    )


def _text(canvas, value, x, y, size=12,
          color="#273449", bold=False):
    style = (
        skia.FontStyle.Bold()
        if bold else skia.FontStyle.Normal()
    )
    font = skia.Font(
        skia.Typeface.MakeFromName(None, style),
        size,
    )
    canvas.drawString(str(value), x, y, font, _paint(color))


def _center_text(canvas, value, cx, y, size, color):
    font = skia.Font(None, size)
    text_width = font.measureText(str(value))
    _text(canvas, value, cx - text_width / 2, y, size, color)


def _polygon(canvas, points, color):
    path = skia.Path()
    path.moveTo(*points[0])

    for point in points[1:]:
        path.lineTo(*point)

    path.close()
    canvas.drawPath(path, _paint(color))


def _draw_landscape(canvas, rng):
    """Draw an original stylized mountain landscape."""
    sky = rng.choice([
        ("#A9D7EE", "#74B7CF", "#426F8F"),
        ("#E9B49C", "#C18D9F", "#755E84"),
        ("#9BC9BD", "#6C9F9D", "#416D77"),
    ])
    _rect(canvas, 0, 0, 640, 360, sky[0])

    canvas.drawCircle(
        rng.randint(420, 555), 85,
        35, _paint("#FFF0C5"),
    )

    _polygon(
        canvas,
        [(0, 260), (130, 115), (245, 267),
         (390, 102), (535, 267), (640, 170),
         (640, 360), (0, 360)],
        sky[1],
    )

    _polygon(
        canvas,
        [(0, 295), (175, 170), (320, 310),
         (480, 158), (640, 291),
         (640, 360), (0, 360)],
        sky[2],
    )

    _rect(canvas, 0, 299, 640, 61, "#427E91")

    for _ in range(9):
        x = rng.randint(0, 640)
        y = rng.randint(306, 351)
        _rect(
            canvas, x, y,
            rng.randint(30, 100), 2,
            "#83B9B7", 1,
        )


def _draw_cityscape(canvas, rng):
    """Draw an original city skyline."""
    night = rng.choice([True, False])

    sky = "#263653" if night else "#9CBEDC"
    building = (
        ["#263044", "#35455C", "#48536A"]
        if night else
        ["#667D98", "#8298AB", "#536B85"]
    )
    window_color = "#F5D99A" if night else "#D4E8F6"

    _rect(canvas, 0, 0, 640, 360, sky)

    canvas.drawCircle(
        515, 80, 30,
        _paint("#E9E6CF" if night else "#FFF1BD"),
    )

    x = 0
    while x < 640:
        bw = rng.randint(43, 92)
        bh = rng.randint(110, 255)
        top = 315 - bh

        _rect(
            canvas, x, top, bw, bh,
            rng.choice(building),
        )

        for wx in range(x + 9, min(x + bw - 4, 640), 15):
            for wy in range(top + 15, 305, 22):
                if rng.random() < 0.58:
                    _rect(
                        canvas, wx, wy, 6, 9,
                        window_color,
                    )

        x += bw + rng.randint(2, 9)

    _rect(canvas, 0, 315, 640, 45, "#35424D")
    _rect(canvas, 0, 332, 640, 3, "#66747D")


def _draw_abstract(canvas, rng):
    """Draw an original geometric abstract composition."""
    palettes = [
        ["#F3D4B3", "#D77F75", "#497E92", "#2A5368"],
        ["#D8CBE9", "#977AB8", "#6C9FA1", "#E5A978"],
        ["#E4EBC7", "#88AC8A", "#3F7985", "#D29C68"],
    ]
    colors = rng.choice(palettes)

    _rect(canvas, 0, 0, 640, 360, colors[0])

    for _ in range(12):
        x = rng.randint(-70, 580)
        y = rng.randint(-65, 310)
        size = rng.randint(45, 165)
        color = rng.choice(colors[1:])

        if rng.random() < 0.5:
            canvas.drawCircle(
                x + size / 2,
                y + size / 2,
                size / 2,
                _paint(color),
            )
        else:
            _rect(
                canvas, x, y,
                size, size,
                color,
                rng.choice((0, 12, 30)),
            )


ARTWORK = {
    "landscape": ("Mountain Lake.jpg", _draw_landscape),
    "cityscape": ("City Lights.jpg", _draw_cityscape),
    "abstract": ("Composition.png", _draw_abstract),
}


def _artwork(canvas, kind, seed, x, y, width, height):
    """Draw artwork in a fixed virtual 640x360 coordinate system."""
    canvas.save()
    canvas.clipRect(skia.Rect.MakeXYWH(x, y, width, height))
    canvas.translate(x, y)
    canvas.scale(width / 640, height / 360)

    rng = random.Random(seed)
    ARTWORK[kind][1](canvas, rng)

    canvas.restore()


def render_media_viewer(
    seed: int = 42,
    *,
    width: int = 860,
    height: int = 600,
    theme: str | None = None,
    media_type: str | None = None,
    scale: float = 1.0,
) -> tuple[skia.Image, dict]:
    """Render a synthetic photo viewer with a thumbnail filmstrip."""
    if width < 560 or height < 420:
        raise ValueError("Media viewer canvas is too small")

    rng = random.Random(seed)

    if theme is None:
        theme = rng.choices(
            ("dark", "light"),
            weights=(65, 35),
            k=1,
        )[0]

    if theme not in THEMES:
        raise ValueError(f"Unknown viewer theme: {theme}")

    if media_type is None:
        media_type = rng.choice(tuple(ARTWORK))

    if media_type not in ARTWORK:
        raise ValueError(f"Unknown media type: {media_type}")

    colors = THEMES[theme]

    scale = validate_scale(scale)
    surface = skia.Surface(pixels(width, scale), pixels(height, scale))
    canvas = surface.getCanvas()
    canvas.scale(scale, scale)
    canvas.clear(_color(colors["background"]))

    toolbar_h = 54
    filmstrip_h = 110
    status_h = 34

    # Viewer toolbar.
    _rect(
        canvas, 0, 0,
        width, toolbar_h,
        colors["toolbar"],
    )

    _text(
        canvas, ARTWORK[media_type][0],
        20, 23, 14,
        colors["text"], True,
    )

    _text(
        canvas, "Photos",
        20, 42, 11,
        colors["muted"],
    )

    # Vector toolbar controls.
    for i, kind in enumerate(("back", "zoom", "menu")):
        xx = width - 124 + i * 43
        yy = 17

        if kind == "back":
            _line(
                canvas, xx + 14, yy + 2,
                xx + 5, yy + 10,
                colors["text"], 2,
            )
            _line(
                canvas, xx + 5, yy + 10,
                xx + 14, yy + 18,
                colors["text"], 2,
            )
        elif kind == "zoom":
            canvas.drawCircle(
                xx + 8, yy + 8, 6,
                _paint(colors["text"], stroke=1.7),
            )
            _line(
                canvas, xx + 13, yy + 13,
                xx + 19, yy + 19,
                colors["text"], 1.7,
            )
        else:
            for j in range(3):
                canvas.drawCircle(
                    xx + 9 + j * 6,
                    yy + 10, 1.5,
                    _paint(colors["text"]),
                )

    _rect(
        canvas, 0, toolbar_h - 1,
        width, 1,
        colors["border"],
    )

    # Main image viewport.
    viewport_top = toolbar_h
    viewport_bottom = height - filmstrip_h - status_h
    viewport_h = viewport_bottom - viewport_top

    image_w = min(width - 70, (viewport_h - 34) * 16 / 9)
    image_h = image_w * 9 / 16
    image_x = (width - image_w) / 2
    image_y = viewport_top + (viewport_h - image_h) / 2

    _rect(
        canvas,
        image_x + 4, image_y + 5,
        image_w, image_h,
        "#15191E",
    )

    _artwork(
        canvas, media_type,
        seed,
        image_x, image_y,
        image_w, image_h,
    )

    # Thumbnail filmstrip.
    strip_y = viewport_bottom

    _rect(
        canvas, 0, strip_y,
        width, filmstrip_h,
        colors["panel"],
    )

    _rect(
        canvas, 0, strip_y,
        width, 1,
        colors["border"],
    )

    kinds = tuple(ARTWORK)
    selected_index = kinds.index(media_type)
    thumb_w = 118
    thumb_h = 67
    thumb_gap = 14
    strip_width = len(kinds) * thumb_w + (len(kinds) - 1) * thumb_gap
    thumb_start = (width - strip_width) / 2

    for i, kind in enumerate(kinds):
        x = thumb_start + i * (thumb_w + thumb_gap)
        y = strip_y + 18

        if i == selected_index:
            _rect(
                canvas,
                x - 4, y - 4,
                thumb_w + 8,
                thumb_h + 8,
                colors["accent"], 5,
            )

        _artwork(
            canvas,
            kind,
            seed + i * 13 if kind != media_type else seed,
            x, y,
            thumb_w, thumb_h,
        )

    # Bottom status bar.
    _rect(
        canvas, 0, height - status_h,
        width, status_h,
        colors["toolbar"],
    )
    _rect(
        canvas, 0, height - status_h,
        width, 1,
        colors["border"],
    )

    _text(
        canvas, f"{selected_index + 1} of {len(kinds)}",
        18, height - 12,
        11, colors["muted"],
    )

    _center_text(
        canvas, "100%",
        width - 47, height - 12,
        11, colors["muted"],
    )

    return surface.makeImageSnapshot(), {}
