
"""Skia desktop renderer for synthetic screenshots.

Renders the sampled shell and visual style.
Application windows are handled separately.
"""

from io import BytesIO

import skia
from PIL import Image


def _color(value: str, alpha: int = 255) -> int:
    """Convert a #RRGGBB color to a Skia color."""
    value = value.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return skia.ColorSetARGB(alpha, r, g, b)


def _paint(color: str, alpha: int = 255) -> skia.Paint:
    return skia.Paint(
        Color=_color(color, alpha),
        AntiAlias=True,
    )


def _round_rect(canvas, x, y, w, h, radius, color, alpha=255):
    canvas.drawRoundRect(
        skia.Rect.MakeXYWH(x, y, w, h),
        radius,
        radius,
        _paint(color, alpha),
    )


def _glass_panel(
    surface: skia.Surface,
    rect: tuple[float, float, float, float],
    *,
    radius: float = 14,
    blur: float = 12,
    tint: str = "#F4F6F9",
    opacity: int = 155,
) -> None:
    """Draw frosted glass over the existing canvas pixels."""
    canvas = surface.getCanvas()
    x, y, width, height = rect

    shape = skia.RRect.MakeRectXY(
        skia.Rect.MakeXYWH(x, y, width, height),
        radius,
        radius,
    )

    # Capture the already composed background.
    backdrop = surface.makeImageSnapshot()

    canvas.save()
    canvas.clipRRect(shape, doAntiAlias=True)

    # Apply blur to the captured background, not the
    # window content that will be drawn afterward.
    canvas.drawImage(
        backdrop,
        0,
        0,
        paint=skia.Paint(
            AntiAlias=True,
            ImageFilter=skia.ImageFilters.Blur(blur, blur),
        ),
    )

    # Tinted glass surface.
    canvas.drawRRect(
        shape,
        _paint(tint, opacity),
    )
    canvas.restore()

    # Thin highlight around the glass.
    border = skia.Paint(
        Color=skia.ColorSetARGB(105, 255, 255, 255),
        AntiAlias=True,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=1,
    )
    canvas.drawRRect(shape, border)


def _edge_panel(
    surface: skia.Surface,
    rect: tuple[float, float, float, float],
    desktop: dict,
    colors: dict,
) -> None:
    """Draw a full-width, edge-attached desktop panel."""
    canvas = surface.getCanvas()
    x, y, w, h = rect
    style = desktop["style"]

    if style == "aero":
        _glass_panel(
            surface,
            rect,
            radius=0,
            blur=12,
            tint=colors["surface"],
            opacity=145,
        )
    else:
        canvas.drawRect(
            skia.Rect.MakeXYWH(x, y, w, h),
            _paint(
                colors["surface"],
                245 if style == "classic" else 225,
            ),
        )

    # Single separating line facing the desktop.
    line_y = y if y > 0 else y + h - 1

    canvas.drawLine(
        x, line_y,
        x + w, line_y,
        skia.Paint(
            Color=_color(colors["subtle"], 160),
            AntiAlias=True,
            StrokeWidth=1,
        ),
    )


def _text(canvas, value, x, y, color, size=13):
    font = skia.Font(None, size)
    blob = skia.TextBlob(str(value), font)
    canvas.drawTextBlob(blob, x, y, _paint(color))


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


def _panel(canvas, rect, desktop, colors):
    """Draw a softly elevated, optionally glass-like panel."""
    x, y, w, h = rect
    radius = colors["radius"]
    style = desktop["style"]

    # Broad, low-opacity shadow.
    shadow_rect = skia.Rect.MakeXYWH(x, y + 3, w, h)
    shadow = skia.Paint(
        Color=_color("#000000", 65 if style != "classic" else 30),
        AntiAlias=True,
        ImageFilter=skia.ImageFilters.Blur(10, 10),
    )
    canvas.drawRoundRect(shadow_rect, radius, radius, shadow)

    _round_rect(
        canvas, x, y, w, h, radius,
        colors["surface"], colors["alpha"],
    )

    # Restrained upper-edge highlight and border.
    outline = skia.Paint(
        Color=_color(colors["subtle"], 130),
        AntiAlias=True,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=1,
    )
    canvas.drawRoundRect(
        skia.Rect.MakeXYWH(x + 0.5, y + 0.5, w - 1, h - 1),
        radius, radius, outline,
    )


def _app_icon(canvas, x, y, size, color, glyph, text_color):
    """Original, simple symbolic application icon."""
    _round_rect(canvas, x, y, size, size, size * 0.27, color)
    _text(
        canvas, glyph,
        x + size * 0.34,
        y + size * 0.69,
        "#FFFFFF",
        size * 0.48,
    )


def _draw_taskbar(surface, size, desktop, colors):
    canvas = surface.getCanvas()
    w, h = size

    classic = desktop["style"] == "classic"
    height = 44 if classic else 48
    y = h - height

    _edge_panel(
        surface, (0, y, w, height), desktop, colors
    )

    icons = [
        ("◈", colors["accent"]),
        ("F", "#DB9F50"),
        ("W", "#5C92C7"),
        ("T", "#4D9C8C"),
    ]

    icon_size = 28
    gap = 12

    total = (
        len(icons) * icon_size
        + (len(icons) - 1) * gap
    )
    start_x = 16 if classic else (w - total) // 2

    for i, (glyph, color) in enumerate(icons):
        x = start_x + i * (icon_size + gap)
        icon_y = y + (height - icon_size) // 2

        _app_icon(
            canvas, x, icon_y,
            icon_size, color, glyph, colors["text"],
        )

        # Active application underline.
        if i == 1:
            _round_rect(
                canvas,
                x + 7, h - 4,
                icon_size - 14, 3,
                1.5, colors["accent"],
            )

    _text(
        canvas, "12:34",
        w - 64, y + height // 2 + 5,
        colors["text"], 13,
    )


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
    _text(canvas, "Desktop", 20, 22, colors["text"], 14)
    _text(canvas, "12:34", w - 66, 22, colors["text"], 13)

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
        ("F", "#DB9F50"),
        ("W", "#5C92C7"),
        ("T", "#4D9C8C"),
        ("C", "#9873C8"),
        ("S", "#6DAD73"),
        ("M", colors["accent"]),
    ]

    for i, (glyph, color) in enumerate(icons):
        x = x0 + padding + i * (icon_size + gap)
        y = y0 + padding
        _app_icon(canvas, x, y, icon_size, color, glyph, colors["text"])

        if i == 1:
            canvas.drawCircle(
                x + icon_size / 2,
                y0 + dock_height - 7,
                2.5,
                _paint(colors["accent"]),
            )


def _draw_top_panel(surface, size, desktop, colors):
    canvas = surface.getCanvas()
    w, _ = size
    height = 36

    _edge_panel(
        surface, (0, 0, w, height), desktop, colors
    )

    # Activities indicator.
    _round_rect(
        canvas, 14, 11, 4, 14,
        2, colors["accent"],
    )

    _text(
        canvas, "Activities",
        26, 24, colors["text"], 13,
    )

    _text(
        canvas, "Workspace 1",
        125, 24, colors["text"], 12,
    )

    # Compact system indicators.
    for i in range(3):
        canvas.drawCircle(
            w - 120 + i * 12,
            18,
            2.5,
            _paint(colors["text"], 170),
        )

    _text(
        canvas, "12:34",
        w - 63, 24, colors["text"], 13,
    )


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
