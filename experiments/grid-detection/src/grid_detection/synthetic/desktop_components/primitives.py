
"""Shared Skia drawing primitives for synthetic desktop shells."""

import skia


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
                255 if style == "classic" else 225,
            ),
        )

    # Classic omits the extra separating line.
    if style != "classic":
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
