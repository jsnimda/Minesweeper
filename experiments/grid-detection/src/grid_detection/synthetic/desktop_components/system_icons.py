"""Reusable vector system-status icons for desktop shells."""

import skia

from .primitives import _color, _paint


def draw_wifi(
    canvas: skia.Canvas,
    cx: float,
    cy: float,
    *,
    color="#253044",
    size=20,
):
    canvas.save()
    try:
        # Shift the complete icon downward.
        canvas.translate(cx, cy + 2)
        canvas.scale(size / 20, size / 20)

        stroke = skia.Paint(
            Color=_color(color),
            AntiAlias=True,
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=1.7,
            StrokeCap=skia.Paint.kRound_Cap,
        )

        # Slightly larger Wi-Fi arcs.
        for radius in (10, 6):
            arc = skia.Path()
            arc.addArc(
                skia.Rect.MakeXYWH(
                    -radius,
                    -radius + 2,
                    radius * 2,
                    radius * 2,
                ),
                225,
                90,
            )
            canvas.drawPath(arc, stroke)

        # Restore breathing room below the inner arc.
        canvas.drawCircle(
            0, 2, 1.5, _paint(color)
        )
    finally:
        canvas.restore()


def draw_volume(
    canvas: skia.Canvas,
    cx: float,
    cy: float,
    *,
    color: str = "#253044",
    size: float = 20,
) -> None:
    """Draw a centered speaker with two sound waves."""
    if size <= 0:
        raise ValueError("Icon size must be positive")

    canvas.save()
    try:
        canvas.translate(cx, cy)
        canvas.scale(size / 20, size / 20)

        # Speaker body.
        speaker = skia.Path()
        speaker.moveTo(-9, -3)
        speaker.lineTo(-5, -3)
        speaker.lineTo(0, -7)
        speaker.lineTo(0, 7)
        speaker.lineTo(-5, 3)
        speaker.lineTo(-9, 3)
        speaker.close()

        canvas.drawPath(speaker, _paint(color))

        stroke = skia.Paint(
            Color=_color(color),
            AntiAlias=True,
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=1.6,
            StrokeCap=skia.Paint.kRound_Cap,
        )

        # Small inner wave.
        inner = skia.Path()
        inner.moveTo(3, -4)
        inner.quadTo(7, 0, 3, 4)
        canvas.drawPath(inner, stroke)

        # Larger outer wave.
        outer = skia.Path()
        outer.moveTo(6, -7)
        outer.quadTo(12, 0, 6, 7)
        canvas.drawPath(outer, stroke)

    finally:
        canvas.restore()


def draw_battery(
    canvas: skia.Canvas,
    cx: float,
    cy: float,
    *,
    color: str = "#253044",
    size: float = 20,
    charge: float = 0.75,
) -> None:
    """Draw a centered battery with a configurable charge level.

    charge ranges from 0.0 (empty) to 1.0 (full).
    """
    if size <= 0:
        raise ValueError("Icon size must be positive")
    if not 0.0 <= charge <= 1.0:
        raise ValueError("Battery charge must be between 0 and 1")

    canvas.save()
    try:
        canvas.translate(cx, cy)
        canvas.scale(size / 20, size / 20)

        stroke = skia.Paint(
            Color=_color(color),
            AntiAlias=True,
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=1.4,
            StrokeJoin=skia.Paint.kRound_Join,
        )

        # Main battery outline.
        canvas.drawRoundRect(
            skia.Rect.MakeXYWH(-10, -6, 17, 12),
            2, 2, stroke,
        )

        # Battery terminal on the right.
        canvas.drawRoundRect(
            skia.Rect.MakeXYWH(8, -2.5, 2, 5),
            0.7, 0.7,
            _paint(color),
        )

        # Filled portion of the battery.
        fill_width = 12 * charge

        if fill_width > 0:
            canvas.drawRoundRect(
                skia.Rect.MakeXYWH(
                    -7.5,
                    -3.5,
                    fill_width,
                    7,
                ),
                min(1, fill_width / 2),
                min(1, fill_width / 2),
                _paint(color),
            )

    finally:
        canvas.restore()
