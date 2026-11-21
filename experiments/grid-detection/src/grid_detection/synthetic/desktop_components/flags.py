
"""Reusable waving flag icon for classic and Aero desktop shells."""

import skia

from .primitives import _paint


def draw_waving_flag(
    canvas: skia.Canvas,
    cx: float,
    cy: float,
    size: float = 24,
    *,
    slant: float = 1.2,
    horizontal_gap: float = 1.8,
    vertical_gap: float = 1.8,
    shadow: bool = True,
) -> None:
    """Draw a four-color waving flag centered at (cx, cy).

    At size=24, geometry matches the existing Aero flag.
    Other sizes scale the entire design proportionally.
    """
    if size <= 0:
        raise ValueError("Flag size must be positive")

    # Original Aero flag coordinates (24-unit design).
    top_y0 = -10.0
    bottom_y1 = 11.0

    # Center the gap at the same position as the Aero flag.
    gap_center_y = -0.7
    top_y1 = gap_center_y - vertical_gap / 2
    bottom_y0 = gap_center_y + vertical_gap / 2

    top_height = top_y1 - top_y0
    bottom_height = bottom_y1 - bottom_y0
    average_height = (top_height + bottom_height) / 2

    if top_height <= 0 or bottom_height <= 0:
        raise ValueError("Vertical gap is too large")

    if not 0 < horizontal_gap < 24:
        raise ValueError("Horizontal gap must be between 0 and 24")

    # Preserve diagonal alignment between the rows.
    row_offset = slant * (1 + vertical_gap / average_height)
    top_offset = row_offset
    bottom_offset = -row_offset

    # Symmetric columns.
    left_x0, left_x1 = -12.0, -horizontal_gap / 2
    right_x0, right_x1 = horizontal_gap / 2, 12.0

    def wave_tile(x0, x1, top, bottom, color, bend):
        path = skia.Path()
        mid_x = (x0 + x1) / 2

        # Top edge, shifted right.
        path.moveTo(x0 + slant, top)

        # Rounded ^ for left tiles, rounded v for right tiles.
        path.quadTo(
            mid_x + slant,
            top + 2 * bend,
            x1 + slant,
            top,
        )

        # Slanted right edge.
        path.lineTo(x1 - slant, bottom)

        # Lower edge follows the same wave.
        path.quadTo(
            mid_x - slant,
            bottom + 2 * bend,
            x0 - slant,
            bottom,
        )

        # Slanted left edge.
        path.close()

        if shadow:
            shadow_paint = skia.Paint(
                Color=skia.ColorSetARGB(105, 15, 30, 45),
                AntiAlias=True,
                ImageFilter=skia.ImageFilters.Blur(0.8, 0.8),
            )

            canvas.save()
            canvas.translate(0.7, 1.0)
            canvas.drawPath(path, shadow_paint)
            canvas.restore()

        canvas.drawPath(path, _paint(color))

    canvas.save()
    try:
        canvas.translate(cx, cy)
        canvas.scale(size / 24, size / 24)

        # Top-left: red, upward arch (^).
        wave_tile(
            left_x0 + top_offset,
            left_x1 + top_offset,
            top_y0, top_y1,
            "#ED5146",
            -3,
        )

        # Top-right: green, downward dip (v).
        wave_tile(
            right_x0 + top_offset,
            right_x1 + top_offset,
            top_y0, top_y1,
            "#7BC453",
            3,
        )

        # Bottom-left: blue, upward arch (^).
        wave_tile(
            left_x0 + bottom_offset,
            left_x1 + bottom_offset,
            bottom_y0, bottom_y1,
            "#48A6E8",
            -3,
        )

        # Bottom-right: yellow, downward dip (v).
        wave_tile(
            right_x0 + bottom_offset,
            right_x1 + bottom_offset,
            bottom_y0, bottom_y1,
            "#F5CA45",
            3,
        )
    finally:
        canvas.restore()
