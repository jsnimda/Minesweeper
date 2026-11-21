
"""Draw application windows on a shared Skia surface.

Content images are drawn at native resolution.
Returned annotations use screenshot-local half-open boxes.
"""

from collections.abc import Mapping, Sequence

import skia
from PIL import Image

from .desktop import _glass_panel, _palette, _skia_image


Box = tuple[int, int, int, int]

TITLE_HEIGHT = 40
MENU_HEIGHT = 28
CONTENT_PADDING = 1
MIN_WIDTH = 240


def _color(hex_color: str, alpha: int = 255) -> int:
    value = hex_color.lstrip("#")
    r, g, b = (
        int(value[i:i + 2], 16)
        for i in (0, 2, 4)
    )
    return skia.ColorSetARGB(alpha, r, g, b)


def _paint(color: str, alpha: int = 255, *,
           stroke: bool = False) -> skia.Paint:
    return skia.Paint(
        Color=_color(color, alpha),
        AntiAlias=True,
        Style=(
            skia.Paint.kStroke_Style
            if stroke else skia.Paint.kFill_Style
        ),
        StrokeWidth=1,
    )


def _text(canvas, text, x, y, color, size=13):
    font = skia.Font(None, size)
    blob = skia.TextBlob(str(text), font)
    canvas.drawTextBlob(blob, x, y, _paint(color))


def _line(canvas, x0, y0, x1, y1, color, width=1.5):
    paint = _paint(color)
    paint.setStrokeWidth(width)
    canvas.drawLine(x0, y0, x1, y1, paint)


def window_size(
    content_image: Image.Image,
    menu_items: Sequence[str] = (),
) -> tuple[int, int]:
    """Return the window size without resizing content."""
    if content_image.width <= 0 or content_image.height <= 0:
        raise ValueError("Content must have positive dimensions")

    width = max(
        content_image.width + CONTENT_PADDING * 2,
        190,  # Enough room for title and caption controls.
    )

    height = (
        TITLE_HEIGHT
        + (MENU_HEIGHT if menu_items else 0)
        + content_image.height
        + CONTENT_PADDING * 2
    )

    return width, height


def _caption_controls(
    canvas,
    x: int,
    y: int,
    width: int,
    desktop: dict,
    foreground: str,
):
    """Draw simplified caption controls."""
    shell = desktop["shell"]
    style = desktop["style"]

    cy = y + TITLE_HEIGHT // 2

    if shell == "dock":
        # Traffic-light-inspired controls.
        colors = ("#E66A65", "#E5B34C", "#68B97B")
        for i, color in enumerate(colors):
            cx = x + 20 + i * 20
            canvas.drawCircle(
                cx, cy, 6,
                _paint(color),
            )
        return

    # Right-aligned controls for taskbar/top-panel.
    centers = [
        x + width - 78,
        x + width - 51,
        x + width - 24,
    ]

    if style == "classic":
        for cx in centers:
            rect = skia.Rect.MakeXYWH(
                cx - 10, cy - 10, 20, 20
            )
            canvas.drawRect(
                rect, _paint("#D5D7DC")
            )
            canvas.drawRect(
                rect, _paint("#737D8A", stroke=True)
            )

        symbol = "#253044"

    else:
        # Only close gets a subtle hover-like surface.
        canvas.drawRoundRect(
            skia.Rect.MakeXYWH(
                centers[2] - 11, cy - 11, 22, 22
            ),
            6, 6,
            _paint(
                "#DFE7F0" if desktop["color_scheme"] != "dark"
                else "#525B68",
                95,
            ),
        )
        symbol = foreground

    # Minimize.
    _line(
        canvas,
        centers[0] - 4, cy + 3,
        centers[0] + 4, cy + 3,
        symbol,
    )

    # Maximize.
    canvas.drawRect(
        skia.Rect.MakeXYWH(
            centers[1] - 4, cy - 4, 8, 8
        ),
        _paint(symbol, stroke=True),
    )

    # Close.
    _line(
        canvas,
        centers[2] - 4, cy - 4,
        centers[2] + 4, cy + 4,
        symbol,
    )
    _line(
        canvas,
        centers[2] + 4, cy - 4,
        centers[2] - 4, cy + 4,
        symbol,
    )


def draw_generic_window(
    surface: skia.Surface,
    position: tuple[int, int],
    content_image: Image.Image,
    content_boxes: Mapping[str, Box],
    title: str,
    menu_items: Sequence[str] = (),
    desktop: dict | None = None,
) -> dict[str, Box]:
    """Draw one window and return screenshot-local annotations.

    The caller places the window. No scaling is performed.
    The window itself, menu and title bar are not game_region.
    """
    if desktop is None:
        raise ValueError("desktop configuration is required")

    for name, (x0, y0, x1, y1) in content_boxes.items():
        if not (
            0 <= x0 < x1 <= content_image.width
            and 0 <= y0 < y1 <= content_image.height
        ):
            raise ValueError(
                f"Invalid content box {name!r}"
            )

    x, y = position
    w, h = window_size(content_image, menu_items)

    if x < 0 or y < 0:
        raise ValueError("Window position must be nonnegative")

    canvas = surface.getCanvas()

    colors = _palette(desktop)
    style = desktop["style"]
    scheme = desktop["color_scheme"]

    dark = scheme == "dark"

    radius = {
        "modern": 11,
        "aero": 10,
        "classic": 3,
    }[style]

    body = "#252A33" if dark else "#FAFBFC"
    foreground = "#F2F4F8" if dark else "#26313F"

    rect = skia.Rect.MakeXYWH(x, y, w, h)
    outer = skia.RRect.MakeRectXY(
        rect, radius, radius
    )

    # 1. Draw a soft outer shadow.
    shadow_paint = skia.Paint(
        Color=_color("#000000", 55),
        AntiAlias=True,
        ImageFilter=skia.ImageFilters.Blur(10, 10),
    )

    canvas.drawRRect(
        skia.RRect.MakeRectXY(
            skia.Rect.MakeXYWH(x, y + 5, w, h),
            radius, radius,
        ),
        shadow_paint,
    )

    # 2. Clip the window to its outer rounded shape.
    canvas.save()
    canvas.clipRRect(outer, doAntiAlias=True)

    # 3. Title bar. Aero blurs what was already drawn.
    if style == "aero":
        _glass_panel(
            surface,
            (x, y, w, TITLE_HEIGHT),
            radius=0,
            blur=14,
            tint="#DCE9F4" if not dark else "#354E68",
            opacity=145,
        )

    elif style == "classic":
        canvas.drawRect(
            skia.Rect.MakeXYWH(
                x, y, w, TITLE_HEIGHT
            ),
            _paint(colors["accent"]),
        )
        foreground = "#FFFFFF"

    else:
        canvas.drawRect(
            skia.Rect.MakeXYWH(
                x, y, w, TITLE_HEIGHT
            ),
            _paint(
                "#303640" if dark else "#F1F4F8"
            ),
        )

    # 4. Opaque window body, below the title bar.
    canvas.drawRect(
        skia.Rect.MakeXYWH(
            x, y + TITLE_HEIGHT,
            w, h - TITLE_HEIGHT,
        ),
        _paint(body),
    )

    # 5. Application title, clipped before controls.
    controls_left = desktop["shell"] == "dock"
    title_x = x + (80 if controls_left else 15)

    available_title_width = (
        w - 95 if controls_left else w - 115
    )

    canvas.save()
    canvas.clipRect(
        skia.Rect.MakeXYWH(
            title_x, y,
            max(0, available_title_width),
            TITLE_HEIGHT,
        ),
    )

    _text(
        canvas,
        title,
        title_x,
        y + 26,
        foreground,
        14,
    )
    canvas.restore()

    _caption_controls(
        canvas, x, y, w, desktop, foreground
    )

    # 6. Optional menu bar.
    menu_height = MENU_HEIGHT if menu_items else 0

    if menu_items:
        menu_y = y + TITLE_HEIGHT

        menu_color = (
            "#303640" if dark else "#F7F8FA"
        )

        canvas.drawRect(
            skia.Rect.MakeXYWH(
                x, menu_y, w, MENU_HEIGHT
            ),
            _paint(menu_color),
        )

        text_x = x + 13
        for item in menu_items:
            _text(
                canvas,
                item,
                text_x,
                menu_y + 19,
                colors["text"],
                12,
            )
            text_x += max(48, len(item) * 8 + 19)

        _line(
            canvas,
            x, menu_y + MENU_HEIGHT - 1,
            x + w, menu_y + MENU_HEIGHT - 1,
            colors["subtle"],
            1,
        )

    # 7. Content origin, preserving native pixel size.
    content_x = x + (w - content_image.width) // 2
    content_y = (
        y + TITLE_HEIGHT + menu_height + CONTENT_PADDING
    )

    canvas.drawImage(
        _skia_image(content_image),
        content_x,
        content_y,
    )

    canvas.restore()

    # 8. Thin outer frame.
    canvas.drawRRect(
        outer,
        _paint(
            "#778493" if style == "classic"
            else colors["subtle"],
            160,
            stroke=True,
        ),
    )

    # 9. Translate content-local boxes exactly once.
    return {
        name: (
            x0 + content_x,
            y0 + content_y,
            x1 + content_x,
            y1 + content_y,
        )
        for name, (x0, y0, x1, y1) in content_boxes.items()
    }
