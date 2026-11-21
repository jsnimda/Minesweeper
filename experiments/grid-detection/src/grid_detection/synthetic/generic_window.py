
"""Draw application windows on a shared Skia surface.

Content images are drawn at native resolution.
Returned annotations use screenshot-local half-open boxes.
"""

from collections.abc import Mapping, Sequence

import skia
from PIL import Image

from .desktop import _palette, _skia_image
from .desktop_components.primitives import _glass_panel


Box = tuple[int, int, int, int]

TITLE_HEIGHT = 40
CLASSIC_TITLE_HEIGHT = 32
MENU_HEIGHT = 28
CONTENT_PADDING = 1
MIN_WIDTH = 240
AERO_FRAME = 6
CLASSIC_FRAME = 4


def title_height(style: str) -> int:
    return CLASSIC_TITLE_HEIGHT if style == "classic" else TITLE_HEIGHT

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
    *,
    style: str = "modern",
) -> tuple[int, int]:
    """Return window dimensions without resizing content."""
    if content_image.width <= 0 or content_image.height <= 0:
        raise ValueError("Content must have positive dimensions")

    frame = (
        AERO_FRAME if style == "aero"
        else CLASSIC_FRAME if style == "classic"
        else 0
    )

    width = max(
        content_image.width + CONTENT_PADDING * 2,
        190,
    ) + frame * 2

    height = (
        title_height(style)
        + (MENU_HEIGHT if menu_items else 0)
        + content_image.height
        + CONTENT_PADDING * 2
        + frame
    )

    return width, height


def _top_panel_caption_controls(
    canvas: skia.Canvas,
    x: int,
    y: int,
    width: int,
    foreground: str,
    dark: bool,
) -> None:
    """Draw compact GNOME-inspired window controls."""

    cy = y + TITLE_HEIGHT / 2

    centers = (
        x + width - 88,
        x + width - 56,
        x + width - 24,
    )

    button_bg = "#464C57" if dark else "#E1E5EB"
    symbol_color = "#F1F3F6" if dark else "#303844"

    for cx in centers:
        canvas.drawCircle(
            cx, cy, 12,
            _paint(button_bg),
        )

    # Minimize.
    _line(
        canvas,
        centers[0] - 4, cy + 1,
        centers[0] + 4, cy + 1,
        symbol_color,
        1.6,
    )

    # Maximize.
    canvas.drawRect(
        skia.Rect.MakeXYWH(
            centers[1] - 4,
            cy - 4,
            8,
            8,
        ),
        _paint(symbol_color, stroke=True),
    )

    # Close: slightly stronger circular surface.
    canvas.drawCircle(
        centers[2], cy, 12,
        _paint("#D7DCE4" if not dark else "#59616E"),
    )

    _line(
        canvas,
        centers[2] - 4, cy - 4,
        centers[2] + 4, cy + 4,
        symbol_color,
        1.7,
    )
    _line(
        canvas,
        centers[2] + 4, cy - 4,
        centers[2] - 4, cy + 4,
        symbol_color,
        1.7,
    )


def _aero_caption_controls(
    canvas: skia.Canvas,
    x: int,
    y: int,
    width: int,
) -> None:
    """Draw compact Windows 7-style glass caption buttons."""

    # Wide, shallow Aero caption buttons with a small right glass margin.
    top = y + 1
    height = 22

    button_widths = (34, 34, 48)
    right_margin = 8

    left = x + width - sum(button_widths) - right_margin

    # Minimize, maximize, close.
    buttons = (
        (left,      button_widths[0], "#A6C5DF"),
        (left + 34, button_widths[1], "#A6C5DF"),
        (left + 68, button_widths[2], "#C84646"),
    )

    for i, (bx, bw, color) in enumerate(buttons):
        rect = skia.Rect.MakeXYWH(bx, top, bw, height)

        # Main translucent or red button surface.
        canvas.drawRect(
            rect,
            _paint(color, 195 if i < 2 else 245),
        )

        # Brighter upper half for a glossy appearance.
        canvas.drawRect(
            skia.Rect.MakeXYWH(
                bx + 1, top + 1,
                bw - 2, height * 0.44,
            ),
            _paint(
                "#FFFFFF",
                68 if i < 2 else 43,
            ),
        )

        # Subtle dark outline.
        canvas.drawRect(
            rect,
            _paint("#40556C", 165, stroke=True),
        )

        # Thin highlight along the upper edge.
        canvas.drawLine(
            bx + 2, top + 1.5,
            bx + bw - 2, top + 1.5,
            _paint("#FFFFFF", 125),
        )

    cy = top + height / 2
    centers = (
        left + 17,
        left + 51,
        left + 92,
    )

    # Minimize.
    _line(
        canvas,
        centers[0] - 5, cy + 4,
        centers[0] + 5, cy + 4,
        "#26384B",
        1.5,
    )

    # Maximize.
    canvas.drawRect(
        skia.Rect.MakeXYWH(
            centers[1] - 5, cy - 5, 10, 9
        ),
        _paint("#26384B", stroke=True),
    )

    # Close: white X on red.
    _line(
        canvas,
        centers[2] - 5, cy - 5,
        centers[2] + 5, cy + 5,
        "#FFFFFF",
        2,
    )
    _line(
        canvas,
        centers[2] + 5, cy - 5,
        centers[2] - 5, cy + 5,
        "#FFFFFF",
        2,
    )


def _classic_caption_controls(
    canvas: skia.Canvas,
    x: int,
    y: int,
    width: int,
    accent: str,
) -> None:
    """Draw glossy, accent-aware Classic caption buttons."""

    def mix(color: str, target: str, amount: float) -> int:
        a = color.lstrip("#")
        b = target.lstrip("#")

        rgb = [
            round(
                int(a[i:i + 2], 16) * (1 - amount)
                + int(b[i:i + 2], 16) * amount
            )
            for i in (0, 2, 4)
        ]

        return skia.ColorSetRGB(*rgb)

    def blend_hex(base: str, accent: str, amount: float) -> str:
        a = base.lstrip("#")
        b = accent.lstrip("#")

        values = [
            round(
                int(a[i:i + 2], 16) * (1 - amount)
                + int(b[i:i + 2], 16) * amount
            )
            for i in (0, 2, 4)
        ]

        return "#{:02X}{:02X}{:02X}".format(*values)

    button_size = 24
    gap = 4
    right_margin = 6

    top = y + (CLASSIC_TITLE_HEIGHT - button_size) / 2
    left = (
        x + width
        - right_margin
        - button_size * 3
        - gap * 2
    )

    # Blend the Close button's red with the desktop accent.
    # It remains recognizably red across different themes.
    close_color = blend_hex("#D94A34", accent, 0.18)

    buttons = (
        (left, accent),
        (left + button_size + gap, accent),
        (left + (button_size + gap) * 2, close_color),
    )

    centers = []

    for bx, base in buttons:
        rect = skia.Rect.MakeXYWH(
            bx, top, button_size, button_size
        )

        centers.append((
            bx + button_size / 2,
            top + button_size / 2,
        ))

        # Subtle glossy gradient based on the button color.
        shader = skia.GradientShader.MakeLinear(
            points=[
                (bx, top),
                (bx, top + button_size),
            ],
            colors=[
                mix(base, "#FFFFFF", 0.22),
                mix(base, "#000000", 0.13),
            ],
            positions=[0.0, 1.0],
            mode=skia.TileMode.kClamp,
        )

        canvas.drawRoundRect(
            rect, 5, 5,
            skia.Paint(
                Shader=shader,
                AntiAlias=True,
            ),
        )

        # Bright outline characteristic of Classic controls.
        outline = skia.Paint(
            Color=_color("#F4F8FF", 235),
            AntiAlias=True,
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=1.5,
        )

        canvas.drawRoundRect(
            rect, 5, 5, outline
        )

        # Gloss highlight across the upper portion.
        canvas.drawLine(
            bx + 5, top + 3,
            bx + button_size - 5, top + 3,
            _paint("#FFFFFF", 105),
        )

    # Minimize — white horizontal line.
    cx, cy = centers[0]
    _line(
        canvas,
        cx - 5, cy + 4,
        cx + 5, cy + 4,
        "#FFFFFF", 2.2,
    )

    # Maximize — white square.
    cx, cy = centers[1]
    symbol = skia.Paint(
        Color=_color("#FFFFFF"),
        AntiAlias=True,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=2,
    )

    canvas.drawRect(
        skia.Rect.MakeXYWH(
            cx - 5, cy - 5, 10, 10
        ),
        symbol,
    )

    # Close — bold white X.
    cx, cy = centers[2]

    _line(
        canvas,
        cx - 5, cy - 5,
        cx + 5, cy + 5,
        "#FFFFFF", 2.5,
    )

    _line(
        canvas,
        cx + 5, cy - 5,
        cx - 5, cy + 5,
        "#FFFFFF", 2.5,
    )


def _caption_controls(
    canvas,
    x: int,
    y: int,
    width: int,
    desktop: dict,
    foreground: str,
):
    """Draw caption controls for each window style."""
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

    if shell == "top_panel":
        _top_panel_caption_controls(
            canvas,
            x,
            y,
            width,
            foreground,
            dark=desktop["color_scheme"] == "dark",
        )
        return

    if style == "aero":
        _aero_caption_controls(canvas, x, y, width)
        return

    if style == "classic":
        _classic_caption_controls(
            canvas,
            x,
            y,
            width,
            accent=_palette(desktop)["accent"],
        )
        return

    # Modern window controls.
    centers = [
        x + width - 78,
        x + width - 51,
        x + width - 24,
    ]

    # Subtle surface behind the Close button.
    canvas.drawRoundRect(
        skia.Rect.MakeXYWH(
            centers[2] - 11,
            cy - 11,
            22,
            22,
        ),
        6, 6,
        _paint(
            "#DFE7F0"
            if desktop["color_scheme"] != "dark"
            else "#525B68",
            95,
        ),
    )

    # Minimize.
    _line(
        canvas,
        centers[0] - 4, cy + 3,
        centers[0] + 4, cy + 3,
        foreground,
    )

    # Maximize.
    canvas.drawRect(
        skia.Rect.MakeXYWH(
            centers[1] - 4,
            cy - 4,
            8,
            8,
        ),
        _paint(foreground, stroke=True),
    )

    # Close.
    _line(
        canvas,
        centers[2] - 4, cy - 4,
        centers[2] + 4, cy + 4,
        foreground,
    )
    _line(
        canvas,
        centers[2] + 4, cy - 4,
        centers[2] - 4, cy + 4,
        foreground,
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
    w, h = window_size(
        content_image,
        menu_items,
        style=desktop["style"],
    )

    if x < 0 or y < 0:
        raise ValueError("Window position must be nonnegative")

    canvas = surface.getCanvas()

    colors = _palette(desktop)
    style = desktop["style"]
    scheme = desktop["color_scheme"]
    header_h = title_height(style)

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

    # Opaque accent-colored shell for Classic windows.
    # The title bar and inset content will be drawn over it.
    if style == "classic":
        canvas.drawRRect(
            outer,
            _paint(colors["accent"]),
        )

    # 3. Title bar. Aero blurs what was already drawn.
    if style == "aero":
        frame = 6

        # Full glass shell: title bar + side/bottom frame.
        _glass_panel(
            surface,
            (x, y, w, h),
            radius=8,
            blur=14,
            tint="#DCE9F4" if not dark else "#354E68",
            opacity=135,
        )

        # Stronger glossy highlight near the top edge.
        canvas.drawRect(
            skia.Rect.MakeXYWH(
                x + 1, y + 1, w - 2, 12
            ),
            _paint("#FFFFFF", 42),
        )

        # Thin bright top edge, flush to the frame.
        canvas.drawLine(
            x + 8, y + 1,
            x + w - 8, y + 1,
            _paint("#FFFFFF", 125),
        )

        # Separator between title bar and content region.
        canvas.drawLine(
            x + frame,
            y + TITLE_HEIGHT - 1,
            x + w - frame,
            y + TITLE_HEIGHT - 1,
            _paint("#526A83", 125),
        )

    elif style == "classic":
        # Subtle accent-aware gradient for Classic windows.
        accent = colors["accent"].lstrip("#")
        r, g, b = (
            int(accent[i:i + 2], 16)
            for i in (0, 2, 4)
        )

        # Slightly brighter at the top, darker at the bottom.
        top_color = skia.ColorSetRGB(
            round(r * 0.95 + 255 * 0.05),
            round(g * 0.95 + 255 * 0.05),
            round(b * 0.95 + 255 * 0.05),
        )

        bottom_color = skia.ColorSetRGB(
            round(r * 0.92),
            round(g * 0.92),
            round(b * 0.92),
        )

        title_shader = skia.GradientShader.MakeLinear(
            points=[
                (x, y),
                (x, y + header_h),
            ],
            colors=[
                top_color,
                bottom_color,
            ],
            positions=[0.0, 1.0],
            mode=skia.TileMode.kClamp,
        )

        canvas.drawRect(
            skia.Rect.MakeXYWH(
                x, y, w, header_h
            ),
            skia.Paint(
                Shader=title_shader,
                AntiAlias=True,
            ),
        )

        # Very subtle highlight along the top.
        canvas.drawLine(
            x + 3, y + 1,
            x + w - 3, y + 1,
            _paint("#FFFFFF", 55),
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

    # 4. Opaque window body, leaving visible side/bottom borders.
    frame = (
        AERO_FRAME if style == "aero"
        else CLASSIC_FRAME if style == "classic"
        else 0
    )

    canvas.drawRect(
        skia.Rect.MakeXYWH(
            x + frame,
            y + header_h,
            w - frame * 2,
            h - header_h - frame,
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
        y + header_h / 2 + 5,
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
        menu_y = y + header_h

        menu_color = (
            "#303640" if dark else "#F7F8FA"
        )

        canvas.drawRect(
            skia.Rect.MakeXYWH(
                x + frame,
                menu_y,
                w - 2 * frame,
                MENU_HEIGHT,
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
    content_x = (
        x + frame
        + (w - frame * 2 - content_image.width) // 2
    )

    content_y = (
        y + header_h + menu_height + CONTENT_PADDING
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
            "#34465D" if style == "classic"
            else colors["subtle"],
            160,
            stroke=True,
        ),
    )

    if style == "classic":
        # Beveled highlight along the exposed colored frame.
        canvas.drawLine(
            x + 1.5,
            y + header_h,
            x + 1.5,
            y + h - 4,
            _paint("#FFFFFF", 95),
        )

        canvas.drawLine(
            x + 2,
            y + h - 2,
            x + w - 2,
            y + h - 2,
            _paint("#FFFFFF", 65),
        )

    if style == "aero":
        inner = skia.RRect.MakeRectXY(
            skia.Rect.MakeXYWH(
                x + 1, y + 1,
                w - 2, h - 2,
            ),
            radius - 1,
            radius - 1,
        )

        canvas.drawRRect(
            inner,
            _paint(
                "#FFFFFF",
                55,
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
