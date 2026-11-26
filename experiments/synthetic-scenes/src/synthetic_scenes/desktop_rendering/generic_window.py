
"""Draw application windows on a shared Skia surface.

Content images are drawn at native resolution.
Returned annotations use screenshot-local half-open boxes.
"""

from collections.abc import Mapping, Sequence

import skia

from .scaling import pixels, validate_scale
from .desktop import _palette


Box = tuple[int, int, int, int]

TITLE_HEIGHT = 40
CLASSIC_TITLE_HEIGHT = 32
MENU_HEIGHT = 28
CONTENT_PADDING = 1
CLASSIC_FRAME = 4


def title_height(style: str, scale: float = 1.0) -> int:
    return pixels(CLASSIC_TITLE_HEIGHT if style == "classic" else TITLE_HEIGHT, scale)

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


def window_geometry(content_size, menu_items=(), *, style="modern", scale=1.0):
    """Return physical window size and content origin for physical content."""
    scale = validate_scale(scale)
    width, height = content_size
    if width <= 0 or height <= 0:
        raise ValueError("Content must have positive dimensions")
    if style not in {"modern", "classic"}:
        raise ValueError(f"Unsupported window style: {style}")
    logical_frame = CLASSIC_FRAME if style == "classic" else 0
    frame = pixels(logical_frame, scale) if logical_frame else 0
    padding = pixels(CONTENT_PADDING, scale)
    header = title_height(style, scale)
    menu_height = pixels(MENU_HEIGHT, scale) if menu_items else 0
    origin = (frame + padding, header + menu_height + padding)
    return ((width + 2 * (frame + padding),
             height + header + menu_height + 2 * padding + frame), origin)


def window_size(content_size, menu_items=(), *, style="modern", scale=1.0):
    return window_geometry(content_size, menu_items, style=style, scale=scale)[0]


def _caption_layout(width, header_height, style, shell):
    """Return bounded control rectangles and the title's horizontal interval.

    Coordinates are logical and relative to the window. Close is retained first;
    maximize and minimize are added only when their complete slots fit.
    """
    left_controls = style == "modern" and shell == "dock"
    kind = "classic" if style == "classic" else "dock" if left_controls else shell
    button_size = 16 if kind == "dock" else 20 if kind == "classic" else 22
    margin, gap = 4, 4
    if header_height < button_size + 2:
        return kind, [], (0, 0)
    count = min(3, max(0, int((width - 2 * margin + gap) // (button_size + gap))))
    names = ("close", "minimize", "maximize") if left_controls else ("close", "maximize", "minimize")
    top = (header_height - button_size) / 2
    controls = []
    for index in range(count):
        left = margin + index * (button_size + gap) if left_controls else (
            width - margin - button_size - index * (button_size + gap))
        controls.append((names[index], (left, top, left + button_size, top + button_size)))
    if left_controls:
        title_left = min(width, controls[-1][1][2] + 6) if controls else min(8, width)
        title_right = max(title_left, width - 8)
    else:
        title_left = min(8, width)
        title_right = max(title_left, (controls[-1][1][0] - 6) if controls else width - 8)
    return kind, controls, (title_left, min(width, title_right))


def _caption_controls(canvas, x, y, kind, controls, foreground, dark):
    """Draw only the controls for which the titlebar has room."""
    dock_colors = {"close": "#E66A65", "minimize": "#E5B34C", "maximize": "#68B97B"}
    for name, (left, top, right, bottom) in controls:
        left, top, right, bottom = x + left, y + top, x + right, y + bottom
        cx, cy = (left + right) / 2, (top + bottom) / 2
        if kind == "dock":
            canvas.drawCircle(cx, cy, 6, _paint(dock_colors[name]))
            continue
        if kind == "classic":
            button = "#50555D" if dark else "#DDE0E4"
            edge = "#9299A2" if dark else "#7F8790"
            canvas.drawRect(skia.Rect.MakeLTRB(left, top, right, bottom), _paint(button))
            canvas.drawRect(skia.Rect.MakeLTRB(left + .5, top + .5, right - .5, bottom - .5),
                            _paint(edge, stroke=True))
        elif kind == "top_panel":
            canvas.drawCircle(cx, cy, 10, _paint("#505761" if dark else "#E0E5EB"))

        if name == "minimize":
            _line(canvas, cx - 4, cy + 3, cx + 4, cy + 3, foreground)
        elif name == "maximize":
            canvas.drawRect(skia.Rect.MakeXYWH(cx - 4, cy - 4, 8, 8),
                            _paint(foreground, stroke=True))
        else:
            _line(canvas, cx - 4, cy - 4, cx + 4, cy + 4, foreground)
            _line(canvas, cx + 4, cy - 4, cx - 4, cy + 4, foreground)


def draw_generic_window(
    surface: skia.Surface,
    position: tuple[int, int],
    content_image: skia.Image,
    content_boxes: Mapping[str, Box],
    title: str,
    menu_items: Sequence[str] = (),
    desktop: dict | None = None,
    *,
    scale: float = 1.0,
) -> dict[str, Box]:
    """Draw a window and translate content-local boxes to physical pixels."""
    if desktop is None:
        raise ValueError("desktop configuration is required")
    for name, (x0, y0, x1, y1) in content_boxes.items():
        if not (0 <= x0 < x1 <= content_image.width()
                and 0 <= y0 < y1 <= content_image.height()):
            raise ValueError(f"Invalid content box {name!r}")

    scale = validate_scale(scale)
    physical_x, physical_y = position
    if physical_x < 0 or physical_y < 0:
        raise ValueError("Window position must be nonnegative")
    style = desktop["style"]
    (physical_w, physical_h), (offset_x, offset_y) = window_geometry(
        (content_image.width(), content_image.height()), menu_items, style=style, scale=scale)
    content_x, content_y = physical_x + offset_x, physical_y + offset_y
    x, y, w, h = physical_x / scale, physical_y / scale, physical_w / scale, physical_h / scale
    header_h = title_height(style, scale) / scale
    menu_h = pixels(MENU_HEIGHT, scale) / scale if menu_items else 0
    frame = pixels(CLASSIC_FRAME, scale) / scale if style == "classic" else 0
    dark = desktop["color_scheme"] == "dark"
    colors = _palette(desktop)
    body = "#252A33" if dark else "#FAFBFC"
    foreground = "#F2F4F8" if dark else "#26313F"
    radius = 0 if style == "classic" else 7

    canvas = surface.getCanvas()
    canvas.save()
    canvas.scale(scale, scale)
    rect = skia.Rect.MakeXYWH(x, y, w, h)
    outer = skia.RRect.MakeRectXY(rect, radius, radius)
    shadow = skia.Paint(Color=_color("#000000", 55), AntiAlias=True,
                        ImageFilter=skia.ImageFilters.Blur(10, 10))
    canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y + 5, w, h),
                                           radius, radius), shadow)

    canvas.save()
    canvas.clipRRect(outer, doAntiAlias=True)
    border_color = "#727C89" if dark else "#9DA7B3"
    canvas.drawRect(rect, _paint(border_color if style == "classic" else body))
    title_color = ("#39404A" if dark else "#D9DEE4") if style == "classic" else (
        "#303640" if dark else "#F1F4F8")
    canvas.drawRect(skia.Rect.MakeXYWH(x, y, w, header_h), _paint(title_color))
    if style == "classic":
        canvas.drawRect(skia.Rect.MakeXYWH(x, y, w, min(header_h, 3)), _paint(colors["accent"]))
    canvas.drawRect(skia.Rect.MakeXYWH(x + frame, y + header_h,
                                       max(0, w - 2 * frame), max(0, h - header_h - frame)),
                    _paint(body))

    kind, controls, (title_left, title_right) = _caption_layout(w, header_h, style, desktop["shell"])
    if title_right > title_left:
        canvas.save()
        canvas.clipRect(skia.Rect.MakeXYWH(x + title_left, y, title_right - title_left, header_h))
        _text(canvas, title, x + title_left, y + header_h / 2 + 5, foreground, 14)
        canvas.restore()
    _caption_controls(canvas, x, y, kind, controls, foreground, dark)

    if menu_items:
        menu_y = y + header_h
        menu_width = max(0, w - 2 * frame)
        canvas.drawRect(skia.Rect.MakeXYWH(x + frame, menu_y, menu_width, menu_h),
                        _paint("#303640" if dark else "#F7F8FA"))
        if menu_width > 0:
            canvas.save()
            canvas.clipRect(skia.Rect.MakeXYWH(x + frame, menu_y, menu_width, menu_h))
            text_x = x + frame + 8
            for item in menu_items:
                if text_x >= x + frame + menu_width:
                    break
                _text(canvas, item, text_x, menu_y + menu_h / 2 + 4, foreground, 12)
                text_x += max(48, len(item) * 8 + 19)
            canvas.restore()
    canvas.restore()

    # Finish all chrome before drawing content, including its border and bevel.
    canvas.drawRRect(outer, _paint(border_color, 180, stroke=True))
    if style == "classic" and w > 4 and h > header_h + 4:
        _line(canvas, x + 1.5, y + header_h, x + 1.5, y + h - 2,
              "#AAB2BC" if dark else "#FFFFFF", 1)
    canvas.restore()

    # Use physical, integer-aligned coordinates, outside the rounded chrome clip.
    # The opaque body supplies the existing backdrop for transparent content.
    canvas.save()
    canvas.resetMatrix()
    canvas.drawRect(skia.Rect.MakeXYWH(content_x, content_y,
                                       content_image.width(), content_image.height()), _paint(body))
    canvas.drawImage(content_image, content_x, content_y)
    canvas.restore()

    return {name: (x0 + content_x, y0 + content_y, x1 + content_x, y1 + content_y)
            for name, (x0, y0, x1, y1) in content_boxes.items()}
