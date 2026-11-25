
"""Reusable Skia vector icons for synthetic desktop shells."""

import math

import skia


ICON_COLORS = {
    "file_manager": "#E6AB4F",
    "finder": "#7BA9D8",
    "browser": "#4F9DC7",
    "terminal": "#364554",
    "spreadsheet": "#45A17B",
    "document_editor": "#578FC8",
    "presentation": "#D38A56",
    "code_editor": "#527CB9",
    "calculator": "#778AA5",
    "settings": "#75879A",
    "sudoku": "#678BB8",
    "other_puzzle": "#BA83B9",
    "media_viewer": "#6AAB9D",
    "minesweeper": "#768EA4",
    "trash": "#A8B5C4",
}


def _color(value: str) -> int:
    value = value.lstrip("#")
    return skia.ColorSetRGB(
        *(int(value[i:i + 2], 16) for i in (0, 2, 4))
    )


def _paint(color: str, stroke: float = 0) -> skia.Paint:
    paint = skia.Paint(Color=_color(color), AntiAlias=True)
    if stroke:
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(stroke)
        paint.setStrokeCap(skia.Paint.kRound_Cap)
        paint.setStrokeJoin(skia.Paint.kRound_Join)
    return paint


def _rect(canvas, x, y, w, h, color, radius=0):
    shape = skia.Rect.MakeXYWH(x, y, w, h)
    if radius:
        canvas.drawRoundRect(
            shape, radius, radius, _paint(color)
        )
    else:
        canvas.drawRect(shape, _paint(color))


def _line(canvas, x1, y1, x2, y2, color, width=2):
    canvas.drawLine(
        x1, y1, x2, y2,
        _paint(color, stroke=width),
    )


def _circle(canvas, x, y, radius, color):
    canvas.drawCircle(x, y, radius, _paint(color))


def _path(canvas, points, color, width=2, closed=False):
    path = skia.Path()
    path.moveTo(*points[0])

    for point in points[1:]:
        path.lineTo(*point)

    if closed:
        path.close()

    canvas.drawPath(
        path,
        _paint(color, 0 if closed else width),
    )


def _icon_art(canvas, kind: str):
    """Draw one icon in a virtual 32 × 32 coordinate system."""

    if kind == "file_manager":
        _rect(canvas, 3, 8, 26, 20, "#D99436", 4)
        _rect(canvas, 3, 5, 13, 9, "#F5C979", 3)
        _rect(canvas, 4, 12, 24, 15, "#EDB753", 3)

    elif kind == "finder":
        # Finder-like two-tone smiling face.
        # Left half is blue; right half is pale blue-white.
        face_rect = skia.Rect.MakeXYWH(3, 3, 26, 26)
        face_shape = skia.RRect.MakeRectXY(face_rect, 5.5, 5.5)

        canvas.save()
        try:
            canvas.clipRRect(face_shape, doAntiAlias=True)

            # Two-tone background.
            _rect(canvas, 3, 3, 13, 26, "#67A8E8")
            _rect(canvas, 16, 3, 13, 26, "#E8F2FC")

            # Subtle color variation on the right.
            _rect(canvas, 16, 3, 13, 7, "#F5F9FE")

            # Central face boundary.
            _line(canvas, 16, 4, 16, 28, "#7AABD9", 0.8)

            # Eyes.
            _line(canvas, 10.5, 11, 10.5, 14, "#20496E", 1.6)
            _line(canvas, 21.5, 11, 21.5, 14, "#20496E", 1.6)

            # Distinctive hooked nose.
            nose = skia.Path()
            nose.moveTo(16, 9)
            nose.lineTo(14.5, 17.5)
            nose.quadTo(14.3, 18.5, 16.5, 18.5)
            nose.lineTo(18, 18)
            canvas.drawPath(nose, _paint("#20496E", stroke=1.4))

            # Continuous curved smile.
            smile = skia.Path()
            smile.moveTo(9, 21)
            smile.cubicTo(
                12, 25.5,
                20.5, 25.5,
                23, 20.5,
            )
            canvas.drawPath(smile, _paint("#20496E", stroke=1.8))

        finally:
            canvas.restore()

    elif kind == "browser":
        _circle(canvas, 16, 16, 14, "#4F9DC7")
        _circle(canvas, 16, 16, 10.5, "#FFFFFF")
        _circle(canvas, 16, 16, 8, "#7FC1D3")
        _path(
            canvas,
            [(8, 16), (24, 16)],
            "#FFFFFF", 2,
        )
        _path(
            canvas,
            [(16, 8), (12, 16), (16, 24)],
            "#FFFFFF", 2,
        )
        _path(
            canvas,
            [(16, 8), (20, 16), (16, 24)],
            "#FFFFFF", 2,
        )

    elif kind == "terminal":
        _rect(canvas, 2, 3, 28, 26, "#364554", 5)
        _rect(canvas, 5, 6, 22, 20, "#1B2430", 3)
        _path(
            canvas,
            [(9, 11), (14, 16), (9, 21)],
            "#95E2BB", 2.5,
        )
        _line(canvas, 17, 21, 24, 21, "#D3E9F1", 2)

    elif kind == "spreadsheet":
        _rect(canvas, 3, 3, 26, 26, "#45A17B", 5)
        _rect(canvas, 8, 8, 16, 16, "#FFFFFF", 2)
        for pos in (13, 19):
            _line(canvas, pos, 8, pos, 24, "#45A17B", 1.4)
            _line(canvas, 8, pos, 24, pos, "#45A17B", 1.4)

    elif kind == "document_editor":
        _rect(canvas, 5, 3, 22, 26, "#578FC8", 4)
        _rect(canvas, 9, 7, 14, 18, "#FFFFFF", 2)
        for yy, ww in ((11, 11), (15, 11), (19, 8)):
            _rect(canvas, 11, yy, ww, 1.6, "#578FC8")

    elif kind == "presentation":
        _rect(canvas, 3, 5, 26, 19, "#D38A56", 4)
        _rect(canvas, 7, 9, 18, 11, "#FFF5E8", 2)
        _rect(canvas, 10, 15, 3, 3, "#E6B18B")
        _rect(canvas, 15, 12, 3, 6, "#D38A56")
        _rect(canvas, 20, 10, 3, 8, "#E6B18B")
        _line(canvas, 16, 24, 16, 28, "#925A3B", 2)
        _line(canvas, 11, 28, 21, 28, "#925A3B", 2)

    elif kind == "code_editor":
        _rect(canvas, 2, 3, 28, 26, "#527CB9", 5)
        _path(
            canvas,
            [(12, 10), (6, 16), (12, 22)],
            "#FFFFFF", 2.8,
        )
        _path(
            canvas,
            [(20, 10), (26, 16), (20, 22)],
            "#FFFFFF", 2.8,
        )
        _line(canvas, 18, 8, 14, 24, "#C9DDF5", 2.4)

    elif kind == "calculator":
        _rect(canvas, 5, 2, 22, 28, "#778AA5", 5)
        _rect(canvas, 8, 6, 16, 7, "#EAF1F8", 2)
        for row in range(2):
            for col in range(3):
                _rect(
                    canvas,
                    8 + col * 6,
                    16 + row * 6,
                    4.3, 4.3,
                    "#FFFFFF" if col < 2 else "#C2D6EF",
                    1,
                )

    elif kind == "settings":
        _circle(canvas, 16, 16, 13, "#75879A")
        for angle in range(0, 360, 45):
            radians = math.radians(angle)
            cx = 16 + math.cos(radians) * 12
            cy = 16 + math.sin(radians) * 12
            _rect(
                canvas, cx - 2, cy - 2,
                4, 4, "#75879A", 1,
            )
        _circle(canvas, 16, 16, 8, "#EAF0F7")
        _circle(canvas, 16, 16, 4, "#75879A")

    elif kind == "sudoku":
        _rect(canvas, 3, 3, 26, 26, "#678BB8", 4)
        _rect(canvas, 6, 6, 20, 20, "#FFFFFF", 1)
        for pos in (12.7, 19.3):
            _line(canvas, pos, 6, pos, 26, "#678BB8", 1.7)
            _line(canvas, 6, pos, 26, pos, "#678BB8", 1.7)
        _circle(canvas, 9.3, 9.3, 1.5, "#386392")
        _circle(canvas, 16, 16, 1.5, "#386392")
        _circle(canvas, 22.7, 22.7, 1.5, "#386392")

    elif kind == "other_puzzle":
        for x, y, value in (
            (3, 3, "#CD8EC6"),
            (17, 3, "#A777B8"),
            (3, 17, "#8D72AB"),
            (17, 17, "#D9A7D0"),
        ):
            _rect(canvas, x, y, 12, 12, value, 3)
        for cx, cy in ((9, 9), (23, 9), (9, 23)):
            _circle(canvas, cx, cy, 2.4, "#FFFFFF")

    elif kind == "media_viewer":
        _rect(canvas, 2, 4, 28, 24, "#6AAB9D", 5)
        _rect(canvas, 5, 7, 22, 18, "#DFF3EE", 3)
        _circle(canvas, 22, 12, 3, "#E9BB70")
        _path(
            canvas,
            [(6, 24), (13, 15), (18, 20), (22, 16),
             (27, 24)],
            "#54978B", closed=True,
        )

    elif kind == "minesweeper":
        _rect(canvas, 2, 2, 28, 28, "#768EA4", 4)
        _rect(canvas, 5, 5, 22, 22, "#DCE5EC")
        for pos in (12.3, 19.7):
            _line(canvas, pos, 5, pos, 27, "#A6B4C3", 1)
            _line(canvas, 5, pos, 27, pos, "#A6B4C3", 1)
        _circle(canvas, 16, 16, 4, "#4B5662")
        for dx, dy in (
            (-7, 0), (7, 0), (0, -7), (0, 7),
            (-5, -5), (5, -5), (-5, 5), (5, 5),
        ):
            _line(
                canvas, 16, 16,
                16 + dx, 16 + dy,
                "#4B5662", 1.5,
            )
        _circle(canvas, 14.5, 14.5, 1.5, "#FFFFFF")

    elif kind == "trash":
        _rect(canvas, 7, 10, 18, 18, "#A8B5C4", 3)
        _rect(canvas, 5, 7, 22, 4, "#687E93", 2)
        _rect(canvas, 12, 4, 8, 3, "#687E93", 1)
        _line(canvas, 12, 15, 12, 24, "#FFFFFF", 1.8)
        _line(canvas, 20, 15, 20, 24, "#FFFFFF", 1.8)

    else:
        raise ValueError(f"Unknown application icon: {kind}")


def draw_app_icon(
    canvas: skia.Canvas,
    kind: str,
    x: float,
    y: float,
    size: float = 32,
) -> None:
    """Draw an icon at any size using a 32 × 32 vector design."""
    if size <= 0:
        raise ValueError("Icon size must be positive")
    if kind not in ICON_COLORS:
        raise ValueError(f"Unknown application icon: {kind}")

    canvas.save()
    try:
        canvas.translate(x, y)
        canvas.scale(size / 32, size / 32)
        _icon_art(canvas, kind)
    finally:
        canvas.restore()
