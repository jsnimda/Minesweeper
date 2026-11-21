
"""Procedural Sudoku application content rendered with Skia."""

import random

import skia
from PIL import Image

from ..desktop import surface_to_pillow


def _paint(color: int, *, stroke: float = 0) -> skia.Paint:
    paint = skia.Paint(Color=color, AntiAlias=True)
    if stroke:
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(stroke)
    return paint


def _text(canvas, value, x, y, size, color):
    font = skia.Font(None, size)
    blob = skia.TextBlob(str(value), font)
    canvas.drawTextBlob(blob, x, y, _paint(color))


def _generate_board(rng: random.Random):
    """Generate a valid filled Sudoku by permuting a base grid."""
    base = 3
    side = 9

    def shuffled(values):
        values = list(values)
        rng.shuffle(values)
        return values

    rows = [
        group * base + row
        for group in shuffled(range(base))
        for row in shuffled(range(base))
    ]
    cols = [
        group * base + col
        for group in shuffled(range(base))
        for col in shuffled(range(base))
    ]
    digits = shuffled(range(1, side + 1))

    def value(r, c):
        return digits[
            (base * (r % base) + r // base + c) % side
        ]

    return [
        [value(r, c) for c in cols]
        for r in rows
    ]


def render_sudoku(
    seed: int = 42,
    *,
    clues: int = 32,
    cell_size: int = 36,
) -> tuple[Image.Image, dict]:
    """Render Sudoku content and empty Minesweeper annotations.

    The generated clues are consistent with a valid solved grid.
    Puzzle uniqueness is not guaranteed.
    """
    if not 0 <= clues <= 81:
        raise ValueError("clues must be between 0 and 81")
    if cell_size < 20:
        raise ValueError("cell_size must be at least 20")

    rng = random.Random(seed)
    solution = _generate_board(rng)

    visible = set(rng.sample(range(81), clues))
    selected = rng.randrange(81)

    margin = 22
    header = 78
    footer = 66

    grid_size = 9 * cell_size
    width = grid_size + 2 * margin
    height = header + grid_size + footer

    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()

    # Application background.
    canvas.clear(skia.ColorWHITE)

    background = skia.ColorSetRGB(250, 251, 253)
    text = skia.ColorSetRGB(40, 49, 62)
    muted = skia.ColorSetRGB(108, 119, 135)
    blue = skia.ColorSetRGB(65, 116, 190)
    highlight = skia.ColorSetRGB(229, 239, 252)
    selected_color = skia.ColorSetRGB(202, 223, 249)

    # Application header.
    _text(canvas, "Sudoku", margin, 30, 20, text)
    _text(canvas, "Easy", margin, 57, 12, muted)
    _text(canvas, "00:42", width - 75, 57, 13, muted)

    # A restrained divider.
    canvas.drawLine(
        margin, 68, width - margin, 68,
        _paint(skia.ColorSetRGB(223, 228, 236), stroke=1),
    )

    grid_x = margin
    grid_y = header

    selected_row, selected_col = divmod(selected, 9)

    for r in range(9):
        for c in range(9):
            x = grid_x + c * cell_size
            y = grid_y + r * cell_size

            if r == selected_row or c == selected_col:
                fill = highlight
            else:
                fill = skia.ColorWHITE

            if (r, c) == (selected_row, selected_col):
                fill = selected_color

            canvas.drawRect(
                skia.Rect.MakeXYWH(x, y, cell_size, cell_size),
                _paint(fill),
            )

            if r * 9 + c in visible:
                digit = str(solution[r][c])
                font_size = cell_size * 0.55
                font = skia.Font(None, font_size)

                # Center glyph using its actual text width.
                glyph_width = font.measureText(digit)
                text_x = x + (cell_size - glyph_width) / 2
                text_y = y + cell_size * 0.72

                _text(
                    canvas, digit,
                    text_x, text_y, font_size, text,
                )

    # Thin cell lines and thicker 3x3 boundaries.
    for i in range(10):
        coordinate = i * cell_size
        line_width = 2.5 if i % 3 == 0 else 0.7
        line_color = (
            skia.ColorSetRGB(47, 63, 87)
            if i % 3 == 0
            else skia.ColorSetRGB(173, 185, 200)
        )

        canvas.drawLine(
            grid_x + coordinate, grid_y,
            grid_x + coordinate, grid_y + grid_size,
            _paint(line_color, stroke=line_width),
        )
        canvas.drawLine(
            grid_x, grid_y + coordinate,
            grid_x + grid_size, grid_y + coordinate,
            _paint(line_color, stroke=line_width),
        )

    # Bottom number strip.
    strip_y = grid_y + grid_size + 33

    for digit in range(1, 10):
        x = grid_x + (digit - 1) * cell_size
        glyph = str(digit)
        font = skia.Font(None, 19)

        _text(
            canvas,
            glyph,
            x + (cell_size - font.measureText(glyph)) / 2,
            strip_y,
            19,
            blue,
        )

    # Sudoku is a negative example for Minesweeper detection.
    return surface_to_pillow(surface), {}
