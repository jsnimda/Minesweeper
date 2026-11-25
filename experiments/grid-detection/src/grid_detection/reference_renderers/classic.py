"""Classic Windows Minesweeper renderer (no desktop or window chrome)."""

import numpy as np
from PIL import Image, ImageDraw

CELL_SIZE = 16
BACKGROUND = "#C0C0C0"

NUMBER_COLORS = {
    1: "#0000FF", 2: "#008000", 3: "#FF0000", 4: "#000080",
    5: "#800000", 6: "#008080", 7: "#000000", 8: "#808080",
}

DIGITS = {
    1: ('0000011000', '0000111000', '0001111000', '0011111000', '0000111000', '0000111000', '0000111000', '0000111000', '0011111110', '0011111110'),
    2: ('0111111110', '1111111111', '1110000111', '0000000111', '0000011110', '0001111100', '0111110000', '1111000000', '1111111111', '1111111111'),
    3: ('1111111110', '1111111111', '0000000111', '0000000111', '0001111110', '0001111110', '0000000111', '0000000111', '1111111111', '1111111110'),
    4: ('0011101110', '0011101110', '0111001110', '0111001110', '1111111111', '1111111111', '0000001110', '0000001110', '0000001110', '0000001110'),
    5: ('1111111111', '1111111111', '1110000000', '1110000000', '1111111110', '1111111111', '0000000111', '0000000111', '1111111111', '1111111110'),
    6: ('0111111110', '1111111110', '1110000000', '1110000000', '1111111110', '1111111111', '1110000111', '1110000111', '1111111111', '0111111110'),
    7: ('1111111111', '1111111111', '0000000111', '0000000111', '0000001110', '0000001110', '0000011100', '0000011100', '0000111000', '0000111000'),
    8: ('0111111110', '1111111111', '1110000111', '1110000111', '0111111110', '0111111110', '1110000111', '1110000111', '1111111111', '0111111110'),
}

SEGMENT_PATTERN = ('0aaaaaaaaa0', 'f0aaaaaaa0b', 'ff0aaaaa0bb', 'fff00000bbb', 'fff00000bbb', 'fff00000bbb', 'fff00000bbb', 'fff00000bbb', 'ff0000000bb',
                   'f0ggggggg0b', '0ggggggggg0', 'e0ggggggg0c', 'ee0000000cc', 'eee00000ccc', 'eee00000ccc', 'eee00000ccc', 'eee00000ccc', 'eee00000ccc',
                   'ee0ddddd0cc', 'e0ddddddd0c', '0ddddddddd0')

SEGMENT_PARITY = {
    "a": 0, "b": 1, "c": 1, "d": 0,
    "e": 1, "f": 1, "g": 0,
}


def _blend_colors(
    first: str,
    second: str,
    fraction: float = 0.5,
) -> tuple[int, int, int]:
    """Blend two hex colors using the given foreground fraction."""
    a = tuple(int(first[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(second[i:i + 2], 16) for i in (1, 3, 5))

    return tuple(
        round(fraction * x + (1 - fraction) * y)
        for x, y in zip(a, b)
    )


def _bevel(draw, box, sunken=False, thickness=2):
    """Draw a constant-width bevel with blended diagonal corners.

    The box uses half-open xyxy coordinates.
    """
    x1, y1, x2, y2 = box
    t = thickness

    if t < 1 or 2 * t >= min(x2 - x1, y2 - y1):
        raise ValueError("Bevel thickness must fit inside the box")

    upper_left = "#808080" if sunken else "#FFFFFF"
    lower_right = "#FFFFFF" if sunken else "#808080"

    # 1. Bottom/right base rectangle
    draw.rectangle(
        (x1, y1, x2 - 1, y2 - 1),
        fill=lower_right,
    )

    # 2. Top/left six-vertex polygon
    draw.polygon(
        [
            (x1, y1),
            (x2 - 1, y1),
            (x2 - t - 1, y1 + t),
            (x1 + t, y1 + t),
            (x1 + t, y2 - t - 1),
            (x1, y2 - 1),
        ],
        fill=upper_left,
    )

    # 3. Flat center
    draw.rectangle(
        (
            x1 + t,
            y1 + t,
            x2 - t - 1,
            y2 - t - 1,
        ),
        fill=BACKGROUND,
    )

    # 4. Blend the short 45-degree corner joins
    corner_color = _blend_colors(upper_left, lower_right)

    if t > 0:
        draw.line(
            [(x2 - 1, y1), (x2 - t, y1 + t - 1)],
            fill=corner_color,
        )
        draw.line(
            [(x1, y2 - 1), (x1 + t - 1, y2 - t)],
            fill=corner_color,
        )


def _number(draw: ImageDraw.ImageDraw, x: int, y: int, value: int) -> None:
    glyph = DIGITS[value]
    color = NUMBER_COLORS[value]

    for row, pixels in enumerate(glyph):
        for col, pixel in enumerate(pixels):
            if pixel == "1":
                draw.point((x + 3 + col, y + 3 + row), fill=color)


def _hidden(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    _bevel(
        draw,
        (x, y, x + CELL_SIZE, y + CELL_SIZE),
        thickness=2,
    )


def _revealed(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    exploded: bool = False,
) -> None:
    draw.rectangle(
        (x, y, x + 15, y + 15),
        fill="#FF0000" if exploded else BACKGROUND,
    )
    draw.line(
        ((x + 15, y), (x, y), (x, y + 15)),
        fill="#808080",
        width=1,
    )


def _flag(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.line(
        (x + 8, y + 3, x + 8, y + 12),
        fill="#000000",
        width=1,
    )
    draw.polygon(
        ((x + 8, y + 3), (x + 7, y + 3), (x + 5, y + 4),
         (x + 4, y + 5), (x + 5, y + 6), (x + 7, y + 7),
         (x + 8, y + 7)),
        fill="#FF0000",
    )
    draw.polygon(
        ((x + 6, y + 10), (x + 9, y + 10), (x + 11, y + 11),
         (x + 11, y + 12), (x + 4, y + 12), (x + 4, y + 11)),
        fill="#000000",
    )


def _mine(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    cx, cy = x + 8, y + 8

    for dx, dy in (
        (-6, 0), (6, 0), (0, -6), (0, 6),
        (-4, -4), (4, -4), (-4, 4), (4, 4),
    ):
        draw.line(
            (cx, cy, cx + dx, cy + dy),
            fill="#000000",
            width=1,
        )

    draw.polygon(
        [
            (cx - 2, cy - 4),
            (cx + 2, cy - 4),
            (cx + 4, cy - 2),
            (cx + 4, cy + 2),
            (cx + 2, cy + 4),
            (cx - 2, cy + 4),
            (cx - 4, cy + 2),
            (cx - 4, cy - 2),
        ],
        fill="#000000",
    )
    draw.rectangle(
        (cx - 2, cy - 2, cx - 1, cy - 1),
        fill="#FFFFFF",
    )


def _wrong_flag(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    _mine(draw, x, y)

    # Main diagonals
    draw.line((x + 2, y + 3, x + 13, y + 14), fill="#FF0000", width=1)
    draw.line((x + 13, y + 3, x + 2, y + 14), fill="#FF0000", width=1)

    # Slight horizontal offset to fake ~1.5px thickness
    draw.line((x + 3, y + 3, x + 14, y + 14), fill="#FF0000", width=1)
    draw.line((x + 14, y + 3, x + 3, y + 14), fill="#FF0000", width=1)


def render_board(
    game: dict,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Render a numerical game state to a classic 16px grid."""
    board = game["board"]
    revealed = game["revealed"]
    flags = game["flags"]
    status = game["status"]
    triggered = game["triggered_mine"]

    if board.ndim != 2 or board.shape != revealed.shape or board.shape != flags.shape:
        raise ValueError("board, revealed, and flags must have the same 2D shape")

    if status not in {"new", "active", "won", "lost"}:
        raise ValueError(f"Unknown status: {status}")

    if triggered is not None and (status != "lost" or board[triggered] != -1):
        raise ValueError("Triggered mine must be a mine in a lost game")

    rows, columns = board.shape
    image = Image.new(
        "RGB",
        (columns * CELL_SIZE, rows * CELL_SIZE),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(image)

    for row in range(rows):
        for column in range(columns):
            x, y = column * CELL_SIZE, row * CELL_SIZE
            pos = (row, column)
            value = int(board[pos])

            if pos == triggered:
                _revealed(draw, x, y, exploded=True)
                _mine(draw, x, y)

            elif bool(flags[pos]):
                if status == "lost" and value != -1:
                    _revealed(draw, x, y)
                    _wrong_flag(draw, x, y)
                else:
                    _hidden(draw, x, y)
                    _flag(draw, x, y)

            elif status == "lost" and value == -1:
                _revealed(draw, x, y)
                _mine(draw, x, y)

            elif bool(revealed[pos]):
                _revealed(draw, x, y)
                if value > 0:
                    _number(draw, x, y, value)

            else:
                _hidden(draw, x, y)

    return image, (0, 0, image.width, image.height)


SEGMENTS = {
    "0": "abcdef",
    "1": "bc",
    "2": "abged",
    "3": "abgcd",
    "4": "fgbc",
    "5": "afgcd",
    "6": "afgecd",
    "7": "abc",
    "8": "abcdefg",
    "9": "abfgcd",
    "-": "g",
}


def _counter(draw, box, value):
    _bevel(draw, box, sunken=True, thickness=1)

    x1, y1, x2, y2 = box

    draw.rectangle(
        (x1 + 1, y1 + 1, x2 - 2, y2 - 2),
        fill="#000000",
    )

    value = max(-99, min(999, value))
    text = f"{value:03d}" if value >= 0 else f"-{abs(value):02d}"

    for i, digit in enumerate(text):
        # Each digit occupies 13x23, with an inner 11x21 pattern.
        x = x1 + 2 + i * 13
        y = y1 + 2
        active = SEGMENTS[digit]

        for py, row in enumerate(SEGMENT_PATTERN):
            for px, segment in enumerate(row):
                if segment == "0":
                    continue

                if segment in active:
                    color = "#FF0000"
                elif (px + py) % 2 == SEGMENT_PARITY[segment]:
                    color = "#800000"
                else:
                    continue

                draw.point((x + px, y + py), fill=color)


def _smiley(draw, box, status):
    x1, y1, x2, y2 = box

    # Outer dark frame, excluding top-right and bottom-left corners.
    draw.line(
        [(x2 - 2, y1), (x1, y1), (x1, y2 - 2)],
        fill="#808080",
    )
    draw.line(
        [(x2 - 1, y1 + 1), (x2 - 1, y2 - 1), (x1 + 1, y2 - 1)],
        fill="#808080",
    )

    # Inner raised button.
    inner_box = (x1 + 1, y1 + 1, x2 - 1, y2 - 1)
    _bevel(draw, inner_box, thickness=2)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Yellow face with black outline.
    draw.ellipse(
        (cx - 8, cy - 8, cx + 8, cy + 8),
        fill="#FFFF00",
        outline="#000000",
    )

    if status == "lost":
        # Crossed eyes and frown.
        for dx in (-3, 3):
            draw.line(
                (cx + dx - 1, cy - 4, cx + dx + 1, cy - 2),
                fill="black",
            )
            draw.line(
                (cx + dx + 1, cy - 4, cx + dx - 1, cy - 2),
                fill="black",
            )

        draw.line(
            (cx - 4, cy + 4, cx - 2, cy + 2,
             cx + 2, cy + 2, cx + 4, cy + 4),
            fill="black",
        )

    elif status == "won":
        # Sunglasses.
        draw.line(
            (cx - 4, cy - 3, cx + 4, cy - 3),
            fill="black",
        )

        draw.ellipse(
            (cx - 4, cy - 3, cx - 1, cy),
            fill="black",
        )
        draw.ellipse(
            (cx + 1, cy - 3, cx + 4, cy),
            fill="black",
        )

        draw.line((cx - 5, cy - 2, cx - 7, cy), fill="black")
        draw.line((cx + 5, cy - 2, cx + 7, cy), fill="black")

        # Shaded pixels at the bottom of the lenses.
        draw.point((cx - 4, cy), fill="#808000")
        draw.point((cx + 4, cy), fill="#808000")

        # Short smile.
        draw.line(
            (cx - 3, cy + 3, cx - 2, cy + 4,
             cx + 2, cy + 4, cx + 3, cy + 3),
            fill="black",
        )

    else:
        # Normal eyes and smile.
        draw.rectangle(
            (cx - 3, cy - 3, cx - 2, cy - 2),
            fill="black",
        )
        draw.rectangle(
            (cx + 2, cy - 3, cx + 3, cy - 2),
            fill="black",
        )
        draw.line(
            (cx - 4, cy + 2, cx - 2, cy + 4,
             cx + 2, cy + 4, cx + 4, cy + 2),
            fill="black",
        )


def render_game_region(
    game: dict,
) -> tuple[Image.Image, dict[str, tuple[int, int, int, int]]]:
    elapsed_seconds = game["elapsed_seconds"]

    grid, _ = render_board(game)
    status = game["status"]

    # Measured XP layout dimensions
    outer_top = 4
    outer_left = 3

    left_gap = 6
    right_gap = 5
    bottom_gap = 5

    header_to_grid = 6
    header_height = 37
    grid_border = 3

    # Header panel and grid frame share the same horizontal boundaries
    panel_x = outer_left + left_gap
    panel_y = outer_top + 6
    panel_width = grid.width + 2 * grid_border

    header_box = (
        panel_x,
        panel_y,
        panel_x + panel_width,
        panel_y + header_height,
    )

    frame_y = header_box[3] + header_to_grid

    grid_frame = (
        panel_x,
        frame_y,
        panel_x + panel_width,
        frame_y + grid.height + 2 * grid_border,
    )

    grid_x = panel_x + grid_border
    grid_y = frame_y + grid_border

    width = grid_frame[2] + right_gap
    height = grid_frame[3] + bottom_gap

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    # White outer panel strips: top and left only.
    draw.rectangle((0, 0, image.width - 1, outer_top - 1), fill="#FFFFFF")
    draw.rectangle((0, 0, outer_left - 1, image.height - 1), fill="#FFFFFF")

    _bevel(draw, header_box, sunken=True, thickness=2)

    counter_y = panel_y + 6
    counter_width = 41
    counter_height = 25
    control_inset = 7

    mine_counter = (
        panel_x + control_inset,
        counter_y,
        panel_x + control_inset + counter_width,
        counter_y + counter_height,
    )

    timer = (
        header_box[2] - control_inset - 2 - counter_width,
        counter_y,
        header_box[2] - control_inset - 2,
        counter_y + counter_height,
    )

    button_x = (header_box[0] + header_box[2] - 26) // 2

    reset_button = (
        button_x,
        counter_y,
        button_x + 26,
        counter_y + 26,
    )

    remaining_mines = (
        int(np.count_nonzero(game["board"] == -1))
        - int(np.count_nonzero(game["flags"]))
    )

    _counter(draw, mine_counter, remaining_mines)
    _counter(draw, timer, min(elapsed_seconds, 999))
    _smiley(draw, reset_button, status)

    # Sunken grid border
    _bevel(
        draw,
        grid_frame,
        sunken=True,
        thickness=grid_border,
    )

    image.paste(grid, (grid_x, grid_y))

    boxes = {
        "game_region": (0, 0, width, height),
        "grid": (
            grid_x,
            grid_y,
            grid_x + grid.width,
            grid_y + grid.height,
        ),
        "mine_counter": mine_counter,
        "reset_button": reset_button,
        "timer": timer,
    }

    return image, boxes