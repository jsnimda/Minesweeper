
"""Procedural 2048 and sliding-number puzzles rendered with Skia."""

import random

import skia

from ..scaling import pixels, validate_scale



TILE_COLORS = {
    0: "#CDC1B4",
    2: "#EEE4DA",
    4: "#EDE0C8",
    8: "#F2B179",
    16: "#F59563",
    32: "#F67C5F",
    64: "#F65E3B",
    128: "#EDCF72",
    256: "#EDCC61",
    512: "#EDC850",
    1024: "#EDC53F",
    2048: "#EDC22E",
}


def _color(value):
    value = value.lstrip("#")
    return skia.ColorSetRGB(
        *(int(value[i:i + 2], 16) for i in (0, 2, 4))
    )


def _paint(color):
    return skia.Paint(
        Color=_color(color),
        AntiAlias=True,
    )


def _rect(canvas, x, y, w, h, color, radius=0):
    shape = skia.Rect.MakeXYWH(x, y, w, h)

    if radius:
        canvas.drawRoundRect(
            shape, radius, radius, _paint(color)
        )
    else:
        canvas.drawRect(shape, _paint(color))


def _text(
    canvas, value, x, y,
    size=14, color="#51483E", bold=False,
):
    style = (
        skia.FontStyle.Bold()
        if bold else skia.FontStyle.Normal()
    )
    font = skia.Font(
        skia.Typeface.MakeFromName(None, style),
        size,
    )
    canvas.drawString(
        str(value), x, y, font, _paint(color)
    )
    return font


def _center_text(
    canvas, value, cx, cy,
    size, color, bold=False,
):
    """Center text using measured font width."""
    value = str(value)
    style = (
        skia.FontStyle.Bold()
        if bold else skia.FontStyle.Normal()
    )
    font = skia.Font(
        skia.Typeface.MakeFromName(None, style),
        size,
    )
    x = cx - font.measureText(value) / 2
    y = cy + size * 0.35
    _text(canvas, value, x, y, size, color, bold)


def _spawn_tile(board, rng):
    empty = [
        (row, col)
        for row in range(4)
        for col in range(4)
        if board[row][col] == 0
    ]

    if not empty:
        return False

    row, col = rng.choice(empty)
    board[row][col] = 4 if rng.random() < 0.10 else 2
    return True


def _merge_line(values):
    """Slide and merge one line toward the beginning."""
    numbers = [value for value in values if value]
    result = []
    points = 0
    i = 0

    while i < len(numbers):
        if (
            i + 1 < len(numbers)
            and numbers[i] == numbers[i + 1]
        ):
            merged = numbers[i] * 2
            result.append(merged)
            points += merged
            i += 2
        else:
            result.append(numbers[i])
            i += 1

    result.extend([0] * (4 - len(result)))
    return result, points


def _move_2048(board, direction):
    """Apply one real 2048 movement rule."""
    result = [row[:] for row in board]
    score = 0

    for index in range(4):
        if direction in ("left", "right"):
            cells = [(index, col) for col in range(4)]
        else:
            cells = [(row, index) for row in range(4)]

        if direction in ("right", "down"):
            cells.reverse()

        original = [result[row][col] for row, col in cells]
        merged, gained = _merge_line(original)
        score += gained

        for (row, col), value in zip(cells, merged):
            result[row][col] = value

    return result, score


def _generate_2048(rng):
    """Generate a reachable 2048 board through legal moves."""
    board = [[0] * 4 for _ in range(4)]
    _spawn_tile(board, rng)
    _spawn_tile(board, rng)

    score = 0
    successful_moves = 0
    target_moves = rng.randint(12, 65)

    for _ in range(target_moves * 5):
        direction = rng.choice(
            ("left", "right", "up", "down")
        )
        next_board, gained = _move_2048(board, direction)

        if next_board == board:
            continue

        board = next_board
        score += gained
        _spawn_tile(board, rng)
        successful_moves += 1

        if successful_moves >= target_moves:
            break

    return board, score


def _generate_sliding(rng):
    """Scramble a solved 15 Puzzle using only legal moves."""
    board = list(range(1, 16)) + [0]
    empty = 15
    previous = None

    move_count = rng.randint(50, 150)

    for _ in range(move_count):
        row, col = divmod(empty, 4)
        neighbors = []

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc

            if 0 <= nr < 4 and 0 <= nc < 4:
                neighbors.append(nr * 4 + nc)

        # Avoid immediately reversing the previous move.
        choices = [
            position for position in neighbors
            if position != previous
        ]
        next_empty = rng.choice(choices or neighbors)

        board[empty], board[next_empty] = (
            board[next_empty], board[empty]
        )
        previous, empty = empty, next_empty

    return (
        [
            board[row * 4:(row + 1) * 4]
            for row in range(4)
        ],
        move_count,
    )


def _draw_board(canvas, board, x, y, size, puzzle_type):
    gap = 10
    cell_size = (size - 5 * gap) / 4

    board_bg = (
        "#BBADA0" if puzzle_type == "2048"
        else "#CBD5E1"
    )
    _rect(canvas, x, y, size, size, board_bg, 9)

    for row in range(4):
        for col in range(4):
            value = board[row][col]
            cell_x = x + gap + col * (cell_size + gap)
            cell_y = y + gap + row * (cell_size + gap)

            if puzzle_type == "2048":
                background = TILE_COLORS.get(
                    value, "#3C3A32"
                )
                foreground = (
                    "#776E65" if value <= 4
                    else "#F9F6F2"
                )
            else:
                background = (
                    "#E2E8F0" if value == 0
                    else "#FFFFFF"
                )
                foreground = "#334155"

            _rect(
                canvas,
                cell_x, cell_y,
                cell_size, cell_size,
                background, 6,
            )

            if value == 0:
                continue

            font_size = (
                29 if value < 100
                else 24 if value < 1000
                else 20
            )

            font_size *= min(1, cell_size / 75)

            _center_text(
                canvas,
                value,
                cell_x + cell_size / 2,
                cell_y + cell_size / 2,
                font_size,
                foreground,
                bold=True,
            )


def render_other_puzzle(
    seed: int = 42,
    *,
    width: int = 520,
    height: int = 610,
    puzzle_type: str | None = None,
    scale: float = 1.0,
) -> tuple[skia.Image, dict]:
    """Render a deterministic non-Minesweeper grid puzzle.

    puzzle_type: "2048" or "sliding"
    Returns no Minesweeper target annotations.
    """
    if width < 360 or height < 460:
        raise ValueError("Puzzle canvas is too small")

    rng = random.Random(seed)

    if puzzle_type is None:
        puzzle_type = rng.choices(
            ("2048", "sliding"),
            weights=(60, 40),
            k=1,
        )[0]

    if puzzle_type not in {"2048", "sliding"}:
        raise ValueError(
            f"Unknown puzzle type: {puzzle_type}"
        )

    scale = validate_scale(scale)
    surface = skia.Surface(pixels(width, scale), pixels(height, scale))
    canvas = surface.getCanvas()
    canvas.scale(scale, scale)

    is_2048 = puzzle_type == "2048"
    background = "#FAF8EF" if is_2048 else "#F5F7FB"
    accent = "#8F7A66" if is_2048 else "#397CC1"
    text_color = "#776E65" if is_2048 else "#283648"

    canvas.clear(_color(background))

    if is_2048:
        board, score = _generate_2048(rng)
        title = "2048"
        subtitle = "Join the numbers, reach 2048!"
    else:
        board, move_count = _generate_sliding(rng)
        score = None
        title = "15 Puzzle"
        subtitle = "Arrange the tiles in order."

    # Heading.
    _text(
        canvas, title,
        30, 65,
        39 if is_2048 else 32,
        text_color, True,
    )

    _text(
        canvas, subtitle,
        31, 99,
        13, text_color,
    )

    # Score or progress card.
    card_w = 105
    card_x = width - card_w - 29

    _rect(
        canvas, card_x, 23,
        card_w, 61, accent, 7,
    )

    _center_text(
        canvas,
        "SCORE" if is_2048 else "MOVES",
        card_x + card_w / 2,
        40,
        11, "#FFFFFF", True,
    )

    _center_text(
        canvas,
        f"{score:,}" if is_2048 else str(move_count),
        card_x + card_w / 2,
        64,
        21, "#FFFFFF", True,
    )

    board_size = min(width - 52, height - 210)
    board_x = (width - board_size) / 2
    board_y = 128

    _draw_board(
        canvas,
        board,
        board_x, board_y,
        board_size,
        puzzle_type,
    )

    # Bottom controls.
    controls_y = board_y + board_size + 18

    _rect(
        canvas,
        width - 154, controls_y,
        128, 40,
        accent, 7,
    )

    _center_text(
        canvas, "New Game",
        width - 90, controls_y + 20,
        14, "#FFFFFF", True,
    )

    _text(
        canvas,
        "Use arrow keys to move"
        if is_2048 else "Slide a tile into the empty space",
        29, controls_y + 25,
        11, text_color,
    )

    return surface.makeImageSnapshot(), {}
