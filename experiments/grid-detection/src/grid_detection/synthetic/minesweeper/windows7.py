"""Windows 7 Minesweeper blue-theme grid renderer (no window chrome)."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSET_DIR = (
    Path(__file__).resolve().parents[4]
    / "data" / "assets" / "minesweeper" / "windows7"
)

CELL_SIZE = 36
ATLAS_PITCH = 40
ATLAS_COLUMNS = 25
ATLAS_LOGICAL_COLUMNS = 30
ATLAS_LOGICAL_ROWS = 24
ATLAS_ORIGIN = (2, 2)
REVEALED_X_OFFSET = 1000

# Fixed sprite crops, grouped by source sheet (half-open xyxy).
BLUE_SPRITES = {
    "flag": (802, 1122, 838, 1158),
    "wrong_flag": (922, 1122, 956, 1156),
    "horizontal_shadow": (962, 1122, 998, 1128),
    "vertical_shadow": (2026, 2, 2031, 38),
    "counter_box": (2, 1162, 82, 1204),
}
MINE_SPRITES = {
    "mine": (56, 2, 92, 38),
    "counter_icon": (2, 2, 52, 52),
}

MINE_OPACITY = 0.5
MINE_BLEND = 0.9

DIGIT_X = 2001
DIGIT_Y = 1
DIGIT_PITCH = 30
DIGIT_CROP_SIZE = (22, 28)

FRAME_SPRITES = {
    "top_left": (256, 1954, 450, 2014),
    "top_repeat": (2044, 680, 2046, 740),
    "top_right": (2, 1954, 252, 2014),

    "left_top": (1964, 534, 2024, 582),
    "left_repeat": (1964, 586, 2024, 588),
    "left_bottom": (1458, 946, 1518, 1290),

    "right_top": (1876, 750, 1936, 894),
    "right_repeat": (1964, 592, 2024, 594),
    "right_bottom": (1812, 610, 1872, 848),

    "bottom_left": (1876, 680, 2034, 746),
    "bottom_repeat": (2038, 680, 2040, 746),
    "bottom_right": (1876, 610, 2042, 676),
}

FRAME_TOP = 60
FRAME_SIDE = 60
FRAME_BOTTOM = 66

OTHER_SPRITES = {
    "timer_icon": (1812, 852, 1862, 902),
}

CONTROL_POSITION = 0.15
CONTROL_GAP = 12
CONTROL_BOTTOM_MARGIN = 8


@lru_cache(maxsize=1)
def _assets() -> tuple[Image.Image, Image.Image, Image.Image]:
    blue = Image.open(ASSET_DIR / "bluesheetx2.png").convert("RGBA")
    mine = Image.open(ASSET_DIR / "minesweepersheetx2.png").convert("RGBA")
    other = Image.open(ASSET_DIR / "othersheetx2.png").convert("RGBA")

    if blue.size != (2033, 1205) or mine.size != (93, 53) or other.size != (2047, 2015):
        raise ValueError("Unexpected Windows 7 atlas dimensions; expected x2 sheets")

    return blue, mine, other


def _paste_centered(canvas: Image.Image, sprite: Image.Image, x: int, y: int) -> None:
    canvas.alpha_composite(sprite, (x + (CELL_SIZE - sprite.width) // 2,
                                    y + (CELL_SIZE - sprite.height) // 2))


def _cell(canvas: Image.Image, sheet: Image.Image, x: int, y: int,
          row: int, column: int, *, opened: bool) -> None:
    index = row * ATLAS_LOGICAL_COLUMNS + column
    atlas_column, atlas_row = index % ATLAS_COLUMNS, index // ATLAS_COLUMNS
    sx = ATLAS_ORIGIN[0] + atlas_column * ATLAS_PITCH
    sy = ATLAS_ORIGIN[1] + atlas_row * ATLAS_PITCH
    if opened:
        sx += REVEALED_X_OFFSET
    canvas.alpha_composite(sheet.crop((sx, sy, sx + CELL_SIZE, sy + CELL_SIZE)), (x, y))


def _number(canvas: Image.Image, sheet: Image.Image, x: int, y: int, value: int) -> None:
    sx = DIGIT_X
    sy = DIGIT_Y + (value - 1) * DIGIT_PITCH
    width, height = DIGIT_CROP_SIZE
    sprite = sheet.crop((sx, sy, sx + width, sy + height))
    _paste_centered(canvas, sprite, x + 1, y + 1)


def _flag(canvas: Image.Image, sheet: Image.Image, x: int, y: int) -> None:
    _paste_centered(canvas, sheet.crop(BLUE_SPRITES["flag"]), x, y)


def _mine(canvas: Image.Image, sheet: Image.Image, x: int, y: int,
          *, triggered: bool = False) -> None:
    sprite = sheet.crop(MINE_SPRITES["mine"])

    color = (255, 0, 0) if triggered else (0, 0, 0)
    rgb = np.asarray(sprite.convert("RGB"), dtype=np.float32)
    tint = np.array(color, dtype=np.float32) / 255
    result = rgb * ((1 - MINE_BLEND) + MINE_BLEND * tint)

    alpha = sprite.getchannel("A").point(lambda a: round(a * MINE_OPACITY))
    sprite = Image.fromarray(np.clip(np.rint(result), 0, 255).astype(np.uint8), "RGB")
    sprite.putalpha(alpha)

    _paste_centered(canvas, sprite, x, y)


def _wrong_flag(canvas: Image.Image, sheet: Image.Image, x: int, y: int) -> None:
    _paste_centered(canvas, sheet.crop(BLUE_SPRITES["wrong_flag"]), x + 1, y + 1)


def _shadow(canvas: Image.Image, sheet: Image.Image, x: int, y: int,
            *, top: bool, left: bool) -> None:
    if top:
        canvas.alpha_composite(sheet.crop(BLUE_SPRITES["horizontal_shadow"]), (x, y))
    if left:
        canvas.alpha_composite(sheet.crop(BLUE_SPRITES["vertical_shadow"]), (x, y))


def _opened_cells(game: dict) -> np.ndarray:
    """Visually open cells, including mines and incorrect flags on loss."""
    opened = np.asarray(game["revealed"], dtype=bool).copy()
    if game["status"] == "lost":
        board = game["board"]
        opened |= board == -1
        opened |= np.asarray(game["flags"], dtype=bool) & (board != -1)
        if game["triggered_mine"] is not None:
            opened[game["triggered_mine"]] = True
    return opened


def render_board(game: dict) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Return RGBA grid and its local half-open xyxy bounding box."""
    board = game["board"]
    revealed = game["revealed"]
    flags = game["flags"]
    status = game["status"]
    triggered = game["triggered_mine"]

    if board.ndim != 2 or board.shape != revealed.shape or board.shape != flags.shape:
        raise ValueError("board, revealed, and flags must have matching 2D shapes")
    if status not in {"new", "active", "won", "lost"}:
        raise ValueError(f"Unknown status: {status}")
    if triggered is not None and (status != "lost" or board[triggered] != -1):
        raise ValueError("Triggered mine must be a mine in a lost game")

    rows, columns = board.shape
    if not (1 <= rows <= ATLAS_LOGICAL_ROWS and 1 <= columns <= ATLAS_LOGICAL_COLUMNS):
        raise ValueError("Windows 7 blue atlas supports 1..24 rows and 1..30 columns")

    row_offset = (ATLAS_LOGICAL_ROWS - rows) // 2
    col_offset = (ATLAS_LOGICAL_COLUMNS - columns) // 2

    blue_sheet, mine_sheet, _ = _assets()
    opened = _opened_cells(game)
    canvas = Image.new("RGBA", (columns * CELL_SIZE, rows * CELL_SIZE))

    for row in range(rows):
        for column in range(columns):
            x, y = column * CELL_SIZE, row * CELL_SIZE
            pos = (row, column)
            value = int(board[pos])
            _cell(canvas, blue_sheet, x, y, row + row_offset, column + col_offset,
                  opened=bool(opened[pos]))

            if revealed[pos] and value > 0:
                _number(canvas, blue_sheet, x, y, value)

            if opened[pos]:
                _shadow(canvas, blue_sheet, x, y,
                        top=row == 0 or not opened[row - 1, column],
                        left=column == 0 or not opened[row, column - 1])

            if status == "lost" and value == -1:
                _mine(canvas, mine_sheet, x, y, triggered=pos == triggered)
                if flags[pos]:
                    _flag(canvas, blue_sheet, x, y)
            elif status == "lost" and flags[pos]:
                _mine(canvas, mine_sheet, x, y)
                _wrong_flag(canvas, blue_sheet, x, y)
            elif flags[pos]:
                _flag(canvas, blue_sheet, x, y)

    return canvas, (0, 0, canvas.width, canvas.height)


def _repeat_strip(canvas: Image.Image, sprite: Image.Image, x: int, y: int,
                  length: int, *, horizontal: bool) -> None:
    step = sprite.width if horizontal else sprite.height

    for offset in range(0, length, step):
        if horizontal:
            piece = sprite.crop((0, 0, min(step, length - offset), sprite.height))
            canvas.alpha_composite(piece, (x + offset, y))
        else:
            piece = sprite.crop((0, 0, sprite.width, min(step, length - offset)))
            canvas.alpha_composite(piece, (x, y + offset))


def _frame(sheet: Image.Image, grid_width: int, grid_height: int) -> Image.Image:
    width = grid_width + 2 * FRAME_SIDE
    height = grid_height + FRAME_TOP + FRAME_BOTTOM
    canvas = Image.new("RGBA", (width, height))

    sprites = {name: sheet.crop(box) for name, box in FRAME_SPRITES.items()}

    # Top edge
    top_left = sprites["top_left"]
    top_right = sprites["top_right"]
    canvas.alpha_composite(top_left, (0, 0))
    _repeat_strip(canvas, sprites["top_repeat"], top_left.width, 0,
                  width - top_left.width - top_right.width, horizontal=True)
    canvas.alpha_composite(top_right, (width - top_right.width, 0))

    # Left edge
    left_top = sprites["left_top"]
    left_bottom = sprites["left_bottom"]
    left_repeat_y = FRAME_TOP + left_top.height
    left_bottom_y = height - left_bottom.height

    canvas.alpha_composite(left_top, (0, FRAME_TOP))
    _repeat_strip(canvas, sprites["left_repeat"], 0, left_repeat_y,
                  max(0, left_bottom_y - left_repeat_y), horizontal=False)
    canvas.alpha_composite(left_bottom, (0, left_bottom_y))

    # Right edge
    right_top = sprites["right_top"]
    right_bottom = sprites["right_bottom"]
    right_x = width - FRAME_SIDE
    right_repeat_y = FRAME_TOP + right_top.height
    right_bottom_y = height - right_bottom.height

    canvas.alpha_composite(right_top, (right_x, FRAME_TOP))
    _repeat_strip(canvas, sprites["right_repeat"], right_x, right_repeat_y,
                  max(0, right_bottom_y - right_repeat_y), horizontal=False)
    canvas.alpha_composite(right_bottom, (right_x, right_bottom_y))

    # Bottom edge
    bottom_left = sprites["bottom_left"]
    bottom_right = sprites["bottom_right"]
    bottom_y = height - FRAME_BOTTOM
    bottom_x = FRAME_SIDE
    bottom_width = width - 2 * FRAME_SIDE

    canvas.alpha_composite(bottom_left, (bottom_x, bottom_y))
    _repeat_strip(canvas, sprites["bottom_repeat"],
                  bottom_x + bottom_left.width, bottom_y,
                  max(0, bottom_width - bottom_left.width - bottom_right.width),
                  horizontal=True)
    canvas.alpha_composite(bottom_right,
                           (bottom_x + bottom_width - bottom_right.width, bottom_y))

    return canvas


def _control(
    canvas: Image.Image,
    blue_sheet: Image.Image,
    icon_sheet: Image.Image,
    icon_box: tuple[int, int, int, int],
    x: int,
    bottom_y: int,
    value: int,
    *,
    icon_left: bool,
) -> tuple[int, int, int, int]:
    box = blue_sheet.crop(BLUE_SPRITES["counter_box"])
    icon = icon_sheet.crop(icon_box)

    if icon_left:
        icon_x = x
        box_x = x + icon.width + CONTROL_GAP
    else:
        box_x = x
        icon_x = x + box.width + CONTROL_GAP

    baseline_y = bottom_y + FRAME_BOTTOM - CONTROL_BOTTOM_MARGIN
    icon_y = baseline_y - icon.height - 1
    box_y = baseline_y - box.height - 1

    canvas.alpha_composite(box, (box_x, box_y))
    canvas.alpha_composite(icon, (icon_x, icon_y))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (box_x + box.width // 2, box_y + box.height // 2),
        str(int(value)),
        fill=(245, 248, 255, 255),
        font=font,
        anchor="mm",
    )

    x1 = min(icon_x, box_x)
    y1 = min(icon_y, box_y)
    x2 = max(icon_x + icon.width, box_x + box.width)
    y2 = max(icon_y + icon.height, box_y + box.height)
    return x1, y1, x2, y2


def render_game_region(
    game: dict,
) -> tuple[Image.Image, dict[str, tuple[int, int, int, int]]]:
    grid, _ = render_board(game)
    blue_sheet, mine_sheet, other_sheet = _assets()

    canvas = _frame(other_sheet, grid.width, grid.height)

    grid_x = FRAME_SIDE
    grid_y = FRAME_TOP
    canvas.alpha_composite(grid, (grid_x, grid_y))

    remaining_mines = (
        int(np.count_nonzero(game["board"] == -1))
        - int(np.count_nonzero(game["flags"]))
    )
    elapsed_seconds = game["elapsed_seconds"]

    bottom_y = canvas.height - FRAME_BOTTOM

    timer_icon = other_sheet.crop(OTHER_SPRITES["timer_icon"])
    mine_icon = mine_sheet.crop(MINE_SPRITES["counter_icon"])
    counter_box = blue_sheet.crop(BLUE_SPRITES["counter_box"])

    timer_center = round(canvas.width * CONTROL_POSITION)
    mine_center = round(canvas.width * (1 - CONTROL_POSITION))

    timer_x = timer_center - timer_icon.width // 2
    mine_x = mine_center - mine_icon.width // 2 - counter_box.width - CONTROL_GAP - 1

    timer_box = _control(
        canvas, blue_sheet, other_sheet, OTHER_SPRITES["timer_icon"],
        timer_x, bottom_y, min(elapsed_seconds, 999), icon_left=True,
    )

    mine_box = _control(
        canvas, blue_sheet, mine_sheet, MINE_SPRITES["counter_icon"],
        mine_x, bottom_y, remaining_mines, icon_left=False,
    )

    boxes = {
        "game_region": (0, 0, canvas.width, canvas.height),
        "grid": (grid_x, grid_y, grid_x + grid.width, grid_y + grid.height),
        "timer": timer_box,
        "mine_counter": mine_box,
    }
    return canvas, boxes