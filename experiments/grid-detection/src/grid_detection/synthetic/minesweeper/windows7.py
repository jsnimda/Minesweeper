"""Windows 7 Minesweeper blue-theme grid renderer (no window chrome)."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

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
    "vertical_shadow": (2026, 2, 2031, 37),
}
MINE_SPRITES = {"mine": (56, 2, 92, 38)}

MINE_OPACITY = 0.5
MINE_BLEND = 0.9

DIGIT_X = 2001
DIGIT_Y = 1
DIGIT_PITCH = 30
DIGIT_CROP_SIZE = (22, 28)


@lru_cache(maxsize=1)
def _assets() -> tuple[Image.Image, Image.Image]:
    blue = Image.open(ASSET_DIR / "bluesheetx2.png").convert("RGBA")
    mine = Image.open(ASSET_DIR / "minesweepersheetx2.png").convert("RGBA")
    if blue.size != (2033, 1205) or mine.size != (93, 53):
        raise ValueError("Unexpected Windows 7 atlas dimensions; expected x2 sheets")
    return blue, mine


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

    blue_sheet, mine_sheet = _assets()
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
