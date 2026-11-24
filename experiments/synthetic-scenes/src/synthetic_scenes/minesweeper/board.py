
import random
from collections import deque
from enum import IntEnum

import numpy as np


def generate_minefield(
    rows: int,
    columns: int,
    mine_count: int,
    rng: random.Random,
    *, target_number: int | None = None,
    target_position: tuple[int, int] | None = None,
) -> np.ndarray:
    if rows <= 0 or columns <= 0:
        raise ValueError("Board dimensions must be positive")

    if not 0 <= mine_count < rows * columns:
        raise ValueError("Invalid mine count")

    board = np.zeros((rows, columns), dtype=np.int8)

    if target_number is None:
        positions = rng.sample(range(rows * columns), mine_count)
    else:
        if target_position is None or not 1 <= target_number <= 8:
            raise ValueError("Target number requires a position and value 1..8")
        r, c = target_position
        if not (0 <= r < rows and 0 <= c < columns):
            raise ValueError("Target position outside board")
        neighbors = [rr * columns + cc
                     for rr in range(max(0, r-1), min(rows, r+2))
                     for cc in range(max(0, c-1), min(columns, c+2))
                     if (rr, cc) != (r, c)]
        excluded = set(neighbors) | {r * columns + c}
        available = [i for i in range(rows * columns) if i not in excluded]
        if not (target_number <= len(neighbors) and
                0 <= mine_count - target_number <= len(available)):
            raise ValueError("Target incompatible with mine count or dimensions")
        positions = (rng.sample(neighbors, target_number) +
                     rng.sample(available, mine_count - target_number))

    for index in positions:
        row, column = divmod(index, columns)
        board[row, column] = -1

    for row in range(rows):
        for column in range(columns):
            if board[row, column] == -1:
                continue

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr = row + dr
                    nc = column + dc

                    if (
                        0 <= nr < rows
                        and 0 <= nc < columns
                        and board[nr, nc] == -1
                    ):
                        board[row, column] += 1

    return board


def reveal_region(
    board: np.ndarray,
    start_row: int,
    start_column: int,
) -> np.ndarray:
    rows, columns = board.shape

    if not (0 <= start_row < rows and 0 <= start_column < columns):
        raise ValueError("Reveal coordinates outside board")

    if board[start_row, start_column] == -1:
        raise ValueError("Cannot reveal a mine")

    revealed = np.zeros(board.shape, dtype=bool)
    queue = deque([(start_row, start_column)])

    while queue:
        row, column = queue.popleft()

        if revealed[row, column]:
            continue

        revealed[row, column] = True

        if board[row, column] != 0:
            continue

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = row + dr
                nc = column + dc

                if (
                    0 <= nr < rows
                    and 0 <= nc < columns
                    and board[nr, nc] != -1
                    and not revealed[nr, nc]
                ):
                    queue.append((nr, nc))

    return revealed


def simulate_progress(
    board: np.ndarray,
    progress: float,
    rng: random.Random,
    allow_win: bool = False,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("Progress must be between 0 and 1")

    revealed = np.zeros(board.shape, dtype=bool)

    if initial is not None:
        revealed |= initial

    safe_positions = [
        (row, column)
        for row in range(board.shape[0])
        for column in range(board.shape[1])
        if board[row, column] != -1
    ]

    total_safe = len(safe_positions)
    target = max(1, round(total_safe * progress))

    rng.shuffle(safe_positions)

    for row, column in safe_positions:
        if np.count_nonzero(revealed) >= target:
            break

        if revealed[row, column]:
            continue

        region = reveal_region(board, row, column)
        candidate = revealed | region

        if not allow_win and np.count_nonzero(candidate) == total_safe:
            continue

        revealed = candidate

    return revealed


def generate_game(
    rows: int = 9,
    columns: int = 9,
    mine_count: int = 10,
    status: str = "new",
    progress: float = 0.0,
    seed: int = 42,
    elapsed_seconds: int = 0,
    *, mode: str = "normal",
    target_number: int | None = None,
) -> dict:
    if status not in {"new", "active", "won", "lost"}:
        raise ValueError(f"Unsupported status: {status}")

    if not 0.0 <= progress <= 1.0:
        raise ValueError("Progress must be between 0 and 1")

    if status == "active" and progress <= 0.0:
        raise ValueError("Active games require positive progress")

    if status == "active" and progress >= 1.0:
        raise ValueError("Active games must have progress below 1")

    if status == "lost" and mine_count == 0:
        raise ValueError("Cannot lose a game without mines")

    if isinstance(elapsed_seconds, (bool, np.bool_)) or not isinstance(
        elapsed_seconds, (int, np.integer)
    ):
        raise ValueError("Elapsed seconds must be an integer")

    if elapsed_seconds < 0:
        raise ValueError("Elapsed seconds cannot be negative")

    if status == "new" and elapsed_seconds != 0:
        raise ValueError("New games must have zero elapsed seconds")

    rng = random.Random(seed)

    if rows <= 0 or columns <= 0 or not 0 <= mine_count < rows * columns:
        raise ValueError("Invalid board dimensions or mine count")
    if mode not in {"normal", "targeted"}:
        raise ValueError("Unknown generation mode")
    target = None
    if mode == "targeted":
        if status == "new" or target_number is None or not 1 <= target_number <= 8:
            raise ValueError("Targeted mode requires a visible number 1..8")
        candidates = []
        for r in range(rows):
            for c in range(columns):
                n = (min(rows, r+2)-max(0, r-1)) * (min(columns, c+2)-max(0, c-1)) - 1
                if target_number <= n and 0 <= mine_count-target_number <= rows*columns-n-1:
                    candidates.append((r, c))
        if not candidates or (status in {"active", "lost"} and rows*columns-mine_count == 1):
            raise ValueError("No feasible target position for this game")
        target = rng.choice(candidates)
    elif target_number is not None:
        raise ValueError("Use targeted mode for target_number")

    board = generate_minefield(
        rows=rows,
        columns=columns,
        mine_count=mine_count,
        rng=rng,
        target_number=target_number,
        target_position=target,
    )

    revealed = np.zeros(board.shape, dtype=bool)
    flags = np.zeros(board.shape, dtype=bool)
    triggered_mine = None

    if status == "won":
        revealed = board != -1
        flags = board == -1

    elif status in {"active", "lost"}:
        revealed = simulate_progress(
            board,
            progress,
            rng,
            initial=reveal_region(board, *target) if target is not None else None,
        )

        if status == "active" and not np.any(revealed):
            raise ValueError(
                "Could not generate an active game "
                "without completing the board"
            )

        mines = [
            (int(row), int(column))
            for row, column in np.argwhere(board == -1)
        ]

        if status == "lost":
            triggered_mine = rng.choice(mines)
            mines.remove(triggered_mine)

        # Approximate number of correctly placed flags.
        max_flags = round(len(mines) * progress)
        flag_count = rng.randint(0, max_flags)

        for row, column in rng.sample(mines, flag_count):
            flags[row, column] = True

        # Occasionally add a wrong flag to a lost game.
        if status == "lost":
            safe_hidden = [
                (int(row), int(column))
                for row, column in np.argwhere((board != -1) & ~revealed)
            ]

            if safe_hidden and rng.random() < 0.5:
                wrong_flag = rng.choice(safe_hidden)
                flags[wrong_flag] = True

    return {
        "generation_mode": mode,
        "target_number": target_number,
        "target_position": target,
        "seed": seed,
        "board": board,
        "revealed": revealed,
        "flags": flags,
        "status": status,
        "triggered_mine": triggered_mine,
        "elapsed_seconds": int(elapsed_seconds),
    }


class CellState(IntEnum):
    REVEALED_EMPTY = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    HIDDEN = 9
    FLAG = 10
    MINE = 11
    EXPLODED_MINE = 12
    WRONG_FLAG = 13


def get_visible_cells(game: dict) -> np.ndarray:
    board = game["board"]
    cells = np.full(board.shape, CellState.HIDDEN, dtype=np.int8)
    cells[game["revealed"]] = board[game["revealed"]]
    if game["status"] == "lost":
        cells[board == -1] = CellState.MINE
    cells[game["flags"]] = CellState.FLAG
    if game["status"] == "lost":
        cells[game["flags"] & (board != -1)] = CellState.WRONG_FLAG
        if game["triggered_mine"] is not None:
            cells[game["triggered_mine"]] = CellState.EXPLODED_MINE
    return cells
