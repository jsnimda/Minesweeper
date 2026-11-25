
import random
from collections import deque

import numpy as np


def generate_minefield(
    rows: int,
    columns: int,
    mine_count: int,
    rng: random.Random,
) -> np.ndarray:
    if rows <= 0 or columns <= 0:
        raise ValueError("Board dimensions must be positive")

    if not 0 <= mine_count < rows * columns:
        raise ValueError("Invalid mine count")

    board = np.zeros((rows, columns), dtype=np.int8)

    positions = rng.sample(range(rows * columns), mine_count)

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
) -> np.ndarray:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("Progress must be between 0 and 1")

    revealed = np.zeros(board.shape, dtype=bool)

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

    board = generate_minefield(
        rows=rows,
        columns=columns,
        mine_count=mine_count,
        rng=rng,
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
        "board": board,
        "revealed": revealed,
        "flags": flags,
        "status": status,
        "triggered_mine": triggered_mine,
        "elapsed_seconds": int(elapsed_seconds),
    }


if __name__ == "__main__":
    game = generate_game(
        rows=9,
        columns=9,
        mine_count=10,
        status="active",
        progress=0.45,
        seed=42,
    )

    print("Minefield:")
    print(game["board"])

    print("\nRevealed:")
    print(game["revealed"].astype(int))

    print("\nFlags:")
    print(game["flags"].astype(int))

    print("\nStatus:", game["status"])
    print("Triggered mine:", game["triggered_mine"])
