
import random
from dataclasses import dataclass, field


Position = tuple[int, int]


@dataclass
class MockGame:
    rows: int = 9
    cols: int = 9
    mine_count: int = 10

    mines: set[Position] = field(default_factory=set, init=False)
    revealed: set[Position] = field(default_factory=set, init=False)
    flags: set[Position] = field(default_factory=set, init=False)

    game_over: bool = field(default=False, init=False)
    won: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Board dimensions must be positive")

        if not 0 <= self.mine_count < self.rows * self.cols:
            raise ValueError("Invalid mine count")

        self.reset()

    def reset(self):
        positions = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
        ]

        self.mines = set(random.sample(positions, self.mine_count))
        self.revealed.clear()
        self.flags.clear()
        self.game_over = False
        self.won = False

    @property
    def remaining_mines(self) -> int:
        return self.mine_count - len(self.flags)

    def neighbors(self, row: int, col: int):
        for r in range(max(0, row - 1), min(self.rows, row + 2)):
            for c in range(max(0, col - 1), min(self.cols, col + 2)):
                if (r, c) != (row, col):
                    yield r, c

    def adjacent_mines(self, row: int, col: int) -> int:
        return sum(
            pos in self.mines
            for pos in self.neighbors(row, col)
        )

    def _validate(self, row: int, col: int):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("Cell outside board")

    def toggle_flag(self, row: int, col: int):
        self._validate(row, col)
        pos = (row, col)

        if self.game_over or pos in self.revealed:
            return

        if pos in self.flags:
            self.flags.remove(pos)
        else:
            self.flags.add(pos)

    def reveal(self, row: int, col: int):
        self._validate(row, col)
        pos = (row, col)

        if (
            self.game_over
            or pos in self.revealed
            or pos in self.flags
        ):
            return

        # First reveal is always safe.
        if not self.revealed and pos in self.mines:
            candidates = [
                (r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if (r, c) not in self.mines and (r, c) != pos
            ]
            self.mines.remove(pos)
            self.mines.add(random.choice(candidates))

        if pos in self.mines:
            self.revealed.add(pos)
            self.game_over = True
            return

        # Flood-fill empty regions, without recursion.
        pending = [pos]

        while pending:
            current = pending.pop()

            if current in self.revealed or current in self.flags:
                continue

            if current in self.mines:
                continue

            self.revealed.add(current)

            if self.adjacent_mines(*current) == 0:
                pending.extend(self.neighbors(*current))

        if len(self.revealed) == self.rows * self.cols - self.mine_count:
            self.game_over = True
            self.won = True
