
import pytest

from minesweeper.devtools.game_model import MockGame


def test_initial_state():
    game = MockGame(rows=9, cols=9, mine_count=10)

    assert len(game.mines) == 10
    assert game.revealed == set()
    assert game.flags == set()
    assert not game.game_over


def test_adjacent_mines():
    game = MockGame(rows=3, cols=3, mine_count=1)
    game.mines = {(1, 1)}

    assert game.adjacent_mines(0, 0) == 1
    assert game.adjacent_mines(0, 1) == 1
    assert game.adjacent_mines(1, 1) == 0


def test_first_click_safe():
    game = MockGame(rows=3, cols=3, mine_count=1)
    game.mines = {(0, 0)}

    game.reveal(0, 0)

    assert (0, 0) not in game.mines
    assert (0, 0) in game.revealed
    assert not game.game_over or game.won


def test_reveal_mine_loses():
    game = MockGame(rows=3, cols=3, mine_count=1)
    game.mines = {(1, 1)}

    game.reveal(0, 0)
    game.reveal(1, 1)

    assert game.game_over
    assert not game.won


def test_empty_cell_expansion():
    game = MockGame(rows=4, cols=4, mine_count=1)
    game.mines = {(3, 3)}

    game.reveal(0, 0)

    assert len(game.revealed) == 15
    assert (3, 3) not in game.revealed
    assert game.won


def test_flags():
    game = MockGame(rows=3, cols=3, mine_count=1)

    game.toggle_flag(1, 1)

    assert (1, 1) in game.flags
    assert game.remaining_mines == 0

    game.reveal(1, 1)
    assert (1, 1) not in game.revealed

    game.toggle_flag(1, 1)
    assert game.flags == set()


def test_reset():
    game = MockGame(rows=3, cols=3, mine_count=1)

    game.toggle_flag(0, 0)
    game.reveal(1, 1)
    game.reset()

    assert game.flags == set()
    assert game.revealed == set()
    assert not game.game_over
    assert not game.won
    assert len(game.mines) == 1


@pytest.mark.parametrize(
    "rows, cols, mines",
    [(0, 3, 1), (3, 0, 1), (2, 2, 4), (3, 3, -1)],
)
def test_invalid_configuration(rows, cols, mines):
    with pytest.raises(ValueError):
        MockGame(rows, cols, mines)
