import numpy as np
import pytest
from synthetic_scenes.minesweeper.board import generate_game, get_visible_cells, reveal_region, CellState


@pytest.mark.parametrize('position', [(-1, 0), (0, -1), (2, 0), (0, 3)])
def test_reveal_region_rejects_outside_coordinates(position):
    with pytest.raises(ValueError, match='Reveal coordinates outside board'):
        reveal_region(np.zeros((2, 3), dtype=np.int8), *position)

@pytest.mark.parametrize('number',range(1,9))
def test_target(number):
    for seed in range(10):
        g=generate_game(rows=16,columns=16,mine_count=40,status='active',progress=.4,mode='targeted',target_number=number,seed=seed)
        p=g['target_position']
        assert get_visible_cells(g)[p]==number
        assert np.count_nonzero(g['board']==-1)==40
        assert not np.any(g['revealed'] & (g['board']==-1))
        assert not np.any(g['revealed'] & g['flags'])
        for r,c in np.argwhere(g['board']>=0):
            assert g['board'][r,c]==np.count_nonzero(g['board'][max(0,r-1):r+2,max(0,c-1):c+2]==-1)

def test_new():
    assert np.all(get_visible_cells(generate_game())==CellState.HIDDEN)

def test_won():
    g=generate_game(status='won')
    assert np.all(get_visible_cells(g)[g['board']==-1]==CellState.FLAG)

def test_lost():
    g=generate_game(status='lost',progress=.5)
    assert get_visible_cells(g)[g['triggered_mine']]==CellState.EXPLODED_MINE
