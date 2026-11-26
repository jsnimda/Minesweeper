import numpy as np
import pytest

pytest.importorskip('skia')

from synthetic_scenes.minesweeper.board import CellState, generate_game
from synthetic_scenes.minesweeper.geometry import measure_game_region
from synthetic_scenes.minesweeper.renderer import render_game_region, STYLES
from synthetic_scenes.render import render_scene


@pytest.mark.parametrize('style', STYLES)
def test_measured_region(style):
    game = generate_game(status='active', progress=.4)
    image, _ = render_game_region(game, style=style, cell_size=24)
    assert image.size == measure_game_region(9, 9, style, 24)


@pytest.mark.parametrize('variant', ['classic', 'flower'])
def test_lost_windows7_open_flags_keep_content_and_context(variant):
    content = {'rows': 9, 'columns': 9, 'mine_count': 10, 'status': 'lost',
               'progress': .85, 'seed': 42,
               'appearance': {'style': 'windows7', 'mine_variant': variant}}
    request = {'mode': 'game', 'seed': 42,
               'windows': [{'type': 'minesweeper', 'content': content}]}
    _, _, metadata = render_scene(request, details=('cell',))
    board = metadata['boards'][0]
    states = np.asarray(board['cell_states']).reshape(-1)
    opened = np.asarray(board['opened_surfaces']).reshape(-1)
    opened_flags = (states == CellState.FLAG) & opened
    ignored = states >= CellState.MINE
    assert opened_flags.any()
    assert set(states[ignored]) == {CellState.MINE, CellState.EXPLODED_MINE, CellState.WRONG_FLAG}
    assert len(states) == content['rows'] * content['columns']


def test_board_id_tracks_minefield_and_state_id_tracks_game():
    def render(**changes):
        content = {'rows': 9, 'columns': 9, 'mine_count': 10, 'status': 'active',
                   'progress': .4, 'seed': 42, 'appearance': {'style': 'classic'}}
        content.update(changes)
        request = {'mode': 'grid', 'seed': 42,
                   'windows': [{'type': 'minesweeper', 'content': content}]}
        return render_scene(request)[2]['boards'][0]

    active = render()
    restyled = render(appearance={'style': 'google'})
    won = render(status='won')
    timer_changed = render(elapsed_seconds=1)
    different_mines = render(mine_count=11)
    assert active['board_id'] == won['board_id'] == restyled['board_id'] == timer_changed['board_id']
    assert active['game_state_id'] == restyled['game_state_id']
    assert active['game_state_id'] != won['game_state_id']
    assert active['game_state_id'] != timer_changed['game_state_id']
    assert active['board_id'] != different_mines['board_id']
    wide = render(rows=2, columns=3, mine_count=0, status='new', progress=0.)
    tall = render(rows=3, columns=2, mine_count=0, status='new', progress=0.)
    assert wide['board_id'] != tall['board_id']
