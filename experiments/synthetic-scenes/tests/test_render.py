import numpy as np
import pytest
pytest.importorskip('skia')
from synthetic_scenes.minesweeper.board import generate_game
from synthetic_scenes.minesweeper.renderer import render_board, STYLES

@pytest.mark.parametrize('style',STYLES)
def test_render(style):
    image, meta = render_board(generate_game(status='active',progress=.4),style=style,cell_size=32)
    expected = 312 if style == 'microsoft' else 288
    assert image.size==(expected,expected)
    assert meta['boxes']['grid']==(0,0,expected,expected)
    assert meta['cell_boxes'][0][0] == (0,0,32,32)
    assert meta['cell_boxes'][0][1][0] == (35 if style == 'microsoft' else 32)
    assert meta['x_edges'][-1] == expected
    assert meta['renderer'].endswith('_hd')
    assert len(meta['x_edges'])==10


@pytest.mark.parametrize('style', STYLES)
@pytest.mark.parametrize('status', ['new', 'active', 'won', 'lost'])
def test_game_region(style, status):
    from synthetic_scenes.minesweeper.renderer import render_game_region

    game = generate_game(status=status, progress=.4, seed=42)
    grid, grid_meta = render_board(game, style=style)
    image, meta = render_game_region(game, style=style)
    gx, gy, right, bottom = meta['boxes']['grid']
    assert (right-gx, bottom-gy) == grid.size
    assert image.crop((gx, gy, right, bottom)).tobytes() == grid.tobytes()
    assert meta['boxes']['game_region'] == (0, 0, *image.size)
    assert meta['x_edges'] == [x+gx for x in grid_meta['x_edges']]
    assert meta['y_edges'] == [y+gy for y in grid_meta['y_edges']]
    assert meta['cell_boxes'][0][0] == (gx, gy, gx+32, gy+32)
    assert meta['cell_states'] == grid_meta['cell_states']
    assert ('reset_button' in meta['boxes']) == (style == 'classic')
    for name, (x,y,r,b) in meta['boxes'].items():
        assert 0 <= x < r <= image.width, name
        assert 0 <= y < b <= image.height, name
        if name not in ('grid', 'game_region'):
            assert r <= gx or x >= right or b <= gy or y >= bottom


def test_flower_variant():
    game = generate_game(status='lost', progress=.4, seed=42)
    classic, a = render_board(game, style='windows7')
    flower, b = render_board(game, style='windows7', mine_variant='flower')
    assert classic.tobytes() != flower.tobytes()
    assert a['cell_states'] == b['cell_states']
    assert a['cell_boxes'] == b['cell_boxes']
    assert b['style_parameters']['mine_variant'] == 'flower'
    with pytest.raises(ValueError):
        render_board(game, style='classic', mine_variant='flower')


@pytest.mark.parametrize('style', STYLES)
def test_oversized_timer_saturates_inside_its_box(style):
    from synthetic_scenes.minesweeper.renderer import render_game_region

    game = generate_game(status='active', progress=.4, elapsed_seconds=0)
    baseline, meta = render_game_region(game, style=style, cell_size=8)
    capped, _ = render_game_region({**game, 'elapsed_seconds': 999}, style=style, cell_size=8)
    overflow, overflow_meta = render_game_region(
        {**game, 'elapsed_seconds': 10**20}, style=style, cell_size=8,
    )
    assert overflow_meta['boxes'] == meta['boxes']
    assert overflow.tobytes() == capped.tobytes()
    left, top, right, bottom = meta['boxes']['timer']
    changed = np.any(np.asarray(baseline) != np.asarray(overflow), axis=2)
    assert changed[top:bottom, left:right].any()
    changed[top:bottom, left:right] = False
    assert not changed.any()
