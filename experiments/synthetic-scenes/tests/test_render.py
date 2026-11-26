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


@pytest.mark.parametrize('style', STYLES)
def test_game_region_cell_scale_and_narrow_layout(style):
    from synthetic_scenes.minesweeper.geometry import measure_game_region, measure_grid, region_layout
    from synthetic_scenes.minesweeper.renderer import render_game_region

    game = generate_game(rows=2, columns=3, mine_count=0, status='new', seed=42)
    results = []
    for cell in (8, 32):
        grid_size = measure_grid(2, 3, style, cell)
        layout = region_layout(grid_size, style, cell)
        image, meta = render_game_region(game, style=style, cell_size=cell)
        assert image.size == measure_game_region(2, 3, style, cell) == layout['size']
        gx, gy = layout['grid_origin']
        assert gx == (image.width-grid_size[0])//2
        assert meta['boxes']['grid'] == (gx, gy, gx+grid_size[0], gy+grid_size[1])
        assert meta['cell_boxes'][0][0] == (gx, gy, gx+cell, gy+cell)
        assert meta['x_edges'][0] == gx and meta['y_edges'][0] == gy
        controls = ['mine_counter', 'timer'] + (['reset_button'] if style == 'classic' else [])
        for name in controls:
            assert meta['boxes'][name] == layout[name]
        for index, name in enumerate(controls):
            x, y, r, b = meta['boxes'][name]
            assert 0 <= x < r <= image.width and 0 <= y < b <= image.height
            for other in controls[index+1:] + ['grid']:
                ox, oy, oright, obottom = meta['boxes'][other]
                assert r <= ox or oright <= x or b <= oy or obottom <= y
        results.append((grid_size, layout, controls))

    small, large = results
    assert small[0][0] < large[0][0] and small[0][1] < large[0][1]
    for name in small[2]:
        def dimensions(box):
            x, y, r, b = box
            return r-x, b-y
        assert dimensions(small[1][name]) == dimensions(large[1][name])
    # The vertical space outside the grid is fixed for each style.
    assert small[1]['size'][1]-small[0][1] == large[1]['size'][1]-large[0][1]


@pytest.mark.parametrize('style', STYLES)
def test_counter_capacity_uses_actual_cell_pitch(style):
    from synthetic_scenes.minesweeper.geometry import measure_grid, region_layout

    layouts = [region_layout(measure_grid(2, 500, style, cell), style, cell)
               for cell in (8, 32, 64)]
    widths = [layout['mine_counter'][2]-layout['mine_counter'][0] for layout in layouts]
    assert widths[0] == widths[1] == widths[2]
    if style == 'classic':
        assert all(layout['slots'] == 4 for layout in layouts)
    smaller = region_layout(measure_grid(2, 50, style, 32), style, 32)
    assert widths[1] > smaller['mine_counter'][2]-smaller['mine_counter'][0]


@pytest.mark.parametrize('style,control_height,region_height', [
    ('classic', 25, 58), ('windows7', 27, 27),
    ('microsoft', 22, 43), ('google', 20, 29),
])
@pytest.mark.parametrize('cell', [12, 24, 48])
def test_selected_ui_preserves_grid_and_measurements(style, control_height, region_height, cell):
    from synthetic_scenes.minesweeper.geometry import measure_game_region, measure_grid, region_layout
    from synthetic_scenes.minesweeper.renderer import render_game_region

    game = generate_game(rows=9, columns=5, mine_count=5, status='active', progress=.4, seed=42)
    grid, grid_meta = render_board(game, style=style, cell_size=cell)
    image, meta = render_game_region(game, style=style, cell_size=cell)
    grid_size = measure_grid(9, 5, style, cell)
    layout = region_layout(grid_size, style, cell)
    assert image.size == measure_game_region(9, 5, style, cell) == layout['size']
    gx, gy = layout['grid_origin']
    assert meta['boxes']['grid'] == (gx, gy, gx+grid_size[0], gy+grid_size[1])
    assert image.crop(meta['boxes']['grid']).tobytes() == grid.tobytes()
    assert meta['style_parameters']['pitch'] == grid_meta['style_parameters']['pitch']
    assert meta['cell_boxes'][0][0] == (gx, gy, gx+cell, gy+cell)
    for name in ('mine_counter', 'timer'):
        assert meta['boxes'][name] == layout[name]
        assert layout[name][3]-layout[name][1] == control_height
    if style == 'classic':
        assert layout['header_box'][3]-layout['header_box'][1] == region_height
        reset = layout['reset_button']
        assert (reset[2]-reset[0], reset[3]-reset[1]) == (37, 37)
    elif style == 'windows7':
        assert layout['timer'][3]-layout['timer'][1] == region_height
    else:
        assert layout['header'] == region_height


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
