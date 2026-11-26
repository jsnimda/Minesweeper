import numpy as np
import pytest
pytest.importorskip('skia')

from synthetic_scenes.scene import GameSpec, render_scene
from synthetic_scenes.minesweeper.geometry import measure_game_region
from synthetic_scenes.minesweeper.renderer import render_game_region, STYLES
from synthetic_scenes.dataset import generate_dataset
from synthetic_scenes.exporters import cell_sample
from synthetic_scenes.minesweeper.board import CellState


@pytest.mark.parametrize('style',STYLES)
def test_measured_region(style):
    spec=GameSpec(appearance={'style':style,'cell_size':24})
    image,meta=render_game_region(spec.generate(),**spec.appearance)
    assert image.size==measure_game_region(9,9,style,24)


@pytest.mark.parametrize('style', STYLES)
def test_legacy_window_placement_uses_complete_game_size(style):
    from synthetic_scenes.desktop import chrome_size, content_origin
    from synthetic_scenes.scene import DEFAULT_CHROME

    spec = GameSpec(rows=2, columns=3, mine_count=0, status='new',
                    appearance={'style': style, 'cell_size': 8})
    _, meta = render_scene([spec], size=(640, 480), seed=4, background_count=0)
    board = meta['boards'][0]
    x, y, r, b = board['window_box']
    chrome = DEFAULT_CHROME[style]
    size = measure_game_region(2, 3, style, 8)
    assert (r-x, b-y) == chrome_size(size, chrome)
    ox, oy = content_origin(chrome)
    assert board['boxes']['game_region'] == (x+ox, y+oy, x+ox+size[0], y+oy+size[1])


def test_scene():
    specs=[GameSpec(seed=i,appearance={'style':style,'cell_size':24})
           for i,style in enumerate(('classic','google'))]
    image,meta=render_scene(specs,seed=4)
    for spec,board in zip(specs,meta['boards']):
        local,_=render_game_region(spec.generate(),**spec.appearance)
        assert image.crop(board['boxes']['game_region']).tobytes()==local.tobytes()
    image,meta=render_scene([],seed=4)
    assert not meta['boards']


def test_dataset(tmp_path):
    result=generate_dataset(tmp_path/'small',train_count=2,val_count=1,augment=False)
    assert result['train']['counts']['scenes']==2
    assert result['val']['counts']['scenes']==1


@pytest.mark.parametrize('variant', ['classic', 'flower'])
def test_lost_windows7_open_flags_keep_content_and_context(variant):
    spec = GameSpec(status='lost', progress=.85, seed=42,
                    appearance={'style': 'windows7', 'mine_variant': variant})
    image, meta = render_scene([spec], mode='game')
    board = meta['boards'][0]
    sample = cell_sample(image, board)
    states = np.asarray(board['cell_states']).reshape(-1)
    opened = np.asarray(board['opened_surfaces']).reshape(-1)
    opened_flags = (states == CellState.FLAG) & opened
    ignored = states >= CellState.MINE

    assert opened_flags.any()
    assert set(states[ignored]) == {CellState.MINE, CellState.EXPLODED_MINE, CellState.WRONG_FLAG}
    np.testing.assert_array_equal(sample['cell_states'], states)
    assert len(sample['images']) == spec.rows * spec.columns
    assert np.all(sample['labels'][opened_flags] == 10)
    assert np.all(sample['content_labels'][opened_flags] == 9)
    assert np.all(sample['surface_labels'][opened_flags] == 1)
    for key in ('labels', 'content_labels', 'surface_labels'):
        assert np.all(sample[key][ignored] == -1)
    np.testing.assert_array_equal(sample['surface_labels'][~ignored], opened[~ignored])


def test_board_id_tracks_minefield_and_state_id_tracks_game():
    def render(spec):
        _, meta = render_scene([spec], mode='grid')
        return meta['boards'][0]

    active = render(GameSpec(seed=42))
    restyled = render(GameSpec(seed=42, appearance={'style': 'google'}))
    won = render(GameSpec(seed=42, status='won'))
    timer_changed = render(GameSpec(seed=42, elapsed_seconds=1))
    different_mines = render(GameSpec(seed=42, mine_count=11))
    assert active['board_id'] == won['board_id'] == restyled['board_id'] == timer_changed['board_id']
    assert active['game_state_id'] == restyled['game_state_id']
    assert active['game_state_id'] != won['game_state_id']
    assert active['game_state_id'] != timer_changed['game_state_id']
    assert active['board_id'] != different_mines['board_id']

    wide = render(GameSpec(rows=2, columns=3, mine_count=0, status='new'))
    tall = render(GameSpec(rows=3, columns=2, mine_count=0, status='new'))
    assert wide['board_id'] != tall['board_id']


@pytest.mark.parametrize('task', ['cells', 'geometry'])
def test_dataset_export_is_independent_of_other_tasks(tmp_path, task):
    full = tmp_path/'full'
    single = tmp_path/'single'
    options = {'train_count': 2, 'val_count': 1, 'geometry_size': (32, 24), 'heatmap_sigma': 1}
    generate_dataset(full, **options)
    generate_dataset(single, tasks=(task,), **options)

    for directory in ('scenes', task):
        full_files = sorted(path.relative_to(full) for path in (full/directory).rglob('*') if path.is_file())
        single_files = sorted(path.relative_to(single) for path in (single/directory).rglob('*') if path.is_file())
        assert full_files == single_files
        assert full_files
        for relative in full_files:
            assert (full/relative).read_bytes() == (single/relative).read_bytes(), relative
