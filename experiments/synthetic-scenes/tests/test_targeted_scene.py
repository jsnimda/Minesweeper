from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from synthetic_scenes.minesweeper.board import generate_game, get_visible_cells
from synthetic_scenes.render import render_scene
from synthetic_scenes.schema import SceneSpec


def scene(mode='game'):
    result = {'mode': mode, 'seed': 20, 'windows': [{'type': 'minesweeper', 'content': {
        'rows': 9, 'columns': 9, 'mine_count': 10, 'status': 'active',
        'progress': .4, 'seed': 42, 'appearance': {'style': 'classic', 'cell_size': 16},
    }}]}
    if mode == 'desktop':
        result.update(screen={'width': 1000, 'height': 700, 'scale': 1.25},
                      desktop={'shell': 'taskbar', 'style': 'modern', 'color_scheme': 'light'},
                      wallpaper={'category': 'procedural', 'seed': 3})
    return result


@pytest.mark.parametrize('fields', [{}, {'generation_mode': 'normal', 'target_number': None},
    {'generation_mode': 'targeted', 'target_number': 1},
    {'generation_mode': 'targeted', 'target_number': 8}])
def test_generation_schema_accepts_valid_fields(fields):
    request = scene()
    request['windows'][0]['content'].update(fields)
    original = deepcopy(request)
    content = SceneSpec.model_validate(request).windows[0].content
    assert content.generation_mode == fields.get('generation_mode', 'normal')
    assert content.target_number == fields.get('target_number')
    assert request == original


@pytest.mark.parametrize('fields', [
    {'generation_mode': 'targeted'}, {'target_number': 1},
    *({'generation_mode': 'targeted', 'target_number': value} for value in (0, -1, 9, True, 1.0, '1')),
    {'generation_mode': 'other'},
    {'generation_mode': 'targeted', 'target_number': 1, 'status': 'new', 'progress': 0.},
])
def test_generation_schema_rejects_invalid_fields(fields):
    request = scene()
    request['windows'][0]['content'].update(fields)
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(request)


@pytest.mark.parametrize('rows,columns,mines,number,position', [
    (2, 2, 3, 3, (0, 0)), (2, 3, 5, 5, (0, 1)), (3, 3, 8, 8, (1, 1)),
])
def test_target_geometry_and_one_safe_won(rows, columns, mines, number, position):
    game = generate_game(rows, columns, mines, status='won', mode='targeted', target_number=number)
    assert game['target_position'] == position or (rows, columns) == (2, 2)
    assert get_visible_cells(game)[game['target_position']] == number
    assert np.count_nonzero(game['board'] == -1) == mines
    for r, c in np.argwhere(game['board'] >= 0):
        assert game['board'][r, c] == np.count_nonzero(game['board'][max(0, r-1):r+2, max(0, c-1):c+2] == -1)


@pytest.mark.parametrize('status,progress', [('active', .01), ('active', .99), ('won', 0.),
                                              ('lost', 0.), ('lost', 1.)])
def test_target_status_and_determinism(status, progress):
    kwargs = dict(status=status, progress=progress, mode='targeted', target_number=8, seed=7)
    first, second = generate_game(**kwargs), generate_game(**kwargs)
    assert first['target_position'] == second['target_position']
    assert get_visible_cells(first)[first['target_position']] == 8
    for key in ('board', 'revealed', 'flags'):
        assert np.array_equal(first[key], second[key])
    assert first['triggered_mine'] == second['triggered_mine']


@pytest.mark.parametrize('status', ['active', 'lost'])
def test_one_safe_cell_rejected_for_unfinished_targeted_game(status):
    with pytest.raises(ValueError, match='No feasible target position'):
        generate_game(3, 3, 8, status=status, progress=.5, mode='targeted', target_number=8)


@pytest.mark.parametrize('mode', ['game', 'grid', 'desktop'])
def test_targeted_scene_annotations(mode):
    request = scene(mode)
    content = request['windows'][0]['content']
    content.update(generation_mode='targeted', target_number=8)
    if mode == 'desktop':
        content['appearance']['zoom'] = 1.5
    original = deepcopy(request)
    image, resolved, annotations = render_scene(request, details=('scene', 'cell', 'mine'))
    board = annotations['boards'][0]
    row, column = board['target_position']
    assert board['generation_mode'] == 'targeted' and board['target_number'] == 8
    assert board['cell_states'][row][column] == 8
    left, top, right, bottom = board['cell_boxes'][row][column]
    gx, gy, gr, gb = board['boxes']['grid']
    assert 0 <= gx <= left < right <= gr <= image.width
    assert 0 <= gy <= top < bottom <= gb <= image.height
    game = generate_game(status='active', progress=.4, seed=42, mode='targeted', target_number=8)
    assert set(map(tuple, board['mine_positions'])) == set(map(tuple, np.argwhere(game['board'] == -1)))
    assert len(board['mine_positions']) == 10
    assert request == original and resolved['windows'] == original['windows']


def test_multiple_independent_games():
    request = scene('desktop')
    first = request['windows'][0]['content']
    first.update(generation_mode='targeted', target_number=8)
    second = deepcopy(request['windows'][0])
    second['content'].update(generation_mode='targeted', target_number=1, seed=43)
    third = deepcopy(request['windows'][0])
    third['content'].update(generation_mode='normal', target_number=None, seed=44)
    request['windows'].extend([second, third])
    _, _, annotations = render_scene(request, details=('cell',))
    for board, target in zip(annotations['boards'], (8, 1, None), strict=True):
        assert board['target_number'] == target
        if target is not None:
            r, c = board['target_position']
            assert board['cell_states'][r][c] == target


def test_infeasible_target_does_not_fall_back():
    request = scene()
    request['windows'][0]['content'].update(rows=2, columns=2, mine_count=1,
                                            generation_mode='targeted', target_number=8)
    with pytest.raises(ValueError, match='No feasible target position'):
        render_scene(request)


def test_explicit_normal_is_identical_to_omitted_defaults():
    omitted = scene('game')
    explicit = deepcopy(omitted)
    explicit['windows'][0]['content'].update(generation_mode='normal', target_number=None)
    image1, resolved1, annotations1 = render_scene(omitted, details=('scene', 'cell', 'mine'))
    image2, resolved2, annotations2 = render_scene(explicit, details=('scene', 'cell', 'mine'))
    assert image1.tobytes() == image2.tobytes()
    assert annotations1 == annotations2
    assert resolved1['layout'] == resolved2['layout']
    assert resolved1['windows'] == omitted['windows'] and resolved2['windows'] == explicit['windows']
