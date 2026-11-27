"""Native scale sampling, placement and compact manifest contracts."""
from copy import deepcopy
import random

import pytest

from grid_detection.dataset import generator, sampler
from synthetic_scenes.render import render_scene
from synthetic_scenes.schema import SceneSpec


def _game(cell_size=20, zoom=1.25, style='microsoft', seed=10):
    return {'rows': 4, 'columns': 5, 'mine_count': 2, 'status': 'new',
            'progress': 0., 'seed': seed, 'appearance': {
                'style': style, 'cell_size': cell_size, 'zoom': zoom}}


def _desktop(games=(), scale=1.5):
    return {'mode': 'desktop', 'seed': 5, 'layout_seed': 7,
            'screen': {'width': 800, 'height': 600, 'scale': scale},
            'desktop': {'shell': 'taskbar', 'style': 'modern', 'color_scheme': 'light'},
            'wallpaper': {'category': 'procedural', 'seed': 9},
            'windows': [{'type': 'minesweeper', 'content': game} for game in games]}


def _task(tmp_path, *, augment=False):
    excluded = tmp_path / 'excluded.json'
    excluded.write_text('[]')
    return {'split': 'train', 'index': 0, 'seed': 42, 'excluded_path': str(excluded),
            'max_attempts': 3, 'scene_details': False, 'augment': augment, 'mode': 'render'}


def _generate(monkeypatch, tmp_path, request, sampled_sizes, *, augment=False):
    def fixed_scene(seed, *, sampling_metadata):
        sampling_metadata['sampled_cell_sizes'] = sampled_sizes
        return deepcopy(request)

    monkeypatch.setattr(generator, 'sample_scene', fixed_scene)
    return generator._generate_one(_task(tmp_path, augment=augment))['record']


def test_singleton_scale_does_not_consume_rng():
    rng = random.Random(5)
    assert sampler.sample_scale(rng, {0.75: 0, 1.0: 2}) == 1.0


@pytest.mark.parametrize('weights', [
    {}, {0.: 1}, {-1.: 1}, {float('inf'): 1}, {float('nan'): 1},
    {True: 1}, {'1': 1}, {1.: -1}, {1.: 0}, {1.: float('inf')},
    {1.: float('nan')}, {1.: '1'}, {1.: True},
])
def test_invalid_scale_distributions(weights):
    with pytest.raises(ValueError):
        sampler.sample_scale(random.Random(1), weights)


def test_unused_scale_configuration_is_still_validated(monkeypatch):
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'scene_mode', {'grid': 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'display_scale', {})
    with pytest.raises(ValueError):
        sampler.sample_scene(1)


@pytest.mark.parametrize('mode', ['desktop', 'game', 'grid'])
def test_neutral_scales_do_not_change_seeded_decisions(monkeypatch, mode):
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'scene_mode', {mode: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'game_zoom', {1.0: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'display_scale', {1.0: 1})
    if mode == 'desktop':
        monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'game_count', {0: 1})
        monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'wallpaper_type', {'procedural': 1})
    actual = sampler.sample_scene(113)
    # Bypassing singleton scale draws must leave the other decisions unchanged.
    monkeypatch.setattr(sampler, 'sample_scale', lambda rng, weights: 1.0)
    baseline = sampler.sample_scene(113)
    assert actual == baseline
    assert all(w['content']['appearance']['zoom'] == 1.0 for w in actual['windows']
               if w['type'] == 'minesweeper')
    if mode == 'desktop':
        assert actual['screen']['scale'] == 1.0
    else:
        assert 'screen' not in actual
    SceneSpec.model_validate(actual)


@pytest.mark.parametrize('mode', ['desktop', 'game', 'grid'])
def test_scale_fields_by_scene_mode(monkeypatch, mode):
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'scene_mode', {mode: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'game_zoom', {1.25: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'display_scale', {1.5: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'game_count', {1: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'wallpaper_type', {'procedural': 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'board_size',
                        {'small': {'weight': 1, 'size': (5, 5), 'mines': 3}})
    request = sampler.sample_scene(19)
    assert request['windows'][0]['content']['appearance']['zoom'] == 1.25
    if mode == 'desktop':
        assert request['screen']['scale'] == 1.5
    else:
        assert 'screen' not in request
    SceneSpec.model_validate(request)


def test_standalone_does_not_sample_display_scale(monkeypatch):
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'scene_mode', {'grid': 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'board_size',
                        {'small': {'weight': 1, 'size': (5, 5), 'mines': 3}})
    calls = []
    original = sampler.sample_scale

    def capture(rng, weights):
        calls.append(weights)
        return original(rng, weights)

    monkeypatch.setattr(sampler, 'sample_scale', capture)
    sampler.sample_scene(19)
    assert len(calls) == 1 and calls[0] is sampler.SAMPLING_CONFIG['game_zoom']


def test_scene_captures_standalone_size_before_fitting(monkeypatch):
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'scene_mode', {'game': 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'standalone_image_budget',
                        {'max_dimension': 250, 'max_pixels': 62500})
    monkeypatch.setattr(sampler, 'sample_game', lambda rng: _game(64, 2.0))
    metadata = {}
    request = sampler.sample_scene(5, sampling_metadata=metadata)
    assert metadata['sampled_cell_sizes'] == [64]
    assert request['windows'][0]['content']['appearance']['cell_size'] < 64
    assert request['windows'][0]['content']['appearance']['zoom'] == 2.0


def test_desktop_precheck_and_renderer_share_scales(monkeypatch):
    import synthetic_scenes.render as rendering

    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'scene_mode', {'desktop': 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'game_count', {2: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'game_zoom', {1.25: 1, .75: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'display_scale', {1.5: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'board_size',
                        {'small': {'weight': 1, 'size': (5, 5), 'mines': 3}})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'cell_size',
                        {'small': {'weight': 1, 'range': (12, 12)}})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'resolution',
                        {'fixed': {'weight': 1, 'options': [(800, 600)]}})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'cleanliness', {'minimal': 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'wallpaper_type', {'procedural': 1})
    original_sample_scale = sampler.sample_scale
    zooms = iter((1.25, 0.75))

    def controlled_scale(rng, weights):
        if weights is sampler.SAMPLING_CONFIG['game_zoom']:
            return next(zooms)
        return original_sample_scale(rng, weights)

    calls = []
    planner = rendering.plan_game_windows

    def capture(*args, **kwargs):
        calls.append((deepcopy(args), deepcopy(kwargs)))
        return planner(*args, **kwargs)

    monkeypatch.setattr(sampler, 'sample_scale', controlled_scale)
    monkeypatch.setattr(sampler, 'plan_game_windows', capture)
    monkeypatch.setattr(rendering, 'plan_game_windows', capture)
    request = sampler.sample_scene(7)
    _, _, annotations = render_scene(request, wallpaper_assets={})
    assert calls[0] == calls[1]
    assert calls[0][0][-1] == 1.5
    assert [b['requested_zoom'] for b in annotations['boards']] == [1.25, .75]
    assert all(b['requested_display_scale'] == 1.5 for b in annotations['boards'])


def test_negative_desktop_manifest(monkeypatch, tmp_path):
    record = _generate(monkeypatch, tmp_path, _desktop(), [])
    assert record['display_scale'] == 1.5
    assert record['boards'] == [] and record['board_ids'] == []
    assert record['negative'] and record['board_count'] == 0
    assert record['native_size'] == record['final_size'] == (800, 600)
    assert 'resolved_scene' not in record


def test_negative_desktop_samples_display_scale(monkeypatch):
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'scene_mode', {'desktop': 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'game_count', {0: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'display_scale', {1.5: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'cleanliness', {'minimal': 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'wallpaper_type', {'procedural': 1})
    request = sampler.sample_scene(5)
    assert request['screen']['scale'] == 1.5
    assert not request['windows']


def test_native_metadata_and_augmentation_for_multiple_boards(monkeypatch, tmp_path):
    request = _desktop([_game(20, 1.25, seed=10), _game(17, .75, seed=11)])
    raw = _generate(monkeypatch, tmp_path, request, [20, 17])
    augmented = _generate(monkeypatch, tmp_path, request, [20, 17], augment=True)
    for original, transformed in zip(raw['boards'], augmented['boards'], strict=True):
        assert {k: v for k, v in original.items() if k != 'final_grid_box'} == {
            k: v for k, v in transformed.items() if k != 'final_grid_box'}
        sx, sy = augmented['augmentation']['actual_scale']
        assert transformed['final_grid_box'] == pytest.approx(tuple(
            value * (sx if index % 2 == 0 else sy)
            for index, value in enumerate(original['final_grid_box'])))
    assert [board['board_id'] for board in raw['boards']] == raw['board_ids']
    assert [board['sampled_cell_size'] for board in raw['boards']] == [20, 17]
    for board, zoom in zip(raw['boards'], (1.25, .75), strict=True):
        assert board['game_state_id'] and board['native_grid_box'] == board['final_grid_box']
        assert board['requested_zoom'] == zoom
        assert board['requested_display_scale'] == 1.5
        assert board['requested_effective_scale'] == zoom * 1.5
        assert board['cell_width'] == board['cell_height'] == round(board['resolved_cell_size'] * zoom * 1.5)
        assert board['pitch_x'] == board['pitch_y'] > board['cell_width']
        sx, sy = augmented['augmentation']['actual_scale']
        assert board['pitch_x'] * sx > board['cell_width'] * sx
        assert board['pitch_y'] * sy > board['cell_height'] * sy


def test_standalone_fitting_preserves_sampled_size_in_manifest(monkeypatch, tmp_path):
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'standalone_image_budget',
                        {'max_dimension': 250, 'max_pixels': 62500})
    game = _game(64, 2.0)
    original = deepcopy(game)
    sampler._fit_standalone_game(game, 'game')
    assert game['appearance']['cell_size'] < 64
    assert game['appearance']['zoom'] == original['appearance']['zoom']
    assert (game['rows'], game['columns'], game['mine_count']) == (4, 5, 2)
    request = {'mode': 'game', 'seed': 5, 'windows': [{'type': 'minesweeper', 'content': game}]}
    board = _generate(monkeypatch, tmp_path, request, [64])['boards'][0]
    assert board['sampled_cell_size'] == 64
    assert board['requested_cell_size'] == board['resolved_cell_size'] == game['appearance']['cell_size']
    assert board['requested_zoom'] == 2.0 and board['requested_display_scale'] == 1.0


def test_desktop_fitting_preserves_scales_and_original_size(monkeypatch, tmp_path):
    request = _desktop([_game(90, 1.25)], scale=1.5)
    # Force fitting with a board that is too wide at the original size.
    request['windows'][0]['content'].update(rows=9, columns=9, mine_count=10)
    record = _generate(monkeypatch, tmp_path, request, [90])
    board = record['boards'][0]
    assert board['sampled_cell_size'] == board['requested_cell_size'] == 90
    assert board['resolved_cell_size'] < 90
    assert board['requested_zoom'] == 1.25 and board['requested_display_scale'] == 1.5


def test_desktop_style_policy_and_valid_requests():
    config = sampler.SAMPLING_CONFIG
    assert config['shell'] == {'taskbar': 60, 'dock': 25, 'top_panel': 15}
    assert config['shell_styles'] == {
        'taskbar': {'modern': 75, 'classic': 25},
        'dock': {'modern': 95, 'classic': 5},
        'top_panel': {'modern': 80, 'classic': 20},
    }
    assert 'taskbar_layout' not in config
    for seed in range(100):
        desktop = sampler.sample_desktop(random.Random(seed))
        assert desktop['style'] in config['shell_styles'][desktop['shell']]
        assert 'taskbar_layout' not in desktop
        request = _desktop()
        request['desktop'] = desktop
        SceneSpec.model_validate(request)
        assert desktop == sampler.sample_desktop(random.Random(seed))


def test_dock_visibility_precheck_matches_rendered_placement(monkeypatch):
    import synthetic_scenes.render as rendering

    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'scene_mode', {'desktop': 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'game_count', {1: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'board_size',
                        {'small': {'weight': 1, 'size': (5, 5), 'mines': 3}})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'cell_size', {'small': {'weight': 1, 'range': (12, 12)}})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'resolution', {'fixed': {'weight': 1, 'options': [(640, 480)]}})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'display_scale', {1.5: 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'cleanliness', {'minimal': 1})
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'wallpaper_type', {'procedural': 1})
    monkeypatch.setattr(sampler, 'sample_desktop', lambda rng: {
        'shell': 'dock', 'style': 'modern', 'color_scheme': 'light',
        'show_dock': False, 'show_status_bar': True, 'dock_height': 120})
    plans = []
    original = rendering.plan_game_windows

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        plans.append(result)
        return result

    monkeypatch.setattr(sampler, 'plan_game_windows', capture)
    monkeypatch.setattr(rendering, 'plan_game_windows', capture)
    request = sampler.sample_scene(7)
    _, _, annotations = render_scene(request, wallpaper_assets={})
    assert plans[0] == plans[1]
    assert annotations['boards'][0]['window_box'] == plans[0][0]['box']
