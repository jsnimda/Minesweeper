"""Detection attempt instrumentation at existing failure boundaries."""
import json
from copy import deepcopy
from grid_detection.dataset import generator
from synthetic_scenes.render import PlacementError


def game(seed):
    return {'rows': 5, 'columns': 5, 'mine_count': 3, 'status': 'new', 'progress': 0., 'seed': seed,
            'appearance': {'style': 'classic', 'theme': 'default', 'cell_size': 16, 'zoom': 1.}}


def scene():
    return {'mode': 'desktop', 'seed': 4, 'layout_seed': 6,
            'screen': {'width': 800, 'height': 600, 'scale': 1.},
            'desktop': {'shell': 'taskbar', 'style': 'modern', 'color_scheme': 'light'},
            'wallpaper': {'category': 'procedural', 'seed': 7},
            'windows': [{'type': 'minesweeper', 'content': game(10)}, {'type': 'minesweeper', 'content': game(11)}]}


def test_rejected_multiboard_scene_then_success_no_extra_render(monkeypatch, tmp_path):
    seen = []; renders = []
    def sample(seed, *, sampling_metadata):
        seen.append(seed)
        games = [game(10), game(11)]
        sampling_metadata['sampled_cell_sizes'] = [16, 16]
        sampling_metadata['resolution_group'] = 'retro'
        sampling_metadata['attempt'] = {'scene_mode': 'desktop', 'game_count': 2, 'resolution_group': 'retro',
            'screen_width': 800, 'screen_height': 600, 'display_scale': 1., 'shell': 'taskbar',
            'desktop_style': 'modern', 'color_scheme': 'light', 'boards': games}
        if len(seen) == 1: raise PlacementError('precheck')
        return deepcopy(scene())
    render = generator.render_scene
    def capture(*args, **kwargs):
        renders.append(1)
        return render(*args, **kwargs)
    monkeypatch.setattr(generator, 'sample_scene', sample)
    monkeypatch.setattr(generator, 'render_scene', capture)
    excluded = tmp_path / 'excluded.json'; excluded.write_text('[]')
    task = {'split': 'train', 'index': 0, 'seed': 42, 'excluded_path': str(excluded),
            'max_attempts': 3, 'scene_details': False, 'augment': False, 'mode': 'render'}
    output = generator._generate_one(task)
    diagnostics = output['attempt_diagnostics'].result()
    assert len(seen) == 2 and len(renders) == 1 and output['record']['attempt'] == 1
    assert diagnostics['attempted'] == 2 and diagnostics['accepted'] == diagnostics['rejected'] == 1
    assert diagnostics['board_observations'] == 4
    assert diagnostics['by']['resolution_group']['retro']['rejected'] == 1
    assert diagnostics['boards_by_containing_attempt_outcome']['game_zoom']['1.0']['rejected'] == 2
    assert diagnostics['rejection_reasons'] == {'placement_retries': 1}


def test_incomplete_attempt_fields_are_not_invented():
    fields, boards = generator._attempt_observations({'attempt': {'scene_mode': 'desktop'}}, None)
    assert fields['scene_mode'] == 'desktop' and fields['display_scale'] is None
    assert fields['game_count'] is None and boards == []


def test_missing_sampled_cell_size_is_unavailable():
    request = {'mode': 'game', 'windows': [{'type': 'minesweeper', 'content': game(10)}]}
    _, boards = generator._attempt_observations({}, request)
    assert len(boards) == 1 and boards[0]['sampled_cell_size'] is None


def test_attempt_finalization_failure_keeps_dataset_incomplete(monkeypatch, tmp_path):
    import pytest
    monkeypatch.setattr(generator, 'run_provenance', lambda *args: {'source': {'commit': 'kept'}})
    def fail(*args, **kwargs): raise RuntimeError('attempt finalization failed')
    monkeypatch.setattr(generator, 'summaries', fail)
    with pytest.raises(RuntimeError, match='attempt finalization failed'):
        generator.generate_dataset(tmp_path / 'dataset', train=0, val=0, progress=False)
    config = json.loads((tmp_path / 'dataset' / 'config.json').read_text())
    assert config['status'] == 'incomplete' and config['provenance']['source']['commit'] == 'kept'


def test_negative_desktop_has_no_board_observations(monkeypatch, tmp_path):
    negative = scene(); negative['windows'] = []
    def sample(seed, *, sampling_metadata):
        sampling_metadata['sampled_cell_sizes'] = []
        sampling_metadata['resolution_group'] = 'retro'
        return deepcopy(negative)
    monkeypatch.setattr(generator, 'sample_scene', sample)
    excluded = tmp_path / 'excluded.json'; excluded.write_text('[]')
    task = {'split': 'train', 'index': 0, 'seed': 42, 'excluded_path': str(excluded),
            'max_attempts': 3, 'scene_details': False, 'augment': False, 'mode': 'render'}
    output = generator._generate_one(task)
    diagnostics = output['attempt_diagnostics'].result()
    assert diagnostics['attempted'] == diagnostics['accepted'] == 1
    assert diagnostics['rejected'] == diagnostics['board_observations'] == 0
    assert diagnostics['by']['composition']['negative']['accepted'] == 1
    assert diagnostics['attempts_per_sample']['first_attempt'] == 1
