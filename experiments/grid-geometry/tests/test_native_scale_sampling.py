"""Native source scaling, optional budget and exported phase geometry."""
from copy import deepcopy
from dataclasses import asdict
import json
import random
from pathlib import Path

import pytest

from grid_geometry.dataset import generator, sampler
from grid_geometry.dataset.sampling_config import GenerationConfig
from synthetic_scenes.minesweeper.geometry import measure_game_region, measure_grid
from synthetic_scenes.schema import SceneSpec


def _game(cell_size=24, zoom=1.25, rows=5, columns=5):
    return {'rows': rows, 'columns': columns, 'mine_count': 3, 'status': 'new',
            'progress': 0., 'seed': 10, 'appearance': {
                'style': 'microsoft', 'theme': 'default', 'cell_size': cell_size, 'zoom': zoom}}


def _task(tmp_path, config):
    (tmp_path / 'train').mkdir(exist_ok=True)
    excluded = tmp_path / 'excluded.json'
    excluded.write_text('[]')
    return {'root': str(tmp_path), 'config': asdict(config), 'split': 'train',
            'index': 0, 'excluded_path': str(excluded)}


def _small_scene(rng, game, negative_kind=None, desktop_fraction=.8, *, display_scale=None, desktop_appearance=None):
    if game is not None and desktop_fraction == 0:
        return {'mode': 'game', 'seed': 5, 'windows': [{'type': 'minesweeper', 'content': game}]}
    windows = ([{'type': 'minesweeper', 'content': game}] if game is not None else
               [{'type': 'sudoku' if negative_kind == 'hard_negative' else 'browser', 'seed': 12}])
    return {'mode': 'desktop', 'seed': 5, 'layout_seed': 7,
            'screen': {'width': 800, 'height': 600,
                       'scale': sampler._sample_scale(rng, display_scale)},
            'desktop': {'shell': 'taskbar', 'style': 'modern', 'color_scheme': 'light'},
            'wallpaper': {'category': 'procedural', 'seed': 9}, 'windows': windows}


def _config(**kwargs):
    defaults = dict(negative_fraction=0., categories={'complete': 1, 'interior': 0, 'side': 0, 'corner': 0},
                    pitch_bins=((12., 12., 1.),), anisotropic_fraction=1., anisotropic_max=.05,
                    padding_example_fraction=0., context_weights={'procedural': 1, 'desktop': 0},
                    progress=False)
    defaults.update(kwargs)
    return GenerationConfig(**defaults)


def test_scale_defaults_budget_and_singleton_rng():
    config = GenerationConfig()
    assert config.game_zoom == {.5: 5, .75: 10, 1.: 60, 1.25: 15, 1.5: 8, 2.: 2}
    assert config.display_scale == {.75: 5, 1.: 55, 1.25: 25, 1.5: 10, 1.75: 3, 2.: 2}
    assert config.standalone_source_budget == {'max_dimension': 4096, 'max_pixels': 12_000_000}
    assert GenerationConfig(standalone_source_budget=None).standalone_source_budget is None
    rng = random.Random(4)
    state = rng.getstate()
    assert sampler._sample_scale(rng, {.75: 0, 1.: 3}) == 1.
    assert state == rng.getstate()


@pytest.mark.parametrize('weights', [
    {}, {0.: 1}, {-1.: 1}, {float('inf'): 1}, {float('nan'): 1},
    {True: 1}, {'1': 1}, {1.: -1}, {1.: 0}, {1.: float('inf')},
    {1.: float('nan')}, {1.: '1'}, {1.: True},
])
@pytest.mark.parametrize('field', ['game_zoom', 'display_scale'])
def test_invalid_scale_configuration(field, weights):
    with pytest.raises(ValueError):
        GenerationConfig(**{field: weights})


@pytest.mark.parametrize('budget', [
    {}, {'max_dimension': 4096}, {'max_dimension': 0, 'max_pixels': 1},
    {'max_dimension': 1, 'max_pixels': -1}, {'max_dimension': True, 'max_pixels': 1},
    {'max_dimension': 1., 'max_pixels': 1}, {'max_dimension': 1, 'max_pixels': 1, 'extra': 2},
])
def test_invalid_budget_configuration(budget):
    with pytest.raises(ValueError):
        GenerationConfig(standalone_source_budget=budget)


@pytest.mark.parametrize('negative_kind,desktop_fraction', [(None, 0.), (None, 1.), ('negative', 1.), ('hard_negative', 1.)])
def test_neutral_sampling_preserves_rng_and_scene(monkeypatch, negative_kind, desktop_fraction):
    def sample():
        rng = random.Random(117)
        game = None if negative_kind else sampler._game(rng, 'complete', 12., (384, 384))[0]
        scene = sampler._scene(rng, game, negative_kind, desktop_fraction)
        return scene, rng.getstate()

    actual, state = sample()
    monkeypatch.setattr(sampler, '_sample_scale', lambda rng, weights: 1.0)
    baseline, baseline_state = sample()
    assert actual == baseline and state == baseline_state
    SceneSpec.model_validate(actual)


@pytest.mark.parametrize('negative_kind', [None, 'negative', 'hard_negative'])
def test_desktop_scale_and_game_zoom(negative_kind):
    rng = random.Random(7)
    game = None if negative_kind else sampler._game(rng, 'complete', 12., (384, 384), game_zoom={1.25: 1})[0]
    if game:
        game.update(rows=9, columns=9, mine_count=10)
    scene = sampler._scene(rng, game, negative_kind, 1., display_scale={1.5: 1})
    assert scene['mode'] == 'desktop' and scene['screen'] == {'width': 1600, 'height': 1000, 'scale': 1.5}
    if game:
        assert game['appearance']['zoom'] == 1.25
    SceneSpec.model_validate(scene)


def test_standalone_does_not_sample_display_scale(monkeypatch):
    calls = []
    original = sampler._sample_scale

    def capture(rng, weights):
        calls.append(weights)
        return original(rng, weights)

    monkeypatch.setattr(sampler, '_sample_scale', capture)
    rng = random.Random(19)
    game, _ = sampler._game(rng, 'complete', 12., (384, 384), game_zoom={1.25: 1})
    scene = sampler._scene(rng, game, desktop_fraction=0., display_scale={1.5: 1})
    assert scene['mode'] == 'game' and 'screen' not in scene
    assert calls == [{1.25: 1}] and game['appearance']['zoom'] == 1.25


def test_standalone_budget_measures_full_game_and_preserves_fields():
    game = _game(40, 1.5)
    original = deepcopy(game)
    width, height = measure_game_region(5, 5, 'microsoft', 40, 1.5)
    generator._fit_standalone_game(game, {'max_dimension': max(width, height), 'max_pixels': width * height})
    assert game == original
    generator._fit_standalone_game(game, {'max_dimension': 250, 'max_pixels': 62500})
    fitted = game['appearance']['cell_size']
    assert 12 <= fitted < 40
    width, height = measure_game_region(5, 5, 'microsoft', fitted, 1.5)
    assert max(width, height) <= 250 and width * height <= 62500
    game['appearance']['cell_size'] = 40
    assert game == original


def test_budget_accounts_for_header_and_margins():
    game = _game(24, 1.25)
    grid_width, grid_height = measure_grid(5, 5, 'microsoft', 24, 1.25)
    width, height = measure_game_region(5, 5, 'microsoft', 24, 1.25)
    assert width * height > grid_width * grid_height
    generator._fit_standalone_game(game, {
        'max_dimension': max(width, height), 'max_pixels': grid_width * grid_height,
    })
    assert game['appearance']['cell_size'] < 24


def test_budget_failure_retries_without_rendering(monkeypatch, tmp_path):
    import synthetic_scenes.render as rendering

    calls = []
    game_calls = []
    def sampled_game(*args, **kwargs):
        game_calls.append(1)
        return _game(), 'hidden'
    monkeypatch.setattr(generator, '_game', sampled_game)
    monkeypatch.setattr(generator, '_scene', _small_scene)
    def forbidden_render(*args, **kwargs):
        calls.append(1)
        raise AssertionError('oversized scene was rendered')
    monkeypatch.setattr(rendering, 'render_scene', forbidden_render)
    config = _config(desktop_fraction=0., max_attempts=3,
                     standalone_source_budget={'max_dimension': 10, 'max_pixels': 100})
    with pytest.raises(RuntimeError, match='after 3 attempts') as error:
        generator._generate_one(_task(tmp_path, config))
    assert isinstance(error.value.__cause__, ValueError)
    assert len(game_calls) == 3 and calls == []


@pytest.mark.parametrize('desktop', [False, True])
def test_scaled_native_and_transformed_metadata(monkeypatch, tmp_path, desktop):
    game = _game(90 if desktop else 40, 1.25, rows=9 if desktop else 5, columns=9 if desktop else 5)
    monkeypatch.setattr(generator, '_game', lambda *args, **kwargs: (deepcopy(game), 'hidden'))
    monkeypatch.setattr(generator, '_scene', _small_scene)
    config = _config(desktop_fraction=float(desktop), display_scale={1.5: 1},
                     standalone_source_budget=None if desktop else {'max_dimension': 250, 'max_pixels': 62500})
    record = generator._generate_one(_task(tmp_path, config))
    assert record['sampled_cell_size'] == game['appearance']['cell_size']
    assert record['requested_zoom'] == 1.25
    assert record['source_size'][0] > 0 and record['source_size'][1] > 0
    assert (record['source_desktop'] is not None) == desktop
    assert record['anisotropy_selected'] is True
    assert record['display_scale'] == record['requested_display_scale'] == (1.5 if desktop else 1.)
    assert record['requested_effective_scale'] == 1.25 * record['display_scale']
    if desktop:
        assert record['requested_cell_size'] == 90 and record['resolved_cell_size'] < 90
    else:
        assert record['requested_cell_size'] < 40
        assert record['resolved_cell_size'] == record['requested_cell_size']
    assert record['appearance']['pitch'] > record['appearance']['cell_width']
    assert record['original_geometry']['pitch_x'] == record['appearance']['pitch']
    assert record['effective_pitch_x'] == pytest.approx(12.)
    assert record['effective_pitch_y'] == pytest.approx(12. * record['transform']['scale_y'] / record['transform']['scale_x'])
    assert record['effective_pitch_y'] != pytest.approx(record['effective_pitch_x'])
    assert record['geometry']['pitch_x'] == pytest.approx(record['effective_pitch_x'])
    from grid_geometry.dataset.loader import PhaseDataset
    (tmp_path / 'config.json').write_text(json.dumps({
        'status': 'complete', 'generator': 'phase-grid-v1',
        'input_size': [384, 384], 'transition_ratio': .25,
    }))
    sample = PhaseDataset(tmp_path, 'train').load_sample(0)
    assert sample.metadata['sampled_cell_size'] == record['sampled_cell_size']
    assert sample.metadata['geometry']['pitch_x'] == pytest.approx(12.)


def test_disabled_budget_never_fits(monkeypatch, tmp_path):
    monkeypatch.setattr(generator, '_game', lambda *args, **kwargs: (_game(), 'hidden'))
    monkeypatch.setattr(generator, '_scene', _small_scene)
    monkeypatch.setattr(generator, '_fit_standalone_game', lambda *args: pytest.fail('disabled budget was used'))
    record = generator._generate_one(_task(tmp_path, _config(desktop_fraction=0., standalone_source_budget=None)))
    assert record['sampled_cell_size'] == record['requested_cell_size'] == record['resolved_cell_size'] == 24


@pytest.mark.parametrize('negative_kind', ['negative', 'hard_negative'])
def test_negative_metadata_retains_display_scale(monkeypatch, tmp_path, negative_kind):
    def negative_scene(rng, game, kind, desktop_fraction, *, display_scale, desktop_appearance):
        return _small_scene(rng, game, negative_kind, desktop_fraction, display_scale=display_scale)
    monkeypatch.setattr(generator, '_scene', negative_scene)
    record = generator._generate_one(_task(tmp_path, _config(negative_fraction=1., display_scale={1.5: 1})))
    assert record['display_scale'] == 1.5
    assert record['source_desktop']['screen']['scale'] == 1.5
    assert record['source_size'] == (800, 600)
    assert record['geometry'] is None and record['board_id'] is None
    assert record['sampled_cell_size'] is None and record['requested_zoom'] is None


def test_context_desktop_remains_neutral(monkeypatch, tmp_path):
    import synthetic_scenes.render as rendering

    real_render = rendering.render_scene
    requests = []
    def capture(request, **kwargs):
        requests.append(deepcopy(request))
        return real_render(request, **kwargs)
    monkeypatch.setattr(rendering, 'render_scene', capture)
    monkeypatch.setattr(generator, '_game', lambda *args, **kwargs: (_game(), 'hidden'))
    monkeypatch.setattr(generator, '_scene', _small_scene)
    appearance = deepcopy(GenerationConfig().desktop_appearance)
    appearance['shell'] = {'top_panel': 1}
    appearance['shell_styles']['top_panel'] = {'classic': 1}
    appearance['visibility']['show_top_panel'] = 0
    config = _config(desktop_fraction=1., display_scale={1.5: 1}, desktop_appearance=appearance,
                     context_weights={'procedural': 0, 'desktop': 1})
    record = generator._generate_one(_task(tmp_path, config))
    assert record['background_visible'] and len(requests) == 2
    assert requests[0]['screen']['scale'] == 1.5
    assert requests[1]['screen']['scale'] == 1.0
    assert record['source_desktop']['screen']['scale'] == 1.5
    assert record['fill_desktop']['screen']['scale'] == 1.0
    assert requests[1]['desktop']['shell'] == 'top_panel'
    assert requests[1]['desktop']['style'] == 'classic'
    assert requests[1]['desktop']['show_top_panel'] is False


def test_desktop_style_policy_and_valid_sources():
    for seed in range(100):
        desktop = sampler._desktop(random.Random(seed))
        assert desktop['shell'] in {'taskbar', 'dock', 'top_panel'}
        assert desktop['style'] in {'modern', 'classic'}
        assert 'taskbar_layout' not in desktop
        request = {'mode': 'desktop', 'seed': seed, 'windows': [],
                   'screen': {'width': 800, 'height': 600}, 'desktop': desktop,
                   'wallpaper': {'category': 'procedural', 'seed': seed}}
        SceneSpec.model_validate(request)
        assert desktop == sampler._desktop(random.Random(seed))


def test_attempt_diagnostics_preserve_objective_and_render_count(monkeypatch, tmp_path):
    import synthetic_scenes.render as rendering
    sampled = []; calls = []
    def sample_game(rng, category, pitch, size, **kwargs):
        sampled.append((category, pitch))
        return _game(24, .75 if len(sampled) == 1 else 1.25), 'hidden'
    monkeypatch.setattr(generator, '_game', sample_game)
    monkeypatch.setattr(generator, '_scene', _small_scene)
    original = rendering.render_scene
    def render(request, **kwargs):
        calls.append(1)
        if len(calls) == 1: raise rendering.PlacementError('first attempt')
        return original(request, **kwargs)
    monkeypatch.setattr(rendering, 'render_scene', render)
    task = _task(tmp_path, _config(desktop_fraction=0., max_attempts=3))
    task['collect_attempt_diagnostics'] = True
    output = generator._generate_one(task)
    summary = output['attempt_diagnostics'].result()
    assert len(sampled) == len(calls) == 2 and sampled[0] == sampled[1]
    assert summary['attempted'] == 2 and summary['accepted'] == summary['rejected'] == 1
    assert summary['by']['game_zoom']['0.75']['rejected'] == 1
    assert summary['by']['game_zoom']['1.25']['accepted'] == 1
    objective = summary['objectives']['by']['category']['complete']
    assert objective['selected'] == 1 and objective['source_attempts'] == 2
    assert output['record']['requested_pitch'] == sampled[0][1]


def test_identity_conflict_is_one_rejected_source_attempt(monkeypatch, tmp_path):
    import synthetic_scenes.render as rendering
    monkeypatch.setattr(generator, '_game', lambda *args, **kwargs: (_game(), 'hidden'))
    monkeypatch.setattr(generator, '_scene', _small_scene)
    original = rendering.render_scene
    calls = []
    def render(request, **kwargs):
        image, resolved, annotations = original(request, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            annotations['boards'][0]['board_id'] = 'excluded'
        return image, resolved, annotations
    monkeypatch.setattr(rendering, 'render_scene', render)
    task = _task(tmp_path, _config(desktop_fraction=0.))
    task['collect_attempt_diagnostics'] = True
    Path(task['excluded_path']).write_text('["excluded"]')
    output = generator._generate_one(task)
    diagnostics = output['attempt_diagnostics'].result()
    assert diagnostics['rejection_reasons'] == {'identity_overlap': 1}
    assert diagnostics['attempted'] == 2 and diagnostics['accepted'] == 1


def test_budget_fitting_failure_is_counted_without_extra_render(monkeypatch, tmp_path):
    import synthetic_scenes.render as rendering
    monkeypatch.setattr(generator, '_game', lambda *args, **kwargs: (_game(), 'hidden'))
    monkeypatch.setattr(generator, '_scene', _small_scene)
    original_fit = generator._fit_standalone_game
    fits = []; renders = []
    def fit(game, budget):
        fits.append(1)
        if len(fits) == 1: raise ValueError('forced budget fitting failure')
        return original_fit(game, budget)
    original_render = rendering.render_scene
    def render(*args, **kwargs):
        renders.append(1)
        return original_render(*args, **kwargs)
    monkeypatch.setattr(generator, '_fit_standalone_game', fit)
    monkeypatch.setattr(rendering, 'render_scene', render)
    task = _task(tmp_path, _config(desktop_fraction=0., max_attempts=3))
    task['collect_attempt_diagnostics'] = True
    output = generator._generate_one(task)
    diag = output['attempt_diagnostics'].result()
    assert len(fits) == 2 and len(renders) == 1
    assert diag['rejection_reasons'] == {'sampling_or_placement': 1}
    assert diag['fitting_observations']['standalone_fitting_stage_rejected_yes'] == 1
    assert diag['fitting_observations']['standalone_budget_fitted_no'] == 1


def test_crop_failure_has_one_terminal_rejection(monkeypatch, tmp_path):
    monkeypatch.setattr(generator, '_game', lambda *args, **kwargs: (_game(), 'hidden'))
    monkeypatch.setattr(generator, '_scene', _small_scene)
    original = generator._crop_box
    calls = []
    def crop(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1: raise ValueError('forced crop failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(generator, '_crop_box', crop)
    task = _task(tmp_path, _config(desktop_fraction=0., max_attempts=3))
    task['collect_attempt_diagnostics'] = True
    output = generator._generate_one(task)
    diag = output['attempt_diagnostics'].result()
    assert len(calls) == 2 and diag['attempted'] == 2
    assert diag['accepted'] == diag['rejected'] == 1
    assert diag['fitting_observations']['crop_stage_rejected_yes'] == 1


def test_attempt_finalization_failure_keeps_dataset_incomplete(monkeypatch, tmp_path):
    import synthetic_scenes.provenance as provenance
    monkeypatch.setattr(provenance, 'run_provenance', lambda *args: {'source': {'commit': 'kept'}})
    def fail(*args, **kwargs): raise RuntimeError('attempt finalization failed')
    monkeypatch.setattr(generator, 'summaries', fail)
    with pytest.raises(RuntimeError, match='attempt finalization failed'):
        generator.generate_dataset(tmp_path / 'dataset', GenerationConfig(train=0, val=0, test=0, progress=False))
    config = json.loads((tmp_path / 'dataset' / 'config.json').read_text())
    assert config['status'] == 'incomplete' and config['provenance']['source']['commit'] == 'kept'


def test_diagnostic_error_does_not_become_sampling_retry(monkeypatch, tmp_path):
    calls = []
    def sample_game(*args, **kwargs):
        calls.append(1)
        return _game(), 'hidden'
    def fail(*args, **kwargs):
        raise ValueError('broken diagnostic')
    monkeypatch.setattr(generator, '_game', sample_game)
    monkeypatch.setattr(generator, 'board_factors', fail)
    task = _task(tmp_path, _config(desktop_fraction=0., max_attempts=3))
    with pytest.raises(RuntimeError, match='attempt diagnostics failed') as error:
        generator._generate_one(task)
    assert isinstance(error.value.__cause__, ValueError) and len(calls) == 1
