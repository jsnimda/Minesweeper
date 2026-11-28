"""Geometry source policy and objective-isolation contracts."""
from copy import deepcopy
from dataclasses import asdict
import json
import random

import pytest

from grid_geometry.dataset import generator, sampler
from grid_geometry.dataset.sampling_config import GenerationConfig, SOURCE_CELL_SIZE
from synthetic_scenes.schema import SceneSpec


def test_policy_defaults_and_independent_mutable_values():
    first, second = GenerationConfig(), GenerationConfig()
    assert SOURCE_CELL_SIZE == {'normal': (16, 40), 'large': (12, 22)}
    assert first.desktop_appearance == {
        'shell': {'taskbar': 70, 'dock': 20, 'top_panel': 10},
        'shell_styles': {'taskbar': {'modern': 75, 'classic': 25},
                         'dock': {'modern': 95, 'classic': 5},
                         'top_panel': {'modern': 80, 'classic': 20}},
        'color_scheme': {'modern': {'light': 50, 'dark': 50}, 'classic': {'default': 70, 'dark': 30}},
        'visibility': {'show_taskbar': .9, 'show_dock': .85, 'show_status_bar': .95, 'show_top_panel': .95},
        'dock_height': {'small': {'range': (32, 47), 'weight': 10},
                        'medium': {'range': (48, 63), 'weight': 25},
                        'normal': {'range': (64, 80), 'weight': 45},
                        'large': {'range': (81, 96), 'weight': 15},
                        'extra_large': {'range': (97, 120), 'weight': 5}}}
    first.desktop_appearance['shell_styles']['dock']['classic'] = 100
    first.game_zoom[1.] = 0
    first.display_scale[1.] = 0
    first.standalone_source_budget['max_dimension'] = 10
    assert second.desktop_appearance['shell_styles']['dock']['classic'] == 5
    assert second.game_zoom[1.] == 60 and second.display_scale[1.] == 55
    assert second.standalone_source_budget['max_dimension'] == 4096


@pytest.mark.parametrize('field,value', [
    ('shell', {'unknown': 1}), ('shell', {'dock': True}), ('shell', {'dock': 0}),
    ('shell', {'dock': float('inf')}), ('shell_styles', {'dock': {'classic': 1}}),
    ('color_scheme', {'classic': {'dark': 1}}), ('visibility', {}), ('dock_height', {}),
])
def test_invalid_top_level_policy(field, value):
    policy = deepcopy(GenerationConfig().desktop_appearance); policy[field] = value
    with pytest.raises(ValueError):
        GenerationConfig(desktop_appearance=policy)


@pytest.mark.parametrize('section,key,value', [
    ('shell_styles', 'dock', {'aero': 1}), ('shell_styles', 'dock', {'modern': 0}),
    ('shell_styles', 'dock', {'classic': float('nan')}),
    ('color_scheme', 'modern', {'default': 1}), ('color_scheme', 'classic', {'light': 1}),
    ('color_scheme', 'classic', {'dark': -1}), ('visibility', 'show_dock', True),
    ('visibility', 'show_dock', float('inf')), ('visibility', 'show_dock', 1.1),
    ('visibility', 'show_dock', -.1),
    ('dock_height', 'small', {'range': (31, 40), 'weight': 1}),
    ('dock_height', 'small', {'range': (33, 32), 'weight': 1}),
    ('dock_height', 'small', {'range': (32, 121), 'weight': 1}),
    ('dock_height', 'small', {'range': (32., 40), 'weight': 1}),
    ('dock_height', 'small', {'range': (32, 40), 'weight': True}),
    ('dock_height', 'small', {'range': (32, 40), 'weight': float('nan')}),
])
def test_invalid_nested_policy(section, key, value):
    policy = deepcopy(GenerationConfig().desktop_appearance); policy[section][key] = value
    with pytest.raises(ValueError):
        GenerationConfig(desktop_appearance=policy)


@pytest.mark.parametrize('shell,show_dock,fields', [
    ('taskbar', 1, {'show_taskbar'}), ('top_panel', 1, {'show_top_panel'}),
    ('dock', 0, {'show_dock', 'show_status_bar'}),
    ('dock', 1, {'show_dock', 'show_status_bar', 'dock_height'}),
])
def test_desktop_conditional_fields_and_classic(shell, show_dock, fields):
    policy = deepcopy(GenerationConfig().desktop_appearance)
    policy['shell'] = {shell: 1}; policy['shell_styles'][shell] = {'classic': 1}
    policy['visibility'].update(show_dock=show_dock, show_status_bar=0)
    config = GenerationConfig(desktop_appearance=policy)
    desktop = sampler._desktop(random.Random(7), config.desktop_appearance)
    assert desktop['style'] == 'classic'
    assert set(desktop) & {'show_taskbar', 'show_top_panel', 'show_dock', 'show_status_bar', 'dock_height'} == fields
    if shell == 'dock':
        assert desktop['show_dock'] == bool(show_dock) and not desktop['show_status_bar']
    request = {'mode': 'desktop', 'seed': 1, 'screen': {'width': 800, 'height': 600},
               'desktop': desktop, 'wallpaper': {'category': 'procedural', 'seed': 1}, 'windows': []}
    SceneSpec.model_validate(request)


def test_hidden_and_irrelevant_settings_do_not_consume_rng():
    policy = deepcopy(GenerationConfig().desktop_appearance)
    policy['shell'] = {'dock': 1}
    policy['visibility'].update(show_dock=0, show_status_bar=0)
    a = random.Random(123); first = sampler._desktop(a, policy)
    other = deepcopy(policy)
    other['visibility'].update(show_taskbar=0, show_top_panel=0)
    other['dock_height'] = {'single': {'range': (120, 120), 'weight': 1}}
    b = random.Random(123); second = sampler._desktop(b, other)
    assert first == second and a.getstate() == b.getstate()


def test_source_selection_threshold_and_desktop_fraction():
    def game(rows, columns):
        return {'rows': rows, 'columns': columns}
    assert sampler._scene(random.Random(1), game(25, 28), desktop_fraction=0)['mode'] == 'game'
    eligible = sampler._scene(random.Random(1), game(25, 28), desktop_fraction=1, display_scale={1.5: 1})
    assert eligible['mode'] == 'desktop' and eligible['screen'] == {'width': 1600, 'height': 1000, 'scale': 1.5}
    assert sampler._scene(random.Random(1), game(25, 29), desktop_fraction=1)['mode'] == 'game'
    for kind in ('negative', 'hard_negative'):
        scene = sampler._scene(random.Random(1), None, kind, 0, display_scale={1.25: 1})
        assert scene['mode'] == 'desktop' and scene['screen']['scale'] == 1.25


def test_config_file_records_source_policy(tmp_path):
    root = tmp_path / 'dataset'
    config = GenerationConfig(train=0, val=0, test=0, progress=False)
    generator.generate_dataset(root, config)
    saved = json.loads((root / 'config.json').read_text())
    assert saved['generator'] == 'phase-grid-v1' and saved['status'] == 'complete'
    assert saved['game_zoom']['0.5'] == 5
    assert saved['desktop_appearance']['shell']['dock'] == 20
    assert saved['standalone_source_budget']['max_pixels'] == 12_000_000


def _task(tmp_path, config):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'train').mkdir()
    excluded = tmp_path / 'excluded.json'; excluded.write_text('[]')
    return {'root': str(tmp_path), 'config': asdict(config), 'split': 'train', 'index': 0, 'excluded_path': str(excluded)}


def _game(rows, columns, cell, zoom):
    return {'rows': rows, 'columns': columns, 'mine_count': 3, 'status': 'new', 'progress': 0., 'seed': 10,
            'appearance': {'style': 'microsoft', 'theme': 'default', 'cell_size': cell, 'zoom': zoom}}


@pytest.mark.parametrize('category,cell,zoom,scale,desktop', [
    ('complete', 12, .5, 1., False), ('complete', 40, 2., 1., False),
    ('complete', 24, 1.25, 1.5, True), ('side', 20, .75, 1.25, True),
])
def test_source_scales_preserve_pitch_objectives(monkeypatch, tmp_path, category, cell, zoom, scale, desktop):
    rows = columns = 5 if category == 'complete' else 26
    game = _game(rows, columns, cell, zoom)
    monkeypatch.setattr(generator, '_game', lambda *args, **kwargs: (deepcopy(game), 'hidden'))
    categories = {name: float(name == category) for name in ('complete', 'interior', 'side', 'corner')}
    config = GenerationConfig(negative_fraction=0, categories=categories, pitch_bins=((20., 20., 1.),),
        anisotropic_fraction=1, anisotropic_max=.05, desktop_fraction=float(desktop), display_scale={scale: 1},
        standalone_source_budget=None, context_weights={'procedural': 1, 'desktop': 0}, progress=False)
    record = generator._generate_one(_task(tmp_path, config))
    assert record['requested_category'] == record['category'] == category
    assert record['requested_pitch'] == pytest.approx(20.)
    assert record['effective_pitch_x'] == pytest.approx(20.)
    assert record['effective_pitch_y'] == pytest.approx(20 * record['transform']['scale_y'] / record['transform']['scale_x'])
    assert record['original_geometry']['pitch_x'] == record['appearance']['pitch']
    assert record['appearance']['pitch'] > record['appearance']['cell_width']
    assert record['requested_effective_scale'] == pytest.approx(zoom * (scale if desktop else 1))


def test_retries_keep_objective_and_resample_source(monkeypatch, tmp_path):
    import synthetic_scenes.render as rendering
    calls = []
    def game(rng, category, pitch, size, **kwargs):
        calls.append((category, pitch))
        return _game(5, 5, 24, 1.0 if len(calls) == 1 else 1.5), 'hidden'
    monkeypatch.setattr(generator, '_game', game)
    render = rendering.render_scene
    def fail_first(request, **kwargs):
        if len(calls) == 1:
            raise rendering.PlacementError('forced first failure')
        return render(request, **kwargs)
    monkeypatch.setattr(rendering, 'render_scene', fail_first)
    config = GenerationConfig(negative_fraction=0, categories={'complete': 1, 'interior': 0, 'side': 0, 'corner': 0},
        pitch_bins=((12., 14., 1.),), anisotropic_fraction=1, desktop_fraction=0,
        context_weights={'procedural': 1, 'desktop': 0}, progress=False)
    record = generator._generate_one(_task(tmp_path, config))
    assert len(calls) == 2 and calls[0] == calls[1]
    assert record['attempt'] == 1 and record['requested_zoom'] == 1.5
    assert record['requested_category'] == calls[0][0] and record['requested_pitch'] == calls[0][1]
    assert record['effective_pitch_x'] == pytest.approx(record['requested_pitch'])
    assert .95 <= record['effective_pitch_y'] / record['effective_pitch_x'] <= 1.05
    monkeypatch.setattr(rendering, 'render_scene', render)
    baseline = generator._generate_one(_task(tmp_path / 'baseline', config))
    assert baseline['requested_category'] == record['requested_category']
    assert baseline['requested_pitch'] == record['requested_pitch']
    assert baseline['effective_pitch_y'] / baseline['effective_pitch_x'] == pytest.approx(
        record['effective_pitch_y'] / record['effective_pitch_x'])


def test_negative_kind_is_preserved_across_retry(monkeypatch, tmp_path):
    import synthetic_scenes.render as rendering
    original_scene = generator._scene
    original_render = rendering.render_scene
    kinds = []
    def capture_scene(rng, game, kind, desktop_fraction, **kwargs):
        assert game is None
        kinds.append(kind)
        return original_scene(rng, game, kind, desktop_fraction, **kwargs)
    def fail_first(request, **kwargs):
        if len(kinds) == 1:
            raise rendering.PlacementError('forced first failure')
        return original_render(request, **kwargs)
    monkeypatch.setattr(generator, '_scene', capture_scene)
    monkeypatch.setattr(rendering, 'render_scene', fail_first)
    config = GenerationConfig(negative_fraction=1., display_scale={1.: 1}, progress=False)
    record = generator._generate_one(_task(tmp_path, config))
    assert len(kinds) == 2 and kinds[0] == kinds[1]
    assert record['category'] == kinds[0] and record['board_id'] is None
