"""Revision 7 detection sampling policy and configuration contracts."""
from copy import deepcopy
import random

import pytest

from grid_detection.dataset import generator, sampler
from synthetic_scenes.render import render_scene


def test_approved_distributions():
    c = sampler.SAMPLING_CONFIG
    assert c['cell_size'] == {
        'small': {'weight': 20, 'range': (12, 19)},
        'normal': {'weight': 65, 'range': (20, 32)},
        'large': {'weight': 13, 'range': (33, 48)},
        'extra_large': {'weight': 2, 'range': (49, 64)}}
    assert c['game_zoom'] == {.5: 5, .75: 10, 1.: 55, 1.25: 15, 1.5: 10, 2.: 5}
    assert c['display_scale'] == dict(zip(
        ('retro', 'laptop', 'widescreen', 'high_res', 'ultrawide', 'portrait'),
        [dict(zip((.75, 1., 1.25, 1.5, 1.75, 2.), weights)) for weights in (
            (15, 75, 8, 2, 0, 0), (5, 50, 25, 15, 3, 2),
            (5, 60, 20, 10, 3, 2), (2, 18, 20, 25, 15, 20),
            (5, 40, 25, 20, 5, 5), (5, 50, 25, 15, 3, 2))]))
    assert c['resolution'] == {
        'retro': {'weight': 15, 'options': [(800, 600), (1024, 768), (1280, 960)]},
        'laptop': {'weight': 20, 'options': [(1280, 800), (1440, 900), (1920, 1200)]},
        'widescreen': {'weight': 35, 'options': [(1280, 720), (1366, 768), (1600, 900), (1920, 1080)]},
        'high_res': {'weight': 20, 'options': [(2560, 1440), (2880, 1800), (3840, 2160), (2560, 1600)]},
        'ultrawide': {'weight': 5, 'options': [(2560, 1080), (3440, 1440)]},
        'portrait': {'weight': 5, 'options': [(1080, 1920)]}}
    assert c['shell_styles'] == {'taskbar': {'modern': 75, 'classic': 25},
        'dock': {'modern': 95, 'classic': 5}, 'top_panel': {'modern': 80, 'classic': 20}}
    assert c['color_scheme'] == {'modern': {'light': 55, 'dark': 45}, 'classic': {'default': 70, 'dark': 30}}
    assert c['component_visibility'] == {'show_taskbar': .9, 'show_dock': .85,
        'show_status_bar': .95, 'show_top_panel': .95}
    assert list(c['dock_height'].values()) == [
        {'weight': 10, 'range': (32, 47)}, {'weight': 25, 'range': (48, 63)},
        {'weight': 45, 'range': (64, 80)}, {'weight': 15, 'range': (81, 96)},
        {'weight': 5, 'range': (97, 120)}]
    assert generator.GENERATOR_REVISION == 7


@pytest.mark.parametrize('policy', [
    {}, {'retro': {1.: 1}}, {'unknown': {1.: 1}},
    {'retro': {1.: 1}, 'other': {1.: 1}}, {1.: True}, {True: 1},
])
def test_invalid_display_policy(monkeypatch, policy):
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'display_scale', policy)
    with pytest.raises(ValueError):
        sampler.sample_scene(1)


def test_unselected_conditional_scale_is_validated(monkeypatch):
    policy = deepcopy(sampler.SAMPLING_CONFIG['display_scale'])
    policy['portrait'] = {1.: 0}
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'display_scale', policy)
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'scene_mode', {'grid': 1})
    with pytest.raises(ValueError):
        sampler.sample_scene(1)


@pytest.mark.parametrize('value', [True, -0.1, 1.1, float('nan'), float('inf'), '0.5'])
def test_invalid_visibility(monkeypatch, value):
    visibility = dict(sampler.SAMPLING_CONFIG['component_visibility'])
    visibility['show_dock'] = value
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'component_visibility', visibility)
    with pytest.raises(ValueError):
        sampler.sample_desktop(random.Random(1))


@pytest.mark.parametrize('group', [
    {'weight': 1, 'range': (31, 40)}, {'weight': 1, 'range': (50, 49)},
    {'weight': 1, 'range': (32, 121)}, {'weight': 1, 'range': (32., 40)},
    {'weight': 1, 'range': (True, 40)}, {'weight': -1, 'range': (32, 40)},
    {'weight': True, 'range': (32, 40)}, {'weight': float('inf'), 'range': (32, 40)},
    {'weight': 0, 'range': (32, 40)},
])
def test_invalid_dock_height(monkeypatch, group):
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'dock_height', {'test': group})
    with pytest.raises(ValueError):
        sampler.sample_desktop(random.Random(1))


@pytest.mark.parametrize('weights', [{'aero': 1}, {'classic': 0}, {'modern': True}, {'modern': float('nan')}])
def test_invalid_shell_styles(monkeypatch, weights):
    styles = deepcopy(sampler.SAMPLING_CONFIG['shell_styles'])
    styles['dock'] = weights
    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'shell_styles', styles)
    with pytest.raises(ValueError):
        sampler.sample_desktop(random.Random(1))


@pytest.mark.parametrize('group', ['retro', 'laptop', 'widescreen', 'high_res', 'ultrawide', 'portrait'])
def test_display_scale_uses_selected_group(monkeypatch, group):
    c = sampler.SAMPLING_CONFIG
    policy = {name: {1.0: 1} for name in c['resolution']}
    policy[group] = {1.75: 1}
    monkeypatch.setitem(c, 'display_scale', policy)
    monkeypatch.setitem(c, 'scene_mode', {'desktop': 1})
    monkeypatch.setitem(c, 'game_count', {0: 1})
    monkeypatch.setattr(sampler, 'choose_group', lambda rng, groups: (group, groups[group]['options'][0]))
    scene = sampler.sample_scene(3)
    assert scene['screen']['scale'] == 1.75
    assert (scene['screen']['width'], scene['screen']['height']) == c['resolution'][group]['options'][0]


@pytest.mark.parametrize('shell, visible, expected', [
    ('taskbar', {}, {'show_taskbar'}),
    ('top_panel', {}, {'show_top_panel'}),
    ('dock', {'show_dock': 0, 'show_status_bar': 0}, {'show_dock', 'show_status_bar'}),
    ('dock', {'show_dock': 1, 'show_status_bar': 0}, {'show_dock', 'show_status_bar', 'dock_height'}),
])
def test_shell_fields_and_classic_style(monkeypatch, shell, visible, expected):
    c = sampler.SAMPLING_CONFIG
    monkeypatch.setitem(c, 'shell', {shell: 1})
    styles = deepcopy(c['shell_styles']); styles[shell] = {'classic': 1}
    monkeypatch.setitem(c, 'shell_styles', styles)
    monkeypatch.setitem(c, 'component_visibility', {**c['component_visibility'], **visible})
    desktop = sampler.sample_desktop(random.Random(8))
    assert desktop['style'] == 'classic'
    assert set(desktop) & {'show_taskbar', 'show_dock', 'show_status_bar', 'show_top_panel', 'dock_height'} == expected
    if shell == 'dock':
        assert desktop['show_dock'] == bool(visible['show_dock'])
        assert not desktop['show_status_bar']


def test_irrelevant_visibility_and_hidden_dock_do_not_consume_draws(monkeypatch):
    c = sampler.SAMPLING_CONFIG
    monkeypatch.setitem(c, 'shell', {'dock': 1})
    visibility = dict(c['component_visibility'], show_dock=0, show_status_bar=0)
    monkeypatch.setitem(c, 'component_visibility', visibility)
    first_rng = random.Random(13); first = sampler.sample_desktop(first_rng)
    monkeypatch.setitem(c, 'component_visibility', dict(visibility, show_taskbar=0, show_top_panel=0))
    monkeypatch.setitem(c, 'dock_height', {'only': {'weight': 1, 'range': (120, 120)}})
    second_rng = random.Random(13); second = sampler.sample_desktop(second_rng)
    assert first == second and first_rng.getstate() == second_rng.getstate()


def test_visible_dock_precheck_and_renderer_receive_same_settings(monkeypatch):
    import synthetic_scenes.render as rendering
    c = sampler.SAMPLING_CONFIG
    monkeypatch.setitem(c, 'scene_mode', {'desktop': 1})
    monkeypatch.setitem(c, 'game_count', {1: 1})
    monkeypatch.setitem(c, 'display_scale', {1.25: 1})
    monkeypatch.setitem(c, 'game_zoom', {1.5: 1})
    monkeypatch.setitem(c, 'shell', {'dock': 1})
    monkeypatch.setitem(c, 'component_visibility', dict(c['component_visibility'], show_dock=1, show_status_bar=0))
    monkeypatch.setitem(c, 'dock_height', {'only': {'weight': 1, 'range': (96, 96)}})
    monkeypatch.setitem(c, 'board_size', {'small': {'weight': 1, 'size': (5, 5), 'mines': 3}})
    monkeypatch.setitem(c, 'cell_size', {'small': {'weight': 1, 'range': (12, 12)}})
    monkeypatch.setitem(c, 'cleanliness', {'minimal': 1})
    monkeypatch.setitem(c, 'wallpaper_type', {'procedural': 1})
    calls = []; planner = rendering.plan_game_windows
    def capture(*args, **kwargs):
        calls.append(deepcopy(args))
        return planner(*args, **kwargs)
    monkeypatch.setattr(sampler, 'plan_game_windows', capture)
    monkeypatch.setattr(rendering, 'plan_game_windows', capture)
    scene = sampler.sample_scene(9)
    _, resolved, annotations = render_scene(scene, details=('scene',), wallpaper_assets={})
    assert calls[0] == calls[1]
    assert calls[0][2]['show_dock'] and not calls[0][2]['show_status_bar']
    assert calls[0][2]['dock_height'] == 96
    assert resolved['desktop'] == scene['desktop']
    assert annotations['boards'][0]['requested_effective_scale'] == 1.25 * 1.5


def test_revision_and_effective_configuration_are_written(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(generator, 'run_splits', lambda *args, **kwargs: {})
    generator.generate_dataset(tmp_path / 'dataset', train=0, val=0, progress=False)
    config = json.loads((tmp_path / 'dataset' / 'config.json').read_text())
    assert config['generator_revision'] == 7 and config['status'] == 'complete'
    assert config['sampling_scope']['display_scale']['retro']['0.75'] == 15
    assert config['sampling_scope']['shell_styles']['dock']['classic'] == 5


def test_global_display_override_with_custom_groups(monkeypatch):
    c = sampler.SAMPLING_CONFIG
    monkeypatch.setitem(c, 'resolution', {'custom': {'weight': 1, 'options': [(800, 600)]}})
    monkeypatch.setitem(c, 'display_scale', {1.5: 1, 2.: 0})
    monkeypatch.setitem(c, 'scene_mode', {'desktop': 1})
    monkeypatch.setitem(c, 'game_count', {0: 1})
    scene = sampler.sample_scene(22)
    assert scene['screen'] == {'width': 800, 'height': 600, 'scale': 1.5}
    assert scene == sampler.sample_scene(22)
