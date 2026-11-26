"""Public scene specification and active rendering boundary checks."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from synthetic_scenes.render import PlacementError, render_scene
from synthetic_scenes.schema import SceneSpec


def game_scene(mode='game'):
    return {
        'mode': mode, 'seed': 'scene:1',
        'windows': [{'type': 'minesweeper', 'content': {
            'rows': 9, 'columns': 9, 'mine_count': 10, 'status': 'new',
            'progress': 0., 'seed': 123, 'appearance': {'style': 'classic'},
        }}],
    }


def desktop_scene(windows=None):
    scene = game_scene('desktop')
    if windows is not None:
        scene['windows'] = windows
    scene.update(screen={'width': 640, 'height': 480},
                 desktop={'shell': 'taskbar', 'style': 'aero', 'color_scheme': 'light',
                          'taskbar_layout': None, 'accent_color': None},
                 wallpaper={'category': 'procedural', 'seed': 3, 'subtype': None})
    return scene


@pytest.mark.parametrize('mode', ['game', 'grid', 'desktop'])
def test_valid_modes_and_supported_defaults(mode):
    scene = desktop_scene() if mode == 'desktop' else game_scene(mode)
    model = SceneSpec.model_validate(scene)
    appearance = model.windows[0].content.appearance
    assert appearance.cell_size == 32 and appearance.theme == 'default'
    assert 'cell_size' not in appearance.model_fields_set
    if mode == 'desktop':
        assert model.screen.scale == 1
        assert model.desktop.accent_color is None
        assert 'accent_color' in model.desktop.model_fields_set


def test_desktop_accepts_negative_and_application_windows():
    scene = desktop_scene([{'type': 'browser', 'seed': 7}, {'type': 'sudoku', 'seed': 8}])
    assert len(SceneSpec.model_validate(scene).windows) == 2
    scene['windows'] = []
    assert SceneSpec.model_validate(scene).windows == []


@pytest.mark.parametrize('mutate', [
    lambda s: s.pop('seed'),
    lambda s: s.update(mode='mobile'),
    lambda s: s.update(typo=True),
    lambda s: s['windows'][0].update(seed=2),
    lambda s: s['windows'][0]['content'].update(mine_count=81),
    lambda s: s['windows'][0]['content'].update(status='active', progress=0.),
    lambda s: s['windows'][0]['content'].update(status='lost', mine_count=0),
    lambda s: s['windows'][0]['content'].update(elapsed_seconds=1),
    lambda s: s['windows'][0]['content']['appearance'].update(cell_size=7),
    lambda s: s['windows'][0]['content']['appearance'].update(font_scale=.1),
    lambda s: s['windows'][0]['content']['appearance'].update(font_sclae=.7),
    lambda s: s['windows'][0]['content']['appearance'].update(mine_variant='flower'),
    lambda s: s['windows'][0]['content']['appearance'].update(theme='bluee'),
])
def test_invalid_game_requests(mutate):
    scene = game_scene()
    mutate(scene)
    with pytest.raises(ValidationError):
        render_scene(scene)


def test_mode_and_window_requirements():
    for scene in (game_scene('desktop'), {**game_scene(), 'windows': []},
                  {**game_scene(), 'windows': [{'type': 'browser', 'seed': 1}]},
                  desktop_scene([{'type': 'browser', 'content': {}}]),
                  desktop_scene([{'type': 'other', 'seed': 1}])):
        with pytest.raises(ValidationError):
            SceneSpec.model_validate(scene)


@pytest.mark.parametrize('scale', [1, 0.5, 1.5, 2])
def test_display_scale_accepts_positive_finite_numbers(scale):
    scene = desktop_scene()
    scene['screen']['scale'] = scale
    assert SceneSpec.model_validate(scene).screen.scale == scale


@pytest.mark.parametrize('scale', [True, False, '1', 0, -1, float('nan'), float('inf'), -float('inf')])
def test_display_scale_rejects_invalid_values(scale):
    scene = desktop_scene()
    scene['screen']['scale'] = scale
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(scene)


@pytest.mark.parametrize('style,theme', [('classic', 'warm'), ('windows7', 'default'),
                                       ('windows7', 'green'), ('microsoft', 'blue'),
                                       ('google', 'warm')])
def test_style_specific_options(style, theme):
    scene = game_scene()
    appearance = scene['windows'][0]['content']['appearance']
    appearance.update(style=style, theme=theme)
    if style == 'windows7':
        appearance['mine_variant'] = 'flower'
    SceneSpec.model_validate(scene)
    appearance['theme'] = 'not-a-theme'
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(scene)


def test_requested_scene_is_not_changed_and_layout_is_separate():
    scene = desktop_scene()
    scene['screen'].update(width=320, height=317)
    scene['windows'][0]['content']['appearance']['cell_size'] = 64
    original = deepcopy(scene)
    _, resolved, _ = render_scene(scene, details=['scene'])
    assert scene == original
    assert resolved['windows'] == original['windows']
    assert resolved['windows'][0]['content']['appearance']['cell_size'] == 64
    layout = resolved['layout'][0]
    assert layout['cell_size'] < 64
    left, top, right, bottom = layout['box']
    assert 0 <= left < right <= 320
    assert 0 <= top < bottom <= 269  # 317 pixels less the Aero taskbar.


@pytest.mark.parametrize('style', ['classic', 'windows7', 'microsoft', 'google'])
def test_desktop_window_uses_complete_game_size(style):
    from synthetic_scenes.desktop_rendering.generic_window import window_geometry
    from synthetic_scenes.minesweeper.geometry import measure_game_region, measure_grid, region_layout

    scene = desktop_scene()
    content = scene['windows'][0]['content']
    content.update(rows=2, columns=3, mine_count=0,
                   appearance={'style': style, 'cell_size': 8})
    _, resolved, annotations = render_scene(scene, details=['scene', 'cell'])
    placement = resolved['layout'][0]
    board = annotations['boards'][0]
    x, y, r, b = placement['box']
    size = measure_game_region(2, 3, style, 8)
    window, (ox, oy) = window_geometry(size, style='aero')
    assert (r-x, b-y) == window
    assert board['window_box'] == placement['box']
    assert board['boxes']['game_region'] == (x+ox, y+oy, x+ox+size[0], y+oy+size[1])
    layout = region_layout(measure_grid(2, 3, style, 8), style, 8)
    for name in ('grid', 'mine_counter', 'timer'):
        if name == 'grid':
            gx, gy = layout['grid_origin']
            width, height = measure_grid(2, 3, style, 8)
            local = (gx, gy, gx+width, gy+height)
        else:
            local = layout[name]
        assert board['boxes'][name] == (local[0]+x+ox, local[1]+y+oy,
                                        local[2]+x+ox, local[3]+y+oy)


def test_placement_failure_is_distinct_from_invalid_input():
    scene = desktop_scene()
    content = scene['windows'][0]['content']
    content.update(rows=100, columns=100, mine_count=1)
    with pytest.raises(PlacementError):
        render_scene(scene)
    content['mine_count'] = 10000
    with pytest.raises(ValidationError):
        render_scene(scene)


def test_details_are_independent_and_order_insensitive():
    scene = game_scene('grid')
    _, basic, annotations = render_scene(scene)
    _, scene_only, _ = render_scene(scene, details=['scene'])
    _, _, cell_only = render_scene(scene, details=['cell'])
    _, _, mine_only = render_scene(scene, details=['mine'])
    _, first, both = render_scene(scene, details=['scene', 'cell', 'mine'])
    _, second, reordered = render_scene(scene, details=['mine', 'scene', 'cell', 'cell'])
    assert 'layout' not in basic and 'layout' in scene_only
    assert 'cell_boxes' not in annotations['boards'][0]
    assert 'mine_positions' not in cell_only['boards'][0]
    assert 'cell_boxes' not in mine_only['boards'][0]
    assert len(mine_only['boards'][0]['mine_positions']) == 10
    assert first == second and both == reordered
    for details in ('scene', ['unknown'], [1], None):
        with pytest.raises(ValueError):
            render_scene(scene, details=details)


def test_custom_wallpaper_category_uses_supplied_assets():
    scene = desktop_scene([])
    scene['wallpaper'] = {'category': 'custom', 'seed': 1}
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(scene)
    SceneSpec.model_validate(scene, context={'wallpaper_assets': {'custom': []}})


def test_omitted_and_explicit_none_layout_seeds_remain_distinct(monkeypatch):
    import synthetic_scenes.render as renderer

    original_plan = renderer.plan_game_windows
    seeds = []

    def capture_plan(games, size, desktop, seed, scale=1.0):
        seeds.append(seed)
        # Fix placement for this check while recording the renderer's argument.
        return original_plan(games, size, desktop, 7, scale)

    monkeypatch.setattr(renderer, 'plan_game_windows', capture_plan)
    scene = desktop_scene([])
    render_scene(scene)
    scene['layout_seed'] = None
    render_scene(scene)
    assert seeds == ['scene:1', None]


def test_appearance_style_default_works_for_desktop_planning():
    scene = desktop_scene()
    del scene['windows'][0]['content']['appearance']['style']
    _, resolved, annotations = render_scene(scene, details=['scene'])
    assert annotations['boards'][0]['style'] == 'classic'
    assert 'style' not in resolved['windows'][0]['content']['appearance']
