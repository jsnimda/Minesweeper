"""Mocked scene generation; no actual renderer or model is used."""
import json

from PIL import Image
import pytest

from grid_geometry.dataset.tiled_evaluation_generate import crop_for_board, generate, read_excluded_ids
from grid_geometry.dataset.targets import PhaseGeometry
from grid_geometry.dataset.tiled_evaluation_plan import PlanConfig
from grid_geometry.phase.paired_evaluate import crop_bounds


def fake_plan(category='large_complete', name='sample', clipping=None):
    return {'sample_id': name, 'category': category, 'scene': {'name': name},
            'crop_intent': {'margin_pitches': .6, 'clipping': clipping},
            'scene_identity': 'identity-' + name}


def board(identity='board-a'):
    return {'board_id': identity, 'rows': 5, 'columns': 8,
            'boxes': {'grid': (20.5, 10.5, 100.5, 60.5)},
            'style_parameters': {'pitch': 10., 'gap': 0.}}


def fake_renderer(scene, *, details, wallpaper_assets):
    assert details == ('scene',) and wallpaper_assets == {}
    boards = [] if scene['name'] == 'negative' else [board()]
    return Image.new('RGB', (120, 80)), {'mode': 'mock'}, {'size': (120, 80), 'boards': boards}


def test_crop_complete_partial_clamping():
    geometry = PhaseGeometry(20.5, 10.5, 10., 10., (20.5, 10.5, 100.5, 60.5))
    complete = crop_for_board(geometry, (120, 80), {'margin_pitches': 10., 'clipping': None}, partial=False)
    assert complete == [0, 0, 120, 80]
    for clipping in ('one_side', 'adjacent_sides', 'opposite_sides', 'interior'):
        crop = crop_for_board(geometry, (120, 80), {'margin_pitches': .6, 'clipping': clipping}, partial=True)
        assert crop[0] > 20.5 and crop[2] > crop[0] and crop[3] > crop[1]
    with pytest.raises(ValueError):
        crop_for_board(geometry, (90, 80), {'margin_pitches': 1., 'clipping': None}, partial=False)


def test_manifest_native_geometry_and_negative(monkeypatch, tmp_path):
    from grid_geometry.dataset import tiled_evaluation_generate as module
    plans = [fake_plan(), fake_plan('partial', 'partial', 'interior'), fake_plan('negative', 'negative')]
    monkeypatch.setattr(module, 'plan', lambda config: plans)
    def renderer(scene, **kwargs):
        image, resolved, annotations = fake_renderer(scene, **kwargs)
        if scene['name'] == 'partial': annotations['boards'][0]['board_id'] = 'board-b'
        return image, resolved, annotations
    root = tmp_path / 'data'
    path = generate(PlanConfig('smoke'), root, renderer=renderer)
    samples = json.loads(path.read_text())['samples']
    first, partial, negative = samples
    assert first['split'] == 'val' and first['geometry']['visible_box'] == [20.5, 10.5, 100.5, 60.5]
    assert first['board_dimensions'] == [5, 8] and first['geometry']['pitch_x'] == 10.
    assert crop_bounds(first, (120, 80), 0) == tuple(first['crop_box'])
    assert partial['complete'] is False and partial['geometry'] == first['geometry']
    assert negative['geometry'] is None and negative['category'] == 'negative'
    with Image.open(root / first['image']) as image:
        assert image.size == (120, 80) and image.format == 'PNG'
    assert json.loads((root / 'generation.json').read_text())['samples'][0]['plan'] == plans[0]


def test_duplicate_boards_and_failure_leave_no_manifest(monkeypatch, tmp_path):
    from grid_geometry.dataset import tiled_evaluation_generate as module
    monkeypatch.setattr(module, 'plan', lambda config: [fake_plan(), fake_plan(name='second')])
    root = tmp_path / 'data'
    with pytest.raises(RuntimeError, match='second.*duplicate'):
        generate(PlanConfig('smoke'), root, renderer=fake_renderer)
    assert not (root / 'manifest.json').exists()
    with pytest.raises(FileExistsError):
        generate(PlanConfig('smoke'), root, renderer=fake_renderer)


def test_identity_exclusion_and_bad_annotations(monkeypatch, tmp_path):
    from grid_geometry.dataset import tiled_evaluation_generate as module
    monkeypatch.setattr(module, 'plan', lambda config: [fake_plan()])
    exclusion = tmp_path / 'manifest.jsonl'
    exclusion.write_text(json.dumps({'board_id': 'board-a'}) + '\n')
    assert read_excluded_ids([exclusion]) == {'board-a'}
    with pytest.raises(RuntimeError, match='sample.*excluded'):
        generate(PlanConfig('development'), tmp_path / 'excluded', exclude_identity_paths=[exclusion], renderer=fake_renderer)
    def broken(scene, **kwargs):
        raise ValueError('renderer failure')
    with pytest.raises(RuntimeError, match='sample.*renderer failure'):
        generate(PlanConfig('test'), tmp_path / 'broken', renderer=broken)
    assert not (tmp_path / 'broken' / 'manifest.json').exists()
    monkeypatch.setattr(module, 'plan', lambda config: [fake_plan('negative')])
    with pytest.raises(RuntimeError, match='sample.*number of target boards'):
        generate(PlanConfig('test'), tmp_path / 'bad-negative', renderer=fake_renderer)
