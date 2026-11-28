"""Integrity checks against tiny generated test fixtures, without inference."""
import json

from PIL import Image
import pytest

from grid_geometry.dataset import tiled_evaluation_validate as validator


def fixture(root, monkeypatch, category='small_complete'):
    root.mkdir()
    Image.new('RGB', (80, 60)).save(root/'image.png')
    planned = {'sample_id': 's', 'category': category}
    monkeypatch.setattr(validator, 'plan', lambda config: [planned])
    entry = {'sample_id': 's', 'split': 'val', 'category': category, 'image': 'image.png',
             'image_size': [80, 60], 'known_box': [0, 0, 80, 60], 'crop_box': [0, 0, 80, 60],
             'complete': True, 'geometry': {'origin_x': 10., 'origin_y': 10., 'pitch_x': 10.,
             'pitch_y': 10., 'visible_box': [10., 10., 50., 40.]},
             'board_dimensions': [3, 4], 'board_id': 'board-1'}
    metadata = {'planner_config': {'split': 'smoke', 'seed': 42}, 'planning_split': 'smoke', 'evaluator_split': 'val',
                'samples': [{'sample_id': 's', 'plan': planned, 'board_id': 'board-1'}]}
    def write():
        (root/'manifest.json').write_text(json.dumps({'samples': [entry]}))
        (root/'generation.json').write_text(json.dumps(metadata))
    write()
    return entry, metadata, write


def test_valid_margins_and_serialization(tmp_path, monkeypatch):
    root = tmp_path/'data'; fixture(root, monkeypatch)
    result = validator.validate(root, tile=32, overlap=8)
    assert result['status'] == 'valid' and result['multi_tile'] == 1
    margins = result['samples'][0]['exterior_margins']
    assert margins['left'] == {'pixels': 10., 'cells': 1.}
    assert margins['right'] == {'pixels': 30., 'cells': 3.}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('change', ['missing', 'dimensions', 'geometry', 'complete', 'crop', 'category', 'large'])
def test_invalid_data(tmp_path, monkeypatch, change):
    root = tmp_path/'data'; e, meta, write = fixture(root, monkeypatch)
    if change == 'missing': (root/'image.png').unlink()
    elif change == 'dimensions': e['image_size'] = [1, 1]
    elif change == 'geometry': e['geometry']['pitch_x'] = 12.
    elif change == 'complete': e['complete'] = False
    elif change == 'crop': e['crop_box'] = [90, 0, 100, 40]
    elif change == 'category': e['category'] = 'negative'
    elif change == 'large':
        e['category'] = 'large_complete'; meta['samples'][0]['plan']['category'] = 'large_complete'
    write()
    result = validator.validate(root, tile=384, overlap=96)
    assert result['errors'] and result['status'] == 'invalid'


def test_identity_sources_and_empty_warning(tmp_path, monkeypatch):
    root = tmp_path/'data'; fixture(root, monkeypatch)
    source = tmp_path/'ids.json'; source.write_text(json.dumps(['board-1']))
    empty = tmp_path/'empty.json'; empty.write_text('[]')
    result = validator.validate(root, identity_sources=[source, empty])
    assert result['identity_sources'][0]['sample_ids'] == ['s']
    assert result['identity_sources'][0]['overlap'] == 1
    assert result['identity_sources'][1]['identities'] == 0
    assert result['errors'] and any('no usable' in x for x in result['warnings'])


def test_partial_excluded_sides_and_single_tile(tmp_path, monkeypatch):
    root = tmp_path/'data'; e, meta, write = fixture(root, monkeypatch, 'partial')
    e['complete'] = False
    e['crop_box'] = [20, 0, 70, 50]
    write()
    result = validator.validate(root)
    assert result['status'] == 'valid' and result['single_tile'] == 1
    assert result['samples'][0]['excluded_sides'] == ['left']


def test_duplicate_sample_identity_rejected(tmp_path, monkeypatch):
    root = tmp_path/'data'; e, _, _ = fixture(root, monkeypatch)
    (root/'manifest.json').write_text(json.dumps({'samples': [e, e]}))
    result = validator.validate(root)
    assert any('duplicate sample' in error for error in result['errors'])
