"""Geometry dataset version, configuration and completion provenance."""
from copy import deepcopy
from dataclasses import asdict
import json
import random

import pytest
from grid_geometry.dataset import generator, sampler
from grid_geometry.dataset.sampling_config import GenerationConfig, SOURCE_CELL_SIZE


def test_configuration_snapshot_and_overrides(monkeypatch, tmp_path):
    import synthetic_scenes.provenance as provenance
    calls = []
    monkeypatch.setattr(provenance, 'run_provenance', lambda *args: calls.append(args) or {'source': {'commit': 'test'}})
    config = GenerationConfig(train=0, val=0, test=0, workers=3, progress=False, game_zoom={1.: 1},
                              standalone_source_budget=None)
    original = deepcopy(asdict(config)); state = random.getstate()
    generator.generate_dataset(tmp_path / 'geometry', config)
    result = json.loads((tmp_path / 'geometry' / 'config.json').read_text())
    assert result['experiment'] == 'grid-geometry' and result['dataset_name'] == 'geometry'
    assert result['generator'] == 'phase-grid-v1' and result['generator_revision'] == 2
    assert result['game_zoom'] == {'1.0': 1} and result['standalone_source_budget'] is None
    assert result['source_policy']['source_cell_size']['normal'] == list(SOURCE_CELL_SIZE['normal'])
    assert result['source_policy']['desktop_board_area_limit'] == sampler.DESKTOP_BOARD_AREA_LIMIT
    assert result['source_policy']['fill']['display_scale'] == generator.FILL_DISPLAY_SCALE
    assert result['source_policy']['augmentation']['blur_choices'] == list(generator.BLUR_CHOICES)
    assert result['provenance']['wallpaper_assets']['status'] == 'not_applicable'
    assert result['status'] == 'complete' and len(calls) == 1
    assert result['workers'] == config.workers == 3
    performance = json.loads((tmp_path / 'geometry' / 'statistics.json').read_text())['performance']
    assert performance['accepted_image_count'] == 0
    assert performance['asset_preparation_seconds'] is None
    assert performance['overall_generation_seconds'] >= performance['sample_generation_export_seconds']
    assert performance['sample_generation_images_per_second'] == 0
    assert asdict(config) == original and random.getstate() == state


def test_failure_retains_incomplete_snapshot(monkeypatch, tmp_path):
    import synthetic_scenes.provenance as provenance
    calls = []
    monkeypatch.setattr(provenance, 'run_provenance', lambda *args: calls.append(args) or {'source': {'commit': 'test'}})
    def fail(*args, **kwargs):
        path = tmp_path / 'failed' / 'config.json'
        assert json.loads(path.read_text())['status'] == 'incomplete'
        raise RuntimeError('failure')
        yield
    monkeypatch.setattr(generator, '_ordered_results', fail)
    config = GenerationConfig(train=1, val=0, test=0, progress=False)
    with pytest.raises(RuntimeError, match='failure'):
        generator.generate_dataset(tmp_path / 'failed', config)
    result = json.loads((tmp_path / 'failed' / 'config.json').read_text())
    assert result['status'] == 'incomplete' and result['provenance']['source']['commit'] == 'test'
    assert not (tmp_path / 'failed' / 'statistics.json').exists()
    assert result['standalone_source_budget'] == {'max_dimension': 4096, 'max_pixels': 12_000_000}
    assert len(calls) == 1


@pytest.mark.parametrize('format_name,revision,valid', [
    ('phase-grid-v1', 1, True), ('phase-grid-v1', 2, True),
    (None, 1, False), ('other', 1, False), ('phase-grid-v1', None, False),
    ('phase-grid-v1', 0, False), ('phase-grid-v1', True, False),
])
def test_dataset_format_and_revision(tmp_path, format_name, revision, valid):
    from grid_geometry.dataset.loader import PhaseDataset
    config = {'status': 'complete', 'input_size': [32, 32], 'transition_ratio': .25}
    if format_name is not None: config['generator'] = format_name
    if revision is not None: config['generator_revision'] = revision
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'train').mkdir()
    if valid:
        assert len(PhaseDataset(tmp_path, 'train')) == 0
    else:
        with pytest.raises(ValueError, match='format|revision'):
            PhaseDataset(tmp_path, 'train')


def test_provenance_does_not_change_seeded_source_sampling(monkeypatch, tmp_path):
    import synthetic_scenes.provenance as provenance
    expected = sampler._game(random.Random(31), 'complete', 12., (384, 384), game_zoom={1.: 1})
    monkeypatch.setattr(provenance, 'run_provenance', lambda *args: {'source': {'commit': 'test'}})
    generator.generate_dataset(tmp_path / 'dataset', GenerationConfig(train=0, val=0, test=0, progress=False))
    actual = sampler._game(random.Random(31), 'complete', 12., (384, 384), game_zoom={1.: 1})
    assert actual == expected
