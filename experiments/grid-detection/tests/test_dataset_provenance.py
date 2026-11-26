"""Dataset identity and passive run-level provenance."""
from copy import deepcopy
import json
import random

import pytest
from grid_detection.dataset import generator


def test_snapshot_and_completion(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(generator, 'run_provenance', lambda *args: calls.append(args) or {'source': {'commit': 'test'}})
    monkeypatch.setitem(generator.SAMPLING_CONFIG, 'display_scale', {1.: 1})
    before = deepcopy(generator.SAMPLING_CONFIG)
    before_rng = random.getstate()
    def splits(*args, **kwargs):
        config = json.loads((tmp_path / 'dataset' / 'config.json').read_text())
        assert config['status'] == 'incomplete'
        assert config['provenance']['source']['commit'] == 'test'
        return {}
    monkeypatch.setattr(generator, 'run_splits', splits)
    generator.generate_dataset(tmp_path / 'dataset', train=0, val=0, progress=False)
    result = json.loads((tmp_path / 'dataset' / 'config.json').read_text())
    assert result['experiment'] == 'grid-detection' and result['dataset_name'] == 'dataset'
    assert result['generator_revision'] == 7 and result['status'] == 'complete'
    assert result['sampling_scope']['display_scale'] == {'1.0': 1}
    assert result['source_policy']['display_scale_policy'] == 'global'
    assert result['source_policy']['capture_augmentation']['resize_range'] == [.65, 1.25]
    assert result['provenance']['source']['commit'] == 'test' and len(calls) == 1
    assert generator.SAMPLING_CONFIG == before and random.getstate() == before_rng


def test_conditional_policy_and_procedural_only(monkeypatch, tmp_path):
    monkeypatch.setitem(generator.SAMPLING_CONFIG, 'wallpaper_type', {'procedural': 1})
    monkeypatch.setattr(generator, 'get_wallpapers', lambda *args: pytest.fail('external assets requested'))
    monkeypatch.setattr(generator, 'run_splits', lambda *args, **kwargs: {})
    generator.generate_dataset(tmp_path / 'procedural', train=1, val=0, progress=False)
    result = json.loads((tmp_path / 'procedural' / 'config.json').read_text())
    assert result['source_policy']['display_scale_policy'] == 'conditional_by_resolution'
    assert result['provenance']['wallpaper_assets']['status'] == 'not_applicable'


def test_failure_leaves_incomplete_config(monkeypatch, tmp_path):
    monkeypatch.setattr(generator, 'run_provenance', lambda *args: {'source': {'commit': 'test'}})
    monkeypatch.setattr(generator, 'run_splits', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('failed')))
    with pytest.raises(RuntimeError, match='failed'):
        generator.generate_dataset(tmp_path / 'failed', train=0, val=0, progress=False)
    saved = json.loads((tmp_path / 'failed' / 'config.json').read_text())
    assert saved['status'] == 'incomplete' and saved['provenance']['source']['commit'] == 'test'


def test_asset_inventory_reuses_actual_prepared_paths(monkeypatch, tmp_path):
    wallpaper = tmp_path / 'wallpaper.png'; wallpaper.write_bytes(b'fixture')
    calls = []
    def get_assets(category):
        calls.append(category)
        return (wallpaper,)
    monkeypatch.setattr(generator, 'get_wallpapers', get_assets)
    captured = []
    def splits(*args, **kwargs):
        captured.append(kwargs['assets'])
        return {}
    monkeypatch.setattr(generator, 'run_splits', splits)
    generator.generate_dataset(tmp_path / 'dataset', train=1, val=0, progress=False)
    saved = json.loads((tmp_path / 'dataset' / 'config.json').read_text())
    assert calls == ['artwork', 'photographs'] and len(captured) == 1
    inventory = saved['provenance']['wallpaper_assets']
    assert inventory['status'] == 'available'
    assert inventory['categories']['artwork']['identifiers'] == ['wallpaper.png']
    assert len(inventory['fingerprint']) == 64


def test_provenance_does_not_change_seeded_game_sampling(monkeypatch, tmp_path):
    from grid_detection.dataset import sampler
    expected = sampler.sample_game(random.Random(31))
    monkeypatch.setattr(generator, 'run_provenance', lambda *args: {'source': {'commit': 'test'}})
    monkeypatch.setattr(generator, 'run_splits', lambda *args, **kwargs: {})
    generator.generate_dataset(tmp_path / 'dataset', train=0, val=0, progress=False)
    assert sampler.sample_game(random.Random(31)) == expected
