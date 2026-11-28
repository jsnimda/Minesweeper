"""Phase dataset generation and loading checks; run manually."""
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest

from grid_geometry.dataset.loader import PhaseDataset, inspect_dataset
from grid_geometry.dataset.generator import GenerationConfig, _crop_box, _geometry, _pitch_bin, generate_dataset
from grid_geometry.dataset.targets import PhaseGeometry


def _write_dataset(root):
    config = {'status': 'complete', 'generator': 'phase-grid-v1', 'generator_revision': 1, 'input_size': [32, 32], 'transition_ratio': .25,
              'train': 2, 'val': 0, 'test': 0}
    root.mkdir()
    (root / 'config.json').write_text(json.dumps(config))
    for split in ('train', 'val', 'test'):
        (root / split).mkdir()
    positive = {'sample_id': '000000', 'split': 'train', 'board_id': 'board-a',
                'geometry': asdict(PhaseGeometry(-20.5, -20.5, 8.25, 8.25, (-20.5, -20.5, 70., 70.))),
                'known_box': [0, 0, 32, 32]}
    negative = {'sample_id': '000001', 'split': 'train', 'board_id': None,
                'geometry': None, 'known_box': [3, 0, 32, 32]}
    for record in (positive, negative):
        name = record['sample_id']
        Image.new('RGB', (32, 32), 'white').save(root / 'train' / f'{name}.png')
        (root / 'train' / f'{name}.json').write_text(json.dumps(record))
    (root / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in (positive, negative)))


def test_loader_positive_negative_and_interior(tmp_path):
    root = tmp_path / 'dataset'; _write_dataset(root)
    dataset = PhaseDataset(root, 'train')
    image, target, phase, occupancy = dataset[0]
    assert image.shape == (3, 32, 32) and target.shape == (5, 32, 32)
    assert phase.all() and occupancy.all() and (target[4] == 1).all()
    _, negative, phase, occupancy = dataset[1]
    assert not phase.any() and not negative.any() and not occupancy[:, :3].any()
    assert inspect_dataset(root)['split_counts'] == {'train': 2, 'val': 0, 'test': 0}


def test_gap_metadata_and_fractional_transform():
    board = {'boxes': {'grid': (4., 5., 33., 34.)}, 'style_parameters': {'pitch': 10., 'gap': 1.}}
    geometry = _geometry(board)
    assert geometry.pitch_x == 10 and geometry.origin_x == 3.5 and geometry.visible_box[2] == 33
    transformed = geometry.transformed(1.25, .75, 3.5, -1.25)
    assert transformed.pitch_x == 12.5 and transformed.origin_y == 2.125
    assert transformed.visible_box[2] == pytest.approx(44.75)
    assert _pitch_bin(12.5, ((6, 11, 1), (12, 19, 1))) == '12-19'


def test_crop_categories_are_deterministic():
    box = (20., 30., 520., 530.); span = (200., 200.)
    for category in ('complete', 'interior', 'side', 'corner'):
        use_box = (20., 30., 120., 130.) if category == 'complete' else box
        crop = _crop_box(random.Random(22), use_box, span, category)
        assert crop == _crop_box(random.Random(22), use_box, span, category)
        sides = sum((crop[0] < use_box[0] < crop[2], crop[0] < use_box[2] < crop[2],
                     crop[1] < use_box[1] < crop[3], crop[1] < use_box[3] < crop[3]))
        assert sides == {'complete': 4, 'interior': 0, 'side': 1, 'corner': 2}[category]


def test_refuses_nonempty_output_and_empty_splits(tmp_path):
    root = tmp_path / 'existing'; root.mkdir(); (root / 'keep').write_text('keep')
    with pytest.raises(FileExistsError):
        generate_dataset(root, GenerationConfig(train=0, val=0, test=0, progress=False))
    empty = tmp_path / 'empty'
    result = generate_dataset(empty, GenerationConfig(train=0, val=0, test=0, progress=False))
    assert all(value['samples'] == 0 for value in result['splits'].values())
    assert inspect_dataset(empty)['split_counts'] == {'train': 0, 'val': 0, 'test': 0}


def test_worker_count_determinism_and_split_isolation(tmp_path):
    config = GenerationConfig(train=4, val=2, test=2, workers=1, progress=False)
    first, second = tmp_path / 'one', tmp_path / 'two'
    generate_dataset(first, config)
    generate_dataset(second, GenerationConfig(train=4, val=2, test=2, workers=2, progress=False))
    assert (first / 'manifest.jsonl').read_bytes() == (second / 'manifest.jsonl').read_bytes()
    for path in first.rglob('*.png'):
        assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
    assert not any(inspect_dataset(first)['board_identity_overlap'].values())


def test_true_boundary_and_unknown_padding(tmp_path):
    root = tmp_path / 'dataset'; _write_dataset(root)
    path = root / 'train' / '000000.json'
    metadata = json.loads(path.read_text())
    metadata['geometry'] = asdict(PhaseGeometry(8.5, 8.5, 8., 8., (8.5, 8.5, 28.5, 28.5)))
    metadata['known_box'] = [2, 0, 32, 32]
    path.write_text(json.dumps(metadata))
    sample = PhaseDataset(root, 'train').load_sample(0)
    assert sample.targets[4, 15, 8].item() == pytest.approx(.5)
    assert sample.targets[4, 15, 0].item() == 0
    assert not sample.occupancy_mask[:, :2].any()


def test_split_overlap_is_reported(tmp_path):
    root = tmp_path / 'dataset'; _write_dataset(root)
    config_path = root / 'config.json'; config = json.loads(config_path.read_text()); config['val'] = 1
    config_path.write_text(json.dumps(config))
    record = json.loads((root / 'train' / '000000.json').read_text())
    record['split'] = 'val'
    (root / 'val' / '000000.json').write_text(json.dumps(record))
    Image.new('RGB', (32, 32)).save(root / 'val' / '000000.png')
    with (root / 'manifest.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    with pytest.raises(ValueError, match='identity overlap'):
        inspect_dataset(root)


def test_source_aware_crops_preserve_category_and_reduce_padding():
    from grid_geometry.dataset.generator import _crop_axis

    # A complete-board crop has a feasible placement inside the source.
    for seed in range(20):
        start = _crop_axis(random.Random(seed), 80, 140, 100, 'complete', source_length=180, prefer_source=True)
        assert 40 <= start <= 75 and 0 <= start <= 80
    # If the source is smaller than the crop, choose a minimum-padding placement.
    start = _crop_axis(random.Random(3), 20, 60, 120, 'complete', source_length=80, prefer_source=True)
    assert -40 <= start <= 0
    crop = _crop_box(random.Random(7), (60., 60., 560., 560.), (200., 200.),
                     'corner', (640, 640), True)
    assert sum((crop[0] < 60 < crop[2], crop[0] < 560 < crop[2],
                crop[1] < 60 < crop[3], crop[1] < 560 < crop[3])) == 2


def test_compact_fractions_match_targets_with_fractional_geometry():
    from grid_geometry.dataset.generator import sample_fractions
    from grid_geometry.dataset.targets import phase_targets

    size = (32, 24)
    geometry = PhaseGeometry(2.25, -2.5, 7.5, 9.25, (2.5, -1.25, 24.5, 19.75))
    known_box = (3.25, 1.5, 29.5, 22.25)
    x = np.arange(size[0]) + .5; y = np.arange(size[1]) + .5
    known = ((x[None, :] >= known_box[0]) & (x[None, :] < known_box[2]) &
             (y[:, None] >= known_box[1]) & (y[:, None] < known_box[3]))
    targets, phase, occupancy_mask = phase_targets(size, geometry, known_mask=known)
    fractions = sample_fractions(size, known_box, geometry, .25)
    assert fractions['phase_supervision_fraction'] == pytest.approx(phase.mean())
    assert fractions['grid_coverage_fraction'] == pytest.approx(phase.mean())
    assert fractions['grid_occupancy_fraction'] == pytest.approx((targets[4] * occupancy_mask).mean(), abs=1e-6)
    negative = sample_fractions(size, known_box, None, .25)
    assert negative['phase_supervision_fraction'] is None and negative['grid_occupancy_fraction'] == 0


def test_coverage_summary_and_config_validation():
    from grid_geometry.dataset.generator import summarize_fractions
    records = [dict(category='complete', actual_pitch_bin_x='6-11', grid_coverage_fraction=.04,
                    phase_supervision_fraction=.04, grid_occupancy_fraction=.04),
               dict(category='negative', actual_pitch_bin_x=None, grid_coverage_fraction=None,
                    phase_supervision_fraction=None, grid_occupancy_fraction=0.)]
    result = summarize_fractions(records)
    assert result['overall']['positive_fraction_below_coverage']['0.05'] == 1
    assert result['by_pitch_bin']['6-11']['positive_samples'] == 1
    assert summarize_fractions([])['overall']['samples'] == 0


def test_sampling_preference_keeps_standard_and_custom_options():
    from grid_geometry.dataset.generator import _game
    dimensions = []
    for seed in range(100):
        game, _ = _game(random.Random(seed), 'complete', 9., (384, 384))
        dimensions.append((game['rows'], game['columns']))
    assert any(size in {(9, 9), (16, 16), (16, 30)} for size in dimensions)
    assert any(size not in {(9, 9), (16, 16), (16, 30), (5, 5), (6, 8)} for size in dimensions)
    assert max(rows * columns for rows, columns in dimensions) > 16 * 16


@pytest.mark.parametrize('negative_fraction', [0., 1.])
def test_generated_samples_have_full_known_region(tmp_path, negative_fraction):
    config = GenerationConfig(train=3, val=0, test=0, workers=1, progress=False,
                              negative_fraction=negative_fraction,
                              context_weights={'procedural': 1, 'desktop': 0})
    root = tmp_path / 'generated'
    stats = generate_dataset(root, config)
    dataset = PhaseDataset(root, 'train')
    obsolete = {'intentional_padding_requested', 'accidental_padding_fraction',
                'padding_type', 'padding_side', 'padding_extent', 'padding_color', 'padding_fraction'}
    for index in range(len(dataset)):
        sample = dataset.load_sample(index)
        assert sample.metadata['known_box'] == [0, 0, *dataset.size]
        assert obsolete.isdisjoint(sample.metadata)
        assert sample.occupancy_mask.all()
        assert not (sample.phase_mask & ~sample.occupancy_mask).any()
        if negative_fraction == 1.:
            assert not sample.phase_mask.any() and not sample.targets[4].any()
    assert 'padding' not in stats['splits']['train']
    saved = json.loads((root / 'config.json').read_text())
    assert saved['generator_revision'] == 2
    assert 'padding_example_fraction' not in saved and 'padding_max_fraction' not in saved


def test_context_configuration_validation():
    with pytest.raises(ValueError):
        GenerationConfig(context_weights={'procedural': 1})


def test_fractional_crop_transform_preserves_true_boundary():
    geometry = PhaseGeometry(10.25, 9.75, 12.5, 11.75, (10.25, 9.75, 60.25, 45.))
    transformed = geometry.transformed(.8, 1.1, -4.3, 7.2)
    assert transformed.origin_x == pytest.approx(3.9)
    assert transformed.pitch_x == pytest.approx(10.)
    assert transformed.visible_box[0] == pytest.approx(3.9)


@pytest.mark.parametrize('context', ['procedural', 'desktop'])
def test_requested_context_is_recorded_without_downloads(tmp_path, context):
    weights = {key: int(key == context) for key in ('procedural', 'desktop')}
    config = GenerationConfig(train=2, val=0, test=0, workers=1, progress=False,
                              context_weights=weights)
    root = tmp_path / 'dataset'
    stats = generate_dataset(root, config)
    assert stats['splits']['train']['requested_context'][context] == 2
    for index in range(2):
        record = PhaseDataset(root, 'train').load_sample(index).metadata
        assert record['requested_context'] == context


def test_scene_validation_errors_are_not_retried(tmp_path, monkeypatch):
    from pydantic import ValidationError
    from synthetic_scenes.schema import SceneSpec
    from grid_geometry.dataset import generator

    calls = []

    def invalid_game(*args, **kwargs):
        calls.append(1)
        SceneSpec.model_validate({})

    monkeypatch.setattr(generator, '_game', invalid_game)
    excluded = tmp_path / 'excluded.json'
    excluded.write_text('[]')
    config = asdict(GenerationConfig(negative_fraction=0., max_attempts=3))
    task = {'config': config, 'split': 'train', 'index': 0, 'excluded_path': str(excluded)}
    with pytest.raises(ValidationError):
        generator._generate_one(task)
    assert len(calls) == 1


def test_geometry_sampled_scenes_match_public_schema():
    from grid_geometry.dataset.sampler import _game, _scene
    from synthetic_scenes.schema import SceneSpec

    rng = random.Random(42)
    for _ in range(40):
        game, _ = _game(rng, 'complete', 12., (384, 384))
        SceneSpec.model_validate(_scene(rng, game))
    SceneSpec.model_validate(_scene(rng, None, 'hard_negative'))


def test_scene_sampling_retries_remain_bounded(tmp_path, monkeypatch):
    from grid_geometry.dataset import generator

    calls = []

    def infeasible_game(*args, **kwargs):
        calls.append(1)
        raise ValueError('infeasible crop')

    monkeypatch.setattr(generator, '_game', infeasible_game)
    excluded = tmp_path / 'excluded.json'
    excluded.write_text('[]')
    config = asdict(GenerationConfig(negative_fraction=0., max_attempts=3))
    task = {'config': config, 'split': 'train', 'index': 0, 'excluded_path': str(excluded)}
    with pytest.raises(RuntimeError, match='after 3 attempts'):
        generator._generate_one(task)
    assert len(calls) == 3
