"""Sampling and retry boundaries for the public scene schema."""
import random

import pytest
from pydantic import ValidationError

from grid_detection.dataset import generator
from grid_detection.dataset.sampler import sample_game
from synthetic_scenes.schema import SceneSpec


def test_sampled_game_appearance_matches_public_schema():
    rng = random.Random(42)
    for _ in range(100):
        game = sample_game(rng)
        SceneSpec.model_validate({'mode': 'game', 'seed': 'sample:42',
                                  'windows': [{'type': 'minesweeper', 'content': game}]})


def test_generator_does_not_retry_invalid_scene(tmp_path, monkeypatch):
    calls = []

    def invalid_scene(seed):
        calls.append(seed)
        return {'mode': 'game', 'seed': seed, 'windows': []}

    monkeypatch.setattr(generator, 'sample_scene', invalid_scene)
    excluded = tmp_path / 'excluded.json'
    excluded.write_text('[]')
    task = {'split': 'train', 'index': 0, 'seed': 42, 'excluded_path': str(excluded),
            'max_attempts': 3, 'scene_details': False}
    with pytest.raises(ValidationError):
        generator._generate_one(task)
    assert len(calls) == 1


def test_placement_retries_remain_bounded(tmp_path, monkeypatch):
    from synthetic_scenes.render import PlacementError

    calls = []

    def cannot_place(seed):
        calls.append(seed)
        raise PlacementError('cannot fit')

    monkeypatch.setattr(generator, 'sample_scene', cannot_place)
    excluded = tmp_path / 'excluded.json'
    excluded.write_text('[]')
    task = {'split': 'train', 'index': 0, 'seed': 42, 'excluded_path': str(excluded),
            'max_attempts': 3, 'scene_details': False}
    with pytest.raises(RuntimeError, match='Failed to generate') as exc:
        generator._generate_one(task)
    assert isinstance(exc.value.__cause__, PlacementError)
    assert len(calls) == 3


def test_standalone_budget_accounts_for_game_zoom(monkeypatch):
    from grid_detection.dataset import sampler
    from synthetic_scenes.minesweeper.geometry import measure_game_region

    monkeypatch.setitem(sampler.SAMPLING_CONFIG, 'standalone_image_budget',
                        {'max_dimension': 400, 'max_pixels': 160000})
    game = {'rows': 4, 'columns': 5, 'appearance': {
        'style': 'classic', 'cell_size': 64, 'zoom': 2.0,
    }}
    sampler._fit_standalone_game(game, 'game')
    cell = game['appearance']['cell_size']
    assert cell < 64 and game['appearance']['zoom'] == 2.0
    width, height = measure_game_region(4, 5, 'classic', cell, 2.0)
    assert max(width, height) <= 400 and width * height <= 160000
