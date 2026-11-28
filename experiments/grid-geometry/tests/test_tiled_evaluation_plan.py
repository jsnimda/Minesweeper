"""Planning tests do not render or load any model or dataset."""
from collections import Counter

import pytest

from grid_geometry.dataset.tiled_evaluation_plan import CATEGORIES, PlanConfig, plan


@pytest.mark.parametrize('split, expected', [
    ('smoke', (4, 2, 2, 1, 1)),
    ('development', (40, 20, 20, 15, 5)),
    ('test', (40, 20, 20, 15, 5)),
])
def test_default_category_counts(split, expected):
    samples = plan(PlanConfig(split))
    assert Counter(s['category'] for s in samples) == dict(zip(CATEGORIES, expected))
    assert len(samples) == sum(expected)


def test_custom_counts_and_category_stability():
    counts = dict(zip(CATEGORIES, (2, 0, 1, 0, 1)))
    config = PlanConfig('smoke', seed=101, counts=counts)
    counts['large_complete'] = 10
    samples = plan(config)
    assert len(samples) == 4
    expanded = plan(PlanConfig('smoke', seed=101, counts=dict(zip(CATEGORIES, (2, 3, 1, 0, 1)))))
    assert [s for s in samples if s['category'] == 'partial'] == [s for s in expanded if s['category'] == 'partial']


def test_reproducible_seeds_and_split_identity():
    a = plan(PlanConfig('smoke', seed=9))
    assert a == plan(PlanConfig('smoke', seed=9))
    b = plan(PlanConfig('smoke', seed=10))
    assert a != b
    groups = [a, b, plan(PlanConfig('development', seed=9)), plan(PlanConfig('test', seed=9))]
    ids = [s['sample_id'] for group in groups for s in group]
    identities = [s['scene_identity'] for group in groups for s in group]
    digests = [s['planning_seed_sha256'] for group in groups for s in group]
    assert len(ids) == len(set(ids))
    assert len(identities) == len(set(identities))
    assert len(digests) == len(set(digests))


def test_board_requests_and_partial_crop_intent():
    for sample in plan(PlanConfig('development', seed=17)):
        scene, category = sample['scene'], sample['category']
        assert scene['mode'] in ('game', 'desktop')
        if category in ('negative', 'hard_negative'):
            assert scene['mode'] == 'desktop'
            assert all(w['type'] != 'minesweeper' for w in scene['windows'])
            continue
        game = next(w['content'] for w in scene['windows'] if w['type'] == 'minesweeper')
        assert 0 < game['mine_count'] < game['rows'] * game['columns']
        assert game['status'] in ('new', 'active', 'won', 'lost')
        assert 0 <= game['progress'] <= 1
        assert game['appearance']['style'] in ('classic', 'windows7', 'microsoft', 'google')
        assert game['appearance']['zoom'] == 1.
        if category in ('large_complete', 'partial'):
            assert scene['mode'] == 'game'
            assert 40 <= game['rows'] <= 80 and 40 <= game['columns'] <= 100
            assert 16 <= game['appearance']['cell_size'] <= 22
            assert min(game['rows'], game['columns']) * game['appearance']['cell_size'] > sample['reference_tile']
        else:
            assert 5 <= game['rows'] <= 14 and 5 <= game['columns'] <= 14
            assert 16 <= game['appearance']['cell_size'] <= 24
        if category == 'partial':
            assert sample['crop_intent']['clipping'] in ('one_side', 'adjacent_sides', 'opposite_sides', 'interior')
        else:
            assert sample['crop_intent']['clipping'] is None
        assert sample['crop_intent']['margin_pitches'] in (.35, .6, 1., 2.)


@pytest.mark.parametrize('kwargs', [
    {'split': 'val'}, {'split': 'smoke', 'seed': -1}, {'split': 'smoke', 'seed': True},
    {'split': 'smoke', 'counts': {}},
    {'split': 'smoke', 'counts': dict(zip(CATEGORIES, (0, 0, 0, 0, 0)))},
    {'split': 'smoke', 'counts': dict(zip(CATEGORIES, (1, 1, -1, 1, 1)))},
    {'split': 'smoke', 'counts': dict(zip(CATEGORIES, (1, 1, True, 1, 1)))},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PlanConfig(**kwargs)
