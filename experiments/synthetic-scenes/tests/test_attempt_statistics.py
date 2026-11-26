"""Logical attempt, board-observation and objective counter semantics."""
from synthetic_scenes.attempt_statistics import AttemptCounters, NOT_APPLICABLE, board_factors, cell_bucket, summaries


def board(zoom=1., cell=20, rows=9, columns=9):
    return board_factors({'rows': rows, 'columns': columns,
                          'appearance': {'style': 'classic', 'cell_size': cell, 'zoom': zoom}})


def test_attempts_board_units_and_partial_fields():
    first = AttemptCounters()
    first.observe({'scene_mode': 'desktop', 'display_scale': 1.5, 'shell': None},
                  [board(1.) for _ in range(4)], 'rejected', 'placement_retries')
    first.observe({'scene_mode': 'desktop', 'display_scale': 1., 'shell': 'dock'},
                  [board(.75), board(1.)], 'accepted')
    first.finish_sample(2)
    second = AttemptCounters()
    second.observe({'scene_mode': 'game', 'display_scale': NOT_APPLICABLE, 'shell': NOT_APPLICABLE}, [], 'accepted')
    second.finish_sample(1)
    overall = AttemptCounters(); overall.merge(first); overall.merge(second)
    result = summaries(overall, {'train': first, 'val': second}, ('train', 'val', 'test'), unit='scene')
    totals = result['overall']; splits = result['splits']
    assert (totals['attempted'], totals['accepted'], totals['rejected']) == (3, 2, 1)
    assert totals['rejection_reasons'] == {'placement_retries': 1}
    assert totals['board_observations'] == 6
    assert totals['boards_by_containing_attempt_outcome']['game_zoom']['1.0']['rejected'] == 4
    assert totals['availability']['shell'] == {'available': 1, 'unavailable': 1, 'inapplicable': 1}
    assert totals['by']['display_scale']['1.5']['rejection_rate'] == 1.
    assert totals['attempts_per_sample']['counts'] == {'1': 1, '2': 1}
    assert totals['attempts_per_sample']['mean'] == 1.5
    assert splits['test']['attempted'] == 0 and splits['test']['attempts_per_sample']['mean'] is None
    assert splits['train']['attempted'] + splits['val']['attempted'] == totals['attempted']


def test_objectives_are_not_multiplied_by_source_attempts():
    stats = AttemptCounters()
    for _ in range(2): stats.observe({'source_mode': 'desktop'}, [board()], 'rejected', 'sampling_or_placement')
    stats.observe({'source_mode': 'game'}, [board(1.5)], 'accepted')
    stats.finish_sample(3, {'objective_type': 'positive', 'category': 'corner', 'pitch_bin': '20-31',
                            'desired_pitch': 24., 'anisotropy_selected': True, 'anisotropy_ratio': 1.05})
    result = stats.result()
    corner = result['objectives']['by']['category']['corner']
    assert corner['selected'] == corner['successful'] == 1
    assert corner['source_attempts'] == 3 and corner['rejected_source_attempts'] == 2
    assert corner['attempts_per_sample'] == {'3': 1}
    assert result['by']['source_mode']['desktop']['rejected'] == 2


def test_fixed_cell_bins_and_no_samples():
    assert [cell_bucket(v) for v in (11, 12, 19, 20, 32, 33, 48, 49, 64, 65)] == [
        'underflow <12', '12-19', '12-19', '20-32', '20-32', '33-48', '33-48', '49-64', '49-64', 'overflow >=65']
    empty = AttemptCounters().result()
    assert empty['attempted'] == empty['accepted'] == empty['rejected'] == 0
    assert empty['attempts_per_sample']['max'] is None
