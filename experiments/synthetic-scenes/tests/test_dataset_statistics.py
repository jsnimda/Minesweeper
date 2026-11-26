"""Numerical summaries and fitting denominators for accepted records."""
import json
import pytest
from synthetic_scenes.dataset_statistics import Summary, numeric_summary


def test_histogram_edges_percentiles_and_empty():
    result = numeric_summary([-1, 0, 1, 2, 3, 4], (0, 1, 2, 3))
    assert result['histogram']['counts'] == [1, 1, 2]
    assert result['histogram']['underflow'] == result['histogram']['overflow'] == 1
    assert result['count'] == 6 and result['min'] == -1 and result['max'] == 4
    assert result['mean'] == 1.5 and result['median'] == 1.5
    assert result['p05'] == pytest.approx(-.75) and result['p95'] == pytest.approx(3.75)
    empty = numeric_summary([None, float('nan'), float('inf')], (0, 1, 2))
    assert empty['count'] == 0 and empty['mean'] is None
    assert empty['histogram']['counts'] == [0, 0]
    json.dumps(empty, allow_nan=False)


def test_fitting_denominators_and_reductions():
    summary = Summary()
    summary.board({'sampled_cell_size': 40, 'requested_cell_size': 30, 'resolved_cell_size': 30,
                   'requested_zoom': 1.5, 'rows': 20, 'columns': 20}, 'game')
    summary.board({'sampled_cell_size': 40, 'requested_cell_size': 40, 'resolved_cell_size': 32,
                   'requested_zoom': 1.5, 'rows': 10, 'columns': 10}, 'desktop', 'retro')
    summary.board({'sampled_cell_size': 20, 'requested_cell_size': 20, 'resolved_cell_size': 20}, 'grid')
    result = summary.result(); totals = result['totals']
    assert totals['boards'] == 3 and totals['budget_applicable_boards'] == 2
    assert totals['budget_fitted_boards'] == 1 and totals['budget_fitting_rate'] == .5
    assert totals['placement_applicable_boards'] == totals['placement_fitted_boards'] == 1
    assert result['numerical']['budget_reduction']['mean'] == 5
    assert result['numerical']['placement_relative_reduction']['mean'] == .2
    assert result['fitting']['overall']['all']['any_fitted'] == 2
    assert result['fitting']['zoom']['1.5']['applicable_boards'] == 2
    assert result['fitting']['resolution_group']['retro']['placement_fitted'] == 1
