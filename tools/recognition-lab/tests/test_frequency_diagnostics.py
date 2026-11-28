from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from grid_geometry.phase.decoder import AxisExtent, DecodeResult, FrequencyCandidate, GridHypothesis
from grid_geometry.phase.tiles import TileObservation
from recognition_lab.frequency_diagnostics import (candidate_rows, profile, profile_markers, resolve_frequency,
                                                    sample_value)
from recognition_lab.recognition_state import RecognitionSelection
from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult


def fixture(mode='tiled'):
    x1 = FrequencyCandidate(64., 63.25, .9, .8)
    x2 = FrequencyCandidate(48., 47.75, .875, .7)
    y1 = FrequencyCandidate(50., 1.25, .85, .65)
    x_extent = AxisExtent(-.75, 48., 2, None, 94.25, True, False)
    y_extent = AxisExtent(1.25, 50., 2, 1.75, None, False, True)
    top = GridHypothesis(x_extent, y_extent, x1, y1, .9, .072, .947)
    alternate = GridHypothesis(x_extent, y_extent, x2, y1, .8, .064, .9265)
    signal = np.array([1+2j, 3-4j, complex(np.nan, 5), 0+0j], np.complex64)
    support = np.array([0., 1., 2., 3.], np.float32)
    occupancy = np.array([.1, .4, .9, .3], np.float32)
    decoded = DecodeResult('ambiguous', (top, alternate),
                           signals={'x': signal, 'x_support': support, 'y': signal, 'y_support': support},
                           frequency_candidates={'x': (x1, x2), 'y': (y1,)},
                           occupancy_profiles={'x': occupancy, 'y': occupancy},
                           reasons=('competing frequency hypotheses',))
    checked = SimpleNamespace(local_accepted=False, local_reasons=('decoder status: ambiguous',))
    obs = TileObservation((0, 0, 4, 4), top, False, decoding=decoded, local_validation=checked)
    other = TileObservation((4, 0, 8, 4), None, False)
    grid = GeometryResult(0, Crop((0, 0, 8, 4), False), None, None, None, None, [],
                          diagnostics={'mode': mode}, phase_data={'observations': [obs] if mode == 'whole' else [obs, other]})
    result = RecognitionResult([Detection('grid', .9, (0, 0, 8, 4))], [grid])
    state = RecognitionSelection(); state.replace(result)
    node = state.snapshot.roots[0].children[0].children[0].children[0]
    return state, node, obs, signal, support, occupancy


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_observation_resolution_and_selected_candidates(mode):
    state, node, obs, _, _, _ = fixture(mode)
    state.select(node.id)
    evidence = resolve_frequency(state, 0)
    assert evidence.observation is obs and evidence.decoder.status == 'ambiguous'
    assert evidence.validation == 'Evaluated rejected' and not evidence.fallback
    rows = candidate_rows(evidence, 'x')
    assert [row[1] for row in rows] == [64., 48.]
    assert rows[0][2] == 1/64 and rows[1][2] == 1/48
    assert rows[0][3:6] == (63.25, .9, .8)
    assert [row[6] for row in rows] == [True, False]
    assert candidate_rows(evidence, 'y')[0][6]
    selected = state.selected
    candidate_rows(evidence, 'x')
    assert state.selected == selected
    state.select(node.children[1].id)
    evidence = resolve_frequency(state, 0)
    assert evidence.validation == 'Not evaluated'
    assert [row[6] for row in candidate_rows(evidence, 'x')] == [False, True]
    assert candidate_rows(evidence, 'y')[0][6]
    state.select(state.snapshot.roots[0].children[0].id)
    assert resolve_frequency(state, 0).fallback
    if mode == 'tiled':
        empty = resolve_frequency(state, 1)
        assert empty.observation_index == 1 and empty.decoder is None
        assert candidate_rows(empty, 'x') == () and profile(empty, 'x', 'Signal real') is None


def test_frequency_identity_does_not_match_equal_unrelated_candidate():
    state, node, obs, *_ = fixture()
    equal_copy = replace(obs.hypothesis.x_evidence)
    replacement = replace(obs.decoding, frequency_candidates={'x': (equal_copy,), 'y': ()})
    state.snapshot.result.grids[0].phase_data['observations'][0] = replace(obs, decoding=replacement)
    state.replace(state.snapshot.result)
    state.select(state.snapshot.roots[0].children[0].children[0].children[0].id)
    assert not candidate_rows(resolve_frequency(state, 0), 'x')[0][6]


def test_retained_profiles_pixel_centers_and_nonfinite_values():
    state, node, _, signal, support, occupancy = fixture(); state.select(node.id)
    evidence = resolve_frequency(state, 0)
    real = profile(evidence, 'x', 'Signal real')
    imag = profile(evidence, 'x', 'Signal imaginary')
    assert np.shares_memory(real, signal) and np.shares_memory(imag, signal)
    assert profile(evidence, 'x', 'Signal support') is support
    assert profile(evidence, 'x', 'Occupancy profile') is occupancy
    assert sample_value(real, .5) == (0, 1.)
    assert sample_value(imag, 1.5) == (1, -4.)
    index, value = sample_value(real, 2.5)
    assert index == 2 and np.isnan(value)
    assert sample_value(real, -.1) is None and sample_value(real, 4.) is None
    assert profile(evidence, 'z', 'Signal support') is None
    assert profile(evidence, 'x', 'Unavailable') is None


def test_fractional_markers_do_not_clamp_and_ignore_censored_crossings():
    state, node, obs, *_ = fixture(); state.select(node.id)
    extent = replace(obs.hypothesis.x, origin=-.75, pitch=2., count=2,
                     visible_start=-1., start_censored=True, visible_end=2.125, end_censored=False)
    replacement = replace(obs.hypothesis, x=extent)
    updated_decode = replace(obs.decoding, hypotheses=(replacement,))
    state.snapshot.result.grids[0].phase_data['observations'][0] = replace(obs, hypothesis=replacement, decoding=updated_decode)
    state.replace(state.snapshot.result); node = state.snapshot.roots[0].children[0].children[0].children[0]
    state.select(node.id)
    markers = profile_markers(resolve_frequency(state, 0), 'x', 4)
    assert ('lattice', 1.25) in markers and ('lattice', 3.25) in markers
    assert ('equivalent boundary', 3.25) in markers and ('equivalent boundary', 0.5) not in markers
    assert ('visible crossing', 2.125) in markers and ('visible crossing', -1.) not in markers


def test_stale_snapshot_and_missing_evidence():
    state, node, obs, *_ = fixture(); state.select(node.id)
    old = node.id
    result = state.snapshot.result
    result.grids[0].phase_data['observations'][0] = replace(obs, decoding=None, hypothesis=None)
    state.replace(result, geometry_update=True)
    assert state.snapshot.resolve(old) is None
    assert not state.select(old) and resolve_frequency(state, 0) is None
    state.select(state.snapshot.roots[0].children[0].children[0].children[0].id)
    evidence = resolve_frequency(state, 0)
    assert evidence.decoder is None and evidence.hypothesis is None
    assert profile_markers(evidence, 'x', 4) == ()


def test_plot_preparation_handles_constant_nonfinite_and_large_arrays():
    from recognition_lab.frequency_panel import EvidencePlot
    plot = SimpleNamespace(update=lambda: None)
    constant = np.array([2., np.nan, 2., np.inf])
    EvidencePlot.set_data(plot, None, constant, 'position', 'signal', values=constant)
    assert plot.points == ((.5, 2., False), (2.5, 2., False))
    assert plot.x_range == (.5, 3.5) and plot.y_range[0] < 2 < plot.y_range[1]
    large = np.ones(20000)
    EvidencePlot.set_data(plot, None, large, 'position', 'signal', values=large)
    assert plot.stride == 5 and len(plot.points) <= 4096
    EvidencePlot.set_data(plot, None, np.array([np.nan]), 'position', 'signal', values=np.array([np.nan]))
    assert not plot.points and plot.x_range == (0., 1.)
    samples = np.array([1., 2., np.nan, 4., 5.])
    EvidencePlot.set_data(plot, None, samples, 'position', 'signal', values=samples)
    assert plot.segments == (((.5, 1.), (1.5, 2.)), ((3.5, 4.), (4.5, 5.)))
    EvidencePlot.set_data(plot, [48., None, 64.], [.8, .7, float('nan')], 'pitch', 'coherence')
    assert plot.points == ((48., .8, False),) and not plot.segments


def test_panel_refreshes_for_distinct_hypothesis_id_even_when_object_repeats():
    from recognition_lab.frequency_panel import FrequencyPanel
    state, _, obs, *_ = fixture()
    decoded = replace(obs.decoding, hypotheses=(obs.hypothesis, obs.hypothesis))
    state.snapshot.result.grids[0].phase_data['observations'][0] = replace(obs, decoding=decoded)
    state.replace(state.snapshot.result)
    node = state.snapshot.roots[0].children[0].children[0].children[0]
    calls = []
    panel = SimpleNamespace(_snapshot=None, _observation=None, _hypothesis=None, _fallback=None,
                            _identity=None, evidence=None, refresh=lambda: calls.append('refresh'))
    state.select(node.children[0].id); FrequencyPanel.set_selection(panel, state, 0)
    assert panel.evidence.validation == 'Evaluated rejected'
    state.select(node.children[1].id); FrequencyPanel.set_selection(panel, state, 0)
    assert panel.evidence.validation == 'Not evaluated' and len(calls) == 2
    FrequencyPanel.set_selection(panel, state, 0)
    assert len(calls) == 2
    state.replace(None); FrequencyPanel.set_selection(panel, state, 0)
    assert panel.evidence is panel._snapshot is panel._observation is panel._hypothesis is None
