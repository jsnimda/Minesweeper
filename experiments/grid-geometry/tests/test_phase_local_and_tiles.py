"""Deterministic geometry fixtures; no model or screenshot required."""
from dataclasses import replace

import numpy as np
import pytest

from grid_geometry.phase.decoder import AxisExtent, FrequencyCandidate, GridHypothesis
from grid_geometry.dataset.targets import PhaseGeometry, phase_targets
from grid_geometry.phase.tiles import TileObservation, consensus, periodic_distance, tile_starts
from grid_geometry.phase.validation import validate_phase


def hypothesis(x, y, pitch=12., columns=8, rows=8, censored=False):
    a = AxisExtent(x, pitch, columns, None if censored else x,
                   None if censored else x + columns*pitch, censored, censored)
    b = AxisExtent(y, pitch, rows, None if censored else y,
                   None if censored else y + rows*pitch, censored, censored)
    return GridHypothesis(a, b, FrequencyCandidate(pitch, x % pitch, 1, 1),
                          FrequencyCandidate(pitch, y % pitch, 1, 1), 1, 0, 1)


def test_partial_local_validity_and_complete_rejection():
    fields = phase_targets((96, 96), PhaseGeometry(-12, -12, 12, 12, (-12, -12, 108, 108)))[0]
    checked = validate_phase(fields, hypothesis(-12, -12, columns=10, rows=10, censored=True))
    assert checked.local_accepted and not checked.accepted
    assert checked.observable.sum() == 64
    assert 'left termination unobservable' in checked.reasons


def test_complete_requires_observed_terminations():
    h = hypothesis(24, 24)
    fields = phase_targets((144, 144), PhaseGeometry(24, 24, 12, 12, (24, 24, 120, 120)))[0]
    assert validate_phase(fields, h).accepted
    assert not validate_phase(fields, replace(h, x=replace(h.x, count=9, visible_end=132))).accepted
    assert not validate_phase(np.zeros_like(fields), h).local_accepted


def test_tile_offsets_periodic_alignment_and_conflicts():
    a = TileObservation((0, 0, 96, 96), hypothesis(4, 8), True)
    b = TileObservation((48, 24, 144, 120), hypothesis(4, 8), True)
    bad = TileObservation((0, 0, 96, 96), hypothesis(9, 8), True)
    result = consensus([a, b, bad])
    assert result.inliers == (0, 1)
    assert result.rejected == (2,)
    assert result.pitch_x == pytest.approx(12)
    assert periodic_distance(result.origin_x, 4, 12) == pytest.approx(0)
    assert consensus([replace(a, local_valid=False)]) is None
    assert tile_starts(1000, 384, 96)[-1] == 616
