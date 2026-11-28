"""Local equivalent extents use inference evidence, not board annotations."""
import numpy as np
import pytest

from grid_geometry.dataset.equivalent_boundaries import equivalent_boundaries
from grid_geometry.dataset.targets import PhaseGeometry, phase_targets
from grid_geometry.phase.decoder import (DecoderSettings, FrequencyCandidate, _axis_extent,
                                         _occupancy_profiles, _profile_extent, decode_phase)
from grid_geometry.phase.validation import validate_phase


@pytest.mark.parametrize('box', [
    (0, 0, 140, 140),       # complete with background
    (11, 11, 109, 109),    # tightly fitted at visible tile edges
    (23, 23, 97, 97),      # interior
    (0, 23, 97, 97),       # side
    (0, 0, 97, 97),        # corner
    (23, 0, 97, 140),      # opposite sides
    (23, 23, 97, 140),     # three sides
    (20, 20, 100, 100),    # lattice-aligned image edges
    (24, 24, 96, 96),      # nearest lines outside
    (26, 26, 94, 94),      # nearest lines inside
])
def test_equivalent_extent_from_ideal_occupancy(box):
    geometry = PhaseGeometry(10, 10, 10, 10, (11, 11, 109, 109))
    fields = phase_targets((140, 140), geometry)[0]
    x0, y0, x1, y1 = box
    known = fields[:, y0:y1, x0:x1]
    profiles = _occupancy_profiles(known[4], .5)
    expected = equivalent_boundaries(geometry, (10, 10), box, midpoint_tolerance=0)
    actual = []
    for profile, offset, length in ((profiles[0], x0, x1-x0), (profiles[1], y0, y1-y0)):
        candidate = FrequencyCandidate(10, (10-offset) % 10, 1, 1)
        extent = _axis_extent(candidate, _profile_extent(profile, .5), length)
        actual.append((extent.origin + offset, extent.end + offset))
    assert actual[0][0] == pytest.approx(expected.left[0])
    assert actual[0][1] == pytest.approx(expected.right[0])
    assert actual[1][0] == pytest.approx(expected.top[0])
    assert actual[1][1] == pytest.approx(expected.bottom[0])


@pytest.mark.parametrize('phase,expected', [(4.9, 4.9), (5., 5.), (5.1, -4.9), (0., 0.)])
def test_censored_nearest_line_and_midpoint_tie(phase, expected):
    candidate = FrequencyCandidate(10., phase, 1., 1.)
    extent = _axis_extent(candidate, (None, None, True, True, False), 80)
    assert extent.origin == pytest.approx(expected)
    assert extent.end > extent.origin


def test_fractional_anisotropic_and_padding_crop():
    geometry = PhaseGeometry(10.25, 12.75, 10.5, 13.25, (-100, -100, 300, 300))
    fields = phase_targets((110, 120), geometry)[0]
    # Unknown padded predictions can contain arbitrary evidence; cropping is
    # the same operation used by Recognition Lab and tile inference.
    padded = np.pad(fields, ((0, 0), (7, 9), (5, 11)), constant_values=1)
    known = padded[:, 7:127, 5:115]
    profiles = _occupancy_profiles(known[4], .5)
    for profile, pitch, origin, length in ((profiles[0], 10.5, 10.25, 110),
                                           (profiles[1], 13.25, 12.75, 120)):
        extent = _axis_extent(FrequencyCandidate(pitch, origin % pitch, 1, 1),
                              _profile_extent(profile, .5), length)
        assert extent.origin == pytest.approx(origin + np.floor(-origin / pitch + .5) * pitch)
        assert extent.end == pytest.approx(origin + np.floor((length-origin) / pitch + .5) * pitch)
        assert extent.start_censored and extent.end_censored


@pytest.mark.parametrize('box', [(0, 0, 140, 140), (23, 23, 97, 97),
                                 (0, 23, 97, 97), (0, 0, 97, 97),
                                 (23, 0, 97, 140), (23, 23, 97, 140)])
def test_decoder_and_local_validation(box):
    geometry = PhaseGeometry(10, 10, 10, 10, (11, 11, 109, 109))
    fields = phase_targets((140, 140), geometry)[0]
    x0, y0, x1, y1 = box
    fields = fields[:, y0:y1, x0:x1]
    result = decode_phase(fields, DecoderSettings(ambiguity_margin=0))
    assert result.hypotheses
    h = result.hypotheses[0]
    expected = equivalent_boundaries(geometry, (10, 10), box, midpoint_tolerance=0)
    for actual, target in ((h.x.origin+x0, expected.left[0]), (h.x.end+x0, expected.right[0]),
                           (h.y.origin+y0, expected.top[0]), (h.y.end+y0, expected.bottom[0])):
        assert actual == pytest.approx(target, abs=.6)
    checked = validate_phase(fields, h, decoder_status=result.status)
    assert checked.local_accepted
    if any((h.x.start_censored, h.x.end_censored, h.y.start_censored, h.y.end_censored)):
        assert not checked.accepted
