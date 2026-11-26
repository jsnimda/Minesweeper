"""Deterministic phase-field decoder and fixed-candidate validation checks."""
from dataclasses import replace

import numpy as np
import pytest

from grid_geometry.phase.decoder import AxisExtent, DecoderSettings, FrequencyCandidate, GridHypothesis, decode_phase
from grid_geometry.phase.targets import PhaseGeometry, phase_targets
from grid_geometry.phase.validation import validate_phase


def fields_for(rows, columns, px, py, left=40.25, top=38.75, margin=40):
    width = int(np.ceil(left + columns * px + margin))
    height = int(np.ceil(top + rows * py + margin))
    geometry = PhaseGeometry(left, top, px, py, (left, top, left + columns * px, top + rows * py))
    return phase_targets((width, height), geometry)[0]


def fixed(rows, columns, px, py, left=40.25, top=38.75):
    x = AxisExtent(left, px, columns, left, left + columns * px, False, False)
    y = AxisExtent(top, py, rows, top, top + rows * py, False, False)
    cx = FrequencyCandidate(px, left % px, 1., 1.)
    cy = FrequencyCandidate(py, top % py, 1., 1.)
    return GridHypothesis(x, y, cx, cy, 1., 0., 1.)


@pytest.mark.parametrize('rows,columns,px,py', [(9, 9, 12.3, 12.3), (16, 16, 17.2, 16.8),
                                               (16, 30, 9.5, 14.2), (9, 9, 32., 32.)])
def test_ideal_recovery(rows, columns, px, py):
    fields = fields_for(rows, columns, px, py)
    result = decode_phase(fields)
    assert result.hypotheses
    candidate = result.hypotheses[0]
    assert (candidate.rows, candidate.columns) == (rows, columns)
    assert candidate.x.pitch == pytest.approx(px, rel=.01)
    assert candidate.y.pitch == pytest.approx(py, rel=.01)
    assert candidate.x.origin == pytest.approx(40.25, abs=.5)
    assert candidate.y.origin == pytest.approx(38.75, abs=.5)
    assert validate_phase(fields, candidate, decoder_status=result.status).accepted


def test_large_extra_column_is_rejected():
    fields = fields_for(60, 100, 20., 20.)
    result = decode_phase(fields)
    assert result.hypotheses
    assert (result.hypotheses[0].rows, result.hypotheses[0].columns) == (60, 100)
    expected = fixed(60, 100, 20., 20.)
    assert validate_phase(fields, expected).accepted
    wrong = replace(expected, x=replace(expected.x, count=101, visible_end=expected.x.visible_end + 20))
    checked = validate_phase(fields, wrong)
    assert not checked.accepted
    assert 'unsupported column' in checked.reasons


def test_negative_weak_and_cropped():
    assert decode_phase(np.zeros((5, 100, 100), np.float32)).status == 'insufficient_evidence'
    fields = fields_for(9, 9, 12., 12.)
    weak = fields.copy(); weak[:4] *= .02
    assert decode_phase(weak).status == 'insufficient_evidence'
    cropped = fields[:, :, 50:]
    result = decode_phase(cropped)
    assert result.hypotheses and result.status == 'fitted'
    checked = validate_phase(cropped, result.hypotheses[0], decoder_status=result.status)
    assert checked.local_accepted
    assert not checked.accepted
    assert 'left termination unobservable' in checked.reasons


def test_local_damage_and_continuation():
    fields = fields_for(9, 9, 12., 12.)
    candidate = fixed(9, 9, 12., 12.)
    damaged = fields.copy(); damaged[4, 38:150, 88:100] = 0
    result = validate_phase(damaged, candidate)
    assert not result.accepted and ('unsupported column' in result.reasons or 'insufficient local cell support' in result.reasons)
    distorted = fields.copy(); distorted[0:2, 60:85, 60:85] *= -1
    assert not validate_phase(distorted, candidate).accepted
    continued = fields.copy(); continued[4, 39:147, 150:165] = 1
    assert not validate_phase(continued, candidate).accepted


def test_ambiguous_status_and_anisotropic_prior():
    fields = fields_for(9, 9, 18., 12.)
    settings = DecoderSettings(scale_x=1.5, scale_y=1.)
    result = decode_phase(fields, settings)
    assert result.hypotheses
    assert result.hypotheses[0].square_preference > .95
    assert not validate_phase(fields, result.hypotheses[0], decoder_status='ambiguous').accepted


def test_noise_and_phase_wrap():
    fields = fields_for(9, 9, 13.7, 13.7, left=53.7, top=52.2)
    rng = np.random.default_rng(9)
    fields[:4] = np.clip(fields[:4] + rng.normal(0, .02, fields[:4].shape), -1, 1)
    result = decode_phase(fields)
    assert result.hypotheses and result.hypotheses[0].columns == 9


def test_harmonic_ambiguity_and_missing_row():
    fields = fields_for(9, 9, 16., 16.)
    candidate = fixed(9, 9, 16., 16.)
    # Twice the actual spatial frequency makes the original candidate fail;
    # the validator does not repair the supplied geometry.
    harmonic = fields.copy()
    x = np.arange(fields.shape[2], dtype=np.float32) + .5
    angle = 4 * np.pi * (x - 40.25) / 16.
    harmonic[0] = np.cos(angle)[None, :]
    harmonic[1] = np.sin(angle)[None, :]
    assert not validate_phase(harmonic, candidate).accepted
    missing = fields.copy()
    missing[4, 38:55, 40:185] = 0
    checked = validate_phase(missing, candidate)
    assert not checked.accepted and 'unsupported row' in checked.reasons


def test_local_phase_magnitude_and_unknown_termination():
    fields = fields_for(9, 9, 12., 12.)
    candidate = fixed(9, 9, 12., 12.)
    weak = fields.copy()
    weak[:4, 60:85, 60:85] *= .01
    assert not validate_phase(weak, candidate).accepted
    censored = replace(candidate, x=replace(candidate.x, start_censored=True, visible_start=None))
    checked = validate_phase(fields, censored)
    assert not checked.accepted and 'left termination unobservable' in checked.reasons


def test_competing_harmonics_are_retained_or_rejected():
    fields = fields_for(16, 16, 16., 16.)
    width, height = fields.shape[2], fields.shape[1]
    x = np.arange(width, dtype=np.float32) + .5
    y = np.arange(height, dtype=np.float32) + .5
    theta_x = 2 * np.pi * (x - 40.25) / 16.
    theta_y = 2 * np.pi * (y - 38.75) / 16.
    fields[0] = (.5 * (np.cos(theta_x) + np.cos(2 * theta_x)))[None, :]
    fields[1] = (.5 * (np.sin(theta_x) + np.sin(2 * theta_x)))[None, :]
    fields[2] = (.5 * (np.cos(theta_y) + np.cos(2 * theta_y)))[:, None]
    fields[3] = (.5 * (np.sin(theta_y) + np.sin(2 * theta_y)))[:, None]
    result = decode_phase(fields, DecoderSettings(ambiguity_margin=.1))
    assert result.status in ('ambiguous', 'insufficient_evidence')
    if result.hypotheses:
        assert not validate_phase(fields, result.hypotheses[0], decoder_status=result.status).accepted
