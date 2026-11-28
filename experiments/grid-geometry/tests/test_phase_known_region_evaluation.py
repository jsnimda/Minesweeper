"""Local evaluation uses only known pixels and reports sample coordinates."""
from types import SimpleNamespace

import numpy as np
import pytest

from grid_geometry.dataset.targets import PhaseGeometry, phase_targets
from grid_geometry.phase import evaluate as evaluation
from grid_geometry.phase.decoder import AxisExtent, DecodeResult, FrequencyCandidate, GridHypothesis, decode_phase
from grid_geometry.phase.validation import validate_phase


def test_full_image_matches_direct_recognition():
    geometry = PhaseGeometry(8, 8, 10, 10, (8, 8, 48, 48))
    fields = phase_targets((56, 56), geometry)[0]
    direct = decode_phase(fields)
    actual, validation = evaluation.local_recognition({'known_box': [0, 0, 56, 56]}, fields)
    assert actual.status == direct.status and actual.hypotheses == direct.hypotheses
    if direct.hypotheses:
        expected = validate_phase(fields, direct.hypotheses[0], decoder_status=direct.status)
        assert (validation.accepted, validation.local_accepted, validation.reasons) == (expected.accepted, expected.local_accepted, expected.reasons)


def test_crop_translation_validation_and_accounting(monkeypatch):
    fields = np.full((5, 65, 55), 99., dtype=np.float32)
    fields[:, 24:54, 13:43] = 0
    x = AxisExtent(-2, 10, 3, None, 28.5, True, False)
    y = AxisExtent(-2, 10, 3, 1.5, None, False, True)
    evidence = FrequencyCandidate(10, 8, 1, 1)
    hypothesis = GridHypothesis(x, y, evidence, evidence, 1, 0, 1)
    decoded = DecodeResult('fitted', (hypothesis,), frequency_candidates={'x': (evidence,), 'y': (evidence,)})
    validation = SimpleNamespace(accepted=False, local_accepted=True, local_reasons=(), reasons=())
    calls = []
    def decode(values):
        calls.append(('decode', values.copy()))
        return decoded
    def validate(values, h, *, decoder_status):
        calls.append(('validate', values.copy()))
        assert h is hypothesis and decoder_status == 'fitted'
        return validation
    monkeypatch.setattr(evaluation, 'decode_phase', decode)
    monkeypatch.setattr(evaluation, 'validate_phase', validate)
    metadata = {'sample_id': 'crop', 'category': 'interior', 'complete': False,
                'known_box': [13, 24, 43, 54], 'board_dimensions': [4, 4],
                'geometry': {'origin_x': 11, 'origin_y': 22, 'pitch_x': 10, 'pitch_y': 10,
                             'visible_box': [11, 22, 51, 62]}}
    actual, checked = evaluation.local_recognition(metadata, fields)
    assert checked is validation
    assert [name for name, _ in calls] == ['decode', 'validate']
    assert all(values.shape == (5, 30, 30) and not values.any() for _, values in calls)
    h = actual.hypotheses[0]
    assert (h.x.origin, h.y.origin, h.x.end, h.y.end) == (11, 22, 41, 52)
    assert (h.x.pitch, h.y.pitch) == (10, 10)
    assert (h.x.visible_start, h.x.visible_end, h.y.visible_start, h.y.visible_end) == (None, 41.5, 25.5, None)
    assert (h.x.start_censored, h.y.end_censored)
    assert (h.x_evidence.origin_mod_pitch, h.y_evidence.origin_mod_pitch) == (1, 2)
    assert actual.frequency_candidates['x'][0].origin_mod_pitch == 1
    record = evaluation.geometry_record(metadata, actual, checked)
    assert record['local_correct'] and record['local_geometry_correct']
    assert record['equivalent_boundaries']['sides']['left']['predicted'] == 11
    summary = evaluation.summarize_local_geometry([record])
    assert summary['lattice']['tp'] == summary['geometry']['tp'] == 1
    assert summary['accepted_boundaries']['correct'] == 4


def test_crop_matches_direct_decoder_after_translation():
    geometry = PhaseGeometry(8, 18, 10, 10, (8, 18, 48, 58))
    fields = phase_targets((55, 65), geometry)[0]
    box = [10, 20, 40, 50]
    crop = fields[:, 20:50, 10:40]
    direct = decode_phase(crop)
    actual, checked = evaluation.local_recognition({'known_box': box}, fields)
    assert actual.status == direct.status and len(actual.hypotheses) == len(direct.hypotheses)
    for a, d in zip(actual.hypotheses, direct.hypotheses):
        for axis, offset in (('x', 10), ('y', 20)):
            got, expected = getattr(a, axis), getattr(d, axis)
            assert got.origin == pytest.approx(expected.origin + offset)
            assert got.end == pytest.approx(expected.end + offset)
            assert got.pitch == expected.pitch
    if direct.hypotheses:
        expected = validate_phase(crop, direct.hypotheses[0], decoder_status=direct.status)
        assert (checked.accepted, checked.local_accepted, checked.reasons) == (expected.accepted, expected.local_accepted, expected.reasons)


def test_global_record_still_passes_crop_relative_observation(monkeypatch):
    fields = np.zeros((5, 60, 50), dtype=np.float32)
    fields[:, 24:54, 13:43] = .25
    decoded = DecodeResult('insufficient_evidence', (), reasons=('no evidence',))
    def decode(values):
        assert values.shape == (5, 30, 30) and np.all(values == .25)
        return decoded
    def recognize(observations, shape):
        observation, = observations
        assert shape == (30, 30) and observation.box == (0, 0, 30, 30)
        assert observation.hypothesis is None and not observation.local_valid
        assert observation.reasons == ('no evidence',)
        assert np.array_equal(observation.fields, fields[:, 24:54, 13:43])
        return SimpleNamespace(accepted=False, reasons=('no evidence',), geometry=None)
    monkeypatch.setattr(evaluation, 'decode_phase', decode)
    monkeypatch.setattr(evaluation, 'recognize_global', recognize)
    result = evaluation.global_record({'known_box': [13, 24, 43, 54]}, fields)
    assert result == {'accepted': False, 'local_accepted': False, 'reasons': ['no evidence'],
                      'correct': None, 'known_region': [13, 24, 43, 54]}
