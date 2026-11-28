"""Deterministic accounting checks for local equivalent-boundary evaluation."""
from types import SimpleNamespace
import json

import pytest

from grid_geometry.phase.evaluate import boundary_record, geometry_record, summarize_local_geometry
from grid_geometry.phase.reporting import recognition_counts


def metadata(category='complete', known_box=(0, 0, 50, 60)):
    return {'sample_id': category, 'category': category, 'complete': category == 'complete',
            'geometry': {'origin_x': 0.25, 'origin_y': 0.75, 'pitch_x': 10.5, 'pitch_y': 12.5,
                         'visible_box': [0.25, 0.75, 42.25, 50.75]},
            'board_dimensions': [4, 4], 'known_box': known_box, 'actual_pitch_bin_x': '10-12'}


def hypothesis(left=.25, top=.75, right=42.25, bottom=50.75, px=10.5, py=12.5):
    x = SimpleNamespace(origin=left, end=right, pitch=px, count=4, visible_start=left, visible_end=right)
    y = SimpleNamespace(origin=top, end=bottom, pitch=py, count=4, visible_start=top, visible_end=bottom)
    return SimpleNamespace(x=x, y=y, columns=4, rows=4)


def record(meta, h, accepted):
    result = SimpleNamespace(status='fitted' if h else 'insufficient', hypotheses=(h,) if h else (), reasons=())
    validation = SimpleNamespace(accepted=False, local_accepted=accepted, local_reasons=(), reasons=()) if h else None
    return geometry_record(meta, result, validation)


def test_fractional_anisotropic_recovery_and_adjacent_line():
    m = metadata()
    good = record(m, hypothesis(), True)
    assert good['local_correct'] and good['local_geometry_correct'] and good['accepted_local_geometry_correct']
    bad = record(m, hypothesis(left=10.75), True)
    assert bad['local_correct'] and not bad['local_geometry_correct']
    side = bad['equivalent_boundaries']['sides']['left']
    assert side['line_correct'] is False and side['error_px'] == 10.5 and not side['correct']


def test_correct_line_with_excessive_localization_and_large_tolerance_cannot_accept_adjacent():
    s = boundary_record(metadata(), hypothesis(left=2.25))['sides']['left']
    assert s['line_correct'] and s['error_px'] == 2 and not s['correct']
    # Even at pitch below twice the localization tolerance, the adjacent line fails.
    m = metadata(); m['geometry'].update(pitch_x=1., visible_box=[.25, .75, 4.25, 50.75])
    s = boundary_record(m, hypothesis(left=1.25, right=4.25, px=1))['sides']['left']
    assert not s['line_correct'] and not s['correct']


@pytest.mark.parametrize('left', [10.75, 21.25])
def test_midpoint_alternatives(left):
    m = metadata('interior', (16, 7, 37, 43))
    result = boundary_record(m, hypothesis(left=left, top=13.25, right=31.75, bottom=38.25))
    assert result['sides']['left']['targets'] == [10.75, 21.25]
    assert result['sides']['left']['correct'] and result['valid_box_correct']


def test_individually_correct_but_invalid_combination():
    m = metadata('interior', (5.5, 7, 5.6, 43))
    result = boundary_record(m, hypothesis(left=.25, right=.25, top=13.25, bottom=38.25))
    assert result['sides']['left']['correct'] and result['sides']['right']['correct']
    assert not result['valid_box_correct']


def test_outside_image_and_crop_categories():
    m = metadata('side', (0, 0, 30, 45))
    m['geometry']['origin_x'] = -10.25
    m['geometry']['visible_box'] = [-10.25, .75, 31.75, 50.75]
    s = boundary_record(m, hypothesis(left=.25, right=31.75, top=.75, bottom=50.75))['sides']
    assert s['left']['targets'] == [.25]
    # A nearest boundary can lie outside the known image.
    m['known_box'] = [-6, 0, 30, 45]
    assert boundary_record(m, None)['sides']['left']['targets'] == [-10.25]
    for category in ('complete', 'interior', 'side', 'corner'):
        assert record(metadata(category), hypothesis(), True)['local_geometry_correct']


def test_outcome_counts_and_rejected_hypotheses():
    good = record(metadata(), hypothesis(), True)
    rejected = record(metadata('corner'), hypothesis(), False)
    wrong = record(metadata('side'), hypothesis(left=10.75), True)
    missing = record(metadata('interior'), None, False)
    negmeta = {'sample_id': 'n', 'category': 'hard_negative', 'geometry': None}
    negative = record(negmeta, hypothesis(), True)
    neg_rejected = record(dict(negmeta, sample_id='n2', category='negative'), None, False)
    records = [good, rejected, wrong, missing, negative, neg_rejected]
    summary = summarize_local_geometry(records)
    assert summary['lattice']['correct_accepted'] == 2
    assert summary['geometry']['correct_hypotheses'] == 2
    assert (summary['geometry']['tp'], summary['geometry']['fp'], summary['geometry']['fn'], summary['geometry']['tn']) == (1, 2, 3, 1)
    raw, accepted = summary['hypothesis_boundaries'], summary['accepted_boundaries']
    assert (raw['expected'], raw['correct'], raw['incorrect'], raw['missing_or_rejected']) == (16, 11, 1, 4)
    assert (accepted['expected'], accepted['correct'], accepted['incorrect'], accepted['missing_or_rejected']) == (16, 7, 1, 8)
    assert accepted['precision'] == 7 / 8 and accepted['recall'] == 7 / 16
    assert sum(summary['by_direction'][s]['accepted']['correct'] for s in ('left', 'top', 'right', 'bottom')) == accepted['correct']
    assert summary['negative_by_category']['hard_negative']['accepted'] == 1
    assert 'equivalent_boundaries' not in negative
    assert rejected['local_geometry_correct'] and not rejected['accepted_local_geometry_correct']


def test_empty_denominators_and_legacy_reporting():
    s = summarize_local_geometry([])
    assert s['geometry']['precision'] is None and s['geometry']['recall'] is None
    assert s['accepted_boundaries']['precision'] is None and s['accepted_boundaries']['line_selection_rate'] is None
    old = {'positive': True, 'complete': False, 'local_accepted': False, 'local_correct': None,
           'global': {'accepted': False, 'correct': None}}
    assert 'Local geometry precision' not in recognition_counts([old])
    with pytest.raises(ValueError, match='equivalent-boundary'):
        summarize_local_geometry([old])


def test_record_and_summary_are_json_serializable():
    records = [record(metadata(), hypothesis(), True), record(metadata('interior'), None, False)]
    encoded = json.dumps({'records': records, 'summary': summarize_local_geometry(records)}, allow_nan=False)
    restored = json.loads(encoded)
    assert restored['summary']['accepted_boundaries']['expected'] == 8
    assert restored['records'][0]['equivalent_boundaries']['sides']['left']['targets'] == [.25]
