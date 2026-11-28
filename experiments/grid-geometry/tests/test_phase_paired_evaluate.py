"""Paired evaluation accounting without model inference or datasets."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from grid_geometry.phase.paired_evaluate import crop_bounds, paired_run, result_record, selection, summarize, _jsonable


def test_single_shared_observation_and_counterbalanced_order():
    calls = []
    observations = [object(), object()]
    def observer(rgb, *, predict, tile, overlap):
        predict(None); predict(None)
        return observations, (2, 1)
    class Predict:
        tile = 384
        overlap = 96
        def __call__(self, image): calls.append('inference')
    def recognize(items, shape, settings):
        assert items is observations
        calls.append(settings.strategy)
        return settings.strategy
    _, _, results, timing = paired_run(np.zeros((4, 4, 3)), Predict(), 1, recognize, observer)
    assert calls == ['inference', 'inference', 'd7.6', 'd7.5']
    assert results == {'d7.5': 'd7.5', 'd7.6': 'd7.6'}
    assert timing['order'] == ['d7.6', 'd7.5']


def test_crop_and_global_coordinates():
    metadata = {'known_box': [1.2, 2.2, 98.8, 99.8], 'crop_box': [10, 20, 90, 80],
                'category': 'complete', 'complete': True, 'board_dimensions': [2, 3],
                'geometry': {'origin_x': 20., 'origin_y': 30., 'pitch_x': 10., 'pitch_y': 10.,
                             'visible_box': [20, 30, 50, 50]}}
    crop = crop_bounds(metadata, (100, 100), 4)
    assert crop == (6, 16, 94, 84)
    lattice = SimpleNamespace(origin_x=4., origin_y=4., pitch_x=10., pitch_y=10.)
    geom = SimpleNamespace(x_edges=np.array([14., 24., 34., 44.]), y_edges=np.array([14., 24., 34.]), rows=2, columns=3)
    result = SimpleNamespace(accepted=True, status='accepted', reasons=(), lattice=lattice, geometry=geom, experimental={})
    record = result_record(result, metadata, crop)
    assert record['correct'] is True and record['candidate']['edges'] == [20, 30, 50, 50]
    metadata['complete'] = False
    assert result_record(result, metadata, crop)['correct'] is None


def record(kind, a, b):
    def result(accepted, correct):
        return {'accepted': accepted, 'correct': correct, 'candidate': {} if correct is not None else None,
                'lattice_available': False, 'lattice': None, 'boundary_errors_px': [0.] * 4 if correct is not None else None,
                'rows_correct': correct, 'columns_correct': correct, 'reasons': ['reason'] if not accepted else []}
    return {'sample_id': kind, 'kind': kind, 'results': {'d7.5': result(*a), 'd7.6': result(*b)},
            'timing': {k: 1. for k in ('shared_ms','inference_ms','local_and_placement_ms','d7.5_ms','d7.6_ms','overhead_ms')}}


def test_summary_denominators_groups_and_missing():
    samples = [record('complete', (False, True), (True, True)),
               record('negative', (False, None), (True, None)),
               record('incomplete', (False, None), (False, None)),
               record('complete', (True, False), (False, None))]
    summary = summarize(samples)
    new = summary['strategies']['d7.6']
    assert new['recall'] == .5 and new['precision'] == .5
    assert new['precision_denominator'] == 2 and new['candidate_missing'] == 1
    assert new['negative_false_acceptance_rate'] == 1
    assert len(summary['paired_groups']['d7.6_only']) == 2
    assert summary['changes']['new_correct_acceptances'] == 1
    assert summary['changes']['new_incorrect_acceptances'] == 1
    assert summary['changes']['corrected_incorrect_acceptances'] == 1
    assert summary['strategies']['d7.5']['reasons']['reason'] == 3
    empty = summarize([])['strategies']['d7.5']
    assert empty['recall'] is None and empty['precision'] is None
    json.dumps(_jsonable(summary), allow_nan=False)


def test_selection_is_deterministic_and_validated():
    assert selection(20, 5, 7) == selection(20, 5, 7)
    assert len(selection(4, 20, 9)) == 4
    with pytest.raises(ValueError): selection(3, 0, 1)


def test_acceptance_and_candidate_transitions_are_independent():
    cases = [
        record('complete', (True, True), (True, False)),
        record('complete', (True, False), (True, True)),
        record('complete', (False, False), (True, False)),
        record('complete', (True, False), (False, False)),
        record('complete', (True, True), (False, True)),
        record('complete', (False, True), (True, True)),
        record('complete', (False, None), (True, True)),
        record('complete', (True, True), (False, None)),
        record('unassessable_multiple', (True, None), (True, None)),
    ]
    summary = summarize(cases)
    transitions = summary['acceptance_transitions']
    for key in ('correct_to_incorrect', 'incorrect_to_correct', 'rejected_to_incorrect',
                'incorrect_to_rejected', 'correct_to_rejected', 'rejected_to_correct'):
        assert transitions[key] >= 1
    assert transitions['unassessable_to_unassessable'] == 1
    assert len(summary['paired_groups']['both_accepted']) == 3
    assert summary['changes']['new_incorrect_acceptances'] == 2
    assert summary['changes']['corrected_incorrect_acceptances'] == 2
    assert summary['changes']['new_correct_acceptances'] == 3
    assert summary['changes']['lost_correct_acceptances'] == 3
    candidates = summary['candidate_transitions']
    assert candidates['missing_to_correct'] == 1
    assert candidates['correct_to_missing'] == 1
    assert candidates['incorrect_to_correct'] == 1
    assert candidates['correct_to_incorrect'] == 1
    assert summary['changes']['candidate_improvements'] == 2
    assert summary['changes']['candidate_regressions'] == 2
