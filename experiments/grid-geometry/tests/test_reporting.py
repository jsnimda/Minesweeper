import torch

from grid_geometry.phase.metrics import TargetMetrics
from grid_geometry.phase.reporting import fraction, occupancy_matrix, outcome_matrix, recognition_counts


def record(positive, complete, local, local_correct, global_accepted, global_correct, global_local):
    return {'positive': positive, 'complete': complete, 'local_accepted': local,
            'local_correct': local_correct, 'global': {'accepted': global_accepted,
            'correct': global_correct, 'local_accepted': global_local}}


def test_recognition_correctness_and_outcome_accounting():
    records = [record(True, True, True, True, True, True, True),
               record(True, True, True, False, True, False, True),
               record(True, True, False, True, False, True, True),
               record(True, False, True, True, True, None, True),
               record(True, False, False, None, False, None, False),
               record(False, None, True, None, True, None, True),
               record(False, None, False, None, False, None, False)]
    counts = recognition_counts(records)
    assert counts['Local geometry precision'] == (2, 4)
    assert counts['Local geometry recall'] == (2, 5)
    assert counts['Complete board precision'] == (1, 4)
    assert counts['Complete board recall'] == (1, 3)
    assert counts['Partial board false acceptance rate'] == (1, 2)
    assert counts['Negative false acceptance rate'] == (1, 2)
    matrix = outcome_matrix(records)
    assert matrix == [[2, 1, 0], [1, 0, 1], [1, 0, 1]]
    assert sum(map(sum, matrix)) == len(records)


def test_empty_denominators():
    assert all(pair == (0, 0) for pair in recognition_counts([]).values())
    assert fraction((0, 0)) == 'Unavailable (0/0)'
    assert outcome_matrix([]) == [[0, 0, 0]] * 3


def test_occupancy_confusion_uses_mask_and_inclusive_threshold():
    logits = torch.zeros(1, 5, 1, 5)
    logits[0, 4, 0] = torch.tensor([0., 1., -1., -1., 1.])
    target = torch.zeros_like(logits)
    target[0, 4, 0] = torch.tensor([.5, 0., 1., 0., 0.])
    mask = torch.tensor([[[True, True, True, True, False]]])
    metrics = TargetMetrics()
    metrics.update(logits, target, torch.zeros_like(mask), mask, torch.tensor([True]))
    result = metrics.compute()
    assert occupancy_matrix({'target_metrics': result}) == [[1, 1], [1, 1]]
    assert result['all']['occupancy_pixels'] == 4
    assert result['all']['iou'] == 1 / 3
