from types import SimpleNamespace

import numpy as np

from recognition_lab.recognition_state import RecognitionSelection, ResultId, ResultSnapshot
from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult


def detection(name='grid'):
    return Detection(name, .875, (1.125, 2.25, 30.5, 40.75))


def geometry(index, accepted=None, lattice=None):
    validation = None if accepted is None else SimpleNamespace(accepted=accepted, lattice=lattice)
    return GeometryResult(index, Crop((1, 2, 30, 40), False), np.array([[[.123456789]]]),
                          object(), object(), validation, [])


def test_empty_and_detection_only_results():
    state = RecognitionSelection()
    state.replace(RecognitionResult(), fallback_to_first_grid=True)
    assert state.snapshot.roots == ()
    assert state.selected is None and state.resolve() is None and state.grid() is None
    result = RecognitionResult([detection('timer'), detection(), detection()])
    state.replace(result)
    assert len(state.snapshot.roots) == 3
    assert all(not node.children for node in state.snapshot.roots)
    assert state.snapshot.resolve(state.snapshot.detection_id(0)) is result.detections[0]
    assert state.snapshot.detection_id(1) != state.snapshot.detection_id(2)
    assert state.snapshot.detection_id(1) == state.snapshot.roots[1].id


def test_geometry_association_validation_and_original_data():
    lattice = object()
    grids = [geometry(1, True, lattice), geometry(2, False), geometry(2)]
    result = RecognitionResult([detection('game_region'), detection(), detection()], grids)
    snapshot = ResultSnapshot(result)
    assert not snapshot.roots[0].children
    first = snapshot.roots[1].children[0]
    assert snapshot.resolve(first.id) is grids[0]
    assert [child.id.kind for child in first.children] == ['lattice', 'validation']
    assert snapshot.resolve(first.children[0].id) is lattice
    assert snapshot.resolve(first.children[1].id) is grids[0].validation
    assert snapshot.grid(first.children[1].id) is grids[0]
    assert snapshot.detection_index(first.children[0].id) == 1
    second, incomplete = snapshot.roots[2].children
    assert snapshot.resolve(second.children[0].id).accepted is False
    assert incomplete.children == ()
    assert snapshot.resolve(incomplete.id).validation is None
    assert snapshot.resolve(first.id).heatmaps is grids[0].heatmaps
    assert snapshot.resolve(first.id).heatmaps[0, 0, 0] == .123456789
    assert snapshot.resolve(snapshot.roots[1].id).box == (1.125, 2.25, 30.5, 40.75)


def test_invalid_identity_and_orphan_geometry():
    grid = geometry(9, False)
    snapshot = ResultSnapshot(RecognitionResult([detection('timer')], [grid]))
    orphan = snapshot.roots[1]
    assert snapshot.resolve(orphan.id) is grid
    assert snapshot.detection_index(orphan.id) is None
    assert snapshot.resolve(ResultId(object(), 'geometry', 0)) is None
    assert snapshot.detection_id(8) is None
    state = RecognitionSelection()
    state.replace(snapshot.result)
    assert not state.select(orphan.id)
    assert state.selected is None


def test_selection_replacement_and_geometry_update():
    first, second = detection(), detection()
    state = RecognitionSelection()
    state.replace(RecognitionResult([first, second]), fallback_to_first_grid=True)
    old = state.snapshot.detection_id(1)
    assert state.select(old) and state.resolve() is second
    grid = geometry(1, True, object())
    state.replace(RecognitionResult([first, second], [grid]), geometry_update=True)
    assert state.snapshot.resolve(old) is None
    assert state.detection_index() == 1 and state.resolve() is second and state.grid() is grid
    derived = state.snapshot.roots[1].children[0].children[0].id
    state.select(derived)
    assert state.resolve() is grid.validation.lattice
    state.replace(RecognitionResult([first, second], [geometry(1, False)]), geometry_update=True)
    assert state.resolve() is second and state.snapshot.resolve(derived) is None
    old = state.selected
    state.replace(RecognitionResult([detection(), detection()]), fallback_to_first_grid=True)
    assert state.snapshot.resolve(old) is None and state.detection_index() == 0
    state.replace(None)
    assert state.snapshot is None and state.selected is None
    state.replace(RecognitionResult([first, second]), geometry_update=True)
    assert state.selected is None


def test_unreliable_geometry_update_does_not_transfer_selection():
    original = detection()
    state = RecognitionSelection()
    state.replace(RecognitionResult([original]))
    state.select(state.snapshot.detection_id(0))
    state.replace(RecognitionResult([original, original]), geometry_update=True)
    assert state.selected is None
    state.replace(RecognitionResult([detection()]), geometry_update=True)
    assert state.selected is None
