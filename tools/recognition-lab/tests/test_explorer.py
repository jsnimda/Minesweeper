from types import SimpleNamespace

from recognition_lab.explorer import explorer_items, viewer_geometry_filter
from recognition_lab.recognition_state import RecognitionSelection
from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult


def detection(name, confidence=.875):
    return Detection(name, confidence, (1.25, 2.5, 30.75, 40.125))


def geometry(index, validation=None):
    return GeometryResult(index, Crop((1, 2, 30, 40), False), None, None,
                          SimpleNamespace(status='fitted'), validation, [])


def validation(accepted, lattice=None, extent=None):
    return SimpleNamespace(accepted=accepted, lattice=lattice, geometry=extent)


def flatten(items):
    for item in items:
        yield item
        yield from flatten(item.children)


def test_empty_and_repeated_detection_presentation():
    state = RecognitionSelection()
    assert explorer_items(None) == ()
    state.replace(RecognitionResult())
    assert explorer_items(state.snapshot) == ()
    state.replace(RecognitionResult([detection('mine_counter', .9), detection('grid'),
                                     detection('mine_counter', .625), detection('grid')]))
    items = explorer_items(state.snapshot)
    assert [item.label for item in items] == ['Mine Counter 1', 'Grid 1', 'Mine Counter 2', 'Grid 2']
    assert [item.summary for item in items] == ['Confidence 0.900', 'Confidence 0.875',
                                                'Confidence 0.625', 'Confidence 0.875']
    assert [item.identity for item in items] == [node.id for node in state.snapshot.roots]
    assert all(state.snapshot.resolve(item.identity) is data
               for item, data in zip(items, state.snapshot.result.detections))


def test_missing_global_decoder_status_is_not_invented():
    grid = geometry(0, validation(False, object()))
    grid.decoding = None
    state = RecognitionSelection(); state.replace(RecognitionResult([detection('grid')], [grid]))
    summary = explorer_items(state.snapshot)[0].children[0].summary
    assert summary == 'Rejected · extent unavailable'


def test_geometry_status_children_and_unassociated_result():
    lattice = object()
    extent = SimpleNamespace(rows=16, columns=30)
    grids = [geometry(0, validation(True, lattice, extent)),
             geometry(1, validation(False, lattice)), geometry(8)]
    state = RecognitionSelection()
    state.replace(RecognitionResult([detection('grid'), detection('grid')], grids))
    items = explorer_items(state.snapshot)
    accepted, rejected = items[0].children[0], items[1].children[0]
    assert accepted.summary == 'Accepted · Decoder: fitted'
    assert [(item.label, item.summary) for item in accepted.children] == [
        ('Fitted lattice', 'Board extent 16 × 30'), ('Validation', 'Accepted')]
    assert rejected.summary == 'Rejected · extent unavailable · Decoder: fitted'
    assert rejected.children[0].summary == 'Board extent unavailable'
    assert rejected.children[1].summary == 'Rejected'
    orphan = items[2]
    assert orphan.label == 'Geometry result (unassociated)'
    assert orphan.summary == 'Incomplete · validation unavailable · Decoder: fitted' and not orphan.children
    for item in flatten(items):
        assert state.snapshot.node(item.identity) is not None
        assert state.snapshot.resolve(item.identity) is not None


def test_selection_identity_on_rebuild_and_replacement():
    first, second = detection('grid'), detection('grid')
    state = RecognitionSelection()
    state.replace(RecognitionResult([first, second], [geometry(1, validation(False))]))
    old_items = explorer_items(state.snapshot)
    child = old_items[1].children[0].children[0]
    assert state.select(child.identity)
    assert next(item for item in flatten(explorer_items(state.snapshot)) if item.identity == state.selected) == child
    stale = state.selected
    state.replace(RecognitionResult([first, second], [geometry(1, validation(True))]), geometry_update=True)
    new_items = tuple(flatten(explorer_items(state.snapshot)))
    assert all(item.identity != stale for item in new_items)
    assert next(item for item in new_items if item.identity == state.selected).label == 'Grid 2'
    state.replace(RecognitionResult([detection('timer')]), fallback_to_first_grid=True)
    assert state.selected is None
    assert all(item.identity != stale for item in flatten(explorer_items(state.snapshot)))


def test_non_grid_and_orphan_viewer_filters():
    grid, orphan = geometry(1), geometry(9)
    state = RecognitionSelection()
    state.replace(RecognitionResult([detection('timer'), detection('grid')], [grid, orphan]))
    assert viewer_geometry_filter(state) == (None, None)
    state.select(state.snapshot.roots[0].id)
    assert state.grid() is None and viewer_geometry_filter(state) == (None, None)
    state.select(state.snapshot.roots[1].id)
    assert state.grid() is grid and viewer_geometry_filter(state) == (1, None)
    state.select(state.snapshot.roots[1].children[0].id)
    assert state.grid() is grid and viewer_geometry_filter(state) == (1, None)
    state.select(state.snapshot.roots[2].id)
    index, exact_grid = viewer_geometry_filter(state)
    assert index is None and exact_grid is orphan and state.grid() is orphan


def test_renderer_filters_exact_unassociated_geometry():
    from recognition_lab.rendering import draw

    class Canvas:
        def __init__(self): self.rectangles = []
        def drawRect(self, rect, paint):
            self.rectangles.append((rect.left(), rect.top(), rect.right(), rect.bottom()))
        def drawString(self, *args): pass

    first, orphan = geometry(0), geometry(9)
    orphan.crop = Crop((40, 50, 60, 70), False)
    result = RecognitionResult([detection('grid')], [first, orphan])
    canvas = Canvas()
    draw(canvas, result, {'crops': True}, selected_geometry=orphan)
    assert canvas.rectangles == [(40, 50, 60, 70)]
    canvas = Canvas()
    draw(canvas, result, {'crops': True})
    assert len(canvas.rectangles) == 2
