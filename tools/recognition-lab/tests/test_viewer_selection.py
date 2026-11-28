from types import SimpleNamespace
import math

import numpy as np
import pytest

from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult
from recognition_lab.recognition_state import RecognitionSelection
from recognition_lab.viewer_selection import (bracket_dimensions, hit_detection, image_point,
                                                is_drag, selection_region)
from recognition_lab.viewport import ViewportTransform, fit_region, physical_scale, MAX_MAGNIFICATION


def grid(index, crop=(4, 5, 80, 90), hypotheses=None):
    return GeometryResult(index, Crop(crop, False), None, None, SimpleNamespace(status='fitted'),
                          SimpleNamespace(accepted=False, status='rejected', lattice=object()), hypotheses or [])


def state():
    detections = [Detection('game_region', .8, (0, 0, 100, 100)),
                  Detection('grid', .9, (10.25, 10.5, 40.25, 40.5)),
                  Detection('grid', .7, (10.25, 10.5, 40.25, 40.5)),
                  Detection('timer', .5, (20, 20, 25, 25))]
    recognition = RecognitionSelection()
    recognition.replace(RecognitionResult(detections, [grid(1)]))
    return recognition


def test_hit_testing_overlap_boundaries_hidden_and_transform():
    selection = state(); snapshot = selection.snapshot
    transform = ViewportTransform(2.5, -4.25, 8.5)
    def hit(point, visible=True):
        return hit_detection(snapshot, transform, transform.to_viewport(*point), (100, 100), visible)
    assert hit((22, 22)) == snapshot.detection_id(3)
    assert hit((12, 12)) == snapshot.detection_id(1)  # equal area: first detection
    assert hit((10.25, 10.5)) == snapshot.detection_id(1)
    assert hit((1, 1)) == snapshot.detection_id(0)
    assert hit((22, 22), False) is None
    assert hit((100, 50)) is None and hit((-1, 0)) is None
    assert image_point(transform, transform.to_viewport(1.125, 2.25), (100, 100)) == pytest.approx((1.125, 2.25))
    assert hit_detection(None, transform, (1, 1), (100, 100), True) is None


def test_snapshot_identity_replacement_and_bad_boxes():
    selection = state(); old = selection.snapshot.detection_id(1)
    selection.replace(RecognitionResult([Detection('grid', .5, (0, 0, 10, 10)),
                                         Detection('bad', .4, (0, 0, math.nan, 3))]))
    identity = hit_detection(selection.snapshot, ViewportTransform(), (2, 2), (20, 20), True)
    assert identity == selection.snapshot.detection_id(0)
    assert selection.snapshot.resolve(old) is None and identity != old
    assert hit_detection(selection.snapshot, ViewportTransform(), (15, 15), (20, 20), True) is None


def test_selection_regions_and_visibility():
    selection = state(); snapshot = selection.snapshot
    root = snapshot.roots[1]; geometry = root.children[0]; lattice = geometry.children[0]
    assert selection_region(snapshot, root.id) == (10.25, 10.5, 40.25, 40.5)
    assert selection_region(snapshot, root.id, {'detections': False}) is None
    assert selection_region(snapshot, geometry.id) == (4, 5, 80, 90)
    assert selection_region(snapshot, lattice.id, {'crops': True}) == (4, 5, 80, 90)
    assert selection_region(snapshot, geometry.id, {'boundaries': True}) is None
    selection.snapshot.result.grids[0].hypotheses = [(np.array([12.25, 37.5]), np.array([13.75, 39.125]), 0)]
    assert selection_region(snapshot, geometry.id, {'boundaries': True}) == (12.25, 13.75, 37.5, 39.125)
    assert selection_region(snapshot, geometry.id, {'crops': True}) == (4, 5, 80, 90)
    selection.snapshot.result.grids[0].crop = Crop((0, 0, 0, 0), False)
    selection.snapshot.result.grids[0].hypotheses = []
    assert selection_region(snapshot, geometry.id) is None
    assert selection_region(snapshot, None) is None


def test_click_threshold():
    assert not is_drag((0, 0), (3, 2))
    assert is_drag((0, 0), (4, 0))
    assert is_drag((1.5, 1.5), (-1.5, -1.5))


@pytest.mark.parametrize('dpr', [1, 1.5, 2, 3])
def test_brackets_are_logical_pixels(dpr):
    magnification = 2 * dpr
    length, outer, inner = bracket_dimensions(magnification, dpr)
    assert (length * 2, outer * 2, inner * 2) == pytest.approx((10, 3, 1.5))


def test_region_fit_padding_limits_and_invalid_bounds():
    tr = fit_region((10.25, 20.5, 110.25, 70.5), (500, 300), 2, padding=20)
    assert tr.to_viewport(60.25, 45.5) == pytest.approx((250, 150))
    assert tr.to_viewport(10.25, 20.5)[0] >= 20
    assert tr.to_viewport(110.25, 70.5)[0] <= 480
    tiny = fit_region((0, 0, .001, .001), (500, 300), 2)
    assert physical_scale(tiny.scale, 2) == pytest.approx(MAX_MAGNIFICATION)
    for bounds in (None, (1, 2, 1, 4), (0, 0, math.inf, 4)):
        assert fit_region(bounds, (500, 300), 1) is None
    assert fit_region((0, 0, 5, 5), (20, 20), 1) is None


def test_viewer_geometry_is_not_filtered_and_desktop_has_no_brackets():
    from recognition_lab.rendering import draw, draw_viewer_visualization, RecognitionStyle

    class Canvas:
        def __init__(self): self.rectangles = []; self.lines = []
        def drawRect(self, rect, paint): self.rectangles.append(rect)
        def drawLine(self, *args): self.lines.append(args)
        def drawString(self, *args): pass

    first, second = grid(0), grid(1, crop=(90, 95, 140, 150))
    result = RecognitionResult(grids=[first, second])
    canvas = Canvas()
    draw_viewer_visualization(canvas, result, {'crops': True}, 0, first, 'None', 0,
                              RecognitionStyle(), 1, 1, first)
    assert len(canvas.rectangles) == 2
    desktop = Canvas()
    draw(desktop, result, {'crops': True})
    assert len(desktop.rectangles) == 2 and desktop.lines == []


def test_selection_brackets_preserve_bounds_and_opacity():
    from recognition_lab.rendering import draw_selection

    class Canvas:
        def __init__(self): self.lines = []
        def drawLine(self, x1, y1, x2, y2, paint):
            self.lines.append(((x1, y1, x2, y2), paint.getStrokeWidth(), paint.getColor()))

    canvas = Canvas()
    draw_selection(canvas, (1.125, 2.25, 30.5, 40.75), 4, 2, .5)
    assert len(canvas.lines) == 16
    assert canvas.lines[0][0] == pytest.approx((1.125, 2.25, 6.125, 2.25))
    assert canvas.lines[0][1] == pytest.approx(1.5)
    assert canvas.lines[8][1] == pytest.approx(.75)
    assert canvas.lines[0][2] >> 24 == 128
    draw_selection(canvas, (0, 0, 10, 10), 4, 2, 0)
    assert len(canvas.lines) == 16
