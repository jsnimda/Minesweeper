from types import SimpleNamespace

import numpy as np
import pytest
import skia

from recognition_lab.rendering import RecognitionStyle, draw
from recognition_lab.results import Crop, GeometryResult, RecognitionResult


def geometry(mode, accepted, offset=0):
    edges = (np.array([10., 20., 30.]) + offset, np.array([10., 20., 30.]), 0.)
    return GeometryResult(0, Crop((0, 0, 90, 50), False), None, None, None,
                          SimpleNamespace(accepted=accepted, status='accepted' if accepted else 'rejected'),
                          [edges], diagnostics={'mode': mode})


@pytest.mark.parametrize('modes', [[('whole', False)], [('tiled', False)],
                                   [('whole', True)], [('tiled', False), ('whole', True)]])
def test_candidate_dash_is_isolated_on_real_skia_canvas(modes):
    surface = skia.Surface(180, 100)

    class Canvas:
        def __init__(self, target): self.target = target; self.effects = []; self.rect_effects = []
        def drawRect(self, rect, paint):
            if paint.getStyle() == skia.Paint.kStroke_Style:
                self.rect_effects.append(paint.getPathEffect() is not None)
            self.target.drawRect(rect, paint)
        def drawLine(self, *args):
            self.effects.append(args[-1].getPathEffect() is not None)
            self.target.drawLine(*args)
        def __getattr__(self, name): return getattr(self.target, name)

    canvas = Canvas(surface.getCanvas())
    grids = [geometry(mode, accepted, i*50) for i, (mode, accepted) in enumerate(modes)]
    draw(canvas, RecognitionResult(grids=grids), {'boundaries': True, 'centers': True, 'crops': True},
         style=RecognitionStyle(), magnification=1., dpr=1.)
    assert canvas.effects == [False for _, accepted in modes for effect in range(6)]
    assert canvas.rect_effects == [effect for _, accepted in modes for effect in (False, not accepted)]


def test_rejected_candidate_does_not_dash_following_diagnostic_lattice():
    from grid_geometry.preprocessing import ImageTransform
    surface = skia.Surface(180, 100)

    class Canvas:
        def __init__(self, target): self.target = target; self.effects = []
        def drawLine(self, *args):
            self.effects.append(args[-1].getPathEffect() is not None)
            self.target.drawLine(*args)
        def __getattr__(self, name): return getattr(self.target, name)

    lattice = SimpleNamespace(origin_x=0., origin_y=0., pitch_x=10., pitch_y=10.)
    later = GeometryResult(1, Crop((50, 0, 70, 20), False), None,
                           ImageTransform((20, 20), (20, 20), (20, 20), (0, 0)), None,
                           SimpleNamespace(accepted=False, status='rejected'), [],
                           phase_data={'global': SimpleNamespace(lattice=lattice), 'observations': []})
    canvas = Canvas(surface.getCanvas())
    draw(canvas, RecognitionResult(grids=[geometry('tiled', False), later]), {'boundaries': True},
         style=RecognitionStyle(), magnification=1., dpr=1.)
    assert canvas.effects[:6] == [False]*6
    assert canvas.effects[6:] == [False]*6
