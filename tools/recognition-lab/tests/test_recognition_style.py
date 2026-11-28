import math

import pytest

from recognition_lab.rendering import DESKTOP_STYLE, RecognitionStyle, draw
from recognition_lab.results import Detection, RecognitionResult
from recognition_lab.viewport import physical_scale


@pytest.mark.parametrize(('logical', 'dpr', 'width'), [
    (1, 1, 1), (.5, 2, 1), (2, 2, .25), (1.5, 2, 1/3),
])
def test_screen_width_tracks_physical_magnification(logical, dpr, width):
    style = RecognitionStyle()
    assert style.dimension(style.width, physical_scale(logical, dpr)) == pytest.approx(width)
    assert RecognitionStyle(width=.5).dimension(.5, physical_scale(logical, dpr)) == pytest.approx(width / 2)


def test_desktop_and_image_space_are_independent():
    assert RecognitionStyle().mode == 'screen'
    assert DESKTOP_STYLE == RecognitionStyle('image', 2, 1)
    assert DESKTOP_STYLE.dimension(DESKTOP_STYLE.width, 64) == 2
    assert RecognitionStyle('image', 3.5).dimension(3.5, .1) == 3.5
    assert DESKTOP_STYLE.color(0x5534d17b) == 0x5534d17b


def test_opacity_multiplies_existing_alpha():
    style = RecognitionStyle(opacity=.5)
    assert style.color(0x8834d17b) == 0x4434d17b
    assert RecognitionStyle(opacity=0).color(0xff123456) == 0x00123456
    assert RecognitionStyle().color(0x5534d17b) == 0x5534d17b


@pytest.mark.parametrize('kwargs', [dict(width=0), dict(width=math.inf), dict(opacity=-.1),
                                    dict(opacity=math.nan), dict(mode='unknown')])
def test_invalid_style(kwargs):
    with pytest.raises(ValueError):
        RecognitionStyle(**kwargs)


def test_fractional_detection_geometry_is_not_changed():
    class Canvas:
        def __init__(self): self.rect = None
        def drawRect(self, rect, paint): self.rect = (rect.left(), rect.top(), rect.right(), rect.bottom())
        def drawString(self, *args): pass

    detection = Detection('grid', .9, (10.125, 11.25, 20.5, 21.75))
    result = RecognitionResult(detections=[detection])
    canvas = Canvas()
    draw(canvas, result, {'detections': True}, style=RecognitionStyle(), magnification=16)
    assert canvas.rect == pytest.approx(detection.box)
    assert detection.box == (10.125, 11.25, 20.5, 21.75)


def test_detection_paint_uses_style_without_changing_coordinates():
    class Canvas:
        def __init__(self): self.rectangles = []
        def drawRect(self, rect, paint):
            self.rectangles.append(((rect.left(), rect.top(), rect.right(), rect.bottom()),
                                    paint.getStrokeWidth(), paint.getColor()))
        def drawString(self, *args): pass

    box = (2.125, 3.25, 20.5, 21.75)
    result = RecognitionResult(detections=[Detection('grid', .9, box)])
    for magnification in (8, 16, 64):
        canvas = Canvas()
        draw(canvas, result, {'detections': True}, style=RecognitionStyle(width=1.5, opacity=.5),
             magnification=magnification)
        coords, width, color = canvas.rectangles[0]
        assert coords == pytest.approx(box)
        assert width == pytest.approx(1.5 / magnification)
        assert color == 0x8036cfff
    assert result.detections[0].box == box
