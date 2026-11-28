import pytest

from recognition_lab.rendering import RecognitionStyle, diagnostic_label


@pytest.mark.parametrize('dpr,magnification,expected_size', [(1., 2., 6.), (2., 2., 12.), (3., 6., 6.)])
def test_diagnostic_label_logical_pixel_size_and_overlap(dpr, magnification, expected_size):
    class Canvas:
        def __init__(self): self.rects = []; self.fonts = []
        def drawRect(self, rect, paint): self.rects.append(rect)
        def drawString(self, text, x, y, font, paint): self.fonts.append(font.getSize())
    canvas = Canvas(); occupied = []
    for _ in range(2):
        diagnostic_label(canvas, 'evidence', 10., 20., 0xffffffff, RecognitionStyle(), magnification, dpr, occupied)
    assert canvas.fonts == pytest.approx([expected_size, expected_size])
    assert len(canvas.rects) == 2 and occupied[1][1] > occupied[0][3]
