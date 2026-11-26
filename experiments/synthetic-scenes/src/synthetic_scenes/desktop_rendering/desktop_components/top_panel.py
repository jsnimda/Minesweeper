"""Traditional desktop top panel."""

import skia

from ..scaling import validate_scale
from .geometry import component_geometry
from .panel_content import draw_panel_content
from .primitives import _edge_panel


def _draw_top_panel(surface, size, desktop, colors, scale=1.0):
    scale = validate_scale(scale)
    canvas = surface.getCanvas()
    canvas.save()
    canvas.scale(scale, scale)
    width, screen_height = size[0] / scale, size[1] / scale
    height = component_geometry(size, desktop, scale)['top_panel'][3] / scale
    canvas.clipRect(skia.Rect.MakeXYWH(0, 0, width, screen_height))
    _edge_panel(surface, (0, 0, width, height), desktop, colors)
    draw_panel_content(canvas, width, height, desktop, colors, workspace=True)
    canvas.restore()
