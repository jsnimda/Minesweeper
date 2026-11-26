"""Floating translucent dock and independently visible top status bar."""

import skia

from ..scaling import validate_scale
from .geometry import component_geometry, dock_layout
from .icons import draw_app_icon
from .panel_content import draw_panel_content
from .primitives import _glass_panel, _paint


DOCK_ICONS = ('file_manager', 'browser', 'terminal', 'code_editor', 'settings', 'trash')


def _draw_dock(surface, size, desktop, colors, scale=1.0):
    scale = validate_scale(scale)
    canvas = surface.getCanvas()
    canvas.save()
    canvas.scale(scale, scale)
    geometry = component_geometry(size, desktop, scale)
    width, screen_height = size[0] / scale, size[1] / scale
    canvas.clipRect(skia.Rect.MakeXYWH(0, 0, width, screen_height))

    if 'status_bar' in geometry:
        status_height = geometry['status_bar'][3] / scale
        _glass_panel(surface, (0, 0, width, status_height), radius=0, blur=9,
                     tint=colors['surface'], opacity=205 if desktop['style'] == 'modern' else 145)
        draw_panel_content(canvas, width, status_height, desktop, colors)

    if 'dock' in geometry:
        layout = dock_layout(size, desktop, scale)
        left, top, right, bottom = geometry['dock']
        x, y = left / scale, top / scale
        dock_width, dock_height = (right - left) / scale, (bottom - top) / scale
        ratio = layout['ratio']
        _glass_panel(surface, (x, y, dock_width, dock_height), radius=18 * ratio, blur=16,
                     tint=colors['surface'], opacity=185 if desktop['style'] == 'modern' else 125)
        canvas.save()
        canvas.clipRect(skia.Rect.MakeXYWH(x, y, dock_width, dock_height))
        for i, ((icon_left, icon_top, icon_right, _), kind) in enumerate(zip(layout['icons'], DOCK_ICONS)):
            icon_size = (icon_right - icon_left) / scale
            draw_app_icon(canvas, kind, icon_left / scale, icon_top / scale, icon_size)
            if i == 1:
                canvas.drawCircle((icon_left + icon_right) / (2 * scale),
                                  (bottom / scale) - 7 * ratio, 2.5 * ratio, _paint(colors['accent']))
        canvas.restore()
    canvas.restore()
