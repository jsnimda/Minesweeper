"""Modern and traditional taskbars with bounded logical layouts."""

import skia

from ..scaling import validate_scale
from .geometry import component_geometry
from .icons import draw_app_icon
from .primitives import _edge_panel, _paint, _round_rect, _text
from .system_icons import draw_battery, draw_volume, draw_wifi


ICONS = ("minesweeper", "file_manager", "browser", "terminal")
LABELS = ("Minesweeper", "Files", "Browser", "Terminal")


def _taskbar_layout(width, style):
    """Return tray boundary and bounded launcher/application rectangles."""
    tray_width = min(178 if style == "classic" else 188, width * .45)
    tray_left = max(0, width - tray_width)
    available = max(0, tray_left - 4)
    if style == "modern":
        if available < 24:
            return tray_left, None, []
        count = min(len(ICONS), max(0, int((available - 24) // 40)))
        group_width = 24 + count * 40
        left = max(0, min((width - group_width) / 2, available - group_width))
        return tray_left, (left, left + 24), [
            (left + 36 + 40 * i, left + 64 + 40 * i) for i in range(count)
        ]

    launcher_width = min(82, available)
    launcher = (0, launcher_width) if launcher_width >= 24 else None
    start = launcher_width + 4 if launcher else 0
    room = max(0, available - start)
    count = min(len(ICONS), max(0, int((room + 4) // 84)))
    slot_width = min(142, (room - 4 * (count - 1)) / count) if count else 0
    return tray_left, launcher, [
        (start + i * (slot_width + 4), start + i * (slot_width + 4) + slot_width)
        for i in range(count)
    ]


def _draw_system_tray(canvas, width, y, height, tray_left, desktop, colors):
    tray_width = max(0, width - tray_left)
    if tray_width <= 0:
        return
    canvas.save()
    canvas.clipRect(skia.Rect.MakeXYWH(tray_left, y, tray_width, height))
    if desktop['style'] == 'classic':
        canvas.drawLine(tray_left + 1, y + 3, tray_left + 1, y + height - 3,
                        _paint(colors['subtle']))
    color = colors['text']
    center_y = y + height / 2
    clock = desktop.get('clock_time', '12:34')
    clock_left = max(tray_left, width - 78)
    canvas.save()
    canvas.clipRect(skia.Rect.MakeXYWH(clock_left, y, max(0, width - clock_left), height))
    if desktop['style'] == 'modern':
        font = skia.Font(None, 12)
        date_font = skia.Font(None, 10)
        date = desktop.get('clock_date', '11/21/2026')
        clock_center = width - 43
        _text(canvas, clock, clock_center - font.measureText(clock) / 2,
              center_y - 2, color, 12)
        _text(canvas, date, clock_center - date_font.measureText(date) / 2,
              center_y + 13, color, 10)
    else:
        _text(canvas, clock, width - 61, center_y + 5, color, 13)
    canvas.restore()
    if tray_width >= 115:
        draw_volume(canvas, width - 96, center_y, color=color, size=20)
    if tray_width >= 150:
        draw_battery(canvas, width - 130, center_y, color=color, size=20, charge=.75)
    if tray_width >= 180:
        draw_wifi(canvas, width - 164, center_y, color=color, size=18)
    canvas.restore()


def _launcher(canvas, x, y, color):
    for row in range(2):
        for col in range(2):
            canvas.drawRect(skia.Rect.MakeXYWH(x + col * 7, y + row * 7, 5, 5), _paint(color))


def _draw_taskbar(surface, size, desktop, colors, scale=1.0):
    scale = validate_scale(scale)
    canvas = surface.getCanvas()
    canvas.save()
    canvas.scale(scale, scale)
    width, screen_height = size[0] / scale, size[1] / scale
    box = component_geometry(size, desktop, scale)['taskbar']
    height = (box[3] - box[1]) / scale
    y = screen_height - height
    canvas.clipRect(skia.Rect.MakeXYWH(0, 0, width, screen_height))
    _edge_panel(surface, (0, y, width, height), desktop, colors)
    style = desktop['style']
    tray_left, launcher, slots = _taskbar_layout(width, style)
    canvas.save()
    canvas.clipRect(skia.Rect.MakeXYWH(0, max(0, y), max(0, tray_left), max(0, min(height, screen_height))))
    if style == 'modern':
        if launcher:
            _launcher(canvas, launcher[0] + 6, y + (height - 12) / 2, colors['accent'])
        for i, (left, right) in enumerate(slots):
            draw_app_icon(canvas, ICONS[i], left, y + (height - 28) / 2, 28)
            if i == 0:
                _round_rect(canvas, left + 7, y + height - 4, right - left - 14,
                            3, 1.5, colors['accent'])
    else:
        dark = desktop['color_scheme'] == 'dark'
        fill = '#4A505A' if dark else '#E2E4E8'
        active = '#58616D' if dark else '#C4CFDA'
        border = '#8D96A3' if dark else '#8993A1'
        if launcher:
            left, right = launcher
            canvas.drawRect(skia.Rect.MakeXYWH(left + 1, y + 3, right - left - 2, height - 6), _paint(fill))
            canvas.drawRect(skia.Rect.MakeXYWH(left + 1, y + 3, right - left - 2, height - 6),
                            skia.Paint(Color=_paint(border).getColor(), Style=skia.Paint.kStroke_Style))
            _launcher(canvas, left + 7, y + (height - 12) / 2, colors['accent'])
            if right - left >= 65:
                _text(canvas, 'Menu', left + 26, y + height / 2 + 5, colors['text'], 13)
        for i, (left, right) in enumerate(slots):
            canvas.drawRect(skia.Rect.MakeXYWH(left, y + 3, right - left, height - 6),
                            _paint(active if i == 0 else fill))
            canvas.drawRect(skia.Rect.MakeXYWH(left, y + 3, right - left, height - 6),
                            skia.Paint(Color=_paint(border).getColor(), Style=skia.Paint.kStroke_Style))
            draw_app_icon(canvas, ICONS[i], left + 5, y + (height - 20) / 2, 20)
            canvas.save()
            canvas.clipRect(skia.Rect.MakeXYWH(left + 29, y + 3, max(0, right - left - 33), height - 6))
            _text(canvas, LABELS[i], left + 29, y + height / 2 + 5, colors['text'], 13)
            canvas.restore()
    canvas.restore()
    _draw_system_tray(canvas, width, y, height, tray_left, desktop, colors)
    canvas.restore()
