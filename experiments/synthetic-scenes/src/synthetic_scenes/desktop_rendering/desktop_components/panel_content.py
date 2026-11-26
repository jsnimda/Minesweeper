"""Small bounded controls shared by the status bar and top panel."""

import skia

from .primitives import _paint, _text
from .system_icons import draw_battery, draw_volume, draw_wifi


def draw_panel_content(canvas, width, height, desktop, colors, *, workspace=False):
    """Draw a launcher and status area without allowing the groups to overlap."""
    status_width = min(160, width * .55)
    left_width = max(0, width - status_width)
    canvas.save()
    canvas.clipRect(skia.Rect.MakeXYWH(0, 0, left_width, height))
    if left_width >= 24:
        for row in range(2):
            for col in range(2):
                canvas.drawRect(skia.Rect.MakeXYWH(10 + col * 7, height / 2 - 6 + row * 7, 5, 5),
                                _paint(colors['accent']))
    if left_width >= 110:
        _text(canvas, 'Applications', 30, height / 2 + 5, colors['text'], 12)
    if workspace and left_width >= 190:
        _text(canvas, 'Workspace 1', 120, height / 2 + 5, colors['text'], 12)
    canvas.restore()

    canvas.save()
    canvas.clipRect(skia.Rect.MakeXYWH(left_width, 0, status_width, height))
    # Each indicator has a reserved slot; small panels omit the leftmost slots.
    if status_width >= 150:
        draw_wifi(canvas, width - 148, height / 2, color=colors['text'], size=18)
    if workspace and status_width >= 125:
        draw_volume(canvas, width - 116, height / 2, color=colors['text'], size=18)
    if status_width >= 95:
        draw_battery(canvas, width - 84, height / 2, color=colors['text'], size=18, charge=.75)
    canvas.save()
    clock_left = max(left_width, width - 66)
    canvas.clipRect(skia.Rect.MakeXYWH(clock_left, 0, max(0, width - clock_left), height))
    _text(canvas, desktop.get('clock_time', '12:34'), width - 60, height / 2 + 5, colors['text'], 13)
    canvas.restore()
    canvas.restore()
