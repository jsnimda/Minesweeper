"""Skia drawing shared by the image viewer and desktop overlay."""
import skia
import numpy as np

COLORS = {'accepted': 0xff34d17b, 'rejected': 0xffff5c66, 'ambiguous': 0xffffc34d}


def draw(canvas, result, layers, selected=None):
    if result is None:
        return
    stroke = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2)
    font = skia.Font(None, 14)
    def label(text, x, y, color):
        canvas.drawString(text, x, max(14, y), font, skia.Paint(Color=color, AntiAlias=True))
    if layers.get('detections'):
        for d in result.detections:
            stroke.setColor(0xff36cfff)
            canvas.drawRect(skia.Rect.MakeLTRB(*d.box), stroke)
            label(f'{d.name} {d.confidence:.2f}', d.box[0], d.box[1]-3, 0xff36cfff)
    for grid in result.grids:
        if selected is not None and grid.detection_index != selected:
            continue
        status = 'ambiguous' if grid.decoding.status == 'ambiguous' else grid.validation.status
        color = COLORS.get(status, 0xffffc34d)
        stroke.setColor(color)
        if layers.get('crops'):
            canvas.drawRect(skia.Rect.MakeLTRB(*grid.crop.box), stroke)
        if not grid.hypotheses:
            continue
        xs, ys, _ = grid.hypotheses[0]
        if layers.get('boundaries'):
            for x in xs: canvas.drawLine(float(x), float(ys[0]), float(x), float(ys[-1]), stroke)
            for y in ys: canvas.drawLine(float(xs[0]), float(y), float(xs[-1]), float(y), stroke)
        if layers.get('centers'):
            paint = skia.Paint(Color=color, AntiAlias=True, StrokeWidth=3, StrokeCap=skia.Paint.kRound_Cap)
            cx, cy = (xs[:-1]+xs[1:])/2, (ys[:-1]+ys[1:])/2
            points = [skia.Point(float(x), float(y)) for y in cy for x in cx]
            canvas.drawPoints(skia.Canvas.kPoints_PointMode, points, paint)
        label(f'{len(ys)-1}×{len(xs)-1} {grid.decoding.status} / {grid.validation.status}', float(xs[0]), float(ys[0])-4, color)
