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
        status = 'ambiguous' if grid.decoding.status == 'ambiguous' else (grid.validation.status if grid.validation else 'rejected')
        color = COLORS.get(status, 0xffffc34d)
        stroke.setColor(color)
        if layers.get('crops'):
            canvas.drawRect(skia.Rect.MakeLTRB(*grid.crop.box), stroke)
        if grid.phase_data:
            tr = grid.heatmap_transform
            left, top = grid.crop.box[:2]
            def point(x, y):
                return (left + (x-tr.offset[0])/tr.scale_x, top + (y-tr.offset[1])/tr.scale_y)
            for observation in grid.phase_data['observations']:
                x0, y0, x1, y1 = observation.box
                if layers.get('tiles'):
                    stroke.setColor(0xff36cfff if observation.local_valid else 0xffffc34d)
                    a, b = point(x0, y0), point(x1, y1)
                    canvas.drawRect(skia.Rect.MakeLTRB(*a, *b), stroke)
                if layers.get('terminations') and observation.hypothesis is not None:
                    h = observation.hypothesis
                    marks = [(h.x.visible_start, (y1-y0)/2, x0, y0), (h.x.visible_end, (y1-y0)/2, x0, y0)]
                    for x, y, ox, oy in marks:
                        if x is not None:
                            px, py = point(x+ox, y+oy)
                            canvas.drawCircle(px, py, 3, skia.Paint(Color=0xff36cfff, AntiAlias=True))
                    for y in (h.y.visible_start, h.y.visible_end):
                        if y is not None:
                            px, py = point((x0+x1)/2, y+y0)
                            canvas.drawCircle(px, py, 3, skia.Paint(Color=0xff36cfff, AntiAlias=True))
            stroke.setColor(color)
        if not grid.hypotheses and grid.phase_data and layers.get('boundaries'):
            lattice = grid.phase_data['global'].lattice
            if lattice is not None:
                tr = grid.heatmap_transform
                width, height = tr.resized_size
                xx = lattice.origin_x + np.arange(int(np.ceil(-lattice.origin_x/lattice.pitch_x)),
                                                   int(np.floor((width-lattice.origin_x)/lattice.pitch_x))+1)*lattice.pitch_x
                yy = lattice.origin_y + np.arange(int(np.ceil(-lattice.origin_y/lattice.pitch_y)),
                                                   int(np.floor((height-lattice.origin_y)/lattice.pitch_y))+1)*lattice.pitch_y
                left, top = grid.crop.box[:2]
                stroke.setColor(COLORS['rejected'])
                for x in xx:
                    xpos = left + (x-tr.offset[0])/tr.scale_x
                    canvas.drawLine(float(xpos), float(top), float(xpos), float(grid.crop.box[3]), stroke)
                for y in yy:
                    ypos = top + (y-tr.offset[1])/tr.scale_y
                    canvas.drawLine(float(left), float(ypos), float(grid.crop.box[2]), float(ypos), stroke)
                label('diagnostic lattice; board extent unknown', left, top-4, COLORS['rejected'])
        if not grid.hypotheses:
            continue
        xs, ys, _ = grid.hypotheses[0]
        if layers.get('cell_support') and grid.phase_data:
            cells = grid.phase_data['global'].cell_occupancy
            if cells is not None:
                for row in range(cells.shape[0]):
                    for col in range(cells.shape[1]):
                        value = cells[row, col]
                        cell_color = 0x5534d17b if np.isfinite(value) and value >= .5 else 0x88ff5c66
                        canvas.drawRect(skia.Rect.MakeLTRB(float(xs[col]), float(ys[row]), float(xs[col+1]), float(ys[row+1])), skia.Paint(Color=cell_color))
        if layers.get('boundaries'):
            for x in xs: canvas.drawLine(float(x), float(ys[0]), float(x), float(ys[-1]), stroke)
            for y in ys: canvas.drawLine(float(xs[0]), float(y), float(xs[-1]), float(y), stroke)
        if layers.get('centers'):
            paint = skia.Paint(Color=color, AntiAlias=True, StrokeWidth=3, StrokeCap=skia.Paint.kRound_Cap)
            cx, cy = (xs[:-1]+xs[1:])/2, (ys[:-1]+ys[1:])/2
            points = [skia.Point(float(x), float(y)) for y in cy for x in cx]
            canvas.drawPoints(skia.Canvas.kPoints_PointMode, points, paint)
        label(f'{len(ys)-1}×{len(xs)-1} {grid.decoding.status} / {grid.validation.status if grid.validation else "rejected"}', float(xs[0]), float(ys[0])-4, color)
