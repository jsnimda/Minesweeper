"""Skia drawing shared by the image viewer and desktop overlay."""
from dataclasses import dataclass
from functools import lru_cache
import math

import skia
import numpy as np

from .viewer_selection import bracket_dimensions, valid_bounds

DETECTION_COLORS = {'grid': 0xff1683c7, 'mine counter': 0xffb84669, 'timer': 0xff8a5abd, 'reset button': 0xffad6200}

COLORS = {'accepted': 0xff34d17b, 'rejected': 0xffff5c66, 'ambiguous': 0xffffc34d}



@dataclass(frozen=True)
class RecognitionStyle:
    """Dimensions are physical pixels in screen mode, source pixels in image mode."""
    mode: str = 'screen'
    width: float = 1.0
    opacity: float = 1.0

    def __post_init__(self):
        if self.mode not in ('screen', 'image') or not math.isfinite(self.width) or self.width <= 0:
            raise ValueError('invalid recognition style')
        if not math.isfinite(self.opacity) or not 0 <= self.opacity <= 1:
            raise ValueError('invalid recognition opacity')

    def dimension(self, size, magnification):
        if not math.isfinite(magnification) or magnification <= 0:
            raise ValueError('invalid physical magnification')
        return size / magnification if self.mode == 'screen' else size

    def color(self, color):
        # Per-primitive opacity preserves existing alpha; overlaps accumulate.
        return (round(((color >> 24) & 255) * self.opacity) << 24) | (color & 0xffffff)


# Desktop uses the original source-pixel dimensions and full opacity.
DESKTOP_STYLE = RecognitionStyle('image', 2.0, 1.0)

@lru_cache(maxsize=1)
def label_typeface():
    """Resolve a named sans-serif face once, falling back to an installed family."""
    manager = skia.FontMgr.RefDefault()
    style = skia.FontStyle.Normal()
    for family in ('Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'):
        face = manager.matchFamilyStyle(family, style)
        if face is not None:
            return face
    for family in manager:
        face = manager.matchFamilyStyle(family, style)
        if face is not None:
            return face
    raise RuntimeError('No usable system font is available for recognition labels')


def draw(canvas, result, layers, selected=None, style=DESKTOP_STYLE, magnification=1.0, dpr=None, selected_geometry=None):
    if result is None:
        return
    dim = lambda size: style.dimension(size, magnification)
    color_with_opacity = style.color
    stroke = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=dim(style.width))
    interactive = dpr is not None
    # Annotation sizes use logical pixels in screen mode; precision strokes use physical pixels.
    logical = lambda size: style.dimension(size * dpr, magnification) if interactive and style.mode == 'screen' else size
    font_size = logical(13) if interactive else dim(14)
    font = skia.Font(label_typeface(), font_size)
    occupied = []
    def label(text, x, y, color):
        if not interactive:
            canvas.drawString(text, x, max(dim(14), y), font, skia.Paint(Color=color_with_opacity(color), AntiAlias=True))
            return
        pad = logical(3)
        height = font_size + 2 * pad
        width = font.measureText(text) + 2 * pad
        top = y - height
        if top < 0: top = y + pad
        for left, upper, right, lower in occupied:
            if x < right and x + width > left and top < lower and top + height > upper:
                top = lower + pad
        occupied.append((x, top, x + width, top + height))
        canvas.drawRect(skia.Rect.MakeXYWH(x, top, width, height), skia.Paint(Color=color_with_opacity(color)))
        r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
        foreground = 0xff000000 if (299*r + 587*g + 114*b) >= 150000 else 0xffffffff
        canvas.drawString(text, x + pad, top + pad + font_size, font, skia.Paint(Color=color_with_opacity(foreground), AntiAlias=True))
    if layers.get('detections'):
        for d in result.detections:
            detection_color = DETECTION_COLORS.get(d.name.lower(), 0xff167d86) if interactive else 0xff36cfff
            stroke.setColor(color_with_opacity(detection_color))
            if interactive: stroke.setStrokeWidth(logical(2))
            canvas.drawRect(skia.Rect.MakeLTRB(*d.box), stroke)
            label(f'{d.name} {d.confidence:.2f}', d.box[0], d.box[1] if interactive else d.box[1]-dim(3), detection_color)
        if interactive: stroke.setStrokeWidth(dim(style.width))
    for grid in result.grids:
        if selected_geometry is not None and grid is not selected_geometry:
            continue
        if selected is not None and grid.detection_index != selected:
            continue
        status = 'ambiguous' if grid.decoding.status == 'ambiguous' else (grid.validation.status if grid.validation else 'rejected')
        color = COLORS.get(status, 0xffffc34d)
        stroke.setColor(color_with_opacity(color))
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
                    stroke.setColor(color_with_opacity(0xff36cfff if observation.local_valid else 0xffffc34d))
                    a, b = point(x0, y0), point(x1, y1)
                    canvas.drawRect(skia.Rect.MakeLTRB(*a, *b), stroke)
                if layers.get('terminations') and observation.hypothesis is not None:
                    h = observation.hypothesis
                    marks = [(h.x.visible_start, (y1-y0)/2, x0, y0), (h.x.visible_end, (y1-y0)/2, x0, y0)]
                    for x, y, ox, oy in marks:
                        if x is not None:
                            px, py = point(x+ox, y+oy)
                            canvas.drawCircle(px, py, dim(3), skia.Paint(Color=color_with_opacity(0xff36cfff), AntiAlias=True))
                    for y in (h.y.visible_start, h.y.visible_end):
                        if y is not None:
                            px, py = point((x0+x1)/2, y+y0)
                            canvas.drawCircle(px, py, dim(3), skia.Paint(Color=color_with_opacity(0xff36cfff), AntiAlias=True))
            stroke.setColor(color_with_opacity(color))
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
                stroke.setColor(color_with_opacity(COLORS['rejected']))
                for x in xx:
                    xpos = left + (x-tr.offset[0])/tr.scale_x
                    canvas.drawLine(float(xpos), float(top), float(xpos), float(grid.crop.box[3]), stroke)
                for y in yy:
                    ypos = top + (y-tr.offset[1])/tr.scale_y
                    canvas.drawLine(float(left), float(ypos), float(grid.crop.box[2]), float(ypos), stroke)
                label('diagnostic lattice; board extent unknown', left, top-dim(4), COLORS['rejected'])
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
                        canvas.drawRect(skia.Rect.MakeLTRB(float(xs[col]), float(ys[row]), float(xs[col+1]), float(ys[row+1])), skia.Paint(Color=color_with_opacity(cell_color)))
        if layers.get('boundaries'):
            if interactive:
                stroke.setStrokeWidth(logical(1.5))
                canvas.drawRect(skia.Rect.MakeLTRB(float(xs[0]), float(ys[0]), float(xs[-1]), float(ys[-1])), stroke)
                stroke.setStrokeWidth(dim(style.width))
            for x in xs: canvas.drawLine(float(x), float(ys[0]), float(x), float(ys[-1]), stroke)
            for y in ys: canvas.drawLine(float(xs[0]), float(y), float(xs[-1]), float(y), stroke)
        if layers.get('centers'):
            paint = skia.Paint(Color=color_with_opacity(color), AntiAlias=True, StrokeWidth=dim(3), StrokeCap=skia.Paint.kRound_Cap)
            cx, cy = (xs[:-1]+xs[1:])/2, (ys[:-1]+ys[1:])/2
            points = [skia.Point(float(x), float(y)) for y in cy for x in cx]
            canvas.drawPoints(skia.Canvas.kPoints_PointMode, points, paint)
        label(f'{len(ys)-1}×{len(xs)-1} {grid.decoding.status} / {grid.validation.status if grid.validation else "rejected"}', float(xs[0]), float(ys[0])-dim(4), color)


def draw_viewer_visualization(canvas, result, layers, selected, grid, mode, tile_index, style, magnification, dpr, selected_geometry=None):
    """Draw viewer content; legacy selection filters are intentionally ignored.

    ``selected`` and ``selected_geometry`` remain accepted for compatibility.
    The selected grid still controls heatmaps; general vectors are unfiltered.
    """
    heat = None
    tile_offset = (0, 0)
    if grid is not None and grid.phase_data:
        phase = grid.phase_data
        if mode == 'Phase occupancy': heat = phase['global'].occupancy
        elif mode == 'Phase coverage': heat = (phase['global'].weights > 0).astype(float)
        elif mode in ('Tile horizontal phase', 'Tile vertical phase'):
            observations = phase['observations']
            o = observations[tile_index] if 0 <= tile_index < len(observations) else None
            if o is not None and o.fields is not None:
                axis = 0 if mode == 'Tile horizontal phase' else 2
                heat = (np.arctan2(o.fields[axis+1], o.fields[axis]) / (2*np.pi) + .5) * o.fields[4]
                tile_offset = o.box[:2]
    if heat is not None:
        valid = np.isfinite(heat)
        heat = np.nan_to_num(np.clip(heat*255, 0, 255)).astype(np.uint8)
        rgba = np.zeros((*heat.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 3] = np.where(valid, heat // 2, 0).astype(np.uint8)
        hm = skia.Image.fromarray(rgba)
        # Model/tile coordinates go through the existing crop transform to source pixels.
        tr = grid.heatmap_transform
        left, top = grid.crop.box[:2]
        rect = skia.Rect.MakeXYWH(left+(tile_offset[0]-tr.offset[0])/tr.scale_x, top+(tile_offset[1]-tr.offset[1])/tr.scale_y, heat.shape[1]/tr.scale_x, heat.shape[0]/tr.scale_y)
        canvas.save()
        canvas.clipRect(skia.Rect.MakeLTRB(*grid.crop.box))
        canvas.drawImageRect(hm, rect)
        canvas.restore()
    # General annotations remain visible regardless of the current selection.
    draw(canvas, result, layers, None, style, magnification, dpr)


def draw_selection(canvas, bounds, magnification, dpr, opacity):
    """Contrasting logical-pixel corner brackets in source-image coordinates."""
    bounds = valid_bounds(bounds)
    if bounds is None or opacity <= 0:
        return
    length, outer_width, inner_width = bracket_dimensions(magnification, dpr)
    x1, y1, x2, y2 = bounds
    dx, dy = min(length, (x2-x1)/2), min(length, (y2-y1)/2)
    segments = [(x1, y1, x1+dx, y1), (x1, y1, x1, y1+dy),
                (x2, y1, x2-dx, y1), (x2, y1, x2, y1+dy),
                (x1, y2, x1+dx, y2), (x1, y2, x1, y2-dy),
                (x2, y2, x2-dx, y2), (x2, y2, x2, y2-dy)]
    alpha = round(255 * opacity) << 24
    for color, width in ((alpha | 0x000000, outer_width), (alpha | 0xffffff, inner_width)):
        paint = skia.Paint(Color=color, StrokeWidth=width, AntiAlias=True)
        for segment in segments:
            canvas.drawLine(*segment, paint)
