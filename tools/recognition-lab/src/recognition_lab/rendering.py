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


def draw(canvas, result, layers, selected=None, style=DESKTOP_STYLE, magnification=1.0, dpr=None, selected_geometry=None, participation_grid=None):
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
        decoder_status = getattr(grid.decoding, 'status', None)
        status = 'ambiguous' if decoder_status == 'ambiguous' else (grid.validation.status if grid.validation else 'rejected')
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
                if layers.get('tiles') and grid is not participation_grid:
                    stroke.setColor(color_with_opacity(0xff36cfff if observation.local_valid else 0xffffc34d))
                    a, b = point(x0, y0), point(x1, y1)
                    canvas.drawRect(skia.Rect.MakeLTRB(*a, *b), stroke)
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
        if layers.get('boundaries'):
            rejected = grid.validation is not None and not grid.validation.accepted
            boundary_stroke = stroke
            if interactive and rejected:
                unit = dim(1)
                boundary_stroke = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style,
                                             StrokeWidth=dim(style.width), Color=color_with_opacity(color))
                boundary_stroke.setPathEffect(skia.DashPathEffect.Make([7*unit, 4*unit], 0))
            if interactive:
                boundary_stroke.setStrokeWidth(logical(1.5))
                canvas.drawRect(skia.Rect.MakeLTRB(float(xs[0]), float(ys[0]), float(xs[-1]), float(ys[-1])), boundary_stroke)
                boundary_stroke.setStrokeWidth(dim(style.width))
            for x in xs: canvas.drawLine(float(x), float(ys[0]), float(x), float(ys[-1]), stroke)
            for y in ys: canvas.drawLine(float(xs[0]), float(y), float(xs[-1]), float(y), stroke)
        if layers.get('centers'):
            paint = skia.Paint(Color=color_with_opacity(color), AntiAlias=True, StrokeWidth=dim(3), StrokeCap=skia.Paint.kRound_Cap)
            cx, cy = (xs[:-1]+xs[1:])/2, (ys[:-1]+ys[1:])/2
            points = [skia.Point(float(x), float(y)) for y in cy for x in cx]
            canvas.drawPoints(skia.Canvas.kPoints_PointMode, points, paint)
        decoder_label = f' decoder: {decoder_status} /' if decoder_status is not None else ''
        board_label = (' accepted board' if grid.validation and grid.validation.accepted else ' candidate extent') if interactive else ''
        label(f'{len(ys)-1}×{len(xs)-1}{board_label}{decoder_label} board: {grid.validation.status if grid.validation else "unavailable"}', float(xs[0]), float(ys[0])-dim(4), color)


def diagnostic_label(canvas, text, x, y, color, style, magnification, dpr, occupied):
    """Logical-pixel label with a contrasting background; geometry stays anchored."""
    unit = dpr / magnification
    font = skia.Font(label_typeface(), 12*unit)
    pad = 3*unit
    width, height = font.measureText(text)+2*pad, 18*unit
    top = y + pad
    for left, upper, right, lower in occupied:
        if x < right and x+width > left and top < lower and top+height > upper:
            top = lower + pad
    occupied.append((x, top, x+width, top+height))
    canvas.drawRect(skia.Rect.MakeXYWH(x, top, width, height), skia.Paint(Color=style.color(0xff202020)))
    canvas.drawString(text, x+pad, top+pad+12*unit, font,
                      skia.Paint(Color=style.color(color), AntiAlias=True))


def draw_global_terminations(canvas, lines, style, magnification, dpr=1.):
    """Projected axis estimates, not reconstructed spatial observations."""
    if style.opacity <= 0: return
    unit = style.dimension(1, magnification)
    paint = skia.Paint(Color=style.color(0xffed90f0), AntiAlias=True,
                       StrokeWidth=style.dimension(style.width, magnification))
    paint.setPathEffect(skia.DashPathEffect.Make([5*unit, 4*unit], 0))
    occupied = []
    for name, coords in lines:
        canvas.drawLine(*coords, paint)
        diagnostic_label(canvas, f'{name} termination estimate (projected)', coords[0], coords[1],
                         0xffed90f0, style, magnification, dpr, occupied)


def draw_global_cells(canvas, cells, opacity):
    if opacity <= 0: return
    alpha = max(0, min(255, round(opacity*255)))
    for _, _, bounds, rgb in cells:
        color = alpha << 24 | rgb[0] << 16 | rgb[1] << 8 | rgb[2]
        canvas.drawRect(skia.Rect.MakeLTRB(*bounds), skia.Paint(Color=color))


def draw_local_geometry(canvas, elements, validation, style, magnification):
    """Draw local evidence with a contrasting halo and validation-specific dashes."""
    if not elements or style.opacity <= 0:
        return
    width = style.dimension(style.width, magnification)
    unit = style.dimension(1, magnification)
    colors = {'Evaluated accepted': 0xff51f2aa, 'Evaluated rejected': 0xffff7283,
              'Not evaluated': 0xfff6d34e, 'Unavailable': 0xffc4b6ff}
    patterns = {'Evaluated accepted': None, 'Evaluated rejected': (7, 4),
                'Not evaluated': (2, 4), 'Unavailable': (8, 3, 2, 3)}
    color = style.color(colors.get(validation, colors['Unavailable']))
    pattern = patterns.get(validation, patterns['Unavailable'])
    for kind, coords in elements:
        for halo in (True, False):
            paint = skia.Paint(Color=style.color(0xff000000) if halo else color,
                               Style=skia.Paint.kStroke_Style, AntiAlias=True,
                               StrokeWidth=width + (2.5*unit if halo else 0))
            if pattern is not None and kind in ('lattice', 'boundary'):
                paint.setPathEffect(skia.DashPathEffect.Make([v*unit for v in pattern], 0))
            if kind in ('lattice', 'boundary'):
                if kind == 'boundary': paint.setStrokeWidth(width + (3*unit if halo else unit))
                canvas.drawLine(*coords, paint)
            else:
                x, y = coords
                size = (3 if kind == 'center' else 5)*unit
                if kind == 'center':
                    canvas.drawCircle(x, y, size/2, paint)
                elif kind == 'crossing':
                    canvas.drawLine(x-size, y-size, x+size, y+size, paint)
                    canvas.drawLine(x-size, y+size, x+size, y-size, paint)
                elif kind == 'censored':
                    canvas.drawRect(skia.Rect.MakeLTRB(x-size, y-size, x+size, y+size), paint)


def draw_participation(canvas, rectangles, style, magnification, dpr=1.):
    """Source-only consensus membership; labels retain local eligibility separately."""
    if style.opacity <= 0: return
    unit = style.dimension(1, magnification)
    occupied = []
    for index, box, local_status, eligible, membership in rectangles:
        color = 0xff43dfa0 if local_status == 'accepted' else 0xffff9a65 if local_status == 'rejected' else 0xffc9c9c9
        paint = skia.Paint(Color=style.color(color), AntiAlias=True, Style=skia.Paint.kStroke_Style,
                           StrokeWidth=style.dimension(style.width, magnification))
        if membership != 'Inlier':
            pattern = (7, 4) if membership == 'Excluded' else (2, 4)
            paint.setPathEffect(skia.DashPathEffect.Make([v*unit for v in pattern], 0))
        canvas.drawRect(skia.Rect.MakeLTRB(*box), paint)
        eligibility = 'eligible' if eligible is True else 'not eligible' if eligible is False else 'eligibility unknown'
        text = f'{index+1}: local {local_status} / {eligibility} / {membership.lower()}'
        diagnostic_label(canvas, text, box[0], box[1], color, style, magnification, dpr, occupied)


def draw_viewer_visualization(canvas, result, layers, selected, grid, mode, tile_index, style, magnification, dpr, selected_geometry=None, prepared_heat=None, participation_rects=None):
    """Draw viewer content; legacy selection filters are intentionally ignored.

    ``selected`` and ``selected_geometry`` remain accepted for compatibility.
    The selected grid still controls heatmaps; general vectors are unfiltered.
    """
    heat = None
    tile_offset = (0, 0)
    if prepared_heat is not None:
        hm = prepared_heat
    elif grid is not None and grid.phase_data:
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
    if prepared_heat is None and heat is not None:
        valid = np.isfinite(heat)
        heat = np.nan_to_num(np.clip(heat*255, 0, 255)).astype(np.uint8)
        rgba = np.zeros((*heat.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 3] = np.where(valid, heat // 2, 0).astype(np.uint8)
        hm = skia.Image.fromarray(rgba)
    if prepared_heat is not None or heat is not None:
        # Model/tile coordinates go through the existing crop transform to source pixels.
        tr = grid.heatmap_transform
        left, top = grid.crop.box[:2]
        height, width = (prepared_heat.height(), prepared_heat.width()) if prepared_heat is not None else heat.shape
        rect = skia.Rect.MakeXYWH(left+(tile_offset[0]-tr.offset[0])/tr.scale_x, top+(tile_offset[1]-tr.offset[1])/tr.scale_y, width/tr.scale_x, height/tr.scale_y)
        canvas.save()
        canvas.clipRect(skia.Rect.MakeLTRB(*grid.crop.box))
        canvas.drawImageRect(hm, rect)
        canvas.restore()
    # General annotations remain visible regardless of the current selection.
    draw(canvas, result, layers, None, style, magnification, dpr,
         participation_grid=grid if participation_rects is not None else None)
    if participation_rects is not None:
        draw_participation(canvas, participation_rects, style, magnification, dpr)


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
