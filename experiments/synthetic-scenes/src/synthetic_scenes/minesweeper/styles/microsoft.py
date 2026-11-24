"""Procedural approximation of Microsoft Minesweeper Modern theme."""
import skia
from ..drawing import color

THEMES = {
    'default': ('#8ce4ff', '#379ee7'),
    'blue': ('#a8ddff', '#507dca'),
    'warm': ('#ffe0a0', '#c58a43'),
}


def background_color():
    return color('#30363c')


def _tile(canvas, x, y, s, center, edge, outline):
    radius = s * .08
    outer = skia.Rect.MakeXYWH(x, y, s, s)
    canvas.drawRoundRect(outer, radius, radius,
                         skia.Paint(Color=color(outline), AntiAlias=True))
    inset = max(.65, s * .035)
    inner = skia.Rect.MakeXYWH(x+inset, y+inset, s-2*inset, s-2*inset)
    paint = skia.Paint(AntiAlias=True)
    paint.setShader(skia.GradientShader.MakeRadial(
        skia.Point(x+s/2, y+s/2), s*.65,
        [color(center), color(center), color(edge)], [0., .42, 1.]))
    canvas.drawRoundRect(inner, max(0.,radius-inset), max(0.,radius-inset), paint)


def draw_surface(canvas,x,y,s,opened,row,col,theme,neighbors):
    center, edge = ('#ffffff','#aeb4b9') if opened else THEMES[theme]
    _tile(canvas,x,y,s,center,edge,'#171b20')


def draw_flag_surface(canvas,x,y,s):
    _tile(canvas,x,y,s,'#ffe78b','#dca82c','#59461b')
