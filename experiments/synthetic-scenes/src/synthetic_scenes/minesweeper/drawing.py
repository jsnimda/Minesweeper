"""Small shared Skia drawing helpers. Colors are RGB hex strings."""

import math

import skia


def color(value: str):
    return skia.Color(*bytes.fromhex(value.lstrip('#')))


def mix(a: str, b: str, t: float) -> str:
    """Interpolate two RGB colors in channel space."""
    first = bytes.fromhex(a.lstrip('#'))
    second = bytes.fromhex(b.lstrip('#'))
    channels = [round(x + (y - x) * t) for x, y in zip(first, second)]
    return '#' + bytes(channels).hex()


def rect(canvas, x, y, w, h, fill):
    bounds = skia.Rect.MakeXYWH(x, y, w, h)
    paint = skia.Paint(Color=color(fill), AntiAlias=True)
    canvas.drawRect(bounds, paint)


def rounded_rect(canvas, x, y, w, h, radius, fill):
    bounds = skia.Rect.MakeXYWH(x, y, w, h)
    paint = skia.Paint(Color=color(fill), AntiAlias=True)
    canvas.drawRoundRect(bounds, radius, radius, paint)


def polygon(canvas, points, fill):
    path = skia.Path()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    path.close()

    paint = skia.Paint(Color=color(fill), AntiAlias=True)
    canvas.drawPath(path, paint)


def line(canvas, a, b, fill, width):
    paint = skia.Paint(
        Color=color(fill),
        StrokeWidth=width,
        AntiAlias=True,
        StrokeCap=skia.Paint.kRound_Cap,
    )
    canvas.drawLine(*a, *b, paint)


def circle(canvas, x, y, radius, fill):
    paint = skia.Paint(Color=color(fill), AntiAlias=True)
    canvas.drawCircle(x, y, radius, paint)


def gradient(canvas, x, y, w, h, top, bottom):
    bounds = skia.Rect.MakeXYWH(x, y, w, h)
    shader = skia.GradientShader.MakeLinear(
        [skia.Point(x, y), skia.Point(x, y + h)],
        [color(top), color(bottom)],
    )
    paint = skia.Paint(AntiAlias=True)
    paint.setShader(shader)
    canvas.drawRect(bounds, paint)


def font_for(family, size, bold=True):
    style = skia.FontStyle.Bold() if bold else skia.FontStyle.Normal()
    face = skia.Typeface.MakeFromName(family, style)
    return skia.Font(face, size)


def text(canvas, value, cx, cy, font, fill):
    bounds = skia.Rect()
    font.measureText(value, bounds=bounds)

    x = cx - (bounds.left() + bounds.right()) / 2
    y = cy - (bounds.top() + bounds.bottom()) / 2
    paint = skia.Paint(Color=color(fill), AntiAlias=True)
    canvas.drawString(value, x, y, font, paint)


def flag(canvas, x, y, s, fill='#e02020', ink='#202020'):
    pole = [(x + .53*s, y + .22*s), (x + .53*s, y + .77*s)]
    banner = [
        (x + .53*s, y + .19*s),
        (x + .20*s, y + .36*s),
        (x + .53*s, y + .48*s),
    ]
    base = [
        (x + .42*s, y + .68*s),
        (x + .64*s, y + .68*s),
        (x + .75*s, y + .79*s),
        (x + .30*s, y + .79*s),
    ]
    line(canvas, *pole, ink, .055*s)
    polygon(canvas, banner, fill)
    polygon(canvas, base, ink)


def mine(canvas, x, y, s, fill='#202020'):
    cx = x + s / 2
    cy = y + s / 2

    for i in range(4):
        angle = i * math.pi / 4
        dx = math.cos(angle) * s * .34
        dy = math.sin(angle) * s * .34
        line(canvas, (cx-dx, cy-dy), (cx+dx, cy+dy), fill, .055*s)

    circle(canvas, cx, cy, s*.24, fill)
    circle(canvas, cx-s*.08, cy-s*.08, s*.05, '#ffffff')
