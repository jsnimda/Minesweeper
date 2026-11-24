"""Procedural symbols; no original sprite assets are embedded."""

import math

import skia

from .drawing import circle, color, flag, line, mine, polygon, rounded_rect


def draw_flag(canvas, x, y, size, style):
    if style == 'classic':
        flag(canvas, x, y, size)
        return

    if style == 'windows7':
        flag(canvas, x, y, size, fill='#ffffff', ink='#d9f1ff')
        flag(canvas, x + .055*size, y + .035*size, .89*size,
             fill='#e5212b', ink='#315875')
        return

    # Modern and Google use flat silhouettes, without a Classic bevel/base.
    pole_color = '#7d2926' if style == 'google' else '#343b44'
    line(canvas, (x+.39*size, y+.20*size),
         (x+.39*size, y+.79*size), pole_color, .065*size)
    banner = [(x+.39*size, y+.19*size), (x+.78*size, y+.32*size),
              (x+.39*size, y+.48*size)]
    polygon(canvas, banner, '#ef4136' if style == 'google' else '#c82738')
    if style == 'microsoft':
        rounded_rect(canvas, x+.23*size, y+.73*size, .36*size,
                     .08*size, .025*size, pole_color)


def _metal_mine(canvas, x, y, size):
    cx, cy = x + size/2, y + size/2
    for index in range(10):
        angle = index * math.tau / 10
        points = []
        for radius, offset in ((.19, -.17), (.38, 0), (.19, .17)):
            points.append((cx + radius*size*math.cos(angle+offset),
                           cy + radius*size*math.sin(angle+offset)))
        polygon(canvas, points, '#68737d')
    paint = skia.Paint(AntiAlias=True)
    paint.setShader(skia.GradientShader.MakeRadial(
        skia.Point(cx-.09*size, cy-.10*size), .37*size,
        [color('#f6f7f8'), color('#a0a8ae'), color('#333c44')],
        [0., .48, 1.],
    ))
    canvas.drawCircle(cx, cy, .245*size, paint)
    circle(canvas, cx-.09*size, cy-.10*size, .045*size, '#ffffff')


def _flower(canvas, x, y, size):
    cx, cy = x + size/2, y + size/2
    for index in range(8):
        angle = index * math.tau / 8
        canvas.save()
        canvas.translate(cx, cy)
        canvas.rotate(math.degrees(angle))
        bounds = skia.Rect.MakeXYWH(-.10*size, -.39*size, .20*size, .31*size)
        paint = skia.Paint(Color=color('#c8d3db'), AntiAlias=True)
        canvas.drawOval(bounds, paint)
        bounds = skia.Rect.MakeXYWH(-.085*size, -.38*size, .17*size, .28*size)
        canvas.drawOval(bounds, skia.Paint(Color=skia.ColorWHITE, AntiAlias=True))
        canvas.restore()
    circle(canvas, cx, cy, .15*size, '#ce9417')
    circle(canvas, cx-.015*size, cy-.025*size, .12*size, '#f6c832')


def draw_mine(canvas, x, y, size, style, variant='classic'):
    if style == 'windows7':
        if variant == 'flower':
            _flower(canvas, x, y, size)
        else:
            _metal_mine(canvas, x, y, size)
    elif style == 'google':
        circle(canvas, x+size/2, y+size/2, .25*size, '#414141')
        circle(canvas, x+.43*size, y+.42*size, .07*size, '#777777')
    elif style == 'microsoft':
        mine(canvas, x, y, size, fill='#49515b')
    else:
        mine(canvas, x, y, size)
