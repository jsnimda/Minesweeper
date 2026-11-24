"""Procedural glass tiles inspired by the Win7 blue and green atlases.

No sprite pixels are used at rendering time. The lighting field is an
approximation, not a reconstruction of the original atlas.
"""

import math

import skia

from ..drawing import color, gradient, mix, rect, rounded_rect


THEMES = {
    'default': {
        'dark': '#263c9e',
        'light': '#a8e0ff',
        'edge': '#192c66',
        'opened_dark': '#a8b1ce',
        'opened_light': '#e6f5fa',
    },
    'green': {
        'dark': '#31732e',
        'light': '#a4ef94',
        'edge': '#244822',
        'opened_dark': '#a1cda0',
        'opened_light': '#edffeb',
    },
}
THEMES['blue'] = THEMES['default']


def lighting(row, col):
    """Broad diagonal light bands over a gradually darker board."""
    band = .5 + .5 * math.cos((row - col * .20) * 1.0)
    falloff = math.exp(-row / 22 - col / 65)
    return min(1., .15 + .65 * band * falloff + .18 * falloff)


def _glass_body(canvas, x, y, s, tone):
    inset = .10 * s
    bounds = skia.Rect.MakeXYWH(x + inset, y + inset, s - 2*inset, s - 2*inset)
    radius = .075 * s
    shader = skia.GradientShader.MakeLinear(
        [skia.Point(x, y), skia.Point(x + s, y + s)],
        [color(mix(tone, '#ffffff', .33)), color(tone),
         color(mix(tone, '#141c37', .23))],
        [0., .43, 1.],
    )
    paint = skia.Paint(AntiAlias=True)
    paint.setShader(shader)
    canvas.drawRoundRect(bounds, radius, radius, paint)

    # A broad curved reflection, clipped to the inner glass face.
    canvas.save()
    canvas.clipRRect(skia.RRect.MakeRectXY(bounds, radius, radius), doAntiAlias=True)
    reflection = skia.Path()
    reflection.moveTo(x, y)
    reflection.lineTo(x + s, y)
    reflection.cubicTo(x + .86*s, y + .50*s, x + .49*s, y + .67*s, x, y + .62*s)
    reflection.close()
    canvas.drawPath(reflection, skia.Paint(Color=skia.ColorWHITE, Alpha=35, AntiAlias=True))
    canvas.restore()


def _hidden(canvas, x, y, s, palette, light):
    tone = mix(palette['dark'], palette['light'], light)
    rect(canvas, x, y, s, s, palette['edge'])
    rounded_rect(canvas, x+.025*s, y+.025*s, .95*s, .95*s, .085*s,
                 mix(tone, '#ffffff', .35))
    rounded_rect(canvas, x+.055*s, y+.055*s, .89*s, .89*s, .08*s,
                 mix(tone, palette['edge'], .36))
    _glass_body(canvas, x, y, s, tone)
    rounded_rect(canvas, x+.14*s, y+.13*s, .68*s, .027*s, .013*s,
                 mix(tone, '#ffffff', .55))


def _opened(canvas, x, y, s, palette, light, neighbors):
    tone = mix(palette['opened_dark'], palette['opened_light'], light)
    rect(canvas, x, y, s, s, mix(palette['edge'], tone, .62))
    rect(canvas, x+.026*s, y+.026*s, .948*s, .948*s, mix(tone, '#ffffff', .65))
    gradient(canvas, x+.075*s, y+.075*s, .85*s, .85*s,
             mix(tone, '#ffffff', .23), tone)
    # Only covered neighbors cast a shadow; not the outer board edge.
    if neighbors.get('top_covered', False):
        gradient(canvas, x, y, s, .14*s, mix(tone, palette['edge'], .38), tone)
    if neighbors.get('left_covered', False):
        rect(canvas, x, y, .055*s, s, mix(tone, palette['edge'], .30))


def draw_surface(canvas, x, y, s, opened, row, col, theme, neighbors):
    palette = THEMES[theme]
    light = lighting(row, col)
    if opened:
        _opened(canvas, x, y, s, palette, light, neighbors)
    else:
        _hidden(canvas, x, y, s, palette, light)
