"""Procedural desktop wallpapers rendered with Skia."""

import colorsys
import math
import random

import skia


PROCEDURAL_SUBTYPE_WEIGHTS = {
    'solid': 15,
    'linear': 30,
    'radial': 25,
    'multi_gradient': 30,
}


def sample_procedural_subtype(rng: random.Random) -> str:
    """Choose a procedural subtype independently of rendering."""
    return rng.choices(
        tuple(PROCEDURAL_SUBTYPE_WEIGHTS),
        weights=tuple(PROCEDURAL_SUBTYPE_WEIGHTS.values()),
        k=1,
    )[0]


def _sample_color(rng, family, alpha=255):
    saturation, lightness = {
        'dark': ((0.15, 0.8), (0.08, 0.32)),
        'light': ((0.12, 0.65), (0.7, 0.92)),
        'neutral': ((0.0, 0.16), (0.2, 0.85)),
        'saturated': ((0.65, 1.0), (0.3, 0.65)),
    }[family]
    rgb = colorsys.hls_to_rgb(
        rng.random(), rng.uniform(*lightness), rng.uniform(*saturation)
    )
    red, green, blue = (round(channel * 255) for channel in rgb)
    return skia.ColorSetARGB(alpha, red, green, blue)


def _stops(rng, count):
    return [0.0, *sorted(rng.uniform(0.15, 0.85) for _ in range(count - 2)), 1.0]


def _draw_linear(canvas, width, height, rng, family):
    count = rng.choice((2, 3))
    angle = rng.uniform(0, math.tau)
    dx, dy = math.cos(angle), math.sin(angle)
    extent = (abs(dx) * width + abs(dy) * height) / 2
    cx, cy = width / 2, height / 2
    shader = skia.GradientShader.MakeLinear(
        [skia.Point(cx - dx * extent, cy - dy * extent),
         skia.Point(cx + dx * extent, cy + dy * extent)],
        [_sample_color(rng, family) for _ in range(count)],
        _stops(rng, count),
    )
    paint = skia.Paint(Shader=shader)
    canvas.drawPaint(paint)


def _draw_radial(canvas, width, height, rng, family, overlay=False):
    center = skia.Point(rng.uniform(0, width), rng.uniform(0, height))
    radius = rng.uniform(0.35, 1.2) * math.hypot(width, height)
    if overlay:
        color = _sample_color(rng, family, rng.randint(70, 180))
        transparent = color & 0x00ffffff
        colors = [color, transparent]
        stops = [0.0, rng.uniform(0.55, 1.0)]
    else:
        count = rng.choice((2, 3))
        colors = [_sample_color(rng, family) for _ in range(count)]
        stops = _stops(rng, count)
    shader = skia.GradientShader.MakeRadial(center, radius, colors, stops)
    canvas.drawPaint(skia.Paint(Shader=shader))


def render_procedural_wallpaper(size, rng: random.Random, subtype: str, *, native=False):
    """Render a wallpaper as Skia when native, otherwise as an RGB Pillow image."""
    if subtype not in PROCEDURAL_SUBTYPE_WEIGHTS:
        raise ValueError(f'Unknown procedural wallpaper subtype: {subtype}')
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError('Wallpaper dimensions must be positive')
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    family = rng.choice(('dark', 'light', 'neutral', 'saturated'))
    if subtype == 'solid':
        canvas.clear(_sample_color(rng, family))
    elif subtype == 'linear':
        _draw_linear(canvas, width, height, rng, family)
    elif subtype == 'radial':
        _draw_radial(canvas, width, height, rng, family)
    else:
        if rng.choice((True, False)):
            canvas.clear(_sample_color(rng, family))
        else:
            _draw_linear(canvas, width, height, rng, family)
        for _ in range(rng.randint(2, 4)):
            _draw_radial(canvas, width, height, rng, family, overlay=True)
    image = surface.makeImageSnapshot()
    if native:
        return image
    from .image_conversion import image_to_pillow
    return image_to_pillow(image)
