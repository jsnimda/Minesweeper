"""Reusable procedural controls drawn directly on a Skia canvas."""
import skia

from .drawing import circle, color, font_for, line, mine, polygon, rect, text
from .symbols import draw_flag, draw_mine


def bevel(canvas, box, thickness, face, light, dark, sunken=False):
    x, y, r, b = box
    t = thickness
    if sunken:
        light, dark = dark, light
    rect(canvas, x, y, r-x, b-y, dark)
    polygon(canvas, [(x,y), (r,y), (r-t,y+t), (x+t,y+t),
                     (x+t,b-t), (x,b)], light)
    rect(canvas, x+t, y+t, r-x-2*t, b-y-2*t, face)


def digital(canvas, box, value, slots, palette, signed=False):
    """Draw an unclipped signed, zero-padded seven-segment display."""
    x, y, r, b = box
    t = max(1, round((b-y)*.06))
    bevel(canvas, box, t, '#080000', palette[1], palette[2], sunken=True)
    value_text = str(abs(value)).zfill(slots)
    width = (r-x-2*t) / (slots + (.65 if signed else 0))
    if signed and value < 0:
        cy = (y+b)/2
        left = x+t+.08*width
        polygon(canvas, [(left,cy), (left+.08*width,cy-.055*(b-y-2*t)),
                         (left+.47*width,cy-.055*(b-y-2*t)),
                         (left+.55*width,cy), (left+.47*width,cy+.055*(b-y-2*t)),
                         (left+.08*width,cy+.055*(b-y-2*t))], '#ff2424')
    height = b-y-2*t
    active = {'0':'012345', '1':'12', '2':'01643', '3':'01236',
              '4':'5612', '5':'05623', '6':'056432', '7':'012',
              '8':'0123456', '9':'012356', '-':'6'}
    for index, digit in enumerate(value_text):
        left = x+t+(index+(.65 if signed else 0))*width
        # Each horizontal or vertical bar has pointed, separated ends.
        def horizontal(cx, cy):
            return [(cx-.34*width,cy), (cx-.25*width,cy-.055*height),
                    (cx+.25*width,cy-.055*height), (cx+.34*width,cy),
                    (cx+.25*width,cy+.055*height), (cx-.25*width,cy+.055*height)]
        def vertical(cx, cy):
            return [(cx,cy-.19*height), (cx+.08*width,cy-.14*height),
                    (cx+.08*width,cy+.14*height), (cx,cy+.19*height),
                    (cx-.08*width,cy+.14*height), (cx-.08*width,cy-.14*height)]
        cx = left+width/2
        top = y+t
        segments = [horizontal(cx,top+.10*height), vertical(left+.84*width,top+.30*height),
                    vertical(left+.84*width,top+.70*height), horizontal(cx,top+.90*height),
                    vertical(left+.16*width,top+.70*height), vertical(left+.16*width,top+.30*height),
                    horizontal(cx,top+.50*height)]
        for number, points in enumerate(segments):
            polygon(canvas, points, '#ff2424' if str(number) in active[digit] else '#3c0808')


def smiley(canvas, box, status, palette):
    x, y, r, b = box
    s = min(r-x, b-y)
    bevel(canvas, box, max(1, round(s*.08)), *palette)
    cx, cy = (x+r)/2, (y+b)/2
    circle(canvas, cx, cy, .37*s, '#333333')
    circle(canvas, cx, cy, .33*s, '#ffdf31')
    for direction in (-1, 1):
        ex = cx+direction*.12*s
        if status == 'lost':
            for sign in (-1, 1):
                line(canvas, (ex-.045*s,cy-.10*s-sign*.04*s),
                     (ex+.045*s,cy-.10*s+sign*.04*s), '#222222', .035*s)
        else:
            circle(canvas, ex, cy-.10*s, .043*s, '#222222')
    if status == 'won':
        rect(canvas, cx-.25*s, cy-.15*s, .50*s, .11*s, '#222222')
    path = skia.Path()
    path.moveTo(cx-.16*s, cy+.12*s)
    path.quadTo(cx, cy+(-.04 if status == 'lost' else .27)*s, cx+.16*s, cy+.12*s)
    canvas.drawPath(path, skia.Paint(Color=color('#222222'), AntiAlias=True,
                    Style=skia.Paint.kStroke_Style, StrokeWidth=.04*s))


def clock(canvas, x, y, size, ink='#36536a', stopwatch=False):
    cx, cy = x+size/2, y+size*.53
    if stopwatch:
        rect(canvas, cx-.09*size, y+.04*size, .18*size, .10*size, ink)
        line(canvas, (cx+.22*size,y+.16*size), (cx+.30*size,y+.09*size), ink, .07*size)
    circle(canvas, cx, cy, .34*size, ink)
    circle(canvas, cx, cy, .25*size, '#fff4bd' if stopwatch else '#eef6fa')
    line(canvas, (cx,cy), (cx,cy-.17*size), ink, .045*size)
    line(canvas, (cx,cy), (cx+.12*size,cy+.06*size), ink, .045*size)


def icon_number(canvas, box, value, kind, style, variant='classic', ink='#ffffff', icon_side='left'):
    x, y, r, b = box
    h = b-y
    icon_x = x if icon_side == 'left' else r-h
    if kind == 'timer':
        clock(canvas, icon_x, y, h, '#f4c431' if style == 'google' else '#608ba5', style == 'google')
    elif style == 'google':
        draw_flag(canvas, icon_x, y, h, style)
    elif style == 'microsoft':
        mine(canvas, icon_x, y, h, fill='#c7dbe5')
    else:
        draw_mine(canvas, icon_x, y, h, style, variant)
    tx1, tx2 = (x+h, r) if icon_side == 'left' else (x, r-h)
    font = font_for('Arial', h*.57, True)
    text(canvas, str(value), (tx1+tx2)/2, (y+b)/2, font, ink)
