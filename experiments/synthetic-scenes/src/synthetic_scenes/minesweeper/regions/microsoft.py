"""Modern Microsoft-style panel, a procedural interface approximation."""
from ..controls import icon_number
from ..drawing import color, font_for, gradient, rect, text
from ..geometry import cell_gap, scaled_measurement
from . import box, digit_slots

UI_REFERENCE = 24


def layout(grid_size, cell_size, zoom=1.0):
    gw,gh = grid_size
    s = UI_REFERENCE * zoom
    pad = max(scaled_measurement(5, zoom), round(.38*s))
    header = max(scaled_measurement(32, zoom), round(1.8*s))
    ch = max(scaled_measurement(17, zoom), round(header*.52))
    gap = max(scaled_measurement(5, zoom), round(.3*s))
    slots = digit_slots(grid_size,cell_size,cell_gap('microsoft', cell_size))
    cw = ch+round(ch*.4*slots)
    minimum_width = 2*cw+3*gap
    width = max(gw+2*pad,minimum_width)
    gx = (width-gw)//2
    gy = header+pad
    y = (header-ch)//2
    return {'size': (width,gy+gh+pad), 'grid_origin': (gx,gy), 'header': header, 'zoom': zoom, 'minimum_width': minimum_width,
            'mine_counter': box(gap,y,cw,ch), 'timer': box(width-gap-cw,y,cw,ch)}


def draw(canvas, layout, grid_size, theme, variant, status, remaining, elapsed, rows, columns):
    w,h = layout['size']
    header = layout['header']
    top,bottom,accent = {'default':('#30485a','#1b2934','#4aa7dc'),
                         'blue':('#35465f','#222d42','#7199d9'),
                         'warm':('#594c3b','#332d28','#d7a45c')}[theme]
    canvas.clear(color(bottom))
    gradient(canvas,0,0,w,header,top,bottom)
    rule = scaled_measurement(2, layout['zoom'])
    rect(canvas,0,header-rule,w,rule,accent)
    for kind,value in (('mine_counter',remaining),('timer',elapsed)):
        icon_number(canvas,layout[kind],value,'timer' if kind == 'timer' else 'mine',
                    'microsoft',variant)
    left = layout['mine_counter'][2]
    right = layout['timer'][0]
    if right-left > header*1.9:
        text(canvas,'MINESWEEPER',(left+right)/2,header/2,
             font_for('Arial',header*.20,True),'#c6d5de')
