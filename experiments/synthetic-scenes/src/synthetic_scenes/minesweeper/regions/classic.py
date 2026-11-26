"""Traditional raised frame, sunken header and LED counters."""
from ..controls import bevel, digital, smiley
from ..drawing import color
from ..styles.classic import THEMES
from ..geometry import scaled_measurement
from . import box

UI_REFERENCE = 24


def layout(grid_size, cell_size, zoom=1.0):
    gw, gh = grid_size
    s = UI_REFERENCE * zoom
    # The sign has its own area in the LED display.
    slots = max(3, len(str((gw//cell_size)*(gh//cell_size))))
    edge = max(scaled_measurement(2, zoom), round(.18*s))
    space = max(scaled_measurement(3, zoom), round(.28*s))
    h = max(scaled_measurement(16, zoom), round(1.05*s))
    digit_width = max(scaled_measurement(5, zoom), round(h*.52))
    cw = slots*digit_width + round(digit_width*.65) + 2*max(scaled_measurement(1, zoom), round(h*.06))
    timer_w = 3*digit_width + 2*max(scaled_measurement(1, zoom), round(h*.06))
    reset_size = max(scaled_measurement(16, zoom), round(1.55*s))
    minimum_width = 2*cw+reset_size+4*space+2*edge
    width = max(gw+2*(edge+space), minimum_width)
    header_y = edge+space
    header_h = reset_size+2*space
    gy = header_y+header_h+space+edge
    height = gy+gh+space+edge
    cy = header_y+space+(reset_size-h)//2
    gx = (width-gw)//2
    return {'size': (width,height), 'grid_origin': (gx,gy), 'slots': slots, 'minimum_width': minimum_width,
            'edge': edge, 'header_box': box(edge,header_y-space//2,width-2*edge,header_h+space),
            'mine_counter': box(edge+space,cy,cw,h),
            'timer': box(width-edge-space-timer_w,cy,timer_w,h),
            'reset_button': box((width-reset_size)//2,header_y+space,reset_size,reset_size)}


def draw(canvas, layout, grid_size, theme, variant, status, remaining, elapsed, rows, columns):
    palette = THEMES[theme]
    w,h = layout['size']
    gx,gy = layout['grid_origin']
    gw,gh = grid_size
    edge = layout['edge']
    canvas.clear(color(palette[0]))
    bevel(canvas, (0,0,w,h), edge, *palette)
    bevel(canvas, layout['header_box'], max(1,edge//2), *palette, sunken=True)
    bevel(canvas, (gx-edge,gy-edge,gx+gw+edge,gy+gh+edge), edge, *palette, sunken=True)
    digital(canvas, layout['mine_counter'], remaining, layout['slots'], palette, signed=True)
    digital(canvas, layout['timer'], elapsed, 3, palette)
    smiley(canvas, layout['reset_button'], status, palette)
