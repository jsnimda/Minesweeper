"""Windows 7 inspired glass frame and bottom controls."""
from ..controls import icon_number
from ..drawing import gradient, rect
from . import box, digit_slots

UI_REFERENCE = 24


def layout(grid_size, cell_size):
    gw, gh = grid_size
    s = UI_REFERENCE
    pad = max(4, round(.5*s))
    h = max(18, round(1.12*s))
    gap = max(4, round(.25*s))
    slots = digit_slots(grid_size,cell_size)
    cw = h + round(h*.39*slots) + gap
    minimum_width = 2*cw+4*gap
    width = max(gw+2*pad, minimum_width)
    gx = (width-gw)//2
    gy = pad
    cy = gy+gh+gap
    height = cy+h+gap
    return {'size': (width,height), 'grid_origin': (gx,gy), 'pad': pad, 'minimum_width': minimum_width,
            'timer': box(gap,cy,cw,h), 'mine_counter': box(width-gap-cw,cy,cw,h)}


def draw(canvas, layout, grid_size, theme, variant, status, remaining, elapsed, rows, columns):
    w,h = layout['size']
    gx,gy = layout['grid_origin']
    gw,gh = grid_size
    top, bottom, edge = (('#c8ecc4','#4a794c','#28542d') if theme == 'green'
                         else ('#d3efff','#4a6798','#263f6b'))
    gradient(canvas, 0,0,w,h,top,bottom)
    rect(canvas, gx-3,gy-3,gw+6,gh+6,edge)
    rect(canvas, gx-2,gy-2,gw+4,gh+4,'#d7ebf3')
    rect(canvas, gx-1,gy-1,gw+2,gh+2,edge)
    for kind, value in (('timer',elapsed),('mine_counter',remaining)):
        x,y,r,b = layout[kind]
        icon_left = kind == 'timer'
        h0 = b-y
        dx1,dx2 = (x+h0,r) if icon_left else (x,r-h0)
        rect(canvas,dx1,y+2,dx2-dx1,h0-4,edge)
        gradient(canvas,dx1+1,y+3,dx2-dx1-2,h0-6,'#7da6bd','#31536e')
        icon_number(canvas,layout[kind],value,'timer' if icon_left else 'mine',
                    'windows7',variant,icon_side='left' if icon_left else 'right')
