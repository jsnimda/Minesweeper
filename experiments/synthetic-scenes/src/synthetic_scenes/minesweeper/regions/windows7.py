"""Windows 7 inspired glass frame and bottom controls."""
from ..controls import icon_number
from ..drawing import gradient, rect
from ..geometry import scaled_measurement
from . import box, digit_slots

UI_REFERENCE = 24


def layout(grid_size, cell_size, zoom=1.0):
    gw, gh = grid_size
    s = UI_REFERENCE * zoom
    pad = max(scaled_measurement(4, zoom), round(.5*s))
    h = max(scaled_measurement(18, zoom), round(1.12*s))
    gap = max(scaled_measurement(4, zoom), round(.25*s))
    slots = digit_slots(grid_size,cell_size)
    cw = h + round(h*.39*slots) + gap
    minimum_width = 2*cw+4*gap
    width = max(gw+2*pad, minimum_width)
    gx = (width-gw)//2
    gy = pad
    cy = gy+gh+gap
    height = cy+h+gap
    return {'size': (width,height), 'grid_origin': (gx,gy), 'pad': pad, 'zoom': zoom, 'minimum_width': minimum_width,
            'timer': box(gap,cy,cw,h), 'mine_counter': box(width-gap-cw,cy,cw,h)}


def draw(canvas, layout, grid_size, theme, variant, status, remaining, elapsed, rows, columns):
    w,h = layout['size']
    gx,gy = layout['grid_origin']
    gw,gh = grid_size
    top, bottom, edge = (('#c8ecc4','#4a794c','#28542d') if theme == 'green'
                         else ('#d3efff','#4a6798','#263f6b'))
    gradient(canvas, 0,0,w,h,top,bottom)
    for inset, shade in ((3, edge), (2, '#d7ebf3'), (1, edge)):
        inset = scaled_measurement(inset, layout['zoom'])
        rect(canvas, gx-inset,gy-inset,gw+2*inset,gh+2*inset,shade)
    for kind, value in (('timer',elapsed),('mine_counter',remaining)):
        x,y,r,b = layout[kind]
        icon_left = kind == 'timer'
        h0 = b-y
        dx1,dx2 = (x+h0,r) if icon_left else (x,r-h0)
        inset = scaled_measurement(2, layout['zoom'])
        inner = scaled_measurement(1, layout['zoom'])
        rect(canvas,dx1,y+inset,dx2-dx1,max(1,h0-2*inset),edge)
        gradient(canvas,dx1+inner,y+inset+inner,max(1,dx2-dx1-2*inner),
                 max(1,h0-2*(inset+inner)),'#7da6bd','#31536e')
        icon_number(canvas,layout[kind],value,'timer' if icon_left else 'mine',
                    'windows7',variant,icon_side='left' if icon_left else 'right')
