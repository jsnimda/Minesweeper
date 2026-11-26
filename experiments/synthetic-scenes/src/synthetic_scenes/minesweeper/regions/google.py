"""Flat Google-inspired toolbar; its selector and utilities are decorative."""
from ..controls import icon_number
from ..drawing import color, font_for, line, polygon, rounded_rect, text
from ..geometry import scaled_measurement
from . import box, digit_slots

UI_REFERENCE = 20


def layout(grid_size, cell_size, zoom=1.0):
    gw,gh = grid_size
    s = UI_REFERENCE * zoom
    h = max(scaled_measurement(24, zoom), round(1.45*s))
    control_h = max(scaled_measurement(16, zoom), round(h*.68))
    gap = max(scaled_measurement(4, zoom), round(h*.22))
    slots = digit_slots(grid_size,cell_size)
    cw = control_h + round(control_h*.40*slots)
    selector = round(h*2.15)
    utility = round(h*.65)
    minimum_width = 2*gap+selector+gap+2*cw+gap+2*utility
    width = max(gw, minimum_width)
    gx = (width-gw)//2
    y = (h-control_h)//2
    utility_x = width-gap-2*utility
    group_x = min(utility_x-2*cw-gap,
                  max(gap+selector+gap, (width-2*cw-gap)//2))
    return {'size': (width,h+gh), 'grid_origin': (gx,h), 'header': h, 'minimum_width': minimum_width,
            'selector': box(gap,0,selector,h),
            'mine_counter': box(group_x,y,cw,control_h),
            'timer': box(group_x+cw+gap,y,cw,control_h),
            'utility_x': utility_x, 'utility': utility}


def draw(canvas, layout, grid_size, theme, variant, status, remaining, elapsed, rows, columns):
    w,h = layout['size']
    header = layout['header']
    toolbar = {'default':'#4a752c', 'blue':'#386c91', 'warm':'#956c30'}[theme]
    canvas.clear(color(toolbar))
    # The toolbar color also frames any empty space beside a narrow grid.
    x,y,r,b = layout['selector']
    label = {(8,10):'Easy', (14,18):'Medium', (20,24):'Hard'}.get((rows,columns), 'Custom')
    font = font_for('Arial', header*.32, True)
    # Board dimensions deterministically select the alternate white dropdown.
    white_dropdown = (rows + columns) % 2 == 1
    ink = toolbar if white_dropdown else '#ffffff'
    if white_dropdown:
        rounded_rect(canvas,x,header*.19,r-x,header*.62,header*.08,'#ffffff')
    text(canvas,label,x+(r-x)*.42,header/2,font,ink)
    polygon(canvas,[(r-header*.32,header*.44),(r-header*.13,header*.44),
                    (r-header*.225,header*.56)],ink)
    icon_number(canvas,layout['mine_counter'],remaining,'mine','google')
    icon_number(canvas,layout['timer'],elapsed,'timer','google')
    u = layout['utility']
    ux = layout['utility_x']
    cy = header/2
    polygon(canvas,[(ux+u*.15,cy-u*.12),(ux+u*.30,cy-u*.12),
                    (ux+u*.49,cy-u*.28),(ux+u*.49,cy+u*.28),
                    (ux+u*.30,cy+u*.12),(ux+u*.15,cy+u*.12)],'#ffffff')
    line(canvas,(ux+u*.61,cy-u*.15),(ux+u*.61,cy+u*.15),'#ffffff',max(1,u*.05))
    cx = ux+u*1.5
    line(canvas,(cx-u*.18,cy-u*.18),(cx+u*.18,cy+u*.18),'#ffffff',max(1,u*.08))
    line(canvas,(cx+u*.18,cy-u*.18),(cx-u*.18,cy+u*.18),'#ffffff',max(1,u*.08))
