"""Game chrome, independent of grid drawing. All boxes are half-open xyxy."""


import numpy as np
import skia

from .drawing import circle, color, font_for, gradient, line, rect, rounded_rect, snapshot, text
from .renderer import render_board
from .geometry import region_layout
from .symbols import draw_flag, draw_mine


def _box(x, y, w, h):
    return (x, y, x + w, y + h)


def _bevel(canvas, box, thickness, face, light, dark, sunken=False):
    x, y, right, bottom = box
    w, h = right-x, bottom-y
    if sunken:
        light, dark = dark, light
    rect(canvas, x, y, w, h, light)
    rect(canvas, x+thickness, y+thickness, w-thickness, h-thickness, dark)
    rect(canvas, x+thickness, y+thickness, w-2*thickness, h-2*thickness, face)


def _digital(canvas, box, value):
    x, y, right, bottom = box
    w, h = right-x, bottom-y
    rect(canvas, x, y, w, h, '#180000')
    value = max(-99, min(999, value))
    digits = f'{value:03d}'
    segments = ((.22,.10,.78,.10),(.82,.16,.82,.46),(.82,.54,.82,.84),
                (.22,.90,.78,.90),(.18,.54,.18,.84),(.18,.16,.18,.46),
                (.22,.50,.78,.50))
    active = {'0':'012345','1':'12','2':'01643','3':'01236','4':'5612',
              '5':'05623','6':'056432','7':'012','8':'0123456','9':'012356','-':'6'}
    dw = w/3
    for index, digit in enumerate(digits):
        for number, (ax,ay,bx,by) in enumerate(segments):
            shade = '#ff2929' if str(number) in active[digit] else '#3b0808'
            left = x+index*dw
            line(canvas, (left+ax*dw,y+ay*h), (left+bx*dw,y+by*h), shade, .09*dw)


def _smiley(canvas, box, status, palette):
    x,y,r,b = box
    s = min(r-x,b-y)
    face, light, dark = palette
    _bevel(canvas, box, max(1,round(s*.07)), face, light, dark)
    cx,cy = x+s/2,y+s/2
    circle(canvas,cx,cy,.37*s,'#242424')
    circle(canvas,cx,cy,.33*s,'#ffdf31')
    for direction in (-1,1):
        ex = cx+direction*.12*s
        if status == 'lost':
            for sign in (-1,1):
                line(canvas,(ex-.045*s,cy-.10*s-sign*.04*s),
                     (ex+.045*s,cy-.10*s+sign*.04*s),'#222222',.035*s)
        else:
            circle(canvas,ex,cy-.10*s,.045*s,'#222222')
    path = skia.Path()
    path.moveTo(cx-.16*s,cy+.12*s)
    bend = -.04 if status == 'lost' else .27
    path.quadTo(cx,cy+bend*s,cx+.16*s,cy+.12*s)
    paint = skia.Paint(Color=color('#222222'), AntiAlias=True,
                       Style=skia.Paint.kStroke_Style, StrokeWidth=.04*s)
    canvas.drawPath(path,paint)
    if status == 'won':
        rect(canvas,cx-.24*s,cy-.15*s,.48*s,.12*s,'#222222')


def _clock(canvas, x, y, s, ink):
    cx,cy=x+s/2,y+s/2
    circle(canvas,cx,cy,.34*s,ink)
    circle(canvas,cx,cy,.27*s,'#f3f5f5')
    line(canvas,(cx,cy),(cx,cy-.17*s),ink,.05*s)
    line(canvas,(cx,cy),(cx+.12*s,cy+.06*s),ink,.05*s)


def _control(canvas, box, value, style, kind, variant, font, theme):
    x,y,r,b=box
    h=b-y
    if style == 'windows7':
        base = '#39713b' if theme == 'green' else '#365f99'
        rounded_rect(canvas,x,y,r-x,h,.12*h,'#e3edf7')
        rounded_rect(canvas,x+2,y+2,r-x-4,h-4,.10*h,base)
        ink='#ffffff'
    elif style == 'microsoft':
        rounded_rect(canvas,x,y,r-x,h,.12*h,'#242c34')
        ink='#ffffff'
    else:
        ink='#ffffff'
    if kind == 'timer':
        _clock(canvas,x+.06*h,y,h, '#334b5c' if style=='windows7' else '#53674c')
    elif style == 'google':
        draw_flag(canvas,x+.06*h,y,h,style)
    else:
        draw_mine(canvas,x+.06*h,y,h,style,variant)
    canvas.save()
    canvas.clipRect(skia.Rect.MakeLTRB(x, y, r, b))
    text(canvas,str(value),(x+h+r)/2,(y+b)/2,font,ink)
    canvas.restore()


def render_game_region(game, **board_options):
    """Render a game panel (not OS title bar), with region-local annotations.

    Accepts the same keyword arguments as render_board. Windows 7 uses
    bottom controls; other styles use top controls. Only Classic exposes
    a reset button in this initial set of procedural panels.
    """
    grid, metadata = render_board(game, **board_options)
    style = metadata['style']
    theme = metadata['theme']
    params = metadata['style_parameters']
    s = params['cell_size']
    variant = params['mine_variant']
    layout = region_layout(grid.size, style, s)
    width, height = layout['size']
    gx, gy = layout['grid_origin']
    pad = layout['pad']
    header = layout['header']
    margin = layout['margin']
    cy = layout['control_y']
    control_w, control_h = layout['control_size']
    left = _box(margin,cy,control_w,control_h)
    right = _box(width-margin-control_w,cy,control_w,control_h)
    surface=skia.Surface(width,height)
    canvas=surface.getCanvas()
    remaining=int(np.count_nonzero(game['board']==-1)-np.count_nonzero(game['flags']))
    elapsed = min(999, int(game['elapsed_seconds']))
    boxes={'game_region':(0,0,width,height),'grid':_box(gx,gy,grid.width,grid.height)}
    if style=='classic':
        from .styles.classic import THEMES
        palette=THEMES[theme]
        canvas.clear(color(palette[0]))
        border=max(1,round(s*.10))
        _bevel(canvas,(0,0,width,height),border,*palette)
        _bevel(canvas,(margin//2,pad//2,width-margin//2,header+pad//2),border,*palette,sunken=True)
        _bevel(canvas,(gx-border,gy-border,gx+grid.width+border,gy+grid.height+border),border,*palette,sunken=True)
        reset=_box((width-control_h)//2,cy,control_h,control_h)
        _digital(canvas,left,remaining)
        _digital(canvas,right,elapsed)
        _smiley(canvas,reset,game['status'],palette)
        boxes['reset_button']=reset
    else:
        if style=='windows7':
            top,bottom=('#c5e3ba','#477743') if theme=='green' else ('#c3e2fb','#344e83')
            gradient(canvas,0,0,width,height,top,bottom)
            # Match original Win7 layout: timer left, mine counter right.
            left,right=right,left
        elif style=='microsoft':
            canvas.clear(color('#30363c'))
            rect(canvas,0,0,width,header,'#414d59')
        else:
            from .styles.google import THEMES
            hidden,_=THEMES[theme]
            canvas.clear(color(hidden[0]))
            rect(canvas,0,0,width,header,'#4a752c' if theme=='default' else '#45636b')
        font=font_for('Arial',control_h*.54,True)
        _control(canvas,left,remaining,style,'mine',variant,font,theme)
        _control(canvas,right,elapsed,style,'timer',variant,font,theme)
    boxes['mine_counter']=left
    boxes['timer']=right
    image=snapshot(surface)
    image.paste(grid,(gx,gy))
    metadata['boxes']=boxes
    metadata['grid_origin']=(gx,gy)
    metadata['x_edges']=[v+gx for v in metadata['x_edges']]
    metadata['y_edges']=[v+gy for v in metadata['y_edges']]
    metadata['cell_boxes']=[[(x+gx,y+gy,r+gx,b+gy) for x,y,r,b in row]
                            for row in metadata['cell_boxes']]
    metadata['coordinate_space']='game_region'
    return image,metadata
