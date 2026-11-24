from ..drawing import rect, polygon

BEVEL_RATIO = .125
THEMES = {
    'default': ('#c6c6c6','#ffffff','#808080'),
    'blue': ('#b8c9dc','#eaf3ff','#6c829e'),
    'green': ('#b9cdb9','#edf9ed','#728b72'),
    'warm': ('#d5c8b2','#fff5e3','#95836b'),
}


def draw_surface(canvas,x,y,s,opened,row,col,theme,neighbors):
    face, highlight, shadow = THEMES[theme]
    if opened:
        rect(canvas,x,y,s,s,shadow)
        t=s/16
        rect(canvas,x+t,y+t,s-t,s-t,face)
    else:
        t=s*BEVEL_RATIO
        rect(canvas,x,y,s,s,shadow)
        polygon(canvas,[(x,y),(x+s,y),(x,y+s)],highlight)
        rect(canvas,x+t,y+t,s-2*t,s-2*t,face)
