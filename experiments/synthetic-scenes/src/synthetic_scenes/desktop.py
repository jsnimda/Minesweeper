"""Lightweight desktop and application chrome, independent of Minesweeper."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .wallpaper import render_procedural_wallpaper, sample_procedural_subtype


CHROME = {
    'classic': (4, 26, 20),
    'aero': (7, 32, 22),
    'modern': (1, 32, 0),
    'browser': (1, 74, 0),
    'none': (0, 0, 0),
}


def font(size=14):
    for name in ('Arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def chrome_size(content_size, chrome):
    border, title, menu = CHROME[chrome]
    w, h = content_size
    return w + 2*border, h + 2*border + title + menu


def content_origin(chrome):
    border, title, menu = CHROME[chrome]
    return border, border + title + menu


def wrap_window(content, chrome, title='Minesweeper'):
    if chrome == 'none':
        return content.copy()
    size = chrome_size(content.size, chrome)
    x, y = content_origin(chrome)
    border, title_h, menu_h = CHROME[chrome]
    background = '#c0c0c0' if chrome == 'classic' else '#dae5ef'
    image = Image.new('RGB', size, background)
    draw = ImageDraw.Draw(image)
    w, h = size
    dark = chrome in ('classic', 'aero')
    title_color = '#183e83' if chrome == 'classic' else '#41688c' if dark else '#f0f2f5'
    ink = 'white' if dark else '#242424'
    draw.rectangle((border, border, w-border-1, border+title_h-1), fill=title_color)
    draw.text((border+9, border+6), title[:max(1, (w-105)//8)], font=font(), fill=ink)
    for i, symbol in enumerate(('×', '□', '–')):
        right = w-border-4-i*24
        left = right-21
        if left < border:
            break
        fill = '#b93535' if i == 0 else '#e4e8ee'
        draw.rectangle((left, border+3, right, border+title_h-7 if chrome != 'browser' else border+24), fill=fill)
        draw.text((left+5, border+3), symbol, fill='white' if i == 0 else '#222222', font=font())
    if menu_h:
        draw.text((border+8, border+title_h+2), 'Game     Help', fill='#252525', font=font(13))
    if chrome == 'browser':
        draw.rounded_rectangle((10, 35, w-11, 63), radius=12, fill='#ffffff')
        draw.text((22, 42), 'Minesweeper', fill='#50545a', font=font(13))
    image.paste(content, (x, y))
    return image


def wallpaper(size, rng, path=None):
    if path is not None:
        with Image.open(Path(path)) as image:
            return ImageOps.fit(image.convert('RGB'), size)
    subtype = sample_procedural_subtype(rng)
    return render_procedural_wallpaper(size, rng, subtype)


def draw_taskbar(image):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    top = h - 40
    draw.rectangle((0, top, w, h), fill='#252d38')
    draw.rectangle((12, top+9, 32, top+29), fill='#8bc8f4')
    draw.line((22, top+9, 22, top+29), fill='#252d38', width=2)
    draw.line((12, top+19, 32, top+19), fill='#252d38', width=2)
    for i, fill in enumerate(('#d8b65e', '#70a3d8', '#89ba93')):
        left = 60+40*i
        draw.rounded_rectangle((left, top+8, left+22, top+30), radius=4, fill=fill)
    draw.text((max(0,w-70), top+12), '12:34', fill='#e1e4ea', font=font(12))


def background_window(size, rng):
    """Generic non-game content: text, browser, or file list."""
    w, h = size
    content = Image.new('RGB', (w, h), '#ffffff')
    draw = ImageDraw.Draw(content)
    kind = rng.choice(('Documents', 'Files', 'Notes'))
    if kind == 'Files':
        draw.rectangle((0, 0, w*.25, h), fill='#e9edf2')
        start = int(w*.3)
    else:
        start = 18
    for y in range(22, h-12, 23):
        length = rng.randint(max(1, (w-start)//4), max(1,w-start-12))
        draw.rounded_rectangle((start, y, start+length, y+5), radius=2, fill=rng.choice(('#ccd3da', '#aeb8c2', '#e0e5ea')))
    return wrap_window(content, 'modern', kind)
