"""Flat checkerboard with an optional covered-side boundary."""

from ..drawing import rect, mix


THEMES = {
    'default': (('#aad751', '#a2d149'), ('#e5c29f', '#d7b899')),
    'blue': (('#80b8de', '#76afd5'), ('#e8e0cf', '#ddd4c2')),
    'warm': (('#d6a65d', '#cda055'), ('#ece0c5', '#e2d5b9')),
}


def draw_surface(canvas, x, y, s, opened, row, col, theme, neighbors):
    hidden, revealed = THEMES[theme]
    palette = revealed if opened else hidden
    rect(canvas, x, y, s, s, palette[(row + col) % 2])


def draw_boundary(canvas, x, y, s, theme, neighbors, ratio):
    """Draw a thin strip inside a covered cell beside each opened neighbor."""
    if ratio == 0:
        return
    hidden, _ = THEMES[theme]
    shade = mix(hidden[1], '#000000', .17)
    t = s * ratio
    if neighbors['top']:
        rect(canvas, x, y, s, t, shade)
    if neighbors['bottom']:
        rect(canvas, x, y+s-t, s, t, shade)
    if neighbors['left']:
        rect(canvas, x, y, t, s, shade)
    if neighbors['right']:
        rect(canvas, x+s-t, y, t, s, shade)
