
"""Procedural calculator content rendered with Skia."""

import random

import skia

from ..scaling import pixels, validate_scale



THEMES = {
    "light": {
        "background": "#F3F5F8",
        "display": "#F3F5F8",
        "button": "#FFFFFF",
        "function": "#E5EAF0",
        "operator": "#E3EAF4",
        "equals": "#477FC0",
        "equals_text": "#FFFFFF",
        "text": "#253347",
        "muted": "#7F8C9E",
        "border": "#DCE2EA",
    },
    "dark": {
        "background": "#20242B",
        "display": "#20242B",
        "button": "#343A43",
        "function": "#2B3038",
        "operator": "#3C4654",
        "equals": "#568DCB",
        "equals_text": "#FFFFFF",
        "text": "#F1F3F7",
        "muted": "#A0AAB8",
        "border": "#414852",
    },
    "classic": {
        "background": "#E8EDF3",
        "display": "#F9FCFF",
        "button": "#F8FAFD",
        "function": "#DCE5EF",
        "operator": "#D8E4F0",
        "equals": "#E9AD55",
        "equals_text": "#293648",
        "text": "#24334B",
        "muted": "#6F7D90",
        "border": "#ADBACA",
    },
}


def _color(value: str) -> int:
    value = value.lstrip("#")
    return skia.ColorSetRGB(
        *(int(value[i:i + 2], 16) for i in (0, 2, 4))
    )


def _paint(color: str, *, stroke: float = 0) -> skia.Paint:
    paint = skia.Paint(
        Color=_color(color),
        AntiAlias=True,
    )
    if stroke:
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(stroke)
    return paint


def _rect(canvas, x, y, w, h, color, radius=0):
    shape = skia.Rect.MakeXYWH(x, y, w, h)
    if radius:
        canvas.drawRoundRect(shape, radius, radius, _paint(color))
    else:
        canvas.drawRect(shape, _paint(color))


def _text(
    canvas,
    value,
    x,
    y,
    size=14,
    color="#253347",
    bold=False,
):
    style = (
        skia.FontStyle.Bold()
        if bold
        else skia.FontStyle.Normal()
    )
    typeface = skia.Typeface.MakeFromName(None, style)
    font = skia.Font(typeface, size)

    canvas.drawString(
        str(value), x, y, font, _paint(color)
    )
    return font


def _right_text(
    canvas,
    value,
    right,
    baseline,
    max_width,
    size,
    color,
    bold=False,
):
    """Draw right-aligned text, reducing font size if needed."""
    value = str(value)

    while size > 12:
        style = (
            skia.FontStyle.Bold()
            if bold
            else skia.FontStyle.Normal()
        )
        typeface = skia.Typeface.MakeFromName(None, style)
        font = skia.Font(typeface, size)

        if font.measureText(value) <= max_width:
            break

        size -= 1

    font = _text(
        canvas,
        value,
        right - font.measureText(value),
        baseline,
        size,
        color,
        bold,
    )
    return font


def _calculator_state(rng: random.Random):
    """Generate a plausible arithmetic display."""
    operation = rng.choice(("+", "-", "×", "÷"))

    if operation == "+":
        a = rng.randint(10, 9999)
        b = rng.randint(10, 9999)
        result = a + b

    elif operation == "-":
        a = rng.randint(100, 9999)
        b = rng.randint(1, a)
        result = a - b

    elif operation == "×":
        a = rng.randint(2, 999)
        b = rng.randint(2, 99)
        result = a * b

    else:
        b = rng.randint(2, 50)
        result = rng.randint(2, 999)
        a = b * result

    expression = f"{a:,} {operation} {b:,} ="
    display = f"{result:,}"

    return expression, display


BUTTONS = [
    ("%", "CE", "C", "⌫"),
    ("1/x", "x²", "√x", "÷"),
    ("7", "8", "9", "×"),
    ("4", "5", "6", "−"),
    ("1", "2", "3", "+"),
    ("±", "0", ".", "="),
]


def render_calculator(
    seed: int = 42,
    *,
    width: int = 340,
    height: int = 520,
    theme: str | None = None,
    scale: float = 1.0,
) -> tuple[skia.Image, dict]:
    """Render a deterministic calculator application.

    Returns a Skia image and no Minesweeper annotations.
    """
    if width < 260 or height < 420:
        raise ValueError("Calculator canvas is too small")

    rng = random.Random(seed)

    if theme is None:
        theme = rng.choices(
            ("light", "dark", "classic"),
            weights=(55, 30, 15),
            k=1,
        )[0]

    if theme not in THEMES:
        raise ValueError(f"Unknown calculator theme: {theme}")

    colors = THEMES[theme]
    expression, display_value = _calculator_state(rng)

    scale = validate_scale(scale)
    surface = skia.Surface(pixels(width, scale), pixels(height, scale))
    canvas = surface.getCanvas()
    canvas.scale(scale, scale)
    canvas.clear(_color(colors["background"]))

    # Calculator heading.
    _text(
        canvas,
        "Standard",
        20, 38,
        18,
        colors["text"],
        bold=True,
    )

    # Draw vector menu icon to avoid unsupported glyphs.
    for i in range(3):
        _rect(
            canvas,
            width - 38,
            23 + i * 5,
            17, 2,
            colors["muted"],
            1,
        )

    # Calculation history and result.
    _right_text(
        canvas,
        expression,
        width - 22,
        102,
        width - 44,
        14,
        colors["muted"],
    )

    _right_text(
        canvas,
        display_value,
        width - 22,
        160,
        width - 44,
        43,
        colors["text"],
        bold=True,
    )

    # Memory controls.
    memory = ("MC", "MR", "M+", "M−", "MS")

    for i, label in enumerate(memory):
        x = 23 + i * (width - 46) / 5
        _text(
            canvas,
            label,
            x, 202,
            11,
            colors["muted"],
        )

    # Button grid.
    gap = 5
    margin_x = 9
    bottom_margin = 9
    grid_top = 220

    rows = len(BUTTONS)
    columns = 4

    button_width = (
        width - 2 * margin_x - gap * (columns - 1)
    ) / columns

    button_height = (
        height - grid_top - bottom_margin
        - gap * (rows - 1)
    ) / rows

    for row, labels in enumerate(BUTTONS):
        for col, label in enumerate(labels):
            x = margin_x + col * (button_width + gap)
            y = grid_top + row * (button_height + gap)

            if label == "=":
                background = colors["equals"]
                foreground = colors["equals_text"]

            elif label in ("÷", "×", "−", "+"):
                background = colors["operator"]
                foreground = colors["text"]

            elif row < 2:
                background = colors["function"]
                foreground = colors["text"]

            else:
                background = colors["button"]
                foreground = colors["text"]

            _rect(
                canvas,
                x, y,
                button_width, button_height,
                background,
                6,
            )

            if label == "⌫":
                # Draw a backspace symbol directly with Skia.
                cx = x + button_width / 2
                cy = y + button_height / 2

                paint = _paint(foreground, stroke=1.8)

                path = skia.Path()
                path.moveTo(cx - 11, cy)
                path.lineTo(cx - 4, cy - 8)
                path.lineTo(cx + 11, cy - 8)
                path.lineTo(cx + 11, cy + 8)
                path.lineTo(cx - 4, cy + 8)
                path.close()

                canvas.drawPath(path, paint)
                canvas.drawLine(cx, cy - 4, cx + 7, cy + 4, paint)
                canvas.drawLine(cx + 7, cy - 4, cx, cy + 4, paint)
                continue

            # Center button labels using font measurements.
            font = skia.Font(
                skia.Typeface.MakeFromName(
                    None, skia.FontStyle.Normal()
                ),
                18 if len(label) == 1 else 14,
            )

            label_size = 18 if len(label) == 1 else 14
            text_width = font.measureText(label)

            _text(
                canvas,
                label,
                x + (button_width - text_width) / 2,
                y + button_height / 2 + label_size * 0.35,
                label_size,
                foreground,
            )

    return surface.makeImageSnapshot(), {}
