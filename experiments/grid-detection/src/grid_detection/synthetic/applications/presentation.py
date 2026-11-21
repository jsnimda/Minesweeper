
"""Procedural presentation editor rendered with Skia."""

import random

import skia
from PIL import Image

from ..desktop import surface_to_pillow


def _color(value: str) -> int:
    value = value.lstrip("#")
    return skia.ColorSetRGB(
        *(int(value[i:i + 2], 16) for i in (0, 2, 4))
    )


def _paint(color: str, *, stroke: float = 0) -> skia.Paint:
    paint = skia.Paint(Color=_color(color), AntiAlias=True)
    if stroke:
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(stroke)
    return paint


def _rect(canvas, x, y, w, h, color, radius=0):
    shape = skia.Rect.MakeXYWH(x, y, w, h)
    if radius:
        canvas.drawRoundRect(
            shape, radius, radius, _paint(color)
        )
    else:
        canvas.drawRect(shape, _paint(color))


def _line(canvas, x1, y1, x2, y2, color="#DCE3EB", width=1):
    canvas.drawLine(
        x1, y1, x2, y2,
        _paint(color, stroke=width),
    )


def _text(canvas, value, x, y, size=13, color="#283446",
          bold=False):
    style = (
        skia.FontStyle.Bold()
        if bold else skia.FontStyle.Normal()
    )
    typeface = skia.Typeface.MakeFromName(None, style)
    font = skia.Font(typeface, size)
    canvas.drawString(str(value), x, y, font, _paint(color))


def _fit_text(canvas, value, x, y, max_width,
              size=13, color="#283446", bold=False):
    canvas.save()
    canvas.clipRect(
        skia.Rect.MakeXYWH(
            x, y - size - 5, max_width, size + 10
        )
    )
    _text(canvas, value, x, y, size, color, bold)
    canvas.restore()


THEMES = [
    {
        "accent": "#3979BD",
        "accent2": "#83B9E1",
        "light": "#E9F3FC",
        "dark": "#203D61",
    },
    {
        "accent": "#4D9C83",
        "accent2": "#98C9B7",
        "light": "#E9F6F0",
        "dark": "#245947",
    },
    {
        "accent": "#A376B9",
        "accent2": "#CFB0DD",
        "light": "#F3EBF8",
        "dark": "#573568",
    },
    {
        "accent": "#D08958",
        "accent2": "#E8BA92",
        "light": "#FBF0E7",
        "dark": "#74462D",
    },
]

PRESENTATIONS = [
    {
        "name": "Quarterly Business Review",
        "slides": [
            ("Quarterly Business Review", "title"),
            ("Key Metrics", "metrics"),
            ("Performance Overview", "chart"),
            ("Strategic Priorities", "bullets"),
            ("Next Steps", "bullets"),
        ],
    },
    {
        "name": "Project Kickoff",
        "slides": [
            ("Project Kickoff", "title"),
            ("Project Goals", "bullets"),
            ("Timeline", "timeline"),
            ("Team Responsibilities", "metrics"),
            ("Upcoming Milestones", "timeline"),
        ],
    },
    {
        "name": "Research Findings",
        "slides": [
            ("Research Findings", "title"),
            ("Research Questions", "bullets"),
            ("Experimental Results", "chart"),
            ("Key Findings", "metrics"),
            ("Recommendations", "bullets"),
        ],
    },
]


def _draw_slide(
    canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    layout: str,
    theme: dict,
    seed: int,
):
    """Draw one 16:9 slide, scalable for canvas or thumbnail."""
    rng = random.Random(seed)

    canvas.save()
    canvas.clipRect(skia.Rect.MakeXYWH(x, y, w, h))
    canvas.translate(x, y)

    # Design in a fixed 960 x 540 coordinate system.
    canvas.scale(w / 960, h / 540)

    accent = theme["accent"]
    dark = theme["dark"]
    light = theme["light"]

    _rect(canvas, 0, 0, 960, 540, "#FFFFFF")

    if layout == "title":
        _rect(canvas, 0, 0, 960, 14, accent)
        _rect(canvas, 690, 0, 270, 540, light)

        # Decorative geometric shapes.
        _rect(canvas, 746, 95, 145, 145, accent, 20)
        _rect(canvas, 805, 205, 110, 110, theme["accent2"], 18)
        _rect(canvas, 723, 336, 183, 21, accent, 10)

        _fit_text(
            canvas, title,
            75, 241, 585,
            47, dark, True,
        )
        _text(
            canvas,
            "Overview and key insights",
            78, 288, 23, "#7B8A9B",
        )
        _rect(canvas, 78, 329, 100, 7, accent, 3)

    else:
        _rect(canvas, 0, 0, 960, 12, accent)

        _fit_text(
            canvas, title,
            68, 98, 835,
            39, dark, True,
        )
        _rect(canvas, 68, 119, 115, 6, accent, 3)

        if layout == "bullets":
            lines = [
                "Improve collaboration across teams",
                "Focus on measurable outcomes",
                "Simplify everyday workflows",
                "Review progress and priorities",
            ]
            rng.shuffle(lines)

            for i, item in enumerate(lines):
                yy = 193 + i * 70
                canvas.drawCircle(
                    94, yy - 7, 8, _paint(accent)
                )
                _text(
                    canvas, item,
                    124, yy, 25, dark,
                )

            _rect(canvas, 750, 168, 133, 258, light, 16)
            for i, height in enumerate((76, 133, 98)):
                _rect(
                    canvas,
                    770 + i * 34,
                    393 - height,
                    20, height,
                    accent if i == 1 else theme["accent2"],
                    4,
                )

        elif layout == "metrics":
            labels = ["Revenue", "Growth", "Projects"]
            values = [
                f"${rng.randint(2, 9)}.{rng.randint(1, 9)}M",
                f"+{rng.randint(8, 36)}%",
                str(rng.randint(20, 180)),
            ]

            for i, (label, value) in enumerate(zip(labels, values)):
                xx = 67 + i * 299

                _rect(
                    canvas, xx, 190,
                    270, 207, light, 17,
                )
                _text(
                    canvas, label,
                    xx + 25, 242, 23, "#778598",
                )
                _text(
                    canvas, value,
                    xx + 25, 324, 50, dark, True,
                )
                _rect(
                    canvas, xx + 25, 354,
                    105, 8, accent, 4,
                )

        elif layout == "chart":
            chart_x = 82
            chart_y = 182
            chart_w = 786
            chart_h = 250

            for i in range(5):
                yy = chart_y + i * chart_h / 4
                _line(
                    canvas, chart_x, yy,
                    chart_x + chart_w, yy,
                    "#DFE6EF",
                )

            bar_count = 9
            gap = 24
            bar_w = (chart_w - gap * (bar_count - 1)) / bar_count

            for i in range(bar_count):
                bar_h = rng.randint(75, 224)
                xx = chart_x + i * (bar_w + gap)
                yy = chart_y + chart_h - bar_h

                _rect(
                    canvas, xx, yy, bar_w, bar_h,
                    accent if i in (3, 7) else theme["accent2"],
                    5,
                )

        elif layout == "timeline":
            _line(
                canvas, 125, 288,
                833, 288, accent, 5,
            )

            phases = [
                "Plan", "Design", "Build", "Review"
            ]

            for i, label in enumerate(phases):
                xx = 137 + i * 221

                canvas.drawCircle(
                    xx, 288, 15, _paint(accent)
                )
                canvas.drawCircle(
                    xx, 288, 6, _paint("#FFFFFF")
                )

                _text(
                    canvas, label,
                    xx - 32, 249,
                    22, dark, True,
                )
                _text(
                    canvas, f"Phase {i + 1}",
                    xx - 36, 345,
                    17, "#7C8999",
                )

        _text(
            canvas, "CONFIDENTIAL",
            69, 495, 13, "#A6B0BD",
        )

    canvas.restore()


def render_presentation(
    seed: int = 42,
    *,
    width: int = 900,
    height: int = 620,
) -> tuple[Image.Image, dict]:
    """Render a deterministic slide editor.

    Returns a Pillow image and no Minesweeper target boxes.
    """
    if width < 640 or height < 460:
        raise ValueError("Presentation canvas is too small")

    rng = random.Random(seed)
    presentation = rng.choice(PRESENTATIONS)
    theme = rng.choice(THEMES)

    slides = presentation["slides"]
    selected = rng.randrange(len(slides))

    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    canvas.clear(_color("#E9EDF2"))

    dark = "#293648"
    muted = "#788699"
    accent = theme["accent"]

    # Ribbon and file title.
    _rect(canvas, 0, 0, width, 104, "#FFFFFF")

    _text(
        canvas, presentation["name"],
        20, 25, 15, dark, True,
    )
    _text(
        canvas,
        "File     Home     Insert     Design     Transitions     View",
        20, 50, 11, muted,
    )
    _line(canvas, 0, 60, width, 60)

    # Draw toolbar controls as vector shapes.
    for i, label in enumerate(
        ("New Slide", "Layout", "Text", "Shape", "Color")
    ):
        xx = 21 + i * 105

        _rect(
            canvas, xx, 69, 21, 19,
            "#E9F1FB", 4,
        )
        _rect(
            canvas, xx + 5, 74, 11, 2,
            accent,
        )
        _rect(
            canvas, xx + 5, 79, 8, 2,
            accent,
        )
        _text(
            canvas, label,
            xx + 28, 83,
            11, dark,
        )

    _line(canvas, 0, 103, width, 103)

    # Slide thumbnail sidebar.
    sidebar_w = 174
    status_h = 29
    _rect(
        canvas, 0, 104,
        sidebar_w, height - 104 - status_h,
        "#F6F8FA",
    )
    _line(
        canvas, sidebar_w, 104,
        sidebar_w, height - status_h,
    )

    thumb_x = 34
    thumb_w = 125
    thumb_h = thumb_w * 9 / 16
    thumb_spacing = 85

    canvas.save()
    canvas.clipRect(
        skia.Rect.MakeXYWH(
            0, 104, sidebar_w,
            height - status_h - 104,
        )
    )

    for i, (title, layout) in enumerate(slides):
        y = 121 + i * thumb_spacing

        if selected == i:
            _rect(
                canvas,
                7, y - 5,
                sidebar_w - 14, thumb_h + 13,
                "#E2EEF9", 7,
            )

        _text(
            canvas, str(i + 1),
            15, y + 15, 11, muted,
        )

        _rect(
            canvas,
            thumb_x - 2, y - 2,
            thumb_w + 4, thumb_h + 4,
            accent if selected == i else "#CBD4DF",
            3,
        )

        _draw_slide(
            canvas,
            thumb_x, y,
            thumb_w, thumb_h,
            title, layout, theme,
            seed * 100 + i,
        )

    canvas.restore()

    # Main editing canvas.
    workspace_x = sidebar_w
    workspace_w = width - workspace_x
    workspace_h = height - 104 - status_h

    padding_x = 36
    padding_y = 34

    max_w = workspace_w - 2 * padding_x
    max_h = workspace_h - 2 * padding_y

    slide_w = min(max_w, max_h * 16 / 9)
    slide_h = slide_w * 9 / 16

    slide_x = workspace_x + (workspace_w - slide_w) / 2
    slide_y = 104 + (workspace_h - slide_h) / 2

    # Slide shadow.
    _rect(
        canvas, slide_x + 4,
        slide_y + 6,
        slide_w, slide_h,
        "#CBD3DE", 2,
    )

    title, layout = slides[selected]
    _draw_slide(
        canvas,
        slide_x, slide_y,
        slide_w, slide_h,
        title, layout, theme,
        seed * 100 + selected,
    )

    # Bottom status bar.
    _rect(
        canvas, 0, height - status_h,
        width, status_h,
        "#FFFFFF",
    )
    _line(
        canvas,
        0, height - status_h,
        width, height - status_h,
    )

    _text(
        canvas,
        f"Slide {selected + 1} of {len(slides)}",
        18, height - 10,
        11, muted,
    )
    _text(
        canvas, "100%",
        width - 57, height - 10,
        11, muted,
    )

    return surface_to_pillow(surface), {}
