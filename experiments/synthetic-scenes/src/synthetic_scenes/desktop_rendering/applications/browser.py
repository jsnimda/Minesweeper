
"""Procedural browser content for synthetic desktop screenshots."""

import random

import skia

from ..scaling import pixels, validate_scale



def _color(value):
    value = value.lstrip("#")
    return skia.ColorSetRGB(
        *(int(value[i:i + 2], 16) for i in (0, 2, 4))
    )


def _paint(color, *, stroke=0):
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


def _line(canvas, x1, y1, x2, y2, color="#E1E6ED"):
    canvas.drawLine(
        x1, y1, x2, y2,
        _paint(color, stroke=1),
    )


def _text(canvas, value, x, y, size=13, color="#283445"):
    canvas.drawString(
        str(value),
        x, y,
        skia.Font(None, size),
        _paint(color),
    )


def _clipped_text(canvas, value, x, y, max_width,
                  size=13, color="#283445"):
    canvas.save()
    canvas.clipRect(
        skia.Rect.MakeXYWH(
            x, y - size - 3, max_width, size + 8
        )
    )
    _text(canvas, value, x, y, size, color)
    canvas.restore()


def _placeholder_lines(canvas, x, y, width, rng,
                       count=3, spacing=19):
    """Draw abstract text lines with variable lengths."""
    for i in range(count):
        factor = rng.uniform(0.55, 0.98)
        _rect(
            canvas, x, y + i * spacing,
            width * factor, 7,
            "#E4E9EF", 3,
        )


def _search_page(canvas, width, height, rng):
    topics = [
        "Python data visualization",
        "How to organize project files",
        "Introduction to machine learning",
        "Modern desktop interface design",
        "Understanding grid algorithms",
    ]

    topic = rng.choice(topics)
    left = max(35, int(width * 0.08))
    content_width = min(650, width - left - 35)

    _text(canvas, "Search", left, 113, 25, "#3977C5")

    _rect(
        canvas, left, 133, content_width, 46,
        "#F7F9FC", 22,
    )
    _clipped_text(
        canvas, topic, left + 18, 162,
        content_width - 36, 14,
    )

    _text(
        canvas, "All     Images     News     Videos",
        left + 7, 208, 12, "#718093",
    )
    _line(
        canvas, left, 221,
        left + content_width, 221,
    )

    titles = [
        f"{topic} — Getting started",
        "A practical guide and examples",
        "Documentation and useful resources",
        "Common questions and explanations",
        "Tips, references and tutorials",
    ]

    y = 249
    for i, title in enumerate(titles):
        if y + 95 > height:
            break

        _text(
            canvas,
            f"example{i + 1}.org  /  resources",
            left, y, 11, "#54806C",
        )
        _clipped_text(
            canvas, title,
            left, y + 27, content_width,
            17, "#2765B0",
        )
        _placeholder_lines(
            canvas, left, y + 41,
            content_width * 0.83, rng, 2, 17,
        )
        y += 111


def _article_page(canvas, width, height, rng):
    titles = [
        "Getting started with Python",
        "A guide to digital workflows",
        "Understanding visual interfaces",
        "Building reliable applications",
    ]

    title = rng.choice(titles)
    left = max(32, int(width * 0.09))
    main_width = min(620, width - left - 38)

    _text(
        canvas, "DOCUMENTATION",
        left, 112, 11, "#487C9B",
    )
    _clipped_text(
        canvas, title, left, 159,
        main_width, 27, "#253247",
    )
    _text(
        canvas, "Updated recently  ·  8 min read",
        left, 185, 12, "#8995A5",
    )

    _line(canvas, left, 205, left + main_width, 205)

    y = 231
    sections = [
        "Overview",
        "Key concepts",
        "Examples",
        "Next steps",
    ]

    for heading in sections:
        if y + 100 > height:
            break

        _text(
            canvas, heading, left, y,
            18, "#293648",
        )
        _placeholder_lines(
            canvas, left, y + 17,
            main_width, rng, 3, 20,
        )
        y += 120


def _dashboard_page(canvas, width, height, rng):
    left = 32
    right = width - 32
    usable = right - left

    _text(
        canvas, "Overview",
        left, 120, 24, "#293648",
    )
    _text(
        canvas, "Workspace dashboard",
        left, 145, 12, "#8793A2",
    )

    gap = 12
    card_w = (usable - gap * 2) / 3

    metrics = [
        ("Visitors", f"{rng.randint(12, 98):,}K"),
        ("Projects", str(rng.randint(12, 250))),
        ("Completion", f"{rng.randint(55, 96)}%"),
    ]

    for i, (name, value) in enumerate(metrics):
        x = left + i * (card_w + gap)

        _rect(
            canvas, x, 170,
            card_w, 94, "#F4F7FA", 9,
        )
        _text(canvas, name, x + 15, 194, 12, "#78869A")
        _text(canvas, value, x + 15, 235, 26, "#2B3D52")

    chart_top = 295
    chart_h = min(230, max(100, height - chart_top - 65))

    _rect(
        canvas, left, chart_top,
        usable, chart_h,
        "#F8FAFC", 9,
    )

    _text(
        canvas, "Activity",
        left + 18, chart_top + 27,
        15, "#293648",
    )

    chart_left = left + 27
    chart_right = right - 23
    chart_bottom = chart_top + chart_h - 25
    chart_start = chart_top + 53

    # Background chart grid.
    for i in range(4):
        y = chart_start + (chart_bottom - chart_start) * i / 3
        _line(
            canvas, chart_left, y,
            chart_right, y, "#E3EAF1",
        )

    # Original vector line chart.
    path = skia.Path()
    count = 12

    for i in range(count):
        x = chart_left + (
            chart_right - chart_left
        ) * i / (count - 1)

        value = rng.uniform(0.20, 0.84)
        y = chart_bottom - value * (
            chart_bottom - chart_start
        )

        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)

    canvas.drawPath(
        path, _paint("#4A91C6", stroke=2.5)
    )


def render_browser(
    seed: int = 42,
    *,
    width: int = 850,
    height: int = 580,
    page_type: str | None = None,
    scale: float = 1.0,
) -> tuple[skia.Image, dict]:
    """Render a synthetic browser and return no target boxes."""
    if width < 500 or height < 360:
        raise ValueError("Browser dimensions are too small")

    rng = random.Random(seed)

    if page_type is None:
        page_type = rng.choices(
            ["search", "article", "dashboard"],
            weights=[40, 35, 25],
            k=1,
        )[0]

    if page_type not in {"search", "article", "dashboard"}:
        raise ValueError(f"Unknown page type: {page_type}")

    scale = validate_scale(scale)
    surface = skia.Surface(pixels(width, scale), pixels(height, scale))
    canvas = surface.getCanvas()
    canvas.scale(scale, scale)
    canvas.clear(_color("#FFFFFF"))

    # Browser tab strip.
    _rect(canvas, 0, 0, width, 43, "#E9EEF4")
    _rect(canvas, 12, 8, 218, 35, "#FFFFFF", 8)

    titles = {
        "search": "Search results",
        "article": "Documentation",
        "dashboard": "Dashboard",
    }

    _clipped_text(
        canvas, titles[page_type],
        28, 31, 175, 12,
    )
    _text(canvas, "×", 207, 30, 14, "#7F8C9C")
    _text(canvas, "+", 246, 31, 17, "#69798B")

    # Navigation and address bar.
    _rect(canvas, 0, 43, width, 48, "#FFFFFF")
    _text(canvas, "‹", 19, 74, 23, "#687B91")
    _text(canvas, "›", 52, 74, 23, "#687B91")
    _text(canvas, "↻", 83, 72, 17, "#687B91")

    _rect(
        canvas, 115, 52,
        width - 187, 31,
        "#F3F6F9", 15,
    )

    addresses = {
        "search": "https://search.example/results",
        "article": "https://docs.example/guide",
        "dashboard": "https://workspace.example/overview",
    }

    _clipped_text(
        canvas, addresses[page_type],
        135, 73, width - 230,
        12, "#67778A",
    )
    _text(canvas, "⋮", width - 31, 74, 19, "#65758A")

    _line(canvas, 0, 91, width, 91)

    # Website content.
    if page_type == "search":
        _search_page(canvas, width, height, rng)
    elif page_type == "article":
        _article_page(canvas, width, height, rng)
    else:
        _dashboard_page(canvas, width, height, rng)

    return surface.makeImageSnapshot(), {}
