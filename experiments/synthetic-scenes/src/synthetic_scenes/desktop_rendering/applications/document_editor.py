
"""Procedural document editor content rendered with Skia."""

import random

import skia



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
    rect = skia.Rect.MakeXYWH(x, y, w, h)
    if radius:
        canvas.drawRoundRect(rect, radius, radius, _paint(color))
    else:
        canvas.drawRect(rect, _paint(color))


def _line(canvas, x1, y1, x2, y2, color="#DFE4EB"):
    canvas.drawLine(
        x1, y1, x2, y2,
        _paint(color, stroke=1),
    )


def _text(
    canvas, value, x, y,
    size=13, color="#263346", bold=False,
):
    style = (
        skia.FontStyle.Bold()
        if bold
        else skia.FontStyle.Normal()
    )
    font = skia.Font(
        skia.Typeface.MakeFromName(None, style),
        size,
    )
    canvas.drawString(str(value), x, y, font, _paint(color))
    return font


def _paragraph(
    canvas, text, x, y, max_width,
    size=12, line_height=20, color="#374357",
):
    """Draw a wrapped paragraph; return the next baseline."""
    font = skia.Font(None, size)
    words = text.split()
    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if current and font.measureText(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate

    if current:
        lines.append(current)

    for line in lines:
        _text(canvas, line, x, y, size, color)
        y += line_height

    return y


DOCUMENTS = [
    {
        "name": "Project Proposal",
        "subtitle": "Planning and implementation overview",
        "sections": [
            (
                "Executive Summary",
                "This document outlines the proposed approach for "
                "delivering a reliable and maintainable solution. "
                "The work will focus on clear requirements, "
                "incremental development, and measurable results.",
            ),
            (
                "Objectives",
                "Our primary goal is to improve the existing "
                "workflow while reducing unnecessary complexity. "
                "The project will be delivered through a series "
                "of manageable milestones.",
            ),
            (
                "Implementation Plan",
                "The team will begin with a review of the current "
                "system, followed by prototype development, "
                "evaluation, and final integration.",
            ),
        ],
        "bullets": [
            "Define project requirements and priorities",
            "Develop and evaluate the initial prototype",
            "Document results and recommended next steps",
        ],
    },
    {
        "name": "Meeting Notes",
        "subtitle": "Weekly team sync",
        "sections": [
            (
                "Overview",
                "The team reviewed recent progress and discussed "
                "the remaining priorities for the current cycle. "
                "Several improvements are ready for testing.",
            ),
            (
                "Discussion",
                "Participants highlighted the importance of "
                "consistent interfaces, clear documentation, "
                "and repeatable quality checks.",
            ),
            (
                "Action Items",
                "The next phase will focus on completing the "
                "remaining tasks and validating the results "
                "before the next review.",
            ),
        ],
        "bullets": [
            "Review open issues and assign owners",
            "Update the shared project documentation",
            "Prepare a summary for the next meeting",
        ],
    },
    {
        "name": "Research Summary",
        "subtitle": "Methods, findings, and observations",
        "sections": [
            (
                "Introduction",
                "This report summarizes the key observations "
                "from a series of exploratory experiments. "
                "The evaluation considers both practical "
                "performance and implementation complexity.",
            ),
            (
                "Methodology",
                "Multiple configurations were examined using "
                "a consistent evaluation procedure. "
                "Results were recorded and compared across "
                "several representative test cases.",
            ),
            (
                "Findings",
                "The preliminary findings indicate that "
                "incremental improvements can provide useful "
                "benefits without requiring major architectural "
                "changes.",
            ),
        ],
        "bullets": [
            "Repeat experiments with additional samples",
            "Compare results across configurations",
            "Record limitations and future directions",
        ],
    },
]


def render_document_editor(
    seed: int = 42,
    *,
    width: int = 820,
    height: int = 650,
) -> tuple[skia.Image, dict]:
    """Render a deterministic synthetic word processor.

    Returns an image and an empty Minesweeper annotation dictionary.
    """
    if width < 560 or height < 400:
        raise ValueError("Document editor canvas is too small")

    rng = random.Random(seed)
    document = rng.choice(DOCUMENTS)

    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()

    dark = "#273447"
    muted = "#7A8798"
    blue = "#3974B9"

    canvas.clear(_color("#E8EBF0"))

    # Application toolbar.
    _rect(canvas, 0, 0, width, 100, "#FFFFFF")
    _text(
        canvas, document["name"],
        19, 23, 15, dark, bold=True,
    )

    _text(
        canvas,
        "File     Edit     View     Insert     Format     Tools",
        20, 49, 11, muted,
    )

    _line(canvas, 0, 60, width, 60)

    # Formatting controls.
    _rect(canvas, 17, 66, 133, 27, "#F4F6F9", 5)
    _text(canvas, "Arial", 27, 84, 12, dark)
    _text(canvas, "v", 132, 84, 10, muted)

    _rect(canvas, 159, 66, 46, 27, "#F4F6F9", 5)
    _text(canvas, "12", 172, 84, 12, dark)

    _text(canvas, "B", 226, 85, 14, dark, bold=True)
    _text(canvas, "I", 255, 85, 14, dark)
    _text(canvas, "U", 281, 85, 13, dark)

    # Left-align icon.
    for i, length in enumerate((19, 14, 19, 11)):
        _rect(
            canvas,
            320,
            72 + i * 4,
            length,
            1.5,
            muted,
        )

    # Center-align icon.
    for i, length in enumerate((19, 14, 19, 11)):
        _rect(
            canvas,
            350 + (19 - length) / 2,
            72 + i * 4,
            length,
            1.5,
            muted,
        )

    _line(canvas, 0, 99, width, 99)

    # Document page and ruler.
    page_width = min(570, width - 72)
    page_x = (width - page_width) / 2
    page_y = 138
    page_height = 760

    _rect(
        canvas, page_x + 4, page_y + 5,
        page_width, page_height,
        "#D4DAE2",
    )
    _rect(
        canvas, page_x, page_y,
        page_width, page_height,
        "#FFFFFF",
    )

    _rect(
        canvas, page_x, 109,
        page_width, 21,
        "#F3F5F8",
    )

    for i in range(13):
        x = page_x + 24 + i * (page_width - 48) / 12
        _line(canvas, x, 118, x, 124, "#A8B3C1")

    # Page content.
    canvas.save()
    canvas.clipRect(
        skia.Rect.MakeXYWH(
            page_x,
            page_y,
            page_width,
            max(0, height - 29 - page_y),
        )
    )

    margin = 53
    text_x = page_x + margin
    text_width = page_width - 2 * margin
    y = page_y + 77

    _text(
        canvas, document["name"],
        text_x, y, 27, dark, bold=True,
    )
    y += 29

    _text(
        canvas, document["subtitle"],
        text_x, y, 13, muted,
    )
    y += 22

    _line(
        canvas, text_x, y,
        text_x + text_width, y,
        "#E2E7ED",
    )
    y += 35

    sections = document["sections"][:]
    rng.shuffle(sections)

    for heading, paragraph in sections:
        _text(
            canvas, heading,
            text_x, y, 17, dark, bold=True,
        )
        y += 26

        y = _paragraph(
            canvas, paragraph,
            text_x, y, text_width,
            size=12,
            line_height=21,
        )
        y += 24

    _text(
        canvas, "Next Steps",
        text_x, y, 17, dark, bold=True,
    )
    y += 28

    for item in document["bullets"]:
        _rect(
            canvas, text_x + 5, y - 5,
            5, 5, blue, 3,
        )
        y = _paragraph(
            canvas, item,
            text_x + 20, y,
            text_width - 20,
            size=12,
            line_height=20,
        )
        y += 10

    canvas.restore()

    # Vertical scrollbar.
    scrollbar_x = width - 10
    scrollbar_top = 104
    scrollbar_height = height - 104 - 29

    _rect(
        canvas,
        scrollbar_x,
        scrollbar_top,
        5,
        scrollbar_height,
        "#DCE2E9",
        2,
    )

    _rect(
        canvas,
        scrollbar_x,
        scrollbar_top + 5,
        5,
        min(110, scrollbar_height * 0.4),
        "#AAB5C3",
        2,
    )

    # Bottom status bar stays visible above the page.
    _rect(
        canvas, 0, height - 29,
        width, 29, "#FAFBFD",
    )
    _line(
        canvas, 0, height - 29,
        width, height - 29,
    )

    _text(
        canvas, "Page 1 of 1",
        18, height - 10,
        11, muted,
    )
    _text(
        canvas, "100%",
        width - 54, height - 10,
        11, muted,
    )

    return surface.makeImageSnapshot(), {}
