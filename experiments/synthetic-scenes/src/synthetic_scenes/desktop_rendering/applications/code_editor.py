
"""Procedural dark-themed code editor rendered with Skia."""

import random

import skia

from ..scaling import pixels, validate_scale



def _color(value: str) -> int:
    value = value.lstrip("#")
    return skia.ColorSetRGB(
        *(int(value[i:i + 2], 16) for i in (0, 2, 4))
    )


def _paint(color: str) -> skia.Paint:
    return skia.Paint(
        Color=_color(color),
        AntiAlias=True,
    )


def _rect(canvas, x, y, w, h, color, radius=0):
    shape = skia.Rect.MakeXYWH(x, y, w, h)

    if radius:
        canvas.drawRoundRect(
            shape, radius, radius, _paint(color)
        )
    else:
        canvas.drawRect(shape, _paint(color))


def _text(
    canvas, value, x, y,
    size=13, color="#D4D4D4",
    *,
    mono=False,
    bold=False,
):
    family = "monospace" if mono else None
    style = (
        skia.FontStyle.Bold()
        if bold else skia.FontStyle.Normal()
    )
    typeface = skia.Typeface.MakeFromName(family, style)
    font = skia.Font(typeface, size)

    canvas.drawString(
        str(value), x, y, font, _paint(color)
    )

    return font


def _clipped_text(canvas, value, x, y, width,
                  size=13, color="#D4D4D4"):
    canvas.save()
    canvas.clipRect(
        skia.Rect.MakeXYWH(
            x, y - size - 4, width, size + 8
        )
    )
    _text(canvas, value, x, y, size, color)
    canvas.restore()


# Each token is (text, syntax category).
# Keeping the token lists explicit avoids parsing arbitrary source code.
CODE_SAMPLES = [
    {
        "project": "grid-detection",
        "file": "renderer.py",
        "files": [
            "renderer.py",
            "sampler.py",
            "layout.py",
            "wallpaper.py",
            "config.py",
        ],
        "lines": [
            [("from", "keyword"), (" pathlib ", "plain"),
             ("import", "keyword"), (" Path", "type")],
            [],
            [("def", "keyword"), (" render_scene", "function"),
             ("(scene):", "plain")],
            [("    ", "plain"), ("\"\"\"Render a synthetic scene.\"\"\"", "string")],
            [("    ", "plain"), ("windows", "variable"),
             (" = scene[", "plain"), ("\"windows\"", "string"),
             ("]", "plain")],
            [("    ", "plain"), ("layers", "variable"),
             (" = []", "plain")],
            [],
            [("    ", "plain"), ("for", "keyword"),
             (" window ", "variable"), ("in", "keyword"),
             (" windows:", "plain")],
            [("        ", "plain"), ("image", "variable"),
             (", boxes = render_application(window)", "plain")],
            [("        ", "plain"), ("layers.append", "function"),
             ("((image, boxes))", "plain")],
            [],
            [("    ", "plain"), ("return", "keyword"),
             (" compose_screenshot(layers)", "plain")],
        ],
    },
    {
        "project": "analytics-dashboard",
        "file": "metrics.py",
        "files": [
            "metrics.py",
            "charts.py",
            "database.py",
            "models.py",
            "settings.py",
        ],
        "lines": [
            [("import", "keyword"), (" statistics", "plain")],
            [],
            [("def", "keyword"), (" summarize", "function"),
             ("(records):", "plain")],
            [("    ", "plain"), ("\"\"\"Calculate summary metrics.\"\"\"", "string")],
            [("    ", "plain"), ("values", "variable"),
             (" = [r[", "plain"), ("\"value\"", "string"),
             ("] ", "plain"), ("for", "keyword"),
             (" r ", "plain"), ("in", "keyword"),
             (" records]", "plain")],
            [],
            [("    ", "plain"), ("if", "keyword"),
             (" not", "keyword"), (" values:", "plain")],
            [("        ", "plain"), ("return", "keyword"),
             (" {", "plain"), ("\"count\"", "string"),
             (": ", "plain"), ("0", "number"), ("}", "plain")],
            [],
            [("    ", "plain"), ("return", "keyword"), (" {", "plain")],
            [("        ", "plain"), ("\"count\"", "string"),
             (": len(values),", "plain")],
            [("        ", "plain"), ("\"average\"", "string"),
             (": statistics.mean(values),", "plain")],
            [("    }", "plain")],
        ],
    },
    {
        "project": "task-manager",
        "file": "tasks.py",
        "files": [
            "tasks.py",
            "storage.py",
            "cli.py",
            "utils.py",
            "tests.py",
        ],
        "lines": [
            [("from", "keyword"), (" dataclasses ", "plain"),
             ("import", "keyword"), (" dataclass", "type")],
            [],
            [("@dataclass", "decorator")],
            [("class", "keyword"), (" Task", "type"), (":", "plain")],
            [("    title: ", "plain"), ("str", "type")],
            [("    completed: ", "plain"), ("bool", "type"),
             (" = ", "plain"), ("False", "keyword")],
            [],
            [("def", "keyword"), (" pending_tasks", "function"),
             ("(tasks):", "plain")],
            [("    ", "plain"), ("return", "keyword"),
             (" [task ", "plain"), ("for", "keyword"),
             (" task ", "plain"), ("in", "keyword"),
             (" tasks", "plain")],
            [("            ", "plain"), ("if", "keyword"),
             (" not", "keyword"), (" task.completed]", "plain")],
        ],
    },
]

SYNTAX_COLORS = {
    "plain": "#D4D4D4",
    "keyword": "#C586C0",
    "string": "#CE9178",
    "function": "#DCDCAA",
    "type": "#4EC9B0",
    "number": "#B5CEA8",
    "variable": "#9CDCFE",
    "decorator": "#DCDCAA",
}


def _draw_code_line(canvas, tokens, x, y, size=13):
    canvas.save()

    font = skia.Font(
        skia.Typeface.MakeFromName(
            "monospace", skia.FontStyle.Normal()
        ),
        size,
    )

    current_x = x

    for value, category in tokens:
        canvas.drawString(
            value,
            current_x,
            y,
            font,
            _paint(SYNTAX_COLORS[category]),
        )
        current_x += font.measureText(value)

    canvas.restore()


def _activity_icon(canvas, kind: str, x: float, y: float, color: str):
    """Draw a 20 × 20 IDE activity-bar icon using Skia."""
    fill = _paint(color)

    outline = skia.Paint(
        Color=_color(color),
        AntiAlias=True,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=1.7,
        StrokeCap=skia.Paint.kRound_Cap,
        StrokeJoin=skia.Paint.kRound_Join,
    )

    def line(x1, y1, x2, y2):
        canvas.drawLine(x1, y1, x2, y2, outline)

    if kind == "explorer":
        # Two overlapping document outlines.
        canvas.drawRect(
            skia.Rect.MakeXYWH(x + 5, y + 1, 12, 15),
            outline,
        )
        canvas.drawRect(
            skia.Rect.MakeXYWH(x + 2, y + 4, 12, 15),
            outline,
        )

    elif kind == "search":
        canvas.drawCircle(x + 9, y + 8, 6, outline)
        line(x + 13.5, y + 12.5, x + 19, y + 18)

    elif kind == "source_control":
        line(x + 6, y + 4, x + 6, y + 16)
        line(x + 6, y + 10, x + 15, y + 10)
        line(x + 15, y + 10, x + 15, y + 4)

        for cx, cy in ((6, 3), (6, 17), (15, 3)):
            canvas.drawCircle(x + cx, y + cy, 2.4, fill)

    elif kind == "run":
        path = skia.Path()
        path.moveTo(x + 4, y + 3)
        path.lineTo(x + 17, y + 10)
        path.lineTo(x + 4, y + 17)
        path.close()
        canvas.drawPath(path, outline)

    elif kind == "extensions":
        for dx, dy in ((2, 2), (11, 2), (2, 11), (11, 11)):
            canvas.drawRoundRect(
                skia.Rect.MakeXYWH(
                    x + dx, y + dy, 7, 7
                ),
                1.2, 1.2,
                outline,
            )


def render_code_editor(
    seed: int = 42,
    *,
    width: int = 900,
    height: int = 620,
    scale: float = 1.0,
) -> tuple[skia.Image, dict]:
    """Render a synthetic IDE with code and terminal panels."""
    if width < 600 or height < 420:
        raise ValueError("Code editor canvas is too small")

    rng = random.Random(seed)
    sample = rng.choice(CODE_SAMPLES)

    scale = validate_scale(scale)
    surface = skia.Surface(pixels(width, scale), pixels(height, scale))
    canvas = surface.getCanvas()
    canvas.scale(scale, scale)
    canvas.clear(_color("#1E1E1E"))

    bg = "#1E1E1E"
    sidebar_bg = "#252526"
    panel_bg = "#181818"
    tab_bg = "#2D2D2D"
    border = "#333333"
    muted = "#858585"
    text_color = "#CCCCCC"
    accent = "#007ACC"

    activity_w = 43
    sidebar_w = 188
    sidebar_right = activity_w + sidebar_w
    title_h = 37
    tabs_h = 35
    status_h = 24
    terminal_h = min(175, int(height * 0.28))

    editor_top = title_h + tabs_h
    terminal_top = height - status_h - terminal_h

    # Top application bar.
    _rect(canvas, 0, 0, width, title_h, "#323233")
    _text(
        canvas, "File   Edit   Selection   View   Go   Run   Terminal   Help",
        16, 24, 12, "#DDDDDD",
    )
    _clipped_text(
        canvas,
        sample["project"],
        width - 205, 24, 180,
        11, "#AFAFAF",
    )

    # Activity bar and explorer.
    _rect(
        canvas, 0, title_h,
        activity_w, height - title_h - status_h,
        "#333333",
    )
    _rect(
        canvas, activity_w, title_h,
        sidebar_w, height - title_h - status_h,
        sidebar_bg,
    )

    activity_icons = (
        "explorer",
        "search",
        "source_control",
        "run",
        "extensions",
    )

    for i, kind in enumerate(activity_icons):
        yy = 61 + i * 47

        _activity_icon(
            canvas,
            kind,
            x=12,
            y=yy,
            color="#D7DCE4" if i == 0 else "#858585",
        )

        if i == 0:
            _rect(
                canvas,
                0, yy - 4,
                2, 28,
                accent,
            )

    _text(
        canvas, "EXPLORER",
        activity_w + 17, 63,
        11, "#BBBBBB",
    )
    _clipped_text(
        canvas,
        sample["project"].upper(),
        activity_w + 13, 98,
        sidebar_w - 20,
        11, "#D7D7D7",
    )

    selected_file = sample["file"]

    for i, filename in enumerate(sample["files"]):
        yy = 132 + i * 29

        if filename == selected_file:
            _rect(
                canvas, activity_w,
                yy - 18,
                sidebar_w, 28,
                "#37373D",
            )

        _rect(
            canvas,
            activity_w + 19,
            yy - 12,
            11, 13,
            "#4D9CCB" if filename.endswith(".py")
            else "#A0A0A0",
            2,
        )
        _clipped_text(
            canvas, filename,
            activity_w + 40, yy,
            sidebar_w - 48, 12,
            "#FFFFFF" if filename == selected_file
            else "#B9B9B9",
        )

    # Editor tab strip.
    _rect(
        canvas, sidebar_right, title_h,
        width - sidebar_right, tabs_h,
        tab_bg,
    )
    _rect(
        canvas, sidebar_right, title_h,
        160, tabs_h, bg,
    )
    _rect(
        canvas, sidebar_right,
        title_h, 160, 2,
        accent,
    )
    _rect(
        canvas, sidebar_right + 15,
        title_h + 12, 11, 13,
        "#4D9CCB", 2,
    )
    _text(
        canvas,
        selected_file,
        sidebar_right + 35,
        title_h + 24,
        12, "#EEEEEE",
    )
    _text(
        canvas, "×",
        sidebar_right + 139,
        title_h + 24, 13, muted,
    )

    # Editor content region.
    editor_h = terminal_top - editor_top
    _rect(
        canvas,
        sidebar_right, editor_top,
        width - sidebar_right, editor_h,
        bg,
    )

    line_height = 22
    first_line_y = editor_top + 35
    highlight = rng.randrange(len(sample["lines"]))

    canvas.save()
    canvas.clipRect(
        skia.Rect.MakeXYWH(
            sidebar_right,
            editor_top,
            width - sidebar_right,
            max(0, editor_h),
        )
    )

    for i, tokens in enumerate(sample["lines"]):
        yy = first_line_y + i * line_height

        if i == highlight:
            _rect(
                canvas,
                sidebar_right, yy - 16,
                width - sidebar_right,
                line_height,
                "#292929",
            )

        _text(
            canvas, f"{i + 1:>3}",
            sidebar_right + 13, yy,
            12, "#858585",
            mono=True,
        )

        _draw_code_line(
            canvas,
            tokens,
            sidebar_right + 68,
            yy,
            size=13,
        )

    canvas.restore()

    # Integrated terminal.
    _rect(
        canvas,
        sidebar_right, terminal_top,
        width - sidebar_right, terminal_h,
        panel_bg,
    )
    _rect(
        canvas,
        sidebar_right, terminal_top,
        width - sidebar_right, 1,
        "#454545",
    )

    _text(
        canvas,
        "PROBLEMS    OUTPUT    DEBUG CONSOLE    TERMINAL",
        sidebar_right + 20,
        terminal_top + 25,
        11, "#BDBDBD",
    )
    _rect(
        canvas,
        sidebar_right + 270,
        terminal_top + 32,
        60, 2,
        accent,
    )

    terminal_lines = [
        f"$ python {selected_file}",
        rng.choice([
            "Loading configuration...",
            "Running project checks...",
            "Initializing modules...",
        ]),
        rng.choice([
            "Completed successfully.",
            "12 tests passed in 0.38s",
            "Process finished with exit code 0",
        ]),
        "$ ",
    ]

    for i, line in enumerate(terminal_lines):
        yy = terminal_top + 54 + i * 24
        if yy > height - status_h - 7:
            break

        _text(
            canvas, line,
            sidebar_right + 22,
            yy, 12,
            "#D1D1D1" if i != 2 else "#78CFA2",
            mono=True,
        )

    # Status bar.
    _rect(
        canvas, 0, height - status_h,
        width, status_h,
        accent,
    )

    _text(
        canvas, "main*",
        13, height - 8,
        11, "#FFFFFF",
    )
    _text(
        canvas,
        f"Ln {highlight + 1}, Col 1     UTF-8     Python",
        width - 244,
        height - 8,
        11, "#FFFFFF",
    )

    return surface.makeImageSnapshot(), {}
