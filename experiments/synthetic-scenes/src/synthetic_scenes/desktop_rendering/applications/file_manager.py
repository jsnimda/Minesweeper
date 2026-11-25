
"""Procedural file manager content rendered with Skia."""

import random

import skia


def _color(hex_color: str) -> int:
    value = hex_color.lstrip("#")
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
    rect = skia.Rect.MakeXYWH(x, y, w, h)
    paint = _paint(color)

    if radius:
        canvas.drawRoundRect(rect, radius, radius, paint)
    else:
        canvas.drawRect(rect, paint)


def _text(canvas, value, x, y, size=13, color="#293444"):
    font = skia.Font(None, size)
    canvas.drawString(str(value), x, y, font, _paint(color))


def _line(canvas, x0, y0, x1, y1, color="#E3E8EF"):
    canvas.drawLine(
        x0, y0, x1, y1,
        _paint(color, stroke=1),
    )


def _file_icon(canvas, x, y, kind, selected=False):
    """Draw simple original file/folder icons."""
    if kind == "folder":
        _rect(canvas, x + 3, y + 7, 32, 25, "#E6B45C", 5)
        _rect(canvas, x + 3, y + 3, 17, 10, "#F2CD81", 3)

    elif kind == "image":
        _rect(canvas, x + 5, y + 3, 29, 32, "#72A6C4", 4)
        _rect(canvas, x + 9, y + 7, 21, 24, "#D9EAF1", 2)
        canvas.drawCircle(
            x + 25, y + 13, 3, _paint("#F1B75A")
        )
        _line(
            canvas, x + 11, y + 26,
            x + 20, y + 18, "#46918E",
        )
        _line(
            canvas, x + 20, y + 18,
            x + 29, y + 27, "#46918E",
        )

    else:
        color = {
            "document": "#7189BE",
            "spreadsheet": "#56A287",
            "presentation": "#CF956B",
            "archive": "#987EBC",
            "code": "#6A95AE",
        }.get(kind, "#8190A3")

        _rect(canvas, x + 7, y + 3, 26, 33, color, 4)
        _rect(canvas, x + 12, y + 12, 16, 2, "#FFFFFF")
        _rect(canvas, x + 12, y + 18, 13, 2, "#FFFFFF")
        _rect(canvas, x + 12, y + 24, 16, 2, "#FFFFFF")


def render_file_manager(
    seed: int = 42,
    *,
    width: int = 760,
    height: int = 520,
) -> tuple[skia.Image, dict]:
    """Render a deterministic file-manager window content.

    No actual filesystem is accessed.
    Returns no Minesweeper annotations.
    """
    if width < 480 or height < 340:
        raise ValueError("File manager canvas is too small")

    rng = random.Random(seed)

    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    canvas.clear(_color("#FFFFFF"))

    dark = "#293444"
    muted = "#7B8899"
    accent = "#477FC0"
    sidebar_bg = "#F5F7FA"
    selected_bg = "#E6EFFB"

    sidebar_w = 166
    toolbar_h = 55
    breadcrumb_h = 42
    status_h = 34

    # Sidebar background.
    _rect(
        canvas, 0, 0,
        sidebar_w, height, sidebar_bg,
    )

    _line(canvas, sidebar_w, 0, sidebar_w, height)

    # Toolbar.
    _rect(
        canvas, sidebar_w, 0,
        width - sidebar_w, toolbar_h,
        "#FAFBFD",
    )

    _text(canvas, "‹", sidebar_w + 20, 34, 26, muted)
    _text(canvas, "›", sidebar_w + 52, 34, 26, muted)
    _text(canvas, "⌂", sidebar_w + 88, 33, 21, accent)

    _text(
        canvas, "Search files",
        width - 138, 33, 12, muted,
    )

    _line(
        canvas, sidebar_w, toolbar_h,
        width, toolbar_h,
    )

    # Sidebar entries.
    _text(
        canvas, "FAVORITES",
        19, 32, 10, muted,
    )

    sidebar_items = [
        ("Home", "⌂"),
        ("Desktop", "▣"),
        ("Documents", "▤"),
        ("Downloads", "↓"),
        ("Pictures", "▧"),
        ("Music", "♫"),
        ("Videos", "▷"),
    ]

    selected_sidebar = rng.choice(
        ["Home", "Documents", "Downloads", "Pictures"]
    )

    for i, (name, symbol) in enumerate(sidebar_items):
        y = 57 + i * 41

        if name == selected_sidebar:
            _rect(
                canvas, 9, y - 22,
                sidebar_w - 18, 34,
                selected_bg, 7,
            )

        _text(
            canvas, symbol,
            22, y, 16,
            accent if name == selected_sidebar else muted,
        )
        _text(
            canvas, name,
            51, y, 12,
            dark,
        )

    _text(canvas, "LOCATIONS", 19, 376, 10, muted)
    _text(canvas, "◉", 22, 406, 14, muted)
    _text(canvas, "Local Disk", 51, 406, 12, dark)

    # Breadcrumb path.
    breadcrumb_y = toolbar_h

    _rect(
        canvas, sidebar_w, breadcrumb_y,
        width - sidebar_w, breadcrumb_h,
        "#FFFFFF",
    )

    path_text = (
        "Home"
        if selected_sidebar == "Home"
        else f"Home   /   {selected_sidebar}"
    )
    _text(
        canvas, path_text,
        sidebar_w + 24,
        breadcrumb_y + 27,
        13, dark,
    )

    _line(
        canvas,
        sidebar_w, breadcrumb_y + breadcrumb_h,
        width, breadcrumb_y + breadcrumb_h,
    )

    # Generate synthetic files and folders.
    folder_names = [
        "Projects", "Archive", "Reports",
        "Designs", "Assets", "Notes",
        "Screenshots", "Backup",
    ]

    file_names = [
        ("Budget.xlsx", "spreadsheet"),
        ("Presentation.pptx", "presentation"),
        ("Meeting Notes.docx", "document"),
        ("Photo_024.jpg", "image"),
        ("Diagram.png", "image"),
        ("main.py", "code"),
        ("Dataset.zip", "archive"),
        ("Summary.pdf", "document"),
        ("Expenses.xlsx", "spreadsheet"),
        ("Mockup.png", "image"),
    ]

    folder_count = rng.randint(2, 5)
    file_count = rng.randint(4, 9)

    items = [
        (name, "folder")
        for name in rng.sample(folder_names, folder_count)
    ]
    items += rng.sample(file_names, file_count)

    # Different seeds select different highlighted files.
    selected_index = rng.randrange(len(items))

    content_left = sidebar_w + 21
    content_top = toolbar_h + breadcrumb_h + 24

    tile_w = 108
    tile_h = 99
    gap_x = 8
    gap_y = 8

    available_w = width - content_left - 14
    columns = max(1, available_w // (tile_w + gap_x))

    available_h = height - content_top - status_h - 12
    visible_rows = max(1, available_h // (tile_h + gap_y))
    visible_items = items[:columns * visible_rows]

    for index, (name, kind) in enumerate(visible_items):
        row, col = divmod(index, columns)

        x = content_left + col * (tile_w + gap_x)
        y = content_top + row * (tile_h + gap_y)

        if index == selected_index:
            _rect(
                canvas,
                x, y, tile_w, tile_h,
                selected_bg, 9,
            )

        icon_x = x + (tile_w - 40) / 2
        icon_y = y + 9

        _file_icon(canvas, icon_x, icon_y, kind)

        # Clip long file names to the tile.
        canvas.save()
        canvas.clipRect(
            skia.Rect.MakeXYWH(
                x + 5, y + 54,
                tile_w - 10, 35,
            )
        )

        font = skia.Font(None, 11)
        text_width = font.measureText(name)
        text_x = x + max(5, (tile_w - text_width) / 2)

        _text(
            canvas, name,
            text_x, y + 73,
            11, dark,
        )
        canvas.restore()

    # Status bar.
    _rect(
        canvas,
        sidebar_w, height - status_h,
        width - sidebar_w, status_h,
        "#F9FAFC",
    )
    _line(
        canvas,
        sidebar_w, height - status_h,
        width, height - status_h,
    )
    _text(
        canvas,
        f"{len(items)} items",
        sidebar_w + 19,
        height - 12,
        11, muted,
    )

    result = surface.makeImageSnapshot()

    return result, {}
