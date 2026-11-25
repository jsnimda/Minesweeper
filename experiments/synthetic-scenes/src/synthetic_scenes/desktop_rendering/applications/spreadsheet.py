
"""Procedural spreadsheet content for synthetic desktops."""

import random

import skia



def _color(value: str) -> int:
    value = value.lstrip("#")
    return skia.ColorSetRGB(
        *(int(value[i:i + 2], 16) for i in (0, 2, 4))
    )


def _paint(color: str, stroke: float = 0) -> skia.Paint:
    paint = skia.Paint(
        Color=_color(color),
        AntiAlias=True,
    )
    if stroke:
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(stroke)
    return paint


def _rect(canvas, x, y, w, h, color):
    canvas.drawRect(
        skia.Rect.MakeXYWH(x, y, w, h),
        _paint(color),
    )


def _text(canvas, value, x, y, size=12, color="#283445"):
    font = skia.Font(None, size)
    blob = skia.TextBlob(str(value), font)
    canvas.drawTextBlob(blob, x, y, _paint(color))


def _line(canvas, x0, y0, x1, y1, color="#DCE3EA", width=1):
    canvas.drawLine(
        x0, y0, x1, y1,
        _paint(color, stroke=width),
    )


def render_spreadsheet(
    seed: int = 42,
    *,
    rows: int = 17,
    columns: int = 8,
    cell_width: int = 92,
    cell_height: int = 27,
) -> tuple[skia.Image, dict]:
    """Render a deterministic spreadsheet with no target boxes."""
    if not 4 <= rows <= 100:
        raise ValueError("rows must be between 4 and 100")
    if not 3 <= columns <= 26:
        raise ValueError("columns must be between 3 and 26")
    if cell_width < 50 or cell_height < 20:
        raise ValueError("Cell dimensions are too small")

    rng = random.Random(seed)

    row_header = 42
    toolbar_h = 43
    formula_h = 38
    col_header_h = 28
    tabbar_h = 38

    grid_top = toolbar_h + formula_h + col_header_h
    grid_left = row_header

    width = grid_left + columns * cell_width
    height = grid_top + rows * cell_height + tabbar_h

    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    canvas.clear(_color("#FFFFFF"))

    # Colors
    toolbar_bg = "#F8FAFC"
    header_bg = "#F1F5F9"
    grid_line = "#DDE4EC"
    dark = "#293647"
    muted = "#738195"
    accent = "#26876B"
    selected_bg = "#E9F5EF"

    # Toolbar
    _rect(canvas, 0, 0, width, toolbar_h, toolbar_bg)
    _text(canvas, "Workbook", 16, 27, 16, dark)

    toolbar_items = ["File", "Edit", "View", "Insert", "Format"]
    cursor_x = 128

    for item in toolbar_items:
        _text(canvas, item, cursor_x, 27, 12, muted)
        cursor_x += len(item) * 8 + 20

    _line(canvas, 0, toolbar_h, width, toolbar_h)

    # Formula bar
    _rect(canvas, 0, toolbar_h, width, formula_h, "#FFFFFF")

    selected_row = rng.randint(3, rows)
    selected_col = rng.randint(1, columns - 1)
    selected_name = f"{chr(65 + selected_col)}{selected_row}"

    _rect(canvas, 11, toolbar_h + 7, 64, 24, header_bg)
    _text(canvas, selected_name, 22, toolbar_h + 24, 12, dark)

    _text(canvas, "fx", 91, toolbar_h + 24, 13, accent)

    _line(
        canvas, 119, toolbar_h + 7,
        119, toolbar_h + formula_h - 7,
    )

    # Generate a coherent business table.
    headings = [
        "Item", "Units", "Price", "Revenue",
        "Cost", "Profit", "Margin", "Status",
    ]

    products = [
        "Monitor", "Keyboard", "Mouse", "Laptop",
        "Tablet", "Webcam", "Headset", "Speaker",
        "Adapter", "Storage", "Router", "Display",
        "Charger", "Dock", "Cable", "Printer",
    ]

    values = {}
    formulas = {}

    for r in range(1, rows + 1):
        product = rng.choice(products)
        units = rng.randint(8, 450)
        price = rng.randint(15, 380)
        revenue = units * price
        cost = round(revenue * rng.uniform(0.48, 0.82))
        profit = revenue - cost
        margin = profit / revenue

        row_values = [
            product,
            str(units),
            f"${price:,}",
            f"${revenue:,}",
            f"${cost:,}",
            f"${profit:,}",
            f"{margin:.1%}",
            rng.choice(["Active", "Active", "Review"]),
        ]

        for c in range(columns):
            values[(r, c)] = row_values[c] if c < len(row_values) else ""

        if columns > 3:
            formulas[(r, 3)] = f"=B{r + 1}*C{r + 1}"
        if columns > 5:
            formulas[(r, 5)] = f"=D{r + 1}-E{r + 1}"

    # Formula preview corresponds to selected cell.
    key = (selected_row - 1, selected_col)
    formula = formulas.get(key, values.get(key, ""))

    _text(
        canvas, formula,
        135, toolbar_h + 24,
        12, dark,
    )

    _line(
        canvas, 0, toolbar_h + formula_h,
        width, toolbar_h + formula_h,
    )

    # Column headers
    _rect(
        canvas, 0, toolbar_h + formula_h,
        width, col_header_h, header_bg,
    )

    for c in range(columns):
        x = grid_left + c * cell_width

        if c == selected_col:
            _rect(
                canvas, x, toolbar_h + formula_h,
                cell_width, col_header_h, selected_bg,
            )

        label = chr(65 + c)
        _text(
            canvas, label,
            x + cell_width / 2 - 5,
            grid_top - 9, 12, dark,
        )

    # Cells and row headers
    for r in range(rows):
        y = grid_top + r * cell_height

        _rect(canvas, 0, y, row_header, cell_height, header_bg)

        if r == selected_row - 1:
            _rect(canvas, 0, y, row_header, cell_height, selected_bg)

        _text(canvas, r + 1, 13, y + 18, 11, muted)

        for c in range(columns):
            x = grid_left + c * cell_width

            if r == 0:
                background = "#E8F0ED"
            elif r % 2 == 0:
                background = "#FBFCFE"
            else:
                background = "#FFFFFF"

            _rect(canvas, x, y, cell_width, cell_height, background)

            if r == 0:
                value = headings[c] if c < len(headings) else ""
                color = "#225F4D"
            else:
                value = values.get((r, c), "")
                color = dark

            # Clip text inside each cell.
            canvas.save()
            canvas.clipRect(
                skia.Rect.MakeXYWH(
                    x + 5, y + 1,
                    cell_width - 10, cell_height - 2,
                )
            )
            _text(
                canvas, value,
                x + 9, y + cell_height * 0.68,
                11, color,
            )
            canvas.restore()

    # Grid lines
    grid_bottom = grid_top + rows * cell_height

    for c in range(columns + 1):
        x = grid_left + c * cell_width
        _line(
            canvas, x, toolbar_h + formula_h,
            x, grid_bottom, grid_line,
        )

    for r in range(rows + 1):
        y = grid_top + r * cell_height
        _line(canvas, 0, y, width, y, grid_line)

    # Selected cell outline (avoid the table heading).
    sx = grid_left + selected_col * cell_width
    sy = grid_top + (selected_row - 1) * cell_height

    canvas.drawRect(
        skia.Rect.MakeXYWH(
            sx + 1, sy + 1,
            cell_width - 2, cell_height - 2,
        ),
        _paint(accent, stroke=2),
    )

    _rect(
        canvas,
        sx + cell_width - 5,
        sy + cell_height - 5,
        6, 6, accent,
    )

    # Bottom sheet tabs
    _rect(canvas, 0, grid_bottom, width, tabbar_h, toolbar_bg)

    _text(canvas, "+", 19, grid_bottom + 24, 17, muted)

    _rect(canvas, 48, grid_bottom + 5, 96, 30, selected_bg)
    _text(canvas, "Sheet 1", 64, grid_bottom + 25, 12, accent)
    _text(canvas, "Sheet 2", 162, grid_bottom + 25, 12, muted)

    _line(
        canvas, 48, grid_bottom + 34,
        144, grid_bottom + 34,
        accent, 2,
    )

    return surface.makeImageSnapshot(), {}
