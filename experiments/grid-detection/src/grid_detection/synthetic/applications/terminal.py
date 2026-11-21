
"""Procedural terminal application rendered with Skia."""

import random

import skia
from PIL import Image

from ..desktop import surface_to_pillow


THEMES = {
    "dark": {
        "background": "#191B20",
        "toolbar": "#282B32",
        "foreground": "#D6DBE4",
        "muted": "#8792A3",
        "accent": "#83C991",
        "blue": "#83B8ED",
        "yellow": "#E5C07B",
        "red": "#E78B8B",
        "border": "#353A44",
    },
    "light": {
        "background": "#FAFAFC",
        "toolbar": "#E9EDF2",
        "foreground": "#30394A",
        "muted": "#778395",
        "accent": "#268653",
        "blue": "#3979B8",
        "yellow": "#A16B24",
        "red": "#C44E53",
        "border": "#D4DAE3",
    },
    "retro": {
        "background": "#07130A",
        "toolbar": "#112017",
        "foreground": "#83E29B",
        "muted": "#47845B",
        "accent": "#B4FFB9",
        "blue": "#6DCB91",
        "yellow": "#C9E88A",
        "red": "#E19D89",
        "border": "#245535",
    },
}


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


def _text(canvas, value, x, y, font, color):
    canvas.drawString(str(value), x, y, font, _paint(color))


def _command_session(rng: random.Random, kind: str):
    """Return terminal lines as (text, semantic color) pairs."""

    if kind == "files":
        directory = rng.choice(
            ["projects", "documents", "workspace", "downloads"]
        )

        return [
            ("$ pwd", "accent"),
            (f"/home/developer/{directory}", "foreground"),
            ("", "foreground"),
            ("$ ls -lah", "accent"),
            ("total 48K", "foreground"),
            ("drwxr-xr-x  8 user  staff  256 Nov 21 .", "blue"),
            ("drwxr-xr-x 16 user  staff  512 Nov 20 ..", "blue"),
            ("drwxr-xr-x  5 user  staff  160 Nov 19 src", "blue"),
            ("drwxr-xr-x  3 user  staff   96 Nov 18 tests", "blue"),
            ("-rw-r--r--  1 user  staff  2.4K Nov 21 README.md", "foreground"),
            ("-rw-r--r--  1 user  staff  1.2K Nov 20 pyproject.toml", "foreground"),
            ("-rw-r--r--  1 user  staff  4.8K Nov 21 main.py", "foreground"),
            ("", "foreground"),
            ("$ ", "accent"),
        ]

    if kind == "git":
        branch = rng.choice(
            ["main", "develop", "feature/ui", "feature/dataset"]
        )

        return [
            ("$ git status", "accent"),
            (f"On branch {branch}", "foreground"),
            ("Your branch is up to date with 'origin/main'.", "foreground"),
            ("", "foreground"),
            ("Changes not staged for commit:", "yellow"),
            ("  modified:   src/renderer.py", "red"),
            ("  modified:   tests/test_render.py", "red"),
            ("", "foreground"),
            ("Untracked files:", "yellow"),
            ("  src/components/", "red"),
            ("", "foreground"),
            ("$ git log --oneline -3", "accent"),
            ("a92d3e1 Add synthetic rendering pipeline", "yellow"),
            ("bc84e07 Refactor application layout", "yellow"),
            ("2d918fa Initialize project structure", "yellow"),
            ("", "foreground"),
            ("$ ", "accent"),
        ]

    if kind == "tests":
        passed = rng.randint(18, 86)
        duration = rng.uniform(0.4, 5.8)

        return [
            ("$ python -m pytest tests/ -v", "accent"),
            ("================ test session starts ================", "muted"),
            ("platform linux -- Python 3.12.7, pytest-8.3.4", "foreground"),
            (f"collected {passed} items", "foreground"),
            ("", "foreground"),
            ("tests/test_config.py::test_defaults PASSED", "accent"),
            ("tests/test_config.py::test_validation PASSED", "accent"),
            ("tests/test_data.py::test_loading PASSED", "accent"),
            ("tests/test_data.py::test_shapes PASSED", "accent"),
            ("tests/test_render.py::test_output PASSED", "accent"),
            ("tests/test_render.py::test_dimensions PASSED", "accent"),
            ("", "foreground"),
            ("==================== summary =======================", "muted"),
            (f"{passed} passed in {duration:.2f}s", "accent"),
            ("", "foreground"),
            ("$ ", "accent"),
        ]

    if kind == "build":
        package = rng.choice(
            ["desktop-ui", "image-tools", "data-platform"]
        )

        return [
            ("$ npm run build", "accent"),
            ("", "foreground"),
            (f"> {package}@1.0.0 build", "foreground"),
            ("> vite build", "foreground"),
            ("", "foreground"),
            ("vite building for production...", "blue"),
            ("transforming modules...", "foreground"),
            ("284 modules transformed.", "accent"),
            ("rendering chunks...", "foreground"),
            ("computing gzip size...", "foreground"),
            ("", "foreground"),
            ("dist/index.html          0.82 kB", "foreground"),
            ("dist/assets/index.css   18.42 kB", "foreground"),
            ("dist/assets/index.js   142.70 kB", "foreground"),
            ("", "foreground"),
            ("Build completed successfully.", "accent"),
            ("", "foreground"),
            ("$ ", "accent"),
        ]

    raise ValueError(f"Unknown terminal session: {kind}")


def render_terminal(
    seed: int = 42,
    *,
    width: int = 760,
    height: int = 490,
    theme: str | None = None,
    session: str | None = None,
) -> tuple[Image.Image, dict]:
    """Render a deterministic terminal emulator.

    Returns a Pillow image and no Minesweeper annotations.
    """
    if width < 480 or height < 300:
        raise ValueError("Terminal canvas is too small")

    rng = random.Random(seed)

    if theme is None:
        theme = rng.choices(
            ("dark", "light", "retro"),
            weights=(60, 25, 15),
            k=1,
        )[0]

    if theme not in THEMES:
        raise ValueError(f"Unknown terminal theme: {theme}")

    if session is None:
        session = rng.choice(
            ("files", "git", "tests", "build")
        )

    colors = THEMES[theme]
    lines = _command_session(rng, session)

    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()

    canvas.clear(_color(colors["background"]))

    # A compact tab strip, inside the application content.
    toolbar_h = 39

    _rect(
        canvas, 0, 0, width, toolbar_h,
        colors["toolbar"],
    )

    _rect(
        canvas, 12, 7, 177, 32,
        colors["background"], 7,
    )

    _rect(
        canvas, 24, 17, 10, 10,
        colors["accent"], 2,
    )

    ui_font = skia.Font(None, 12)

    _text(
        canvas, "Terminal",
        44, 27, ui_font,
        colors["foreground"],
    )

    _text(
        canvas, "+",
        206, 27,
        skia.Font(None, 17),
        colors["muted"],
    )

    # Vector-drawn menu icon to avoid unsupported glyphs.
    for i in range(3):
        _rect(
            canvas, width - 30,
            16 + i * 5,
            13, 1.5,
            colors["muted"],
        )

    _rect(
        canvas, 0, toolbar_h - 1,
        width, 1, colors["border"],
    )

    # Draw command history with a monospace font.
    font_size = 14
    line_height = 23
    left = 19
    top = toolbar_h + 29

    monospace = skia.Font(
        skia.Typeface.MakeFromName(
            "monospace",
            skia.FontStyle.Normal(),
        ),
        font_size,
    )

    canvas.save()
    canvas.clipRect(
        skia.Rect.MakeXYWH(
            0,
            toolbar_h,
            width,
            height - toolbar_h,
        )
    )

    # Shift the history upward if it exceeds the viewport.
    available_height = height - top - 16
    visible_lines = max(
        1, int(available_height // line_height) + 1
    )
    displayed = lines[-visible_lines:]

    for i, (value, role) in enumerate(displayed):
        y = top + i * line_height

        if value:
            canvas.save()
            canvas.clipRect(
                skia.Rect.MakeXYWH(
                    left,
                    y - font_size - 3,
                    width - left - 15,
                    line_height,
                )
            )
            _text(
                canvas,
                value,
                left, y,
                monospace,
                colors[role],
            )
            canvas.restore()

    # Block cursor at the final prompt.
    last_y = top + (len(displayed) - 1) * line_height

    if displayed and displayed[-1][0] == "$ ":
        cursor_x = left + monospace.measureText("$ ")

        _rect(
            canvas,
            cursor_x,
            last_y - font_size + 2,
            8, font_size + 2,
            colors["accent"],
        )

    canvas.restore()

    return surface_to_pillow(surface), {}
