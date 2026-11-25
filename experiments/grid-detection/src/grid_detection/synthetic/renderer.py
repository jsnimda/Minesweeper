
"""Render sampled synthetic desktop scenes."""

import hashlib
from pathlib import Path

from PIL import Image

from synthetic_scenes.minesweeper.game_region import render_game_region

from .minesweeper.board import generate_game
from .minesweeper import classic, windows7

from .applications.sudoku import render_sudoku
from .applications.spreadsheet import render_spreadsheet
from .applications.file_manager import render_file_manager
from .applications.browser import render_browser
from .applications.document_editor import render_document_editor
from .applications.presentation import render_presentation
from .applications.code_editor import render_code_editor
from .applications.terminal import render_terminal
from .applications.calculator import render_calculator
from .applications.settings import render_settings
from .applications.other_puzzle import render_other_puzzle
from .applications.media_viewer import render_media_viewer

from .wallpaper import resolve_wallpaper
from .screenshot import WindowLayer, compose_screenshot
from .generic_window import window_size
from .layout import place_windows


Box = tuple[int, int, int, int]


# Non-Minesweeper application renderers.
# Each renderer accepts seed=... and returns (image, boxes).
APPLICATIONS = {
    "sudoku": {
        "render": render_sudoku,
        "title": "Sudoku",
    },
    "spreadsheet": {
        "render": render_spreadsheet,
        "title": "Spreadsheet",
    },
    "file_manager": {
        "render": render_file_manager,
        "title": "Files",
    },
    "browser": {
        "render": render_browser,
        "title": "Browser",
    },
    "document_editor": {
        "render": render_document_editor,
        "title": "Document Editor",
    },
    "presentation": {
        "render": render_presentation,
        "title": "Presentation",
    },
    "code_editor": {
        "render": render_code_editor,
        "title": "Code Editor",
    },
    "terminal": {
        "render": render_terminal,
        "title": "Terminal",
    },
    "calculator": {
        "render": render_calculator,
        "title": "Calculator",
    },
    "settings": {
        "render": render_settings,
        "title": "Settings",
    },
    "other_puzzle": {
        "render": render_other_puzzle,
        "title": "Puzzle",
    },
    "media_viewer": {
        "render": render_media_viewer,
        "title": "Photos",
    },
}


def application_title(app_type: str) -> str:
    """Return the default window title."""
    if app_type == "minesweeper":
        return "Minesweeper"

    if app_type not in APPLICATIONS:
        raise NotImplementedError(
            f"Application not implemented: {app_type}"
        )

    return APPLICATIONS[app_type]["title"]


def render_minesweeper(
    spec: dict,
) -> tuple[Image.Image, dict[str, Box]]:
    """Render one Minesweeper game at native resolution."""
    renderer = spec["renderer"]
    board = spec["board"]
    params = spec["game"]

    if renderer not in {"classic", "windows7", "windows7_reference"}:
        raise NotImplementedError(
            f"Unsupported Minesweeper renderer: {renderer}"
        )

    game = generate_game(
        rows=board["rows"],
        columns=board["columns"],
        mine_count=board["mine_count"],
        status=params["status"],
        progress=params["progress"],
        seed=params["seed"],
        elapsed_seconds=params["elapsed_seconds"],
    )

    if renderer == "windows7":
        image, metadata = render_game_region(game, style="windows7", cell_size=36)
        return image, metadata["boxes"]
    if renderer == "windows7_reference":
        return windows7.render_game_region(game)
    return classic.render_game_region(game)


def render_application(
    window: dict,
    scene_seed: int,
) -> tuple[Image.Image, dict[str, Box]]:
    """Render one application's content."""

    app_type = window["type"]

    if app_type == "minesweeper":
        return render_minesweeper(window["content"])

    if app_type not in APPLICATIONS:
        raise NotImplementedError(
            f"Application renderer not implemented: {app_type}"
        )

    # Deterministic, independent content seed.
    key = f"{scene_seed}:{app_type}:{window['id']}"
    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    seed = int.from_bytes(digest[:4], "big")

    renderer = APPLICATIONS[app_type]["render"]
    return renderer(seed=seed)


def render_scene(
    scene: dict,
    *,
    wallpaper_root: Path | None = None,
) -> tuple[Image.Image, list[dict[str, Box]], dict]:
    """Render one sampled scene.

    Returns:
        image: RGB screenshot
        annotations: per-window screenshot-local boxes
        resolved_scene: scene with asset and layout information

    Does not modify the input scene.
    """
    screen = scene["screen"]
    size = (screen["width"], screen["height"])
    desktop = scene["desktop"]
    scene_seed = scene.get("_meta", {}).get("seed", 42)

    # Check unsupported requests before doing rendering work.
    if screen.get("scale", 1.0) != 1.0:
        raise NotImplementedError(
            "Display scaling is not implemented yet"
        )

    for index, window in enumerate(scene["windows"]):
        app_type = window["type"]

        # Generic applications require no game-specific checks.
        if app_type in APPLICATIONS:
            continue

        if app_type != "minesweeper":
            raise NotImplementedError(
                f"Window {index}: unsupported application "
                f"{app_type!r}"
            )

        # Minesweeper-specific validation.
        content = window["content"]

        if content["renderer"] not in {"classic", "windows7", "windows7_reference"}:
            raise NotImplementedError(
                f"Window {index}: unsupported game renderer "
                f"{content['renderer']!r}"
            )

        if content.get("host", "native") != "native":
            raise NotImplementedError(
                f"Window {index}: browser hosting is not ready"
            )

        arrangement = content.get("arrangement", "foreground")
        if arrangement not in {"foreground", "beside_other_apps"}:
            raise NotImplementedError(
                f"Window {index}: unsupported arrangement "
                f"{arrangement!r}"
            )

        visibility = content.get("visibility", {})
        if (
            visibility.get("category", "full") != "full"
            or visibility.get("fraction", 1.0) != 1.0
        ):
            raise NotImplementedError(
                f"Window {index}: partial visibility "
                "is not implemented yet"
            )

    # 1. Resolve the actual wallpaper asset.
    if wallpaper_root is None:
        wallpaper, resolved = resolve_wallpaper(scene)
    else:
        wallpaper, resolved = resolve_wallpaper(
            scene, root=Path(wallpaper_root)
        )

    # 2. Generate native game content.
    windows = []
    scalable = []
    original_sizes = []

    for index, spec in enumerate(scene["windows"]):
        try:
            content_image, boxes = render_application(
                spec, scene_seed
            )
        except (ValueError, RuntimeError) as exc:
            raise ValueError(
                f"Could not render window {index}"
            ) from exc

        original_sizes.append(content_image.size)

        windows.append(
            WindowLayer(
                content_image=content_image,
                content_boxes=boxes,
                position=(0, 0),
                title=spec.get(
                    "title",
                    application_title(spec["type"]),
                ),
                menu_items=tuple(
                    spec.get(
                        "menu_items",
                        ("Game", "Help")
                        if spec["type"] == "minesweeper"
                        else (),
                    )
                ),
            )
        )

        scalable.append(
            spec["type"] == "minesweeper"
            and spec["content"]["renderer"] in {"windows7", "windows7_reference"}
        )

    # 3. Fit and position windows.
    # Minesweeper windows cannot overlap each other.
    # Background applications may overlap any window.
    protected_indices = [
        i
        for i, spec in enumerate(scene["windows"])
        if spec["type"] == "minesweeper"
    ]

    windows = place_windows(
        windows,
        size=size,
        desktop=desktop,
        seed=scene_seed,
        scalable=scalable,
        protected_indices=protected_indices,
        allow_background_overlap=True,
    )

    # 4. Compose the screenshot using Skia.
    # Draw background applications first and Minesweeper last.
    # Preserve relative order within each group.
    protected = set(protected_indices)

    draw_order = [
        i for i in range(len(windows))
        if i not in protected
    ] + [
        i for i in range(len(windows))
        if i in protected
    ]

    image, drawn_annotations = compose_screenshot(
        wallpaper=wallpaper,
        size=size,
        desktop=desktop,
        windows=[windows[i] for i in draw_order],
    )

    # Restore annotation order to match the original scene.
    annotations = [None] * len(windows)

    for index, boxes in zip(draw_order, drawn_annotations):
        annotations[index] = boxes

    # 5. Record the actual rendered geometry.
    for spec, window, original_size in zip(
        resolved["windows"], windows, original_sizes
    ):
        final_size = window.content_image.size

        spec["position"] = list(window.position)
        spec["render"] = {
            "original_content_size": list(original_size),
            "content_size": list(final_size),
            "window_size": list(
                window_size(
                    window.content_image,
                    window.menu_items,
                    style=desktop["style"],
                )
            ),
            "scale": round(
                final_size[0] / original_size[0], 6
            ),
        }

    return image, annotations, resolved
