
"""Fit and position synthetic application windows.

Windows remain fully visible and non-overlapping.
Minesweeper content is scaled only when necessary.
"""

import random
from collections.abc import Sequence
from dataclasses import replace

from PIL import Image

from .generic_window import (
    AERO_FRAME,
    CLASSIC_FRAME,
    CONTENT_PADDING,
    MENU_HEIGHT,
    MIN_WIDTH,
    Box,
    title_height,
    window_size,
)
from .screenshot import WindowLayer, _intersects, _work_area


def fit_content(
    image: Image.Image,
    boxes: dict[str, Box],
    max_size: tuple[int, int],
    *,
    min_scale: float = 0.65,
) -> tuple[Image.Image, dict[str, Box]]:
    """Shrink content uniformly while preserving annotation boxes."""
    max_w, max_h = max_size

    if max_w <= 0 or max_h <= 0:
        raise ValueError("No space available for window content")

    scale = min(
        1.0,
        max_w / image.width,
        max_h / image.height,
    )

    if scale < min_scale:
        raise ValueError(
            f"Required scale {scale:.2f} is below "
            f"minimum {min_scale:.2f}"
        )

    if scale == 1.0:
        return image, dict(boxes)

    new_w = max(1, int(image.width * scale))
    new_h = max(1, int(image.height * scale))

    resized = image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS,
    )

    # Use actual dimensions after integer rounding.
    sx = new_w / image.width
    sy = new_h / image.height

    scaled_boxes = {
        name: (
            round(x0 * sx),
            round(y0 * sy),
            round(x1 * sx),
            round(y1 * sy),
        )
        for name, (x0, y0, x1, y1) in boxes.items()
    }

    return resized, scaled_boxes


def prepare_window(
    window: WindowLayer,
    work_size: tuple[int, int],
    *,
    allow_scaling: bool = False,
    style: str = "modern",
) -> WindowLayer:
    """Prepare one window for the available desktop area."""
    area_w, area_h = work_size

    menu_h = MENU_HEIGHT if window.menu_items else 0

    frame = (
        AERO_FRAME if style == "aero"
        else CLASSIC_FRAME if style == "classic"
        else 0
    )

    max_content_size = (
        area_w - 2 * CONTENT_PADDING - 2 * frame,
        area_h - title_height(style) - menu_h - 2 * CONTENT_PADDING - frame,
    )

    if allow_scaling:
        image, boxes = fit_content(
            window.content_image,
            dict(window.content_boxes),
            max_content_size,
        )
        window = replace(
            window,
            content_image=image,
            content_boxes=boxes,
        )

    w, h = window_size(
        window.content_image,
        window.menu_items,
        style=style,
    )

    if w > area_w or h > area_h or MIN_WIDTH > area_w:
        raise ValueError(
            f"Window {w}x{h} cannot fit in "
            f"work area {area_w}x{area_h}"
        )

    return window


def place_windows(
    windows: Sequence[WindowLayer],
    size: tuple[int, int],
    desktop: dict,
    seed: int = 42,
    *,
    scalable: Sequence[bool] | None = None,
    protected_indices: Sequence[int] = (),
    allow_background_overlap: bool = False,
    gap: int = 12,
    attempts: int = 200,
) -> list[WindowLayer]:
    """Fit and position windows inside the desktop work area.

    Protected windows are positioned first and cannot overlap
    one another.

    When allow_background_overlap=True, background windows may
    overlap any other window. The compositor must draw protected
    windows last to keep them fully visible.

    Returned windows remain in their original input order.
    """
    if gap < 0 or attempts < 0:
        raise ValueError("gap and attempts must be nonnegative")

    if scalable is None:
        scalable = [False] * len(windows)

    if len(scalable) != len(windows):
        raise ValueError("scalable must match windows length")

    protected = set(protected_indices)

    if any(
        not isinstance(i, int) or i < 0 or i >= len(windows)
        for i in protected
    ):
        raise ValueError("Invalid protected window index")

    left, top, right, bottom = _work_area(size, desktop)
    area_size = (right - left, bottom - top)
    style = desktop["style"]

    prepared = [
        prepare_window(
            window,
            area_size,
            allow_scaling=can_scale,
            style=style,
        )
        for window, can_scale in zip(windows, scalable)
    ]

    # Position protected windows first.
    # Within each group, position larger windows first.
    def placement_priority(item):
        index, window = item

        w, h = window_size(
            window.content_image,
            window.menu_items,
            style=style,
        )

        return (index in protected, w * h)

    ordered = sorted(
        enumerate(prepared),
        key=placement_priority,
        reverse=True,
    )

    rng = random.Random(seed)

    # Only needed for collision checking.
    occupied: list[Box] = []

    # Preserve the association between each window and its index.
    result: dict[int, WindowLayer] = {}

    for index, window in ordered:
        w, h = window_size(
            window.content_image,
            window.menu_items,
            style=style,
        )

        max_x = right - w
        max_y = bottom - h

        def fits(x: int, y: int) -> bool:
            # Background windows may overlap anything, but they
            # will be drawn behind protected windows later.
            if allow_background_overlap and index not in protected:
                return True

            candidate = (x, y, x + w, y + h)

            return not any(
                _intersects(
                    candidate,
                    (
                        ox0 - gap,
                        oy0 - gap,
                        ox1 + gap,
                        oy1 + gap,
                    ),
                )
                for ox0, oy0, ox1, oy1 in occupied
            )

        position = None

        # Try deterministic random placements.
        for _ in range(attempts):
            x = rng.randint(left, max_x)
            y = rng.randint(top, max_y)

            if fits(x, y):
                position = (x, y)
                break

        # Fall back to scanning the remaining work area.
        if position is None:
            xs = list(range(left, max_x + 1, 8))
            ys = list(range(top, max_y + 1, 8))

            if not xs or xs[-1] != max_x:
                xs.append(max_x)

            if not ys or ys[-1] != max_y:
                ys.append(max_y)

            for y in ys:
                for x in xs:
                    if fits(x, y):
                        position = (x, y)
                        break

                if position is not None:
                    break

        if position is None:
            raise ValueError(
                f"Could not position window {index} ({w}x{h})"
            )

        x, y = position

        occupied.append((x, y, x + w, y + h))

        result[index] = replace(
            window,
            position=position,
        )

    # Keep original input order for metadata and annotations.
    return [result[i] for i in range(len(windows))]
