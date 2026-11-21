
"""Fit and position synthetic application windows.

Windows remain fully visible and non-overlapping.
Minesweeper content is scaled only when necessary.
"""

import random
from math import ceil
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
    min_scale: float = 0.5,
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
    layout_attempts: int = 60,
) -> list[WindowLayer]:
    """Sample target scales and retry complete window layouts.

    Protected windows cannot overlap.
    Scalable protected windows sample scales in [0.5, 1.0].
    Fixed protected windows retain their native content size.
    Background windows may overlap when enabled.
    """
    if gap < 0 or attempts < 0 or layout_attempts < 1:
        raise ValueError("Invalid placement attempt settings")

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
    area_w = right - left
    area_h = bottom - top
    style = desktop["style"]

    frame = (
        AERO_FRAME if style == "aero"
        else CLASSIC_FRAME if style == "classic"
        else 0
    )

    rng = random.Random(seed)

    # Background applications only need to be prepared once.
    backgrounds = {}

    for i, window in enumerate(windows):
        if i not in protected:
            backgrounds[i] = prepare_window(
                window,
                (area_w, area_h),
                allow_scaling=True,
                style=style,
            )

    def dimensions(window):
        return window_size(
            window.content_image,
            window.menu_items,
            style=style,
        )

    def sample_targets():
        """Create fresh game windows for one layout attempt."""
        prepared = {}

        for i in sorted(protected):
            window = windows[i]

            if not scalable[i]:
                # Classic: never shrink.
                prepared[i] = prepare_window(
                    window,
                    (area_w, area_h),
                    allow_scaling=False,
                    style=style,
                )
                continue

            # Windows 7: independent random scale.
            scale = rng.uniform(0.5, 1.0)

            menu_h = (
                MENU_HEIGHT if window.menu_items else 0
            )

            max_content_w = (
                area_w - 2 * CONTENT_PADDING - 2 * frame
            )
            max_content_h = (
                area_h
                - title_height(style)
                - menu_h
                - 2 * CONTENT_PADDING
                - frame
            )

            target_w = min(
                max_content_w,
                ceil(window.content_image.width * scale),
            )
            target_h = min(
                max_content_h,
                ceil(window.content_image.height * scale),
            )

            # Always resize from original content, never from
            # an image resized during a previous attempt.
            image, boxes = fit_content(
                window.content_image,
                dict(window.content_boxes),
                (target_w, target_h),
                min_scale=0.5,
            )

            prepared[i] = replace(
                window,
                content_image=image,
                content_boxes=boxes,
            )

        return prepared

    def try_placement(prepared):
        """Attempt one complete arrangement."""
        prepared = {**backgrounds, **prepared}

        # Reserve space for Minesweeper first.
        ordered = sorted(
            prepared,
            key=lambda i: (
                i in protected,
                dimensions(prepared[i])[0]
                * dimensions(prepared[i])[1],
            ),
            reverse=True,
        )

        result = {}
        occupied = []

        for i in ordered:
            window = prepared[i]
            w, h = dimensions(window)

            max_x = right - w
            max_y = bottom - h

            if max_x < left or max_y < top:
                return None

            def fits(x, y):
                if (
                    allow_background_overlap
                    and i not in protected
                ):
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

            # Try random positions first for better spatial diversity.
            candidates = [
                (
                    rng.randint(left, max_x),
                    rng.randint(top, max_y),
                )
                for _ in range(attempts)
            ]

            # Use corners as a fallback for difficult layouts.
            corners = [
                (left, top),
                (max_x, top),
                (left, max_y),
                (max_x, max_y),
            ]

            rng.shuffle(corners)
            candidates.extend(corners)

            position = next(
                (
                    (x, y)
                    for x, y in candidates
                    if fits(x, y)
                ),
                None,
            )

            if position is None:
                return None

            x, y = position

            result[i] = replace(
                window,
                position=position,
            )

            # Only protected windows need to reserve space
            # when background overlap is enabled.
            if i in protected or not allow_background_overlap:
                occupied.append(
                    (x, y, x + w, y + h)
                )

        return [
            result[i]
            for i in range(len(windows))
        ]

    # Retry the complete layout, including new target scales.
    for _ in range(layout_attempts):
        try:
            targets = sample_targets()
        except ValueError:
            # A sampled scale could be too small or too large
            # for the available work area.
            continue

        placed = try_placement(targets)

        if placed is not None:
            return placed

    raise ValueError(
        f"Could not place {len(protected)} protected windows "
        f"after {layout_attempts} layout attempts"
    )
