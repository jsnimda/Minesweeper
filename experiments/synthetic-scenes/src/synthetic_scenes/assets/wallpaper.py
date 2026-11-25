"""Resolve synthetic desktop wallpapers from local assets."""

import random
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from PIL import Image

from . import _ensure_assets
from .paths import ASSET_DIR


WALLPAPER_DIR = ASSET_DIR / "wallpapers"

FAMILY_DIRS = {
    "xp": "windows_xp",
    "windows7": "windows_7",
    "windows10": "windows_10",
    "windows11": "windows_11",
    "macos": "macos",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff",
}


def _wallpaper_candidates(
    directory: Path,
    image_weights: Mapping[str, float],
) -> tuple[list[Path], list[float]]:
    """Collect images and their sampling weights."""
    images = sorted(
        (
            path for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(directory).as_posix(),
    )

    weights = []
    for image in images:
        relative = image.relative_to(directory)
        key = "other/" if relative.parts[0] == "other" else relative.as_posix()
        weights.append(image_weights.get(key, 1.0))

    return images, weights


def sample_wallpaper(
    wallpaper: Mapping[str, str],
    *,
    seed: int = 0,
    root: Path | None = None,
    family_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> Path:
    """Select a local wallpaper, preparing the shared collections on first use."""
    root = WALLPAPER_DIR if root is None else Path(root)
    category = wallpaper["type"]
    family = wallpaper.get("family")

    if category == "os_defaults":
        if family not in FAMILY_DIRS:
            raise ValueError(
                f"Unknown wallpaper family: {family}"
            )
        directory = root / "os_defaults" / FAMILY_DIRS[family]

    elif category == "photograph":
        directory = root / "photographs"

    elif category == "artwork":
        directory = root / "artwork"

    else:
        raise ValueError(
            f"Unknown wallpaper type: {category}"
        )

    if root == WALLPAPER_DIR:
        _ensure_assets("wallpapers")

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Wallpaper directory not found: {directory}"
        )

    # An already resolved scene uses its exact asset.
    # Do not sample a second wallpaper.
    if "asset" in wallpaper:
        selected = root / wallpaper["asset"]
        if not selected.is_relative_to(directory) or (
            not selected.is_file()
            or selected.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            raise ValueError(
                f"Invalid resolved wallpaper: {selected}"
            )
    else:
        image_weights = {}
        if category == "os_defaults" and family_weights:
            image_weights = family_weights.get(family, {})
        images, weights = _wallpaper_candidates(directory, image_weights)

        if not images:
            raise FileNotFoundError(
                f"No wallpapers found in: {directory}"
            )

        # Independent from the sampler's RNG stream.
        # Stable for the same scene seed and asset inventory.
        rng = random.Random(
            f"{seed}:{category}:{family or ''}"
        )

        selected = rng.choices(
            images, weights=weights, k=1
        )[0]

    return selected


def resolve_wallpaper(
    scene: dict,
    root: Path | None = None,
    *,
    family_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[Image.Image, dict]:
    """Load the selected wallpaper and record its relative path in a scene copy."""
    root = WALLPAPER_DIR if root is None else Path(root)
    selected = sample_wallpaper(
        scene["desktop"]["wallpaper"],
        seed=scene.get("_meta", {}).get("seed", 0),
        root=root,
        family_weights=family_weights,
    )
    with Image.open(selected) as image:
        result = image.convert("RGB")

    resolved = deepcopy(scene)
    resolved["desktop"]["wallpaper"]["asset"] = (
        selected.relative_to(root).as_posix()
    )

    return result, resolved
