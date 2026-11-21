
"""Resolve synthetic desktop wallpapers from local assets."""

import random
from copy import deepcopy
from pathlib import Path

from PIL import Image


WALLPAPER_DIR = (
    Path(__file__).resolve().parents[3]
    / "data" / "assets" / "wallpapers"
)

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

WEIGHTED_FAMILIES = {
    "xp", "windows7", "windows10", "windows11",
}


def _read_weight(path: Path, default: float = 1.0) -> float:
    """Read a positive weight from an optional text file."""
    if not path.is_file():
        return default

    weight = float(path.read_text(encoding="utf-8").strip())

    if weight <= 0:
        raise ValueError(f"Invalid wallpaper weight: {path}")

    return weight


def _wallpaper_candidates(
    directory: Path,
    weighted: bool,
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
        weight = 1.0

        if weighted:
            relative = image.relative_to(directory)

            if relative.parts[0] == "other":
                # All images under other/ share the parent
                # directory's other.txt weight.
                weight = _read_weight(
                    directory / "other.txt"
                )
            else:
                # Example: Bliss.webp -> Bliss.txt
                weight = _read_weight(
                    image.with_suffix(".txt")
                )

        weights.append(weight)

    return images, weights


def resolve_wallpaper(
    scene: dict,
    root: Path = WALLPAPER_DIR,
) -> tuple[Image.Image, dict]:
    """Load a wallpaper and return the resolved scene.

    The original scene and asset files are unchanged.
    """
    root = Path(root)
    wallpaper = scene["desktop"]["wallpaper"]
    category = wallpaper["type"]
    family = wallpaper.get("family")

    if category == "os_defaults":
        if family not in FAMILY_DIRS:
            raise ValueError(
                f"Unknown wallpaper family: {family}"
            )
        directory = root / "os_defaults" / FAMILY_DIRS[family]
        weighted = family in WEIGHTED_FAMILIES

    elif category == "photograph":
        directory = root / "photographs"
        weighted = False

    elif category == "artwork":
        directory = root / "artwork"
        weighted = False

    else:
        raise ValueError(
            f"Unknown wallpaper type: {category}"
        )

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
        images, weights = _wallpaper_candidates(
            directory, weighted
        )

        if not images:
            raise FileNotFoundError(
                f"No wallpapers found in: {directory}"
            )

        # Independent from the sampler's RNG stream.
        # Stable for the same scene seed and asset inventory.
        seed = scene.get("_meta", {}).get("seed", 0)
        rng = random.Random(
            f"{seed}:{category}:{family or ''}"
        )

        selected = rng.choices(
            images, weights=weights, k=1
        )[0]

    with Image.open(selected) as image:
        result = image.convert("RGB")

    resolved = deepcopy(scene)
    resolved["desktop"]["wallpaper"]["asset"] = (
        selected.relative_to(root).as_posix()
    )

    return result, resolved
