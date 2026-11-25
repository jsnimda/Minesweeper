"""Resolve wallpapers using the grid-detection sampling weights."""

from pathlib import Path

from PIL import Image

from synthetic_scenes.assets.wallpaper import WALLPAPER_DIR, resolve_wallpaper as resolve_shared_wallpaper

from ..dataset.scope import WALLPAPER_WEIGHTS


def resolve_wallpaper(
    scene: dict,
    root: Path = WALLPAPER_DIR,
) -> tuple[Image.Image, dict]:
    return resolve_shared_wallpaper(scene, root=root, family_weights=WALLPAPER_WEIGHTS)
