from functools import cache
from pathlib import Path

from . import artwork, photographs
from ..paths import ASSET_DIR


@cache
def get_wallpapers(category: str) -> tuple[Path, ...]:
    if category not in {"artwork", "photographs"}:
        raise ValueError(f"Unknown wallpaper category: {category}")

    artwork.download()
    photographs.download()

    directory = ASSET_DIR / "wallpapers" / category

    return tuple(sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ))