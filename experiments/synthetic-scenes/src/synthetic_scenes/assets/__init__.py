"""Prepare optional rendering assets on demand."""

from threading import Lock
from typing import Literal


AssetCategory = Literal["wallpapers", "minesweeper_windows7"]
_preparation_lock = Lock()
_prepared: set[AssetCategory] = set()


def prepare_assets(*categories: AssetCategory) -> None:
    """Prepare selected categories, or all categories when called without arguments."""
    from .catalogs import (
        artwork,
        macos,
        minesweeper_windows7,
        photographs,
        windows,
        windows_7,
        windows_10,
        windows_11,
        windows_xp,
    )

    steps = {
        "wallpapers": (artwork, macos, photographs, windows, windows_xp,
                       windows_7, windows_10, windows_11),
        "minesweeper_windows7": (minesweeper_windows7,),
    }
    selected = tuple(dict.fromkeys(categories)) or tuple(steps)
    for category in selected:
        if category not in steps:
            raise ValueError(f"Unknown asset category: {category}")

    total = sum(len(steps[category]) for category in selected)
    with _preparation_lock:
        index = 0
        for category in selected:
            for group in steps[category]:
                index += 1
                name = group.__name__.rsplit(".", 1)[-1]
                if category in _prepared:
                    print(f"[{index}/{total}] {name}: already complete, skipping", flush=True)
                else:
                    print(f"[{index}/{total}] Preparing {name}", flush=True)
                    group.download()
            _prepared.add(category)


def _ensure_assets(category: AssetCategory) -> None:
    if category not in _prepared:
        prepare_assets(category)
