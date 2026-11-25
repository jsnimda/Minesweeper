"""Resource loading for the original Windows 7 reference artwork."""

from functools import cache

from PIL import Image, ImageFont

from . import _ensure_assets
from .catalogs.minesweeper_windows7 import DESTINATION


@cache
def windows7_sprites() -> tuple[Image.Image, Image.Image, Image.Image]:
    _ensure_assets("minesweeper_windows7")
    blue = Image.open(DESTINATION / "bluesheetx2.png").convert("RGBA")
    mine = Image.open(DESTINATION / "minesweepersheetx2.png").convert("RGBA")
    other = Image.open(DESTINATION / "othersheetx2.png").convert("RGBA")

    if blue.size != (2033, 1205) or mine.size != (93, 53) or other.size != (2047, 2015):
        raise ValueError("Unexpected Windows 7 atlas dimensions; expected x2 sheets")

    return blue, mine, other


@cache
def windows7_font(size: int) -> ImageFont.FreeTypeFont:
    _ensure_assets("minesweeper_windows7")
    return ImageFont.truetype(DESTINATION / "fonts" / "segoeuil.ttf", size)
