import shutil
from functools import cache

from PIL import Image, ImageFont
from tqdm import tqdm

from ..git_download import clone_repository, sparse_checkout
from ..paths import ASSET_DIR


DESTINATION = ASSET_DIR / "minesweeper" / "windows7"
SOURCES = ASSET_DIR / "sources"

SPRITES = {
    "repo": "https://github.com/jsnimda/minesweeper-win7-assets.git",
    "paths": [
        "bluesheetx2.png",
        "minesweepersheetx2.png",
        "othersheetx2.png",
    ],
}

FONT = {
    "repo": "https://github.com/microsoft/WingtipTicketsSaaS-DbPerTenant.git",
    "path": "App/src/Events-TenantUserApp/wwwroot/Fonts/segoeuil.ttf",
}


def download() -> None:
    marker = DESTINATION / ".complete"

    if marker.is_file():
        return

    sprite_source = SOURCES / "minesweeper-win7-assets"
    font_source = SOURCES / "WingtipTicketsSaaS-DbPerTenant"

    clone_repository(SPRITES["repo"], sprite_source)
    sparse_checkout(sprite_source, SPRITES["paths"])

    clone_repository(FONT["repo"], font_source)
    sparse_checkout(font_source, [FONT["path"]])

    DESTINATION.mkdir(parents=True, exist_ok=True)

    font_dir = DESTINATION / "fonts"
    font_dir.mkdir(exist_ok=True)

    files = [
        (sprite_source / path, DESTINATION / path)
        for path in SPRITES["paths"]
    ]

    font = font_source / FONT["path"]
    files.append((font, font_dir / font.name))

    for source, destination in tqdm(
        files,
        desc="Copying Windows 7 assets",
        unit="file",
    ):
        shutil.copyfile(source, destination)

    marker.touch()


@cache
def windows7_sprites() -> tuple[Image.Image, Image.Image, Image.Image]:
    download()

    blue = Image.open(DESTINATION / "bluesheetx2.png").convert("RGBA")
    mine = Image.open(DESTINATION / "minesweepersheetx2.png").convert("RGBA")
    other = Image.open(DESTINATION / "othersheetx2.png").convert("RGBA")

    if blue.size != (2033, 1205) or mine.size != (93, 53) or other.size != (2047, 2015):
        raise ValueError("Unexpected Windows 7 atlas dimensions; expected x2 sheets")

    return blue, mine, other


@cache
def windows7_font(size: int) -> ImageFont.FreeTypeFont:
    download()

    return ImageFont.truetype(
        DESTINATION / "fonts" / "segoeuil.ttf",
        size,
    )