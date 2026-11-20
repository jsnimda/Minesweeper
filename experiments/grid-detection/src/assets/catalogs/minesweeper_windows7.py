import shutil
from pathlib import Path

from PIL import Image, ImageFont

from assets.git_download import clone_sparse


ROOT = Path(__file__).resolve().parents[3]
DESTINATION = ROOT / "data" / "assets" / "minesweeper" / "windows7"
SOURCES = ROOT / "data" / "assets" / "sources"

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
    if (DESTINATION / ".complete").exists():
        print("minesweeper_windows7: already complete, skipping")
        return

    sprite_source = SOURCES / "minesweeper-win7-assets"
    font_source = SOURCES / "WingtipTicketsSaaS-DbPerTenant"

    clone_sparse(SPRITES["repo"], sprite_source, SPRITES["paths"])
    clone_sparse(FONT["repo"], font_source, [FONT["path"]])

    DESTINATION.mkdir(parents=True, exist_ok=True)

    for path in SPRITES["paths"]:
        source = sprite_source / path
        with Image.open(source) as image:
            image.verify()
        shutil.copy2(source, DESTINATION / source.name)

    font = font_source / FONT["path"]
    ImageFont.truetype(font, 30)

    font_dir = DESTINATION / "fonts"
    font_dir.mkdir(exist_ok=True)
    shutil.copy2(font, font_dir / font.name)

    (DESTINATION / ".complete").touch()
    print("minesweeper_windows7: assets prepared")


if __name__ == "__main__":
    download()