import shutil

from tqdm import tqdm

from ..git_download import clone_sparse
from ..paths import ASSET_DIR


SOURCE = ASSET_DIR / "sources" / "gnome-backgrounds"
DESTINATION = ASSET_DIR / "wallpapers" / "artwork"

ARTWORK = {
    "repo": "https://github.com/GNOME/gnome-backgrounds.git",
    "paths": [
        "backgrounds/adwaita-thumbnail.png",
        "backgrounds/amber-thumbnail.png",
        "backgrounds/balls-thumbnail.png",
        "backgrounds/blendpills-thumbnail.png",
        "backgrounds/blobs-thumbnail.png",
        "backgrounds/curvaturingster-thumbnail.png",
        "backgrounds/curvy-thumbnail.png",
        "backgrounds/dithered-sun-thumbnail.png",
        "backgrounds/drool-thumbnail.png",
        "backgrounds/fold-thumbnail.png",
        "backgrounds/glass-chip-thumbnail.png",
        "backgrounds/glass-stripes-thumbnail.png",
        "backgrounds/lcd-rainbow-thumbnail.png",
        "backgrounds/map-thumbnail.png",
        "backgrounds/morphogenesis-thumbnail.png",
        "backgrounds/pills-thumbnail.png",
        "backgrounds/pixel-pusher-thumbnail.png",
        "backgrounds/progress-thumbnail.png",
        "backgrounds/progressive-trout-thumbnail.png",
        "backgrounds/sheet-thumbnail.png",
        "backgrounds/symbolic-thumbnail.png",
        "backgrounds/tarka-thumbnail.png",
        "backgrounds/tarkov-pills-thumbnail.png",
        "backgrounds/vnc-thumbnail.png",
    ],
}


def download() -> None:
    if (DESTINATION / ".complete").exists():
        print("artwork: already complete, skipping")
        return

    clone_sparse(ARTWORK["repo"], SOURCE, ARTWORK["paths"])

    DESTINATION.mkdir(parents=True, exist_ok=True)

    for path in tqdm(ARTWORK["paths"], desc="artwork", unit="file"):
        source = SOURCE / path
        shutil.copy2(source, DESTINATION / source.name)

    (DESTINATION / ".complete").touch()


if __name__ == "__main__":
    download()
