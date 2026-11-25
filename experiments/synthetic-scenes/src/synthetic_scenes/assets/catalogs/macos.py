import random
import shutil

from tqdm import tqdm

from ..git_download import clone_sparse
from ..paths import ASSET_DIR


SOURCE = ASSET_DIR / "sources" / "macos-wallpapers"
DESTINATION = ASSET_DIR / "wallpapers" / "os_defaults" / "macos"

MACOS = {
    "repo": "https://github.com/LAYTAT/macOS-Wallpapers.git",
    "candidates": [
        "10-0_10.1.png",
        "10-2.png",
        "10-3.png",
        "10-8.jpg",
        "11-0-Color-Day.jpg",
        "13-Ventura-Dark.jpg",
        "14-Sonoma-Dark.jpg",
        "14-Sonoma-Light.jpg",
        "Desert 4.jpg",
        "Earth Horizon.jpg",
        "Earth and Moon.jpg",
        "Foggy Forest.jpg",
        "Ink Cloud.jpg",
        "Lake.jpg",
        "Lion.jpg",
        "Mountain Range.jpg",
        "Rolling Waves.jpg",
        "Snow.jpg",
        "Yosemite 5.jpg",
    ],
}


def download(count: int = 15, seed: int = 42) -> None:
    if (DESTINATION / ".complete").exists():
        print("macos: already complete, skipping")
        return

    selected = random.Random(seed).sample(MACOS["candidates"], count)

    clone_sparse(MACOS["repo"], SOURCE, selected)

    DESTINATION.mkdir(parents=True, exist_ok=True)

    for path in tqdm(selected, desc="macos", unit="file"):
        source = SOURCE / path
        shutil.copy2(source, DESTINATION / source.name)

    (DESTINATION / ".complete").touch()
    print(f"macos: copied {len(selected)} wallpapers")


if __name__ == "__main__":
    download()
