import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .paths import ASSET_DIR


def download_wallpapers(catalog: dict, destination: Path) -> None:
    if (destination / ".complete").exists():
        print(f"{destination.name}: already complete, skipping")
        return

    repo_name = Path(catalog["repo"]).stem
    source = ASSET_DIR / "sources" / repo_name

    def copy_image(path: str, target: Path) -> None:
        src = source / path
        if src.suffix.lower() == ".bmp":
            with Image.open(src) as image:
                image.save(target / f"{src.stem}.webp", "WEBP")
        else:
            shutil.copy2(src, target / src.name)

    other_dir = destination / "other"
    other_dir.mkdir(parents=True, exist_ok=True)

    files = [(path, destination) for path in catalog["fixed"]]
    files.extend((path, other_dir) for path in catalog["other"])
    for path, target in tqdm(files, desc=destination.name, unit="file"):
        copy_image(path, target)

    (destination / ".complete").touch()
    print(f"{destination.name}: wallpapers prepared")
