import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def download_wallpapers(catalog: dict, destination: Path) -> None:
    if (destination / ".complete").exists():
        print(f"{destination.name}: already complete, skipping")
        return

    repo_name = Path(catalog["repo"]).stem
    source = ROOT / "data" / "assets" / "sources" / repo_name

    def copy_image(path: str, target: Path) -> str:
        src = source / path
        name = src.name

        if src.suffix.lower() == ".bmp":
            name = f"{Path(name).stem}.webp"
            with Image.open(src) as image:
                image.save(target / name, "WEBP")
        else:
            shutil.copy2(src, target / name)

        return name

    destination.mkdir(parents=True, exist_ok=True)

    for path, weight in catalog["fixed"].items():
        name = copy_image(path, destination)
        (destination / f"{Path(name).stem}.txt").write_text(str(weight))

    other = catalog["other"]
    other_dir = destination / "other"
    other_dir.mkdir(exist_ok=True)

    for path in other["candidates"]:
        copy_image(path, other_dir)

    (destination / "other.txt").write_text(str(other["weight"]))
    (destination / ".complete").touch()

    print(f"{destination.name}: wallpapers prepared")