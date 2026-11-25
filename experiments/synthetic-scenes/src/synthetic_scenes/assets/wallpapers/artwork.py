import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm

from ..git_download import clone_repository, sparse_checkout
from ..paths import ASSET_DIR


REPO_URL = "https://github.com/DenverCoder1/minimalistic-wallpaper-collection.git"

CATALOG_PATH = "api/generated/images.json"
SOURCE = ASSET_DIR / "sources" / "minimalistic-wallpaper-collection"
DESTINATION = ASSET_DIR / "wallpapers" / "artwork"

MAX_SIZE = 1024 ** 2
RASTER = {".png", ".jpg", ".jpeg", ".webp"}


def catalog() -> list[dict]:
    clone_repository(REPO_URL, SOURCE)

    metadata = SOURCE / CATALOG_PATH

    if not metadata.is_file():
        sparse_checkout(SOURCE, [CATALOG_PATH])

    return json.loads(metadata.read_text(encoding="utf-8"))


def candidates(entries: list[dict]) -> list[dict]:
    selected = {}

    for item in entries:
        if item["type"] != "file":
            continue

        if item["size"] > MAX_SIZE:
            continue

        if Path(item["name"]).suffix.lower() not in RASTER:
            continue

        selected.setdefault(item["sha"], item)

    return list(selected.values())


def download(
    count: int | None = 200,
    seed: int = 42,
) -> None:
    marker = DESTINATION / ".complete"
    configuration = f"{count}:{seed}"

    if marker.is_file() and marker.read_text() == configuration:
        return

    selected = candidates(catalog())

    if not selected:
        raise ValueError("No eligible artwork found")

    if count is not None:
        rng = random.Random(seed)
        selected = rng.sample(selected, min(count, len(selected)))

    paths = [item["path"] for item in selected]

    if any(not (SOURCE / path).is_file() for path in paths):
        sparse_checkout(SOURCE, [CATALOG_PATH, *paths])

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)

    DESTINATION.mkdir(parents=True, exist_ok=True)

    for item in tqdm(selected, desc="Copying artwork", unit="image"):
        source = SOURCE / item["path"]
        destination = DESTINATION / item["name"]

        shutil.copyfile(source, destination)

    marker.write_text(configuration)