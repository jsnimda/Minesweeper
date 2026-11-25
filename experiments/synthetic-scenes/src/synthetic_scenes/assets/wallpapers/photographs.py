import json
import random
import shutil

from tqdm import tqdm

from ..git_download import clone_repository, sparse_checkout
from ..paths import ASSET_DIR


REPO_URL = "https://github.com/joshkaspar/vistarium.git"

CATALOG_PATH = "docs/data.json"
SOURCE = ASSET_DIR / "sources" / "vistarium"
DESTINATION = ASSET_DIR / "wallpapers" / "photographs"


def catalog() -> list[dict]:
    clone_repository(REPO_URL, SOURCE)

    metadata = SOURCE / CATALOG_PATH

    if not metadata.is_file():
        sparse_checkout(SOURCE, [CATALOG_PATH])

    return json.loads(metadata.read_text(encoding="utf-8"))


def candidates(entries: list[dict]) -> list[str]:
    paths = [
        f"docs/{photo['thumb']}"
        for photo in entries
        if photo.get("license") == "Public domain/Full"
        and photo.get("license_confidence") == "confirmed"
        and photo.get("people_prominence") == "none"
        and photo.get("thumb")
    ]

    return list(dict.fromkeys(paths))


def download(
    count: int | None = 500,
    seed: int = 42,
) -> None:
    marker = DESTINATION / ".complete"
    configuration = f"{count}:{seed}"

    if marker.is_file() and marker.read_text() == configuration:
        return

    selected = candidates(catalog())

    if not selected:
        raise ValueError("No eligible photographs found")

    if count is not None:
        rng = random.Random(seed)
        selected = rng.sample(selected, min(count, len(selected)))

    if any(not (SOURCE / path).is_file() for path in selected):
        sparse_checkout(SOURCE, [CATALOG_PATH, *selected])

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)

    DESTINATION.mkdir(parents=True, exist_ok=True)

    for path in tqdm(selected, desc="Copying photographs", unit="image"):
        source = SOURCE / path
        destination = DESTINATION / source.name

        shutil.copyfile(source, destination)

    marker.write_text(configuration)