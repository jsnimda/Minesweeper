import json
import random
import shutil
from pathlib import Path

from assets.git_download import clone_sparse


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data" / "assets" / "sources" / "vistarium"
DESTINATION = ROOT / "data" / "assets" / "wallpapers" / "photographs"

REPO_URL = "https://github.com/joshkaspar/vistarium.git"


def download(count: int = 100, seed: int = 42) -> None:
    if count < 0:
        raise ValueError("count must be non-negative")

    if (DESTINATION / ".complete").exists():
        print("photographs: already complete, skipping")
        return

    # Stage 1: fetch metadata.
    if not (SOURCE / "docs" / "data.json").is_file():
        clone_sparse(REPO_URL, SOURCE, ["docs/data.json"])

    photos = json.loads(
        (SOURCE / "docs" / "data.json").read_text(encoding="utf-8")
    )

    candidates = [
        photo
        for photo in photos
        if photo.get("license") == "Public domain/Full"
        and photo.get("license_confidence") == "confirmed"
        and photo.get("people_prominence") == "none"
        and photo.get("thumb")
    ]

    selected = random.Random(seed).sample(
        candidates, k=min(count, len(candidates))
    )

    # Stage 2: fetch selected thumbnails.
    paths = ["docs/data.json"] + [
        f"docs/{photo['thumb']}" for photo in selected
    ]

    if any(not (SOURCE / path).is_file() for path in paths):
        clone_sparse(REPO_URL, SOURCE, paths)

    DESTINATION.mkdir(parents=True, exist_ok=True)

    for photo in selected:
        src = SOURCE / "docs" / photo["thumb"]
        shutil.copy2(src, DESTINATION / src.name)

    (DESTINATION / ".complete").touch()
    print(f"photographs: copied {len(selected)} images")


if __name__ == "__main__":
    download()