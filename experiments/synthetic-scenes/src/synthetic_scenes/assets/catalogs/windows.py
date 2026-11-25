from ..git_download import clone_sparse
from ..paths import ASSET_DIR


DESTINATION = ASSET_DIR / "sources" / "windows-ui-assets"

WINDOWS = {
    "repo": "https://github.com/bartekl1/windows-ui-assets.git",
    "paths": [
        "Wallpapers/Windows XP",
        "Wallpapers/Windows 7/Desktop",
        "Wallpapers/Windows 10",
        "Wallpapers/Windows 11",
    ],
}


def download() -> None:
    if (DESTINATION / ".complete").exists():
        print("windows: already complete, skipping")
        return

    clone_sparse(WINDOWS["repo"], DESTINATION, WINDOWS["paths"])

    (DESTINATION / ".complete").touch()
    print("windows: source repository prepared")


if __name__ == "__main__":
    download()
