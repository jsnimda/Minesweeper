from pathlib import Path

from assets.wallpaper_download import download_wallpapers


ROOT = Path(__file__).resolve().parents[3]
DESTINATION = (
    ROOT / "data" / "assets" / "wallpapers"
    / "os_defaults" / "windows_11"
)

WINDOWS_11 = {
    "repo": "https://github.com/bartekl1/windows-ui-assets.git",
    "fixed": {
        "Wallpapers/Windows 11/Desktop/Windows/img0.jpg": 5,
        "Wallpapers/Windows 11/Desktop/Windows/img19.jpg": 5,
    },
    "other": {
        "weight": 3,
        "candidates": [
            "Wallpapers/Windows 11/Desktop/Captured Motion/img24.jpg",
            "Wallpapers/Windows 11/Desktop/Captured Motion/img25.jpg",
            "Wallpapers/Windows 11/Desktop/Captured Motion/img26.jpg",
            "Wallpapers/Windows 11/Desktop/Captured Motion/img27.jpg",
            "Wallpapers/Windows 11/Desktop/Flow/img32.jpg",
            "Wallpapers/Windows 11/Desktop/Flow/img33.jpg",
            "Wallpapers/Windows 11/Desktop/Flow/img34.jpg",
            "Wallpapers/Windows 11/Desktop/Flow/img35.jpg",
            "Wallpapers/Windows 11/Desktop/Glow/img20.jpg",
            "Wallpapers/Windows 11/Desktop/Glow/img21.jpg",
            "Wallpapers/Windows 11/Desktop/Glow/img22.jpg",
            "Wallpapers/Windows 11/Desktop/Glow/img23.jpg",
            "Wallpapers/Windows 11/Desktop/Sunrise/img28.jpg",
            "Wallpapers/Windows 11/Desktop/Sunrise/img29.jpg",
            "Wallpapers/Windows 11/Desktop/Sunrise/img30.jpg",
            "Wallpapers/Windows 11/Desktop/Sunrise/img31.jpg",
            "Wallpapers/Windows 11/Lock screen/img100.jpg",
            "Wallpapers/Windows 11/Lock screen/img101.jpg",
            "Wallpapers/Windows 11/Lock screen/img102.jpg",
            "Wallpapers/Windows 11/Lock screen/img103.jpg",
            "Wallpapers/Windows 11/Lock screen/img104.jpg",
            "Wallpapers/Windows 11/Lock screen/img105.jpg",
        ],
    },
}


def download() -> None:
    download_wallpapers(WINDOWS_11, DESTINATION)


if __name__ == "__main__":
    download()