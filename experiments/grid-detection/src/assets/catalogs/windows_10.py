from pathlib import Path

from assets.wallpaper_download import download_wallpapers


ROOT = Path(__file__).resolve().parents[3]
DESTINATION = (
    ROOT / "data" / "assets" / "wallpapers"
    / "os_defaults" / "windows_10"
)

WINDOWS_10 = {
    "repo": "https://github.com/bartekl1/windows-ui-assets.git",
    "fixed": {
        "Wallpapers/Windows 10/Desktop/Windows/img0.jpg": 10,
    },
    "other": {
        "weight": 3,
        "candidates": [
            "Wallpapers/Windows 10/Desktop/Flowers/img10.jpg",
            "Wallpapers/Windows 10/Desktop/Flowers/img11.jpg",
            "Wallpapers/Windows 10/Desktop/Flowers/img12.jpg",
            "Wallpapers/Windows 10/Desktop/Flowers/img7.jpg",
            "Wallpapers/Windows 10/Desktop/Flowers/img8.jpg",
            "Wallpapers/Windows 10/Desktop/Flowers/img9.jpg",
            "Wallpapers/Windows 10/Desktop/Windows 10/img1.jpg",
            "Wallpapers/Windows 10/Desktop/Windows 10/img13.jpg",
            "Wallpapers/Windows 10/Desktop/Windows 10/img2.jpg",
            "Wallpapers/Windows 10/Desktop/Windows 10/img3.jpg",
            "Wallpapers/Windows 10/Desktop/Windows 10/img4.jpg",
            "Wallpapers/Windows 10/Lock screen/img100.jpg",
            "Wallpapers/Windows 10/Lock screen/img101.png",
            "Wallpapers/Windows 10/Lock screen/img102.jpg",
            "Wallpapers/Windows 10/Lock screen/img103.png",
            "Wallpapers/Windows 10/Lock screen/img104.jpg",
            "Wallpapers/Windows 10/Lock screen/img105.jpg",
        ],
    },
}


def download() -> None:
    download_wallpapers(WINDOWS_10, DESTINATION)


if __name__ == "__main__":
    download()