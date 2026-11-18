from pathlib import Path

from assets.wallpaper_download import download_wallpapers


ROOT = Path(__file__).resolve().parents[3]
DESTINATION = (
    ROOT / "data" / "assets" / "wallpapers"
    / "os_defaults" / "windows_7"
)

WINDOWS_7 = {
    "repo": "https://github.com/bartekl1/windows-ui-assets.git",
    "fixed": {
        "Wallpapers/Windows 7/Desktop/Windows/img0.jpg": 10,
    },
    "other": {
        "weight": 3,
        "candidates": [
            "Wallpapers/Windows 7/Desktop/Architecture/img13.jpg",
            "Wallpapers/Windows 7/Desktop/Architecture/img14.jpg",
            "Wallpapers/Windows 7/Desktop/Architecture/img15.jpg",
            "Wallpapers/Windows 7/Desktop/Architecture/img16.jpg",
            "Wallpapers/Windows 7/Desktop/Architecture/img17.jpg",
            "Wallpapers/Windows 7/Desktop/Architecture/img18.jpg",
            "Wallpapers/Windows 7/Desktop/Characters/img19.jpg",
            "Wallpapers/Windows 7/Desktop/Characters/img20.jpg",
            "Wallpapers/Windows 7/Desktop/Characters/img21.jpg",
            "Wallpapers/Windows 7/Desktop/Characters/img22.jpg",
            "Wallpapers/Windows 7/Desktop/Characters/img23.jpg",
            "Wallpapers/Windows 7/Desktop/Characters/img24.jpg",
            "Wallpapers/Windows 7/Desktop/Landscapes/img10.jpg",
            "Wallpapers/Windows 7/Desktop/Landscapes/img11.jpg",
            "Wallpapers/Windows 7/Desktop/Landscapes/img12.jpg",
            "Wallpapers/Windows 7/Desktop/Landscapes/img7.jpg",
            "Wallpapers/Windows 7/Desktop/Landscapes/img8.jpg",
            "Wallpapers/Windows 7/Desktop/Landscapes/img9.jpg",
            "Wallpapers/Windows 7/Desktop/Nature/img1.jpg",
            "Wallpapers/Windows 7/Desktop/Nature/img2.jpg",
            "Wallpapers/Windows 7/Desktop/Nature/img3.jpg",
            "Wallpapers/Windows 7/Desktop/Nature/img4.jpg",
            "Wallpapers/Windows 7/Desktop/Nature/img5.jpg",
            "Wallpapers/Windows 7/Desktop/Nature/img6.jpg",
            "Wallpapers/Windows 7/Desktop/Scenes/img25.jpg",
            "Wallpapers/Windows 7/Desktop/Scenes/img26.jpg",
            "Wallpapers/Windows 7/Desktop/Scenes/img27.jpg",
            "Wallpapers/Windows 7/Desktop/Scenes/img28.jpg",
            "Wallpapers/Windows 7/Desktop/Scenes/img29.jpg",
            "Wallpapers/Windows 7/Desktop/Scenes/img30.jpg",
        ],
    },
}


def download() -> None:
    download_wallpapers(WINDOWS_7, DESTINATION)


if __name__ == "__main__":
    download()