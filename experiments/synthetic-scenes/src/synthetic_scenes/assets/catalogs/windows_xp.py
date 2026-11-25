from ..wallpaper_download import download_wallpapers
from ..paths import ASSET_DIR


DESTINATION = (
    ASSET_DIR / "wallpapers"
    / "os_defaults" / "windows_xp"
)

WINDOWS_XP = {
    "repo": "https://github.com/bartekl1/windows-ui-assets.git",
    "fixed": [
        "Wallpapers/Windows XP/Desktop/Bliss.bmp",
    ],
    "other": [
        "Wallpapers/Windows XP/Desktop/Ascent.jpg",
        "Wallpapers/Windows XP/Desktop/Autumn.jpg",
        "Wallpapers/Windows XP/Desktop/Azul.jpg",
        "Wallpapers/Windows XP/Desktop/Crystal.jpg",
        "Wallpapers/Windows XP/Desktop/Follow.jpg",
        "Wallpapers/Windows XP/Desktop/Friend.jpg",
        "Wallpapers/Windows XP/Desktop/Home.jpg",
        "Wallpapers/Windows XP/Desktop/Moon flower.jpg",
        "Wallpapers/Windows XP/Desktop/Peace.jpg",
        "Wallpapers/Windows XP/Desktop/Power.jpg",
        "Wallpapers/Windows XP/Desktop/Purple flower.jpg",
        "Wallpapers/Windows XP/Desktop/Radiance.jpg",
        "Wallpapers/Windows XP/Desktop/Red moon desert.jpg",
        "Wallpapers/Windows XP/Desktop/Ripple.jpg",
        "Wallpapers/Windows XP/Desktop/Stonehenge.jpg",
        "Wallpapers/Windows XP/Desktop/Tulips.jpg",
        "Wallpapers/Windows XP/Desktop/Vortec space.jpg",
        "Wallpapers/Windows XP/Desktop/Wind.jpg",
        "Wallpapers/Windows XP/Desktop/Windows XP.jpg",
    ],
}


def download() -> None:
    download_wallpapers(WINDOWS_XP, DESTINATION)


if __name__ == "__main__":
    download()
