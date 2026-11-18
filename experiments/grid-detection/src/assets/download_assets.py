from assets.catalogs import (
    artwork,
    macos,
    photographs,
    windows,
    windows_xp,
    windows_7,
    windows_10,
    windows_11,
)


def download() -> None:
    artwork.download()
    macos.download()
    photographs.download()

    windows.download()
    windows_xp.download()
    windows_7.download()
    windows_10.download()
    windows_11.download()


if __name__ == "__main__":
    download()