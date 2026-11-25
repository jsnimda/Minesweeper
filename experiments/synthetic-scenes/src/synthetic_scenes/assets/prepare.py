from .wallpapers import artwork, photographs
from .minesweeper import windows7


def prepare() -> None:
    artwork.download()
    photographs.download()
    windows7.download()


if __name__ == "__main__":
    prepare()