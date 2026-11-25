"""Prepare shared rendering assets from the command line."""

import argparse

from . import prepare_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("categories", nargs="*", metavar="CATEGORY",
                        help="wallpapers or minesweeper_windows7 (default: both)")
    args = parser.parse_args()
    prepare_assets(*args.categories)


if __name__ == "__main__":
    main()
