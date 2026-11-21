
"""Sample logical synthetic desktop scenes without rendering."""

import colorsys
import random
from collections.abc import Mapping
from typing import TypeVar

from .scope import SCOPE

T = TypeVar("T")


def choose(rng: random.Random, weights: Mapping[T, float]) -> T:
    """Choose one key using relative weights."""
    if not weights:
        raise ValueError("Weights cannot be empty")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Weights cannot be negative")
    if not any(w > 0 for w in weights.values()):
        raise ValueError("At least one weight must be positive")

    return rng.choices(
        list(weights),
        weights=list(weights.values()),
        k=1,
    )[0]


def choose_group(rng: random.Random, groups: dict):
    """Choose a weighted group, then one option."""
    group = choose(
        rng,
        {name: data["weight"] for name, data in groups.items()},
    )
    return group, rng.choice(groups[group]["options"])


def sample_accent_color(rng: random.Random) -> str:
    """Generate a restrained, moderately saturated accent."""
    hue = rng.random()
    saturation = rng.uniform(0.35, 0.65)
    lightness = rng.uniform(0.40, 0.60)

    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#{:02X}{:02X}{:02X}".format(
        *(round(channel * 255) for channel in rgb)
    )


def sample_desktop(rng: random.Random) -> dict:
    shell = choose(rng, SCOPE["shell"])

    allowed = SCOPE["shell_styles"][shell]
    style = choose(
        rng,
        {
            name: weight
            for name, weight in SCOPE["style"].items()
            if name in allowed
        },
    )

    color_scheme = choose(
        rng, SCOPE["color_scheme"][style]
    )

    custom_accent = choose(
        rng, SCOPE["custom_accent"][style]
    )
    accent_color = (
        sample_accent_color(rng) if custom_accent else None
    )

    wallpaper_type = choose(rng, SCOPE["wallpaper_type"])
    wallpaper = {"type": wallpaper_type}

    if wallpaper_type == "os_defaults":
        weights = SCOPE["wallpaper_family"].get(
            (style, shell)
        )
        if weights is None:
            weights = SCOPE["wallpaper_family_fallback"][style]
        wallpaper["family"] = choose(rng, weights)

    return {
        "shell": shell,
        "style": style,
        "color_scheme": color_scheme,
        "accent_color": accent_color,
        "wallpaper": wallpaper,
    }


def sample_game(rng: random.Random) -> dict:
    renderer = choose(rng, SCOPE["game_renderer"])

    if renderer in {"google", "other_web"}:
        host = "browser"
    else:
        host = choose(rng, SCOPE["game_host"])

    preset = choose(
        rng,
        {
            name: config["weight"]
            for name, config in SCOPE["board_size"].items()
        },
    )

    config = SCOPE["board_size"][preset]

    if preset == "custom":
        rows = rng.randint(*config["rows"])
        columns = rng.randint(*config["columns"])

        density = rng.uniform(*config["mine_density"])
        mine_count = round(rows * columns * density)
    else:
        rows, columns = config["size"]
        mine_count = config["mines"]

    mine_count = max(1, min(
        mine_count, rows * columns - 1
    ))

    progress_category = choose(
        rng, SCOPE["game_progress"]
    )

    if progress_category == "hidden":
        status = "new"
        progress = 0.0
    elif progress_category == "finished":
        status = choose(rng, SCOPE["finished_status"])
        progress = (
            1.0 if status == "won"
            else rng.uniform(0.20, 0.95)
        )
    else:
        status = "active"
        low, high = SCOPE["progress_range"][progress_category]
        progress = rng.uniform(low, high)

    arrangement_weights = {
        name: weight
        for name, weight in SCOPE["game_arrangement"].items()
        if name != "browser_embedded" or host == "browser"
    }
    arrangement = choose(rng, arrangement_weights)

    if arrangement in {"partly_covered", "partly_offscreen"}:
        visibility_weights = {
            name: weight
            for name, weight in SCOPE["grid_visibility"].items()
            if name != "full"
        }
        visibility_category = choose(rng, visibility_weights)
    else:
        visibility_category = "full"

    low, high = SCOPE["visibility_range"][visibility_category]
    visible_fraction = (
        1.0 if visibility_category == "full"
        else rng.uniform(low, high)
    )

    return {
        "renderer": renderer,
        "host": host,
        "board": {
            "rows": rows,
            "columns": columns,
            "mine_count": mine_count,
        },
        "game": {
            "status": status,
            "progress": round(progress, 4),
            "seed": rng.randrange(2**32),
            "elapsed_seconds": (
                0 if status == "new"
                else rng.randint(1, 999)
            ),
        },
        "arrangement": arrangement,
        "visibility": {
            "category": visibility_category,
            "fraction": round(visible_fraction, 4),
        },
    }


def sample_components(
    rng: random.Random,
    cleanliness: str,
) -> list[str]:
    factor = SCOPE["component_density"][cleanliness]
    components = []

    for name, probability in SCOPE[
        "desktop_component_probability"
    ].items():
        chance = min(1.0, probability / 100.0 * factor)
        if rng.random() < chance:
            components.append(name)

    return components


def sample_scene(seed: int = 42) -> dict:
    """Generate one reproducible logical scene request."""
    rng = random.Random(seed)

    desktop = sample_desktop(rng)

    _, (width, height) = choose_group(
        rng, SCOPE["resolution"]
    )
    scale = choose(rng, SCOPE["display_scale"])

    cleanliness = choose(rng, SCOPE["cleanliness"])
    game_count = choose(rng, SCOPE["game_count"])

    windows = []

    for _ in range(game_count):
        windows.append({
            "id": f"window_{len(windows)}",
            "type": "minesweeper",
            "content": sample_game(rng),
        })

    extra_count = choose(
        rng, SCOPE["extra_window_count"][cleanliness]
    )

    for _ in range(extra_count):
        windows.append({
            "id": f"window_{len(windows)}",
            "type": choose(rng, SCOPE["application"]),
        })

    components = sample_components(rng, cleanliness)

    return {
        "_meta": {"seed": seed},
        "screen": {
            "width": width,
            "height": height,
            "scale": scale,
        },
        "desktop": desktop,
        "cleanliness": cleanliness,
        "components": components,
        "windows": windows,
    }


def sample_scenes(
    count: int,
    seed: int = 42,
) -> list[dict]:
    if count < 0:
        raise ValueError("count must be nonnegative")

    return [
        sample_scene(seed + index)
        for index in range(count)
    ]
