
"""Sample logical synthetic desktop scenes without rendering."""

import colorsys
import random
from collections.abc import Mapping
from datetime import date, timedelta
from typing import TypeVar

from .scope import SCOPE
from synthetic_scenes.minesweeper.styles import THEMES

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
    taskbar_layout = (
        choose(rng, SCOPE["taskbar_layout"])
        if shell == "taskbar" and style == "modern"
        else None
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

    # A reproducible, synthetic desktop timestamp.
    clock_day = date(2020, 1, 1) + timedelta(
        days=rng.randrange(365 * 7)
    )

    clock_hour = rng.randrange(24)
    clock_minute = rng.randrange(60)

    clock_time = f"{clock_hour:02d}:{clock_minute:02d}"
    clock_date = clock_day.strftime("%m/%d/%Y")

    return {
        "shell": shell,
        "style": style,
        "taskbar_layout": taskbar_layout,
        "color_scheme": color_scheme,
        "accent_color": accent_color,
        "clock_time": clock_time,
        "clock_date": clock_date,
    }


def sample_game(rng: random.Random) -> dict:
    renderer = choose(rng, SCOPE["game_renderer"])

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

    theme = THEMES[renderer][0] if rng.random() < .65 else rng.choice(THEMES[renderer])
    appearance = {
        "style": renderer, "theme": theme, "cell_size": rng.randint(20, 40),
        "font_family": rng.choice(("Arial", "Tahoma", "Verdana")),
        "font_scale": rng.uniform(.65, .77), "bold": rng.random() < .9,
    }
    if renderer == "google":
        appearance["boundary_ratio"] = rng.choice((0., .06, .08, .10))
    if renderer == "windows7":
        appearance["mine_variant"] = rng.choice(("classic", "flower"))
    return {
        "rows": rows, "columns": columns, "mine_count": mine_count,
        "status": status, "progress": progress, "seed": rng.getrandbits(64),
        "elapsed_seconds": 0 if status == "new" else rng.randint(1, 999),
        "appearance": appearance,
    }


def sample_scene(seed: int | str = 42) -> dict:
    """Sample a descriptive request with independent component seeds."""
    rng = random.Random(seed)
    mode = choose(rng, SCOPE["scene_mode"])
    game_count = choose(rng, SCOPE["game_count"]) if mode == "desktop" else 1
    windows = [{"type": "minesweeper", "content": sample_game(rng)}
               for _ in range(game_count)]
    result = {"mode": mode, "seed": seed, "layout_seed": rng.getrandbits(64), "windows": windows}
    if mode == "desktop":
        _, (width, height) = choose_group(rng, SCOPE["resolution"])
        desktop = sample_desktop(rng)
        cleanliness = choose(rng, SCOPE["cleanliness"])
        extra_count = choose(rng, SCOPE["extra_window_count"][cleanliness])
        windows.extend({"type": choose(rng, SCOPE["application"]), "seed": rng.getrandbits(64)}
                       for _ in range(extra_count))
        category = choose(rng, SCOPE["wallpaper_type"])
        wallpaper = {"category": category, "seed": rng.getrandbits(64)}
        if category != "procedural":
            wallpaper["placement"] = choose(rng, SCOPE["wallpaper_placement"])
        result.update(screen={"width": width, "height": height}, desktop=desktop,
                      wallpaper=wallpaper, cleanliness=cleanliness)
    return result


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
