
"""Sample logical synthetic desktop scenes without rendering."""

import colorsys
import math
import random
from collections.abc import Mapping
from datetime import date, timedelta
from typing import TypeVar

from synthetic_scenes.minesweeper.geometry import measure_game_region, measure_grid
from synthetic_scenes.minesweeper.styles import THEMES
from synthetic_scenes.render import PlacementError, plan_game_windows

from .sampling_config import SAMPLING_CONFIG

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


def _scale_choices(weights: Mapping) -> dict:
    """Validate a scale distribution and return its positive-weight choices."""
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("Scale distribution must be a nonempty mapping")
    choices = {}
    for scale, weight in weights.items():
        try:
            valid_scale = type(scale) in (int, float) and math.isfinite(scale) and scale > 0
            valid_weight = type(weight) in (int, float) and math.isfinite(weight) and weight >= 0
        except OverflowError:
            valid_scale = valid_weight = False
        if not valid_scale or not valid_weight:
            raise ValueError("Scale factors and weights must be finite; scales must be positive and weights nonnegative")
        if weight > 0:
            choices[scale] = weight
    try:
        valid_total = bool(choices) and math.isfinite(sum(choices.values()))
    except OverflowError:
        valid_total = False
    if not valid_total:
        raise ValueError("Scale distribution must have a finite positive total weight")
    return choices


def sample_scale(rng: random.Random, weights: Mapping) -> float:
    choices = _scale_choices(weights)
    if len(choices) == 1:
        return next(iter(choices))
    return choose(rng, choices)


def _display_scale_policy() -> Mapping:
    """Validate a global scale mapping or a complete mapping by resolution group."""
    policy = SAMPLING_CONFIG["display_scale"]
    if not isinstance(policy, Mapping) or not policy:
        raise ValueError("Display scale policy must be a nonempty mapping")
    if any(isinstance(value, Mapping) for value in policy.values()):
        if set(policy) != set(SAMPLING_CONFIG["resolution"]):
            raise ValueError("Display scale groups must match resolution groups")
        for weights in policy.values():
            _scale_choices(weights)
    else:
        _scale_choices(policy)
    return policy


def _validate_desktop_sampling() -> None:
    styles = SAMPLING_CONFIG["shell_styles"]
    shells = {"taskbar", "dock", "top_panel"}
    if not isinstance(styles, Mapping) or set(styles) != shells:
        raise ValueError("Shell styles must specify every supported shell")
    for weights in styles.values():
        if not isinstance(weights, Mapping) or not weights or not set(weights) <= {"modern", "classic"}:
            raise ValueError("Shell style distributions contain unsupported styles")
        _validate_weights(weights, "Shell style")
    visibility = SAMPLING_CONFIG["component_visibility"]
    fields = {"show_taskbar", "show_dock", "show_status_bar", "show_top_panel"}
    if not isinstance(visibility, Mapping) or set(visibility) != fields:
        raise ValueError("Component visibility must specify every supported component")
    for probability in visibility.values():
        try:
            valid = (type(probability) in (int, float) and math.isfinite(probability)
                     and 0 <= probability <= 1)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Visibility probabilities must be finite numbers between 0 and 1")
    groups = SAMPLING_CONFIG["dock_height"]
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("Dock height groups must be a nonempty mapping")
    weights = {}
    for name, group in groups.items():
        if not isinstance(group, Mapping) or set(group) != {"weight", "range"}:
            raise ValueError("Dock height groups require weight and range")
        bounds = group["range"]
        if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
                or any(type(value) is not int for value in bounds)
                or not 32 <= bounds[0] <= bounds[1] <= 120):
            raise ValueError("Dock height ranges must be inclusive integers within 32..120")
        weights[name] = group["weight"]
    _validate_weights(weights, "Dock height")


def _validate_weights(weights: Mapping, label: str) -> None:
    try:
        valid = all(type(weight) in (int, float) and math.isfinite(weight) and weight >= 0
                    for weight in weights.values())
        total = sum(weights.values()) if valid else 0
        valid = valid and math.isfinite(total) and total > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} weights must be finite, nonnegative and have a positive total")


def sample_accent_color(rng: random.Random) -> str:
    """Generate a restrained, moderately saturated accent."""
    hue = rng.random()
    saturation = rng.uniform(*SAMPLING_CONFIG["accent_saturation"])
    lightness = rng.uniform(*SAMPLING_CONFIG["accent_lightness"])

    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#{:02X}{:02X}{:02X}".format(
        *(round(channel * 255) for channel in rgb)
    )


def sample_desktop(rng: random.Random) -> dict:
    _validate_desktop_sampling()
    shell = choose(rng, SAMPLING_CONFIG["shell"])
    style = choose(rng, SAMPLING_CONFIG["shell_styles"][shell])
    color_scheme = choose(
        rng, SAMPLING_CONFIG["color_scheme"][style]
    )

    custom_accent = choose(
        rng, SAMPLING_CONFIG["custom_accent"][style]
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

    desktop = {
        "shell": shell,
        "style": style,
        "color_scheme": color_scheme,
        "accent_color": accent_color,
        "clock_time": clock_time,
        "clock_date": clock_date,
    }
    visibility = SAMPLING_CONFIG["component_visibility"]
    relevant = {"taskbar": ("show_taskbar",), "dock": ("show_dock", "show_status_bar"),
                "top_panel": ("show_top_panel",)}[shell]
    for field in relevant:
        desktop[field] = rng.random() < visibility[field]
    if shell == "dock" and desktop["show_dock"]:
        groups = SAMPLING_CONFIG["dock_height"]
        group = choose(rng, {name: item["weight"] for name, item in groups.items()})
        desktop["dock_height"] = rng.randint(*groups[group]["range"])
    return desktop


def sample_game(rng: random.Random) -> dict:
    renderer = choose(rng, SAMPLING_CONFIG["game_renderer"])

    preset = choose(
        rng,
        {
            name: config["weight"]
            for name, config in SAMPLING_CONFIG["board_size"].items()
        },
    )

    config = SAMPLING_CONFIG["board_size"][preset]

    if "rows" in config:
        rows = rng.randint(*config["rows"])
        columns = rng.randint(*config["columns"])

        density = rng.uniform(*SAMPLING_CONFIG["custom_mine_density"])
        mine_count = round(rows * columns * density)
    else:
        rows, columns = config["size"]
        mine_count = config["mines"]

    mine_count = max(1, min(
        mine_count, rows * columns - 1
    ))

    progress_category = choose(
        rng, SAMPLING_CONFIG["game_progress"]
    )

    if progress_category == "hidden":
        status = "new"
        progress = 0.0
    elif progress_category == "finished":
        status = choose(rng, SAMPLING_CONFIG["finished_status"])
        progress = (
            1.0 if status == "won"
            else rng.uniform(*SAMPLING_CONFIG["lost_progress_range"])
        )
    else:
        status = "active"
        low, high = SAMPLING_CONFIG["progress_range"][progress_category]
        progress = rng.uniform(low, high)

    themes = THEMES[renderer]
    if len(themes) == 1 or rng.random() < SAMPLING_CONFIG["default_theme_probability"]:
        theme = themes[0]
    else:
        theme = rng.choice(themes[1:])
    category = choose(rng, {name: config["weight"]
                            for name, config in SAMPLING_CONFIG["cell_size"].items()})
    appearance = {
        "style": renderer, "theme": theme,
        "cell_size": rng.randint(*SAMPLING_CONFIG["cell_size"][category]["range"]),
        "font_family": rng.choice(SAMPLING_CONFIG["font_families"]),
        "font_scale": rng.uniform(*SAMPLING_CONFIG["font_scale"]),
        "bold": rng.random() < SAMPLING_CONFIG["bold_probability"],
    }
    appearance["zoom"] = sample_scale(rng, SAMPLING_CONFIG["game_zoom"])
    if renderer == "google":
        appearance["boundary_ratio"] = rng.choice(SAMPLING_CONFIG["google_boundary_ratios"])
    if renderer == "windows7":
        appearance["mine_variant"] = rng.choice(SAMPLING_CONFIG["windows7_mine_variants"])
    return {
        "rows": rows, "columns": columns, "mine_count": mine_count,
        "status": status, "progress": progress, "seed": rng.getrandbits(64),
        "elapsed_seconds": 0 if status == "new" else rng.randint(*SAMPLING_CONFIG["elapsed_seconds"]),
        "appearance": appearance,
    }


def _fit_standalone_game(game: dict, mode: str) -> None:
    """Shrink cell size, keeping the sampled board, to meet the image budget."""
    appearance = game["appearance"]
    requested = appearance["cell_size"]
    minimum = SAMPLING_CONFIG["cell_size"]["small"]["range"][0]
    budget = SAMPLING_CONFIG["standalone_image_budget"]
    measure = measure_grid if mode == "grid" else measure_game_region
    for cell_size in range(requested, minimum - 1, -1):
        width, height = measure(
            game["rows"], game["columns"], appearance["style"], cell_size,
            appearance.get("zoom", 1.0),
        )
        if (max(width, height) <= budget["max_dimension"]
                and width * height <= budget["max_pixels"]):
            appearance["cell_size"] = cell_size
            return
    raise PlacementError("Board cannot fit the standalone image budget at minimum cell size")


def sample_scene(seed: int | str = 42, *, sampling_metadata: dict | None = None) -> dict:
    """Sample a scene, resolving cell sizes against its measured rendering area.

    Feasibility shrinks cells before retrying a scene; infeasible desktop layouts
    and standalone budgets bias the accepted distributions toward smaller boards.
    If supplied, sampling_metadata receives pre-fit cell sizes in game order.
    """
    rng = random.Random(seed)
    # Validate both distributions even when a scene does not use one of them.
    _scale_choices(SAMPLING_CONFIG["game_zoom"])
    display_policy = _display_scale_policy()
    _validate_desktop_sampling()
    mode = choose(rng, SAMPLING_CONFIG["scene_mode"])
    if mode == "desktop":
        resolution_group, (width, height) = choose_group(rng, SAMPLING_CONFIG["resolution"])
        scale_weights = (display_policy[resolution_group]
                         if resolution_group in display_policy else display_policy)
        display_scale = sample_scale(rng, scale_weights)
        desktop = sample_desktop(rng)
        game_count = choose(rng, SAMPLING_CONFIG["game_count"])
    else:
        game_count = 1

    windows = [{"type": "minesweeper", "content": sample_game(rng)}
               for _ in range(game_count)]
    if sampling_metadata is not None:
        sampling_metadata["sampled_cell_sizes"] = [
            window["content"]["appearance"]["cell_size"] for window in windows
        ]
    layout_seed = rng.getrandbits(64)
    result = {"mode": mode, "seed": seed, "layout_seed": layout_seed, "windows": windows}
    if mode == "desktop":
        games = [(index, window["content"]) for index, window in enumerate(windows)]
        # Check feasibility here while leaving the requested cell sizes intact.
        # The renderer repeats this deterministic plan and reports its resolution.
        plan_game_windows(games, (width, height), desktop, layout_seed, display_scale)

        cleanliness = choose(rng, SAMPLING_CONFIG["cleanliness"])
        extra_count = choose(rng, SAMPLING_CONFIG["extra_window_count"][cleanliness])
        windows.extend({"type": choose(rng, SAMPLING_CONFIG["application"]), "seed": rng.getrandbits(64)}
                       for _ in range(extra_count))
        category = choose(rng, SAMPLING_CONFIG["wallpaper_type"])
        wallpaper = {"category": category, "seed": rng.getrandbits(64)}
        if category != "procedural":
            wallpaper["placement"] = choose(rng, SAMPLING_CONFIG["wallpaper_placement"])
        result.update(screen={"width": width, "height": height, "scale": display_scale}, desktop=desktop,
                      wallpaper=wallpaper, cleanliness=cleanliness)
    else:
        _fit_standalone_game(windows[0]["content"], mode)
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
