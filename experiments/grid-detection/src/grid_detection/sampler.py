
import random

from grid_detection.scope import SCOPE


def choose(rng: random.Random, category: dict):
    weights = [
        value["weight"] if isinstance(value, dict) else value
        for value in category.values()
    ]

    selected = rng.choices(
        list(category),
        weights=weights,
        k=1,
    )[0]

    value = category[selected]

    if isinstance(value, dict) and "options" in value:
        return rng.choice(value["options"])

    return selected


def sample_scene(rng: random.Random) -> dict:
    game_count = choose(rng, SCOPE["game_count"])

    if isinstance(game_count, tuple):
        game_count = rng.choice(game_count)

    games = []

    for _ in range(game_count):
        rows, columns = choose(rng, SCOPE["board_size"])

        games.append(
            {
                "style": choose(rng, SCOPE["game_style"]),
                "rows": rows,
                "columns": columns,
                "progress": choose(rng, SCOPE["game_progress"]),
            }
        )

    return {
        "os": choose(rng, SCOPE["os"]),
        "resolution": choose(rng, SCOPE["resolution"]),
        "wallpaper": choose(rng, SCOPE["wallpaper"]),
        "games": games,
    }


def sample_scenes(count: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return [sample_scene(rng) for _ in range(count)]


if __name__ == "__main__":
    from pprint import pprint

    for scene in sample_scenes(5):
        pprint(scene)
