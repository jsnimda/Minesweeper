
SCOPE = {
    "os": {
        "windows_xp": 15,
        "windows_7": 15,
    },
    "resolution": {
        "retro": {
            "weight": 15,
            "options": [
                (1024, 768),
            ],
        },
        "widescreen": {
            "weight": 40,
            "options": [
                (1366, 768),
                (1920, 1080),
            ],
        },
    },
    "wallpaper": {
        "os_default": 30,
        "photograph": 35,
        "artwork": 20,
        "solid": 10,
        "busy": 5,
    },
    "game_count": {
        0: 10,
        1: 75,
        2: 12,
        (3, 4): 3,
    },
    "game_style": {
        "classic": 35,
    },
    "board_size": {
        "beginner": {
            "weight": 35,
            "options": [(9, 9)],
        },
        "intermediate": {
            "weight": 30,
            "options": [(16, 16)],
        },
        "expert": {
            "weight": 20,
            "options": [(16, 30)],
        },
        "other_presets": {
            "weight": 10,
            "options": [(8, 8)],
        },
        "custom": {
            "weight": 5,
            "options": [
                (10, 10),
                (12, 12),
                (20, 20),
            ],
        },
    },
    "game_progress": {
        "hidden": 10,
        "early": 25,
        "mid": 40,
        "late": 20,
        "finished": 5,
    },
}
