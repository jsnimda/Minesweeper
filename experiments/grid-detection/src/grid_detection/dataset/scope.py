
"""Distributions for synthetic desktop dataset sampling.

Weights are relative; they do not have to sum to 100.
"""

SCOPE = {
    "scene_mode": {"desktop": 80, "game": 15, "grid": 5},
    "wallpaper_placement": {"cover": 80, "fit": 15, "repeat": 5},
    # Desktop appearance
    "shell": {
        "taskbar": 60,
        "dock": 25,
        "top_panel": 15,
    },
    "style": {
        "modern": 75,
        "aero": 15,
        "classic": 10,
    },
    "shell_styles": {
        "taskbar": ("modern", "aero", "classic"),
        "dock": ("modern",),
        "top_panel": ("modern",),
    },
    "taskbar_layout": {
        "centered": 60,
        "left": 40,
    },
    "color_scheme": {
        "modern": {"light": 55, "dark": 45},
        "aero": {"light": 80, "dark": 20},
        "classic": {"default": 100},
    },
    "custom_accent": {
        "modern": {True: 70, False: 30},
        "aero": {True: 35, False: 65},
        "classic": {True: 10, False: 90},
    },

    # Wallpaper categories
    "wallpaper_type": {"procedural": 20, "artwork": 40, "photographs": 40},

    # Screen resolution
    "resolution": {
        "retro": {
            "weight": 15,
            "options": [
                (800, 600), (1024, 768), (1280, 960),
            ],
        },
        "laptop": {
            "weight": 20,
            "options": [
                (1280, 800), (1440, 900), (1920, 1200),
            ],
        },
        "widescreen": {
            "weight": 35,
            "options": [
                (1280, 720), (1366, 768),
                (1600, 900), (1920, 1080),
            ],
        },
        "high_res": {
            "weight": 20,
            "options": [
                (2560, 1440), (2880, 1800), (3840, 2160), (2560, 1600),
            ],
        },
        "ultrawide": {
            "weight": 5,
            "options": [
                (2560, 1080), (3440, 1440),
            ],
        },
        "portrait": {
            "weight": 5,
            "options": [(1080, 1920)],
        },
    },
    # Scene complexity
    "cleanliness": {
        "minimal": 10,
        "clean": 25,
        "normal": 35,
        "busy": 20,
        "chaotic": 10,
    },
    "extra_window_count": {
        "minimal": {0: 100},
        "clean": {0: 50, 1: 50},
        "normal": {1: 45, 2: 55},
        "busy": {2: 30, 3: 45, 4: 25},
        "chaotic": {4: 40, 5: 40, 6: 20},
    },
    "application": {
        "browser": 22,
        "file_manager": 14,
        "spreadsheet": 12,
        "document_editor": 8,
        "presentation": 7,
        "code_editor": 8,
        "terminal": 7,
        "calculator": 4,
        "settings": 5,
        "sudoku": 5,
        "other_puzzle": 4,
        "media_viewer": 4,
    },

    # Minesweeper
    "game_count": {
        0: 10,
        1: 75,
        2: 12,
        3: 1.5,
        4: 1.5,
    },
    "game_renderer": {"classic": 25, "windows7": 25, "microsoft": 25, "google": 25},
    "board_size": {
        "beginner": {"weight": 25, "size": (9, 9), "mines": 10},
        "intermediate": {"weight": 25, "size": (16, 16), "mines": 40},
        "expert": {"weight": 20, "size": (16, 30), "mines": 99},
        "small_custom": {"weight": 5, "rows": (5, 12), "columns": (5, 12)},
        "normal_custom": {"weight": 15, "rows": (8, 30), "columns": (8, 40)},
        "large_custom": {"weight": 7, "rows": (20, 60), "columns": (20, 100)},
        "extreme_custom": {"weight": 3, "rows": (5, 100), "columns": (5, 100)},
    },
    "custom_mine_density": (0.12, 0.20),
    "cell_size": {
        "small": {"weight": 20, "range": (12, 19)},
        "normal": {"weight": 50, "range": (20, 32)},
        "large": {"weight": 25, "range": (33, 48)},
        "extra_large": {"weight": 5, "range": (49, 64)},
    },
    # Limits apply to the rendered game/grid before capture augmentation.
    "standalone_image_budget": {"max_dimension": 4096, "max_pixels": 12_000_000},
    "default_theme_probability": 0.65,
    "font_families": ("Arial", "Tahoma", "Verdana"),
    "font_scale": (0.65, 0.77),
    "bold_probability": 0.9,
    "google_boundary_ratios": (0.0, 0.06, 0.08, 0.10),
    "windows7_mine_variants": ("classic", "flower"),
    "accent_saturation": (0.35, 0.65),
    "accent_lightness": (0.40, 0.60),
    "lost_progress_range": (0.20, 0.95),
    "elapsed_seconds": (1, 999),
    "game_progress": {
        "hidden": 20,
        "early": 25,
        "mid": 30,
        "late": 20,
        "finished": 5,
    },
    "progress_range": {
        "early": (0.05, 0.25),
        "mid": (0.25, 0.65),
        "late": (0.65, 0.95),
    },
    "finished_status": {
        "won": 40,
        "lost": 60,
    },

}
