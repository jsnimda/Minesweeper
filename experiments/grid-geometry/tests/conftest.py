import os
from pathlib import Path

import pytest


@pytest.fixture
def diagnostic_root():
    project = Path(__file__).resolve().parents[1]
    path = os.environ.get("GRID_GEOMETRY_DIAGNOSTICS")
    if path is not None:
        return Path(path)
    default = project / "outputs" / "geometry-diagnostics" / "geometry"
    if not default.is_dir():
        pytest.skip("Export diagnostic samples with scripts/export_geometry_diagnostics.py")
    return default


@pytest.fixture
def annotation():
    tile_x = [2.25, 10.25, 18.25]
    tile_y = [1.5, 8.5]
    return {
        "board_id": "board-with-gaps",
        "game_state_id": "game-state-01",
        "style": "microsoft",
        "rows": 2,
        "columns": 3,
        "size": [31, 19],
        "source_crop": [3.25, 4.125, 34.0, 22.9],
        "grid_box": [2.25, 1.5, 25.25, 14.5],
        "x_edges": [2.25, 9.75, 17.75, 25.25],
        "y_edges": [1.5, 8.0, 14.5],
        "cell_boxes": [
            [[x, y, x + 7, y + 6] for x in tile_x]
            for y in tile_y
        ],
        "centers": [
            [[x + 3.5, y + 3] for x in tile_x]
            for y in tile_y
        ],
    }
