"""Pure measurement functions shared by planning and rasterization."""


def cell_gap(style: str, cell_size: int) -> int:
    return max(1, round(cell_size * .09)) if style == 'microsoft' else 0


def measure_grid(rows, columns, style='classic', cell_size=32):
    if rows < 1 or columns < 1 or cell_size < 8:
        raise ValueError('Invalid grid dimensions or tile size')
    if style not in ('classic', 'windows7', 'microsoft', 'google'):
        raise ValueError(f'Unknown style: {style}')
    gap = cell_gap(style, cell_size)
    pitch = cell_size + gap
    return columns * pitch - gap, rows * pitch - gap


def region_layout(grid_size, style, cell_size):
    """Return the exact panel dimensions and half-open control coordinates."""
    from .regions import STYLES
    if style not in STYLES:
        raise ValueError(f'Unknown style: {style}')
    return STYLES[style].layout(grid_size, cell_size)


def measure_game_region(rows, columns, style='classic', cell_size=32):
    grid_size = measure_grid(rows, columns, style, cell_size)
    return region_layout(grid_size, style, cell_size)['size']
