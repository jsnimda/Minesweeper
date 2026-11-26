"""Pure measurement functions shared by planning and rasterization."""
import math


def validate_zoom(zoom):
    """Accept only ordinary finite positive numbers, without coercing strings."""
    if type(zoom) not in (int, float):
        raise ValueError('zoom must be a positive finite number')
    try:
        valid = math.isfinite(zoom) and zoom > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('zoom must be a positive finite number')
    return zoom


def scaled_measurement(value, zoom, minimum=1):
    """Round each measurement to nearest pixel (ties to even)."""
    return max(minimum, round(value * zoom))


def effective_cell_size(cell_size, zoom=1.0):
    validate_zoom(zoom)
    if type(cell_size) is not int or cell_size < 8:
        raise ValueError('cell_size must be an integer >= 8')
    return scaled_measurement(cell_size, zoom, minimum=8)


def cell_gap(style: str, cell_size: int) -> int:
    return max(1, round(cell_size * .09)) if style == 'microsoft' else 0


def measure_grid(rows, columns, style='classic', cell_size=32, zoom=1.0):
    cell_size = effective_cell_size(cell_size, zoom)
    if rows < 1 or columns < 1 or cell_size < 8:
        raise ValueError('Invalid grid dimensions or tile size')
    if style not in ('classic', 'windows7', 'microsoft', 'google'):
        raise ValueError(f'Unknown style: {style}')
    gap = cell_gap(style, cell_size)
    pitch = cell_size + gap
    return columns * pitch - gap, rows * pitch - gap


def region_layout(grid_size, style, cell_size, zoom=1.0):
    """Return the exact panel dimensions and half-open control coordinates."""
    from .regions import STYLES
    if style not in STYLES:
        raise ValueError(f'Unknown style: {style}')
    validate_zoom(zoom)
    # Tiny zooms retain usable controls and avoid zero-sized drawing primitives.
    return STYLES[style].layout(grid_size, cell_size, max(0.25, zoom))


def measure_game_region(rows, columns, style='classic', cell_size=32, zoom=1.0):
    effective = effective_cell_size(cell_size, zoom)
    grid_size = measure_grid(rows, columns, style, effective)
    return region_layout(grid_size, style, effective, zoom)['size']
