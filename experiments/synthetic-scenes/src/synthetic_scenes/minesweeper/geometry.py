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
    """Return the exact panel dimensions and control coordinates."""
    grid_w, grid_h = grid_size
    pad = max(3, round(cell_size * .35)) if style != 'google' else 0
    header = max(20, round(cell_size * 1.85))
    control_h = max(12, round(cell_size * 1.15))
    control_w = max(32, round(cell_size * 2.65))
    margin = max(3, pad)

    reset_w = control_h if style == 'classic' else 0
    min_width = 2 * control_w + reset_w + 4 * margin
    width = max(grid_w + 2 * pad, min_width)
    height = grid_h + header + 2 * pad
    gx = (width - grid_w) // 2
    gy = pad if style == 'windows7' else header + pad

    control_y = pad + (header - control_h) // 2
    if style == 'windows7':
        control_y = grid_h + 2 * pad + (header - control_h) // 2

    return {
        'size': (width, height),
        'grid_origin': (gx, gy),
        'pad': pad,
        'header': header,
        'margin': margin,
        'control_y': control_y,
        'control_size': (control_w, control_h),
    }


def measure_game_region(rows, columns, style='classic', cell_size=32):
    grid_size = measure_grid(rows, columns, style, cell_size)
    return region_layout(grid_size, style, cell_size)['size']
