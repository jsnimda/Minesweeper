"""Style-specific panel measurement and drawing."""


def digit_slots(grid_size, cell_size, gap=0):
    """Reserve room for any signed mine count on this grid."""
    width, height = grid_size
    columns = (width+gap)//(cell_size+gap)
    rows = (height+gap)//(cell_size+gap)
    return max(3, len(str(rows*columns)) + 1)


def box(x, y, width, height):
    return (x, y, x+width, y+height)


from . import classic, google, microsoft, windows7

STYLES = {'classic': classic, 'windows7': windows7,
          'microsoft': microsoft, 'google': google}
