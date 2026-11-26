"""Compose style-specific panels and grid-local annotations using Skia."""
import numpy as np
import skia

from ..image_conversion import image_to_pillow
from .geometry import region_layout
from .regions import STYLES
from .renderer import render_board


def render_game_region(game, *, native=False, **board_options):
    """Render a game panel (excluding OS chrome) and region-local annotations."""
    grid, metadata = render_board(game, native=True, **board_options)
    style = metadata['style']
    parameters = metadata['style_parameters']
    grid_size = (grid.width(), grid.height())
    layout = region_layout(grid_size, style, parameters['cell_size'], parameters['zoom'])
    width, height = layout['size']
    gx, gy = layout['grid_origin']
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    remaining = int(np.count_nonzero(game['board'] == -1) - np.count_nonzero(game['flags']))
    elapsed = min(999, max(0, int(game['elapsed_seconds'])))
    STYLES[style].draw(canvas, layout, grid_size, metadata['theme'],
                       parameters['mine_variant'], game['status'], remaining, elapsed,
                       metadata['rows'], metadata['columns'])
    canvas.drawImage(grid, gx, gy)
    boxes = {'game_region': (0, 0, width, height),
             'grid': (gx, gy, gx+grid.width(), gy+grid.height()),
             'mine_counter': layout['mine_counter'], 'timer': layout['timer']}
    if 'reset_button' in layout:
        boxes['reset_button'] = layout['reset_button']
    metadata['boxes'] = boxes
    metadata['grid_origin'] = (gx, gy)
    if 'cell_boxes' in metadata:
        metadata['x_edges'] = [v+gx for v in metadata['x_edges']]
        metadata['y_edges'] = [v+gy for v in metadata['y_edges']]
        metadata['cell_boxes'] = [[(x+gx,y+gy,r+gx,b+gy) for x,y,r,b in row]
                                  for row in metadata['cell_boxes']]
    metadata['coordinate_space'] = 'game_region'
    image = surface.makeImageSnapshot()
    return (image if native else image_to_pillow(image)), metadata
