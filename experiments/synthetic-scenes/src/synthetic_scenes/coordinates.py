"""Axis-aligned geometry; boxes use half-open xyxy and float coordinates."""
from copy import deepcopy


def transform_box(box, sx=1., sy=1., dx=0., dy=0.):
    left, top, right, bottom = box
    return (left*sx + dx, top*sy + dy, right*sx + dx, bottom*sy + dy)


def transform_board(metadata, sx=1., sy=1., dx=0., dy=0.):
    """Transform image-space annotations without modifying source style parameters."""
    result = deepcopy(metadata)
    result['boxes'] = {
        name: transform_box(box, sx, sy, dx, dy)
        for name, box in metadata['boxes'].items()
    }
    result['cell_boxes'] = [
        [transform_box(box, sx, sy, dx, dy) for box in row]
        for row in metadata['cell_boxes']
    ]
    result['x_edges'] = [x*sx + dx for x in metadata['x_edges']]
    result['y_edges'] = [y*sy + dy for y in metadata['y_edges']]
    result['grid_origin'] = (result['x_edges'][0], result['y_edges'][0])
    result['coordinate_space'] = 'image'
    if 'window_box' in metadata:
        result['window_box'] = transform_box(metadata['window_box'], sx, sy, dx, dy)
    return result


def intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
