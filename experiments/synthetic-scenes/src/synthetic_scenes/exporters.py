"""Export model-specific views from one canonical scene.

No PyTorch dependency. Labels and tensor dimensions are explicitly documented.
"""
from collections import Counter
import json
from math import ceil, isfinite
from pathlib import Path
import random

import numpy as np
from PIL import Image

from .coordinates import transform_board


YOLO_CLASSES = ('game_region', 'grid', 'mine_counter', 'reset_button', 'timer')
CELL_CLASSES = ('hidden', 'revealed_empty', '1', '2', '3', '4', '5', '6', '7', '8', 'flag')
# Canonical CellState 0..13 -> model class IDs; lost-game special states ignored.
CELL_LUT = np.array([1,2,3,4,5,6,7,8,9,0,10,-1,-1,-1], dtype=np.int8)
SURFACE_LUT = np.array([1]*9 + [0,0,-1,-1,-1], dtype=np.int8)
CONTENT_LUT = np.array(list(range(9)) + [0,9,-1,-1,-1], dtype=np.int8)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def yolo_lines(metadata):
    w, h = metadata['size']
    lines = []
    for board in metadata['boards']:
        for name, box in board['boxes'].items():
            if name not in YOLO_CLASSES:
                continue
            x, y, right, bottom = box
            if not (0 <= x < right <= w+1e-7 and 0 <= y < bottom <= h+1e-7):
                raise ValueError(f'Invalid detection box: {name}: {box}')
            values = ((x+right)/(2*w), (y+bottom)/(2*h), (right-x)/w, (bottom-y)/h)
            lines.append(f'{YOLO_CLASSES.index(name)} ' + ' '.join(f'{v:.8f}' for v in values))
    return lines


def export_yolo(root, split, name, image, metadata):
    root = Path(root)
    images = root / 'images' / split
    labels = root / 'labels' / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    image.save(images / f'{name}.png')
    lines = yolo_lines(metadata)
    (labels / f'{name}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def write_yolo_config(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # JSON string escaping is also valid YAML quoting.
    lines = ['path: ' + json.dumps(root.resolve().as_posix()), 'train: images/train',
             'val: images/val', 'names:']
    lines += [f'  {i}: {name}' for i, name in enumerate(YOLO_CLASSES)]
    (root/'data.yaml').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def crop_extent(image, box, size):
    """Fractional xyxy crop, using Pillow's inverse mapping (no edge rounding)."""
    return image.transform(tuple(size), Image.Transform.EXTENT, tuple(box), Image.Resampling.BICUBIC)


def geometry_sample(image, board, *, rng=None, margin=.3, output_size=None):
    """Grid crop with optional random outward margin measured in cell pitches.

    Returns exact crop-local coordinates. output_size=(width,height) optionally
    warps the crop; leave None to retain its approximate native resolution.
    """
    if margin < 0:
        raise ValueError('margin must be nonnegative')
    rng = rng or random.Random(0)
    x, y, right, bottom = board['boxes']['grid']
    px = (right-x)/board['columns']
    py = (bottom-y)/board['rows']
    box = (max(0,x-rng.uniform(0,margin)*px), max(0,y-rng.uniform(0,margin)*py),
           min(image.width,right+rng.uniform(0,margin)*px), min(image.height,bottom+rng.uniform(0,margin)*py))
    width, height = box[2]-box[0], box[3]-box[1]
    size = tuple(output_size) if output_size is not None else (ceil(width),ceil(height))
    if min(size) < 1:
        raise ValueError('Invalid output size')
    sx, sy = size[0]/width, size[1]/height
    local = transform_board(board, sx=sx, sy=sy, dx=-box[0]*sx, dy=-box[1]*sy)
    centers = [[((l+r)/2,(t+b)/2) for l,t,r,b in row] for row in local['cell_boxes']]
    # Only grid geometry is exported, since UI elements can lie outside this crop.
    target = {'board_id': board['board_id'], 'rows': board['rows'], 'columns': board['columns'],
              'size': size, 'source_crop': box, 'grid_box': local['boxes']['grid'],
              'x_edges': local['x_edges'], 'y_edges': local['y_edges'],
              'cell_boxes': local['cell_boxes'], 'centers': centers, 'style': board['style']}
    if 'game_state_id' in board:
        target['game_state_id'] = board['game_state_id']
    return crop_extent(image, box, size), target


def validate_heatmap_sigma(sigma: float) -> None:
    if not isfinite(sigma) or sigma <= 0:
        raise ValueError('Heatmap sigma must be finite and positive')


def _gaussian(squared_distance, sigma):
    # Dividing twice avoids underflow or overflow from squaring sigma.
    with np.errstate(over='ignore'):
        scaled_distance = squared_distance / sigma / sigma
    return np.exp(-.5 * scaled_distance)


def geometry_heatmaps(target, *, sigma=1.5, include_outer=True):
    """Float32 [3,H,W]: cell centers, vertical lines, horizontal lines.

    Gaussian peaks use pixel-center coordinates. Boundary lines are masked
    to the true grid extent; combine peaks by max rather than sum.
    """
    validate_heatmap_sigma(sigma)
    w, h = target['size']
    x = np.arange(w, dtype=np.float64)+.5
    y = np.arange(h, dtype=np.float64)+.5
    centers = np.asarray(target['centers'], dtype=np.float64)
    # Rectangular lattice: separable nearest-center distances.
    dx = np.min((x[:,None]-centers[0,:,0])**2, axis=1)
    dy = np.min((y[:,None]-centers[:,0,1])**2, axis=1)
    center_map = _gaussian(dy[:, None] + dx[None, :], sigma)
    xe = target['x_edges'] if include_outer else target['x_edges'][1:-1]
    ye = target['y_edges'] if include_outer else target['y_edges'][1:-1]
    gx,gy,gr,gb = target['grid_box']
    vx = _gaussian(np.min((x[:, None] - np.asarray(xe)) ** 2, axis=1), sigma) if xe else np.zeros(w)
    hy = _gaussian(np.min((y[:, None] - np.asarray(ye)) ** 2, axis=1), sigma) if ye else np.zeros(h)
    vertical = ((y>=gy)&(y<gb))[:,None]*vx[None,:]
    horizontal = hy[:,None]*((x>=gx)&(x<gr))[None,:]
    return np.stack((center_map,vertical,horizontal)).astype(np.float32)


def export_geometry(root, split, name, image, board, *, rng, output_size=None,
                    margin=.3, sigma=None):
    crop, target = geometry_sample(image, board, rng=rng, margin=margin, output_size=output_size)
    heatmaps = geometry_heatmaps(target, sigma=sigma) if sigma is not None else None
    folder = Path(root)/split
    folder.mkdir(parents=True, exist_ok=True)
    crop.save(folder/f'{name}.png')
    write_json(folder/f'{name}.json', target)
    if heatmaps is not None:
        np.savez_compressed(folder/f'{name}.npz', heatmaps=heatmaps)


def _jitter_box(box, shift, scale, image_size):
    left, top, right, bottom = box
    w, h = right-left, bottom-top
    cx = (left+right)/2 + shift[0]*w
    cy = (top+bottom)/2 + shift[1]*h
    factor = scale
    # Keep extent inside the screenshot, without adding synthetic black pixels.
    return (max(0,cx-w*factor/2), max(0,cy-h*factor/2),
            min(image_size[0],cx+w*factor/2), min(image_size[1],cy+h*factor/2))


def cell_sample(image, board, *, cell_size=32, rng=None, jitter=.04, use_slots=False):
    """All cells in row-major order, including ignored special states for context.

    Jitter is shared across the board to mimic a small lattice-estimation error.
    Cell size here is the CNN input dimension, not the renderer's native tile size.
    """
    if cell_size < 1 or not 0 <= jitter <= .25:
        raise ValueError('Invalid CNN input size or jitter')
    rng = rng or random.Random(0)
    rows, cols = board['rows'], board['columns']
    states = np.asarray(board['cell_states'], dtype=np.int8).reshape(-1)
    images = np.empty((rows*cols,cell_size,cell_size,3), dtype=np.uint8)
    crop_boxes = []
    shift = (rng.uniform(-jitter,jitter), rng.uniform(-jitter,jitter))
    scale = 1+rng.uniform(-jitter,jitter)
    for row in range(rows):
        for col in range(cols):
            box = board['cell_boxes'][row][col]
            if use_slots:
                xe,ye = board['x_edges'],board['y_edges']
                box = (xe[col],ye[row],xe[col+1],ye[row+1])
            box = _jitter_box(box,shift,scale,image.size)
            index = row*cols+col
            images[index] = np.asarray(crop_extent(image,box,(cell_size,cell_size)).convert('RGB'))
            crop_boxes.append(box)
    labels = CELL_LUT[states]
    if 'opened_surfaces' in board:
        surface_labels = np.asarray(board['opened_surfaces'], dtype=np.int8).flatten()
        surface_labels[labels == -1] = -1
    else:
        # Legacy callers may provide only canonical states.
        surface_labels = SURFACE_LUT[states]
    return {'images': images, 'cell_states': states, 'labels': labels,
            'surface_labels': surface_labels, 'content_labels': CONTENT_LUT[states],
            'crop_boxes': np.asarray(crop_boxes, dtype=np.float32),
            'rows': np.int32(rows), 'columns': np.int32(cols)}


def export_cells(root, split, name, image, board, *, rng, cell_size=32, jitter=.04, use_slots=False):
    folder = Path(root)/split
    folder.mkdir(parents=True,exist_ok=True)
    sample = cell_sample(image,board,cell_size=cell_size,rng=rng,jitter=jitter,use_slots=use_slots)
    np.savez_compressed(folder/f'{name}.npz', **sample)
    metadata = {key: board[key] for key in
                ('board_id','style','renderer','theme','style_parameters','status','generation_mode','target_number')}
    if 'game_state_id' in board:
        metadata['game_state_id'] = board['game_state_id']
    write_json(folder/f'{name}.json', metadata)
    return Counter(int(label) for label in sample['labels'] if label >= 0)


def select_targets(labels, count, *, rng=None, class_weights=None):
    """Select distinct supervised cells; context remains the full board.

    Weights may come from global training-set frequencies. Ignore label -1.
    Fewer targets are returned if fewer eligible cells have positive weight.
    """
    if count < 0:
        raise ValueError('count must be nonnegative')
    rng = rng or random.Random()
    remaining = [i for i,label in enumerate(labels) if label >= 0]
    weights = None
    if class_weights is not None:
        candidates = [(i, float(class_weights[int(labels[i])])) for i in remaining]
        if any(not isfinite(weight) or weight < 0 for _, weight in candidates):
            raise ValueError('Target weights must be finite and nonnegative')
        remaining = [i for i, weight in candidates if weight > 0]
        weights = [weight for _, weight in candidates if weight > 0]

    chosen = []
    for _ in range(min(count,len(remaining))):
        position = rng.choices(range(len(remaining)),weights=weights,k=1)[0]
        chosen.append(remaining.pop(position))
        if weights is not None:
            weights.pop(position)
    return np.asarray(chosen,dtype=np.int64)
