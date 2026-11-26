"""Overlapping native-resolution tiles, based on experiment notebook 07."""
import numpy as np


def tile_starts(length, tile, overlap):
    if tile <= 0 or overlap < 0 or overlap >= tile or length < tile:
        raise ValueError('Invalid tile dimensions or overlap')
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def axis_weight(tile, start, length, overlap):
    weight = np.ones(tile, dtype=np.float32)
    if overlap:
        ramp = np.sin(np.linspace(0, np.pi / 2, overlap + 2)[1:-1]) ** 2
        if start > 0:
            weight[:overlap] = ramp
        if start + tile < length:
            weight[-overlap:] = ramp[::-1]
    return weight


def stitch_tiles(image, tile, overlap, batch_size, predict):
    if tile % 8 or tile <= 0 or not 0 <= overlap < tile or batch_size < 1:
        raise ValueError('Tile size must be a positive multiple of 8; overlap must be smaller')
    h, w = image.shape[:2]
    ph, pw = max(tile, (h + 7) // 8 * 8), max(tile, (w + 7) // 8 * 8)
    padded = np.pad(image, ((0, ph-h), (0, pw-w), (0, 0)))
    positions = [(x, y) for y in tile_starts(ph, tile, overlap) for x in tile_starts(pw, tile, overlap)]
    total = np.zeros((3, ph, pw), np.float32)
    weights = np.zeros((ph, pw), np.float32)
    for start in range(0, len(positions), batch_size):
        batch = positions[start:start + batch_size]
        maps = predict(np.stack([padded[y:y+tile, x:x+tile] for x, y in batch]))
        for (x, y), values in zip(batch, maps):
            weight = axis_weight(tile, y, ph, overlap)[:, None] * axis_weight(tile, x, pw, overlap)[None, :]
            total[:, y:y+tile, x:x+tile] += values * weight
            weights[y:y+tile, x:x+tile] += weight
    if not np.all(weights > 0):
        raise RuntimeError('Tile stitching left uncovered pixels')
    return (total / weights[None])[:, :h, :w]
