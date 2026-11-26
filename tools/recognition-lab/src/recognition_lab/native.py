"""Prepare native-resolution crops for GridUNet without resampling."""
import numpy as np


def pad_native(rgb):
    """Center an RGB crop on the smallest black canvas divisible by eight."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0:
        raise ValueError('Expected a nonempty RGB image')
    height, width = rgb.shape[:2]
    extra_x, extra_y = (-width) % 8, (-height) % 8
    left, top = extra_x // 2, extra_y // 2
    padding = (left, top, extra_x - left, extra_y - top)
    padded = np.pad(rgb, ((top, padding[3]), (left, padding[2]), (0, 0)))
    return padded, padding


def unpad_native(maps, padding):
    """Remove left, top, right, bottom padding from channel-first maps."""
    left, top, right, bottom = padding
    height, width = maps.shape[-2:]
    if min(padding) < 0 or left + right >= width or top + bottom >= height:
        raise ValueError('Invalid output padding')
    return maps[..., top:height-bottom, left:width-right]
