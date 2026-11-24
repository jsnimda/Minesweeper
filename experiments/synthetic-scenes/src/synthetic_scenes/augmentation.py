"""Capture-like augmentation with synchronized annotations."""
from copy import deepcopy
from io import BytesIO

from PIL import Image, ImageFilter

from .coordinates import transform_board


def augment_scene(image, metadata, *, scale=1., blur=0., jpeg_quality=None,
                  resampling=Image.Resampling.LANCZOS):
    """Resize the composed image, then optionally blur and JPEG roundtrip.

    Metadata stays in final image coordinates, without rounding box edges.
    No crops, clipping, rotation, or occlusion are introduced here.
    """
    if scale <= 0 or blur < 0:
        raise ValueError('Invalid scale or blur')
    if jpeg_quality is not None and not 1 <= jpeg_quality <= 100:
        raise ValueError('Invalid JPEG quality')
    w, h = image.size
    size = (max(1, round(w*scale)), max(1, round(h*scale)))
    result = image.resize(size, resampling) if size != image.size else image.copy()
    if blur:
        result = result.filter(ImageFilter.GaussianBlur(blur))
    if jpeg_quality is not None:
        buffer = BytesIO()
        result.convert('RGB').save(buffer, format='JPEG', quality=jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            result = decoded.convert('RGB')
    sx, sy = size[0]/w, size[1]/h
    meta = deepcopy(metadata)
    meta['boards'] = [transform_board(b, sx=sx, sy=sy) for b in metadata['boards']]
    meta['size'] = size
    meta.setdefault('augmentations', []).append({'scale': scale, 'actual_scale': (sx,sy),
        'blur': blur, 'jpeg_quality': jpeg_quality, 'resampling': resampling.name})
    return result, meta
