"""Capture-like augmentation with synchronized annotations."""
from io import BytesIO

from PIL import Image, ImageEnhance, ImageFilter

from .coordinates import transform_board


def adjust_image(image, *, blur=0., brightness=1., contrast=1., jpeg_quality=None):
    """Apply capture effects to an image in blur, brightness, contrast, JPEG order."""
    if blur < 0 or brightness < 0 or contrast < 0:
        raise ValueError('Invalid image adjustment')
    if jpeg_quality is not None and not 1 <= jpeg_quality <= 100:
        raise ValueError('Invalid JPEG quality')
    result = image
    if blur:
        result = result.filter(ImageFilter.GaussianBlur(blur))
    if brightness != 1.:
        result = ImageEnhance.Brightness(result).enhance(brightness)
    if contrast != 1.:
        result = ImageEnhance.Contrast(result).enhance(contrast)
    if jpeg_quality is not None:
        buffer = BytesIO()
        (result if result.mode == 'RGB' else result.convert('RGB')).save(buffer, format='JPEG', quality=jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            result = decoded.convert('RGB')
    return result


def augment_scene(image, metadata, *, resize_factor=1., blur=0., brightness=1., contrast=1., jpeg_quality=None,
                  resampling=Image.Resampling.LANCZOS):
    """Resize the composed image, then apply blur, brightness, contrast and JPEG.

    Metadata stays in final image coordinates, without rounding box edges.
    No crops, clipping, rotation, or occlusion are introduced here. Unchanged
    nested metadata may be shared with the input; a no-op may also reuse the image.
    """
    if resize_factor <= 0 or blur < 0 or brightness < 0 or contrast < 0:
        raise ValueError('Invalid scale or blur')
    if jpeg_quality is not None and not 1 <= jpeg_quality <= 100:
        raise ValueError('Invalid JPEG quality')
    w, h = image.size
    size = (max(1, round(w*resize_factor)), max(1, round(h*resize_factor)))
    result = image.resize(size, resampling) if size != image.size else image
    result = adjust_image(result, blur=blur, brightness=brightness,
                          contrast=contrast, jpeg_quality=jpeg_quality)
    sx, sy = size[0]/w, size[1]/h
    meta = dict(metadata)
    meta['boards'] = [transform_board(b, sx=sx, sy=sy) for b in metadata['boards']]
    meta['size'] = size
    meta['augmentations'] = list(metadata.get('augmentations', ()))
    meta['augmentations'].append({'resize_factor': resize_factor, 'actual_scale': (sx,sy),
        'brightness': brightness, 'contrast': contrast,
        'blur': blur, 'jpeg_quality': jpeg_quality, 'resampling': resampling.name})
    return result, meta
