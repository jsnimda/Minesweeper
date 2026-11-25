"""Owned pixel-buffer conversions between Skia and Pillow."""

import skia
from PIL import Image


def image_to_pillow(image: skia.Image) -> Image.Image:
    """Copy explicit unpremultiplied RGBA pixels into an owned RGB image."""
    pixels = image.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType,
                           alphaType=skia.AlphaType.kUnpremul_AlphaType)
    return Image.fromarray(pixels).convert("RGB")


def pillow_to_image(image: Image.Image) -> skia.Image:
    """Copy Pillow RGB(A) pixels into a Skia-owned image."""
    rgba = image.convert("RGBA")
    return skia.Image.frombytes(rgba.tobytes(), rgba.size,
                                skia.kRGBA_8888_ColorType,
                                skia.kUnpremul_AlphaType, copy=True)


