"""Formatting and parsing physical magnification percentages."""
import math


def parse_percent(text):
    text = text.strip()
    if text.endswith('%'):
        text = text[:-1].strip()
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('magnification must be finite and positive')
    return min(6400.0, max(1.0, value)) / 100


def format_percent(magnification):
    if not math.isfinite(magnification) or magnification <= 0:
        raise ValueError('magnification must be finite and positive')
    percent = magnification * 100
    if percent < 1:
        return f'{percent:.6g}%'
    return f'{percent:.2f}'.rstrip('0').rstrip('.') + '%'
