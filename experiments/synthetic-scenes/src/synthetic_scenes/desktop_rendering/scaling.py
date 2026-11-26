"""Internal conversion between logical UI units and physical pixels."""

import math


def validate_scale(scale: float) -> float:
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("scale must be a positive finite number")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")
    return float(scale)


def pixels(value: float, scale: float) -> int:
    """Round a positive UI extent to pixels, retaining one drawable pixel."""
    return max(1, round(value * validate_scale(scale)))
