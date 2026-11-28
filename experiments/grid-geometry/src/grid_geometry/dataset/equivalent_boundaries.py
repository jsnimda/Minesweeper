"""Ground-truth lattice extents within a rectangular known image region."""
from dataclasses import dataclass
from itertools import product
import math

from .targets import PhaseGeometry


@dataclass(frozen=True)
class EquivalentBoundaries:
    """Acceptable lattice coordinates, in continuous sample coordinates.

    Side alternatives are not necessarily independent for very narrow regions;
    use ``valid_boxes`` to enumerate strictly ordered combinations.
    """
    left: tuple[float, ...]
    top: tuple[float, ...]
    right: tuple[float, ...]
    bottom: tuple[float, ...]

    def valid_boxes(self):
        """Return acceptable (left, top, right, bottom) boxes with positive area."""
        return tuple((left, top, right, bottom)
                     for left, top, right, bottom in product(
                         self.left, self.top, self.right, self.bottom)
                     if left < right and top < bottom)


def equivalent_boundaries(geometry: PhaseGeometry | None, board_dimensions, known_box, *,
                          midpoint_tolerance: float = .05) -> EquivalentBoundaries | None:
    """Derive lattice-aligned boundaries from sample geometry and known_box.

    ``board_dimensions`` is (rows, columns); ``known_box`` is the continuous
    (left, top, right, bottom) extent, in the same coordinates as geometry.
    Visible tile edges determine whether a physical termination is observed;
    the corresponding ideal lattice edge is returned. A negative has no geometry.
    The tolerance is a fraction of pitch, with only floating-point roundoff at
    the two thresholds included in the midpoint interval.
    """
    if not math.isfinite(midpoint_tolerance) or not 0 <= midpoint_tolerance < .5:
        raise ValueError('midpoint_tolerance must be finite and in [0, 0.5)')
    if geometry is None:
        return None
    if (not isinstance(board_dimensions, (tuple, list)) or len(board_dimensions) != 2
            or any(type(n) is not int or n <= 0 for n in board_dimensions)):
        raise ValueError('board_dimensions must contain positive integer rows and columns')
    if len(known_box) != 4 or not all(math.isfinite(v) for v in known_box):
        raise ValueError('known_box must contain four finite coordinates')
    kl, kt, kr, kb = known_box
    if kl >= kr or kt >= kb:
        raise ValueError('known_box must have positive area')
    left, top, right, bottom = geometry.visible_box
    if left >= kr or right <= kl or top >= kb or bottom <= kt:
        raise ValueError('board must overlap the known region')
    rows, columns = board_dimensions

    def side(edge, visible_edge, origin, pitch, count, leading):
        # The visible physical edge, not the half-gap ideal edge, decides
        # observability. Equality is a physical termination.
        physical = visible_edge >= edge if leading else visible_edge <= edge
        if physical:
            return (origin if leading else origin + count * pitch,)
        position = (edge - origin) / pitch
        lower = math.floor(position)
        fraction = position - lower
        # Account for rounding in the normalized position, without introducing
        # a coordinate localization tolerance or a broad ambiguity region.
        epsilon = 8 * math.ulp(max(1., abs(position)))
        low, high = .5 - midpoint_tolerance, .5 + midpoint_tolerance
        if fraction < low - epsilon:
            indices = (lower,)
        elif fraction > high + epsilon:
            indices = (lower + 1,)
        else:
            indices = (lower, lower + 1)
        # Visible edges can differ from ideal board edges by half a tile gap.
        # Never extrapolate beyond the finite physical lattice.
        indices = tuple(i for i in indices if 0 <= i <= count)
        if not indices:
            raise ValueError('known edge has no selected lattice line on the board')
        return tuple(origin + i * pitch for i in indices)

    result = EquivalentBoundaries(
        side(kl, left, geometry.origin_x, geometry.pitch_x, columns, True),
        side(kt, top, geometry.origin_y, geometry.pitch_y, rows, True),
        side(kr, right, geometry.origin_x, geometry.pitch_x, columns, False),
        side(kb, bottom, geometry.origin_y, geometry.pitch_y, rows, False))
    if not result.valid_boxes():
        raise ValueError('no positive-area equivalent boundary combination')
    return result
