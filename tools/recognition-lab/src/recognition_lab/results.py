"""Recognition data and coordinate transforms, independent of the GUI."""
from dataclasses import dataclass, field
import math
import numpy as np
from .provenance import DetectionProvenance, GeometryProvenance


@dataclass(frozen=True)
class DesktopMapping:
    origin_x: float
    origin_y: float
    logical_width: float
    logical_height: float
    pixel_width: int
    pixel_height: int

    def to_desktop(self, x, y):
        return (self.origin_x + x * self.logical_width / self.pixel_width,
                self.origin_y + y * self.logical_height / self.pixel_height)


@dataclass(frozen=True)
class Crop:
    box: tuple[int, int, int, int]
    clipped: bool

    def to_image(self, points):
        return np.asarray(points, dtype=float) + np.array(self.box[:2])


def crop_box(box, image_size, padding):
    if padding < 0:
        raise ValueError('Padding must be nonnegative')
    x1, y1, x2, y2 = box
    if not all(math.isfinite(v) for v in box) or x2 <= x1 or y2 <= y1:
        raise ValueError('Invalid detection box')
    raw = (math.floor(x1 - padding), math.floor(y1 - padding),
           math.ceil(x2 + padding), math.ceil(y2 + padding))
    w, h = image_size
    clipped = (max(0, min(w, raw[0])), max(0, min(h, raw[1])),
               max(0, min(w, raw[2])), max(0, min(h, raw[3])))
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError('Grid proposal lies outside the image')
    return Crop(clipped, raw != clipped)


@dataclass
class Detection:
    name: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass
class GeometryResult:
    detection_index: int
    crop: Crop
    heatmaps: np.ndarray
    heatmap_transform: object
    decoding: object
    validation: object
    hypotheses: list
    timings: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    phase_data: object | None = None

    @property
    def accepted_geometry(self):
        """Screenshot-coordinate edges only when complete validation passed."""
        if self.validation is None or not self.validation.accepted or not self.hypotheses:
            return None
        return self.hypotheses[0][:2]


@dataclass
class RecognitionResult:
    detections: list[Detection] = field(default_factory=list)
    grids: list[GeometryResult] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    detection_provenance: DetectionProvenance | None = None
    geometry_provenance: GeometryProvenance | None = None


def validate_region(region, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Monitor dimensions must be positive")
    if region is None:
        return (0, 0, width, height)
    if len(region) != 4 or any(not isinstance(v, int) or isinstance(v, bool) for v in region):
        raise ValueError("Region must contain four integers")
    x, y, w, h = region
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError("Capture region must be nonempty and contained within the monitor")
    return x, y, w, h
