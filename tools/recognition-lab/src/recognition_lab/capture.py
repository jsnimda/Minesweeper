"""Capture on the Qt thread; mappings are explicit and monitor-specific."""
from dataclasses import dataclass
import sys
from time import perf_counter
from PIL import Image
from PySide6.QtGui import QGuiApplication, QImage
from .results import DesktopMapping, validate_region


@dataclass
class Frame:
    image: Image.Image
    mapping: DesktopMapping | None
    diagnostics: str


def monitors(backend):
    if backend == 'Qt':
        return [(f'{s.name()} | {s.geometry().x()},{s.geometry().y()} | {s.geometry().width()}×{s.geometry().height()} | DPR {s.devicePixelRatio():.2f}', s) for s in QGuiApplication.screens()]
    import mss
    with mss.MSS() as capture:
        return [(f'MSS {i} | {m["left"]},{m["top"]} | {m["width"]}×{m["height"]}', dict(m)) for i, m in enumerate(capture.monitors[1:], 1)]


def capture_frame(backend, monitor, region=None, high_resolution=True):
    """region is monitor-local logical xywh; MSS mappings require an exact Qt match."""
    t = perf_counter()
    if backend == 'Qt':
        screen = monitor
        geo = screen.geometry()
        x, y, w, h = validate_region(region, geo.width(), geo.height())
        pixmap = screen.grabWindow(0, x, y, w, h)
        if pixmap.isNull():
            raise RuntimeError('Qt capture failed; check screen recording permission')
        qimage = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        raw = bytes(qimage.constBits())
        image = Image.frombytes('RGBA', (qimage.width(), qimage.height()), raw, 'raw', 'RGBA', qimage.bytesPerLine()).convert('RGB')
        mapping = DesktopMapping(geo.x()+x, geo.y()+y, w, h, image.width, image.height)
        identity = screen.name()
        dpr = screen.devicePixelRatio()
    else:
        x, y, w, h = validate_region(region, monitor['width'], monitor['height'])
        import mss
        if sys.platform == 'darwin':
            import mss.darwin
            # MSS 10.2 documents this module option for Retina scaling.
            mss.darwin.IMAGE_OPTIONS = 0 if high_resolution else (mss.darwin.kCGWindowImageBoundsIgnoreFraming | mss.darwin.kCGWindowImageShouldBeOpaque | mss.darwin.kCGWindowImageNominalResolution)
        rect = {'left': monitor['left']+x, 'top': monitor['top']+y, 'width': w, 'height': h}
        with mss.MSS() as capture:
            shot = capture.grab(rect)
            image = Image.frombytes('RGB', shot.size, shot.rgb)
        matches = [s for s in QGuiApplication.screens() if (s.geometry().x(), s.geometry().y(), s.geometry().width(), s.geometry().height()) == (monitor['left'], monitor['top'], monitor['width'], monitor['height'])]
        mapping = DesktopMapping(rect['left'], rect['top'], w, h, image.width, image.height) if len(matches) == 1 else None
        identity = str(monitor)
        dpr = matches[0].devicePixelRatio() if len(matches) == 1 else None
    if w <= 0 or h <= 0:
        raise ValueError('Capture region must be nonempty')
    duration = (perf_counter()-t)*1000
    origin = (mapping.origin_x, mapping.origin_y) if mapping else 'unknown'
    return Frame(image, mapping,
                 f'{backend} | {identity}\nDesktop origin: {origin}; region offset: {x}, {y}; logical size: {w}×{h}\nPixels: {image.width}×{image.height}; DPR: {dpr}; scale: {image.width/w:.3f} × {image.height/h:.3f}\nCapture: {duration:.1f} ms; desktop mapping: {"available" if mapping else "unavailable (monitor mismatch)"}')
