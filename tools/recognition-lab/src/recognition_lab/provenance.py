"""Immutable stage provenance, validation and pending-settings comparison."""
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class DetectionProvenance:
    requested_device: str
    resolved_device: str
    checkpoint_path: str | None
    model_id: int | None
    requested_image_size: int
    confidence: float
    effective_image_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class GeometryProvenance:
    requested_device: str
    resolved_device: str
    checkpoint_path: str | None
    model_id: int | None
    padding: int
    mode: str
    tile: int | None = None
    overlap: int | None = None
    whole_input_size: tuple[int, int] | None = None


def validation_errors(settings, stage, available_devices=None):
    """Only validate fields used by the requested stage(s)."""
    if stage not in ('detection', 'geometry', 'all'):
        raise ValueError('Unknown inference stage')
    errors = []
    if settings.device not in ('auto', 'cpu', 'cuda', 'mps'):
        errors.append('Choose a supported device')
    elif available_devices is not None and settings.device != 'auto' and settings.device not in available_devices:
        errors.append(f'{settings.device} device is unavailable')
    if stage in ('detection', 'all'):
        if type(settings.imgsz) is not int or not 32 <= settings.imgsz <= 8192:
            errors.append('YOLO image size must be an integer from 32 to 8192')
        if not isinstance(settings.confidence, (int, float)) or isinstance(settings.confidence, bool) or not math.isfinite(settings.confidence) or not 0 <= settings.confidence <= 1:
            errors.append('Confidence must be between 0 and 1')
    if stage in ('geometry', 'all'):
        if type(settings.padding) is not int or not 0 <= settings.padding <= 2000:
            errors.append('Crop padding must be an integer from 0 to 2000 pixels')
        if settings.mode not in ('whole', 'tiled'):
            errors.append('Choose whole or tiled geometry mode')
        if settings.mode == 'tiled':
            if type(settings.tile) is not int or settings.tile != 384:
                errors.append('Tiled geometry requires 384-pixel tiles')
            if type(settings.overlap) is not int or not 0 <= settings.overlap < 384:
                errors.append('Tile overlap must be an integer from 0 to 383 pixels')
    return tuple(errors)


def require_settings(settings, stage):
    errors = validation_errors(settings, stage)
    if errors:
        raise ValueError('; '.join(errors))


def _path(path):
    if not path:
        return None
    try:
        return str(Path(path).expanduser().resolve())
    except (OSError, ValueError, RuntimeError):
        return str(path)



def stage_mismatches(settings, provenance, loaded_path, model_id, editable_path):
    """Compare requests and model identity, never compare auto with its resolution."""
    if provenance is None:
        return None
    differences = []
    for label, current, old in [('requested device', settings.device, provenance.requested_device),
                                ('loaded model identity', model_id, provenance.model_id),
                                ('loaded checkpoint', _path(loaded_path), _path(provenance.checkpoint_path)),
                                ('editable checkpoint path', _path(editable_path), _path(loaded_path))]:
        if current != old:
            differences.append(label)
    if provenance.model_id is None:
        differences.append('recorded model identity unavailable')
    if provenance.checkpoint_path is None:
        differences.append('recorded checkpoint unavailable')
    if isinstance(provenance, DetectionProvenance):
        pairs = [('YOLO image size', settings.imgsz, provenance.requested_image_size),
                 ('confidence', settings.confidence, provenance.confidence)]
    else:
        pairs = [('crop padding', settings.padding, provenance.padding), ('geometry mode', settings.mode, provenance.mode)]
        if settings.mode == provenance.mode == 'tiled':
            pairs += [('tile size', settings.tile, provenance.tile), ('tile overlap', settings.overlap, provenance.overlap)]
    differences.extend(label for label, current, old in pairs if current != old)
    return tuple(differences)


def tile_index_range(grid):
    phase = grid.phase_data if grid is not None else None
    observations = phase.get('observations') if isinstance(phase, dict) else None
    return (0, len(observations) - 1) if observations is not None and len(observations) else None


def tile_selection(grid, previous_grid, previous_index):
    """Keep a valid index for the same geometry; otherwise start at zero."""
    limits = tile_index_range(grid)
    if limits is None:
        return None, 0
    index = previous_index if grid is previous_grid and limits[0] <= previous_index <= limits[1] else 0
    return limits, index


def stage_summary(label, settings, provenance, loaded_path, model_id, editable_path):
    if provenance is None:
        return f'{label}: no recorded completed configuration'
    mismatch = stage_mismatches(settings, provenance, loaded_path, model_id, editable_path)
    state = 'Matches pending settings' if not mismatch else 'Differs: ' + ', '.join(mismatch)
    common = (f'requested device {provenance.requested_device}; resolved device {provenance.resolved_device}; '
              f'checkpoint {provenance.checkpoint_path or "unavailable"}; load ID {provenance.model_id}')
    if isinstance(provenance, DetectionProvenance):
        detail = f'requested size {provenance.requested_image_size}; confidence {provenance.confidence}'
    else:
        detail = f'mode {provenance.mode}; padding {provenance.padding}'
        if provenance.mode == 'tiled':
            detail += f'; tile {provenance.tile}; overlap {provenance.overlap}'
        else:
            detail += f'; input size {provenance.whole_input_size}'
    return f'{label}: {state}\n{common}\n{detail}'
