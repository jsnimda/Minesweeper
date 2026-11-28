"""Read-only property presentation over the authoritative recognition snapshot."""
from dataclasses import dataclass
from numbers import Number

from .recognition_state import hypothesis_validation
from .viewer_selection import observation_source_bounds


@dataclass(frozen=True)
class Property:
    name: str
    value: str


@dataclass(frozen=True)
class Section:
    title: str
    properties: tuple[Property, ...]


def _value(value):
    if value is None:
        return 'Unavailable'
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    if isinstance(value, Number):
        # numpy scalar .item() retains its actual stored value, including float32.
        return repr(value.item() if hasattr(value, 'item') else value)
    if isinstance(value, str):
        return value if value else '(empty)'
    if isinstance(value, (tuple, list)) and len(value) <= 8 and all(
            v is None or isinstance(v, (str, Number, bool)) for v in value):
        return ', '.join(_value(v) for v in value) if value else '(empty)'
    return 'Unavailable'


def _section(title, pairs):
    return Section(title, tuple(Property(name, _value(value)) for name, value in pairs))


def _space(grid):
    mode = grid.diagnostics.get('mode')
    if mode == 'whole':
        return 'Resized crop pixels (valid region, padding removed)'
    if mode == 'tiled':
        return 'Crop pixels (native resolution)'
    return 'Unavailable (inference mode not recorded)'


def _status(validation):
    accepted = getattr(validation, 'accepted', None)
    if accepted is True:
        return 'Accepted'
    if accepted is False:
        return 'Rejected'
    return 'Unavailable'


def _extent(validation):
    geometry = getattr(validation, 'geometry', None)
    if geometry is None:
        return [('Board extent', None)]
    xs, ys = geometry.x_edges, geometry.y_edges
    return [('Rows', geometry.rows), ('Columns', geometry.columns),
            ('Left edge (px)', xs[0] if len(xs) else None),
            ('Right edge (px)', xs[-1] if len(xs) else None),
            ('Top edge (px)', ys[0] if len(ys) else None),
            ('Bottom edge (px)', ys[-1] if len(ys) else None),
            ('X lattice index range', geometry.x_indices),
            ('Y lattice index range', geometry.y_indices)]


def _measurements(validation):
    measurements = getattr(validation, 'measurements', None)
    if not isinstance(measurements, dict):
        return [('Measurements', None)]
    # Only scalar values are suitable for this compact view; retain zero/False.
    values = [(str(key), value) for key, value in measurements.items()
              if value is None or isinstance(value, (str, Number, bool))]
    if len(values) > 40:
        return values[:40] + [('Additional scalar measurements', len(values) - 40)]
    return values if values else [('Measurements', '(no scalar measurements recorded)')]


def _reasons(reasons):
    return None if reasons is None else ('; '.join(reasons) if reasons else '(none recorded)')


def _observation_space(grid):
    mode = grid.diagnostics.get('mode')
    if mode == 'whole':
        return 'Resized crop pixels (valid region, padding removed)'
    if mode == 'tiled':
        return 'Tile-local pixels (native resolution)'
    return 'Unavailable (inference mode not recorded)'


def _array_info(array):
    return (getattr(array, 'shape', None), str(array.dtype) if hasattr(array, 'dtype') else None)


def _observation_sections(grid, observation, identity):
    box = getattr(observation, 'box', None)
    local_box = (0, 0, box[2]-box[0], box[3]-box[1]) if box is not None else None
    transform = getattr(observation, 'input_transform', None)
    full = getattr(observation, 'full_fields', None)
    valid = getattr(observation, 'fields', None)
    input_array = getattr(observation, 'input_array', None)
    decoded = getattr(observation, 'decoding', None)
    local = getattr(observation, 'local_validation', None)
    full_shape, full_dtype = _array_info(full)
    valid_shape, valid_dtype = _array_info(valid)
    input_shape, input_dtype = _array_info(input_array)
    accepted = getattr(local, 'local_accepted', None)
    local_status = 'Accepted' if accepted is True else 'Rejected' if accepted is False else 'Unavailable'
    return (
        _section('Observation', [('Observation index (zero-based)', identity.observation_index),
            ('Inference mode', grid.diagnostics.get('mode')), ('Coordinate space', _observation_space(grid)),
            ('Observation bounds (local px)', local_box), ('Box coordinate space', _space(grid)),
            ('Observation box (px)', box), ('Source bounds coordinate space', 'Screenshot pixels'),
            ('Source bounds (px)', observation_source_bounds(grid, observation))]),
        _section('Model input and preprocessing', [('Input shape (CHW)', input_shape), ('Input dtype', input_dtype),
            ('Input dimensions (width, height)', getattr(transform, 'canvas_size', None)),
            ('Representation', getattr(observation, 'input_representation', None)),
            ('Resize method', getattr(observation, 'resize_method', None)),
            ('Source dimensions', getattr(transform, 'source_size', None)),
            ('Resized dimensions', getattr(transform, 'resized_size', None)),
            ('Canvas offset', getattr(transform, 'offset', None)),
            ('Scale X', getattr(transform, 'scale_x', None)), ('Scale Y', getattr(transform, 'scale_y', None)),
            ('Known rectangle (model canvas px)', getattr(observation, 'known_box', None)),
            ('Padding left, top, right, bottom (px)', getattr(observation, 'padding', None))]),
        _section('Decoder and local validation', [('Decoder status', getattr(decoded, 'status', None)),
            ('Decoder reasons', _reasons(getattr(decoded, 'reasons', None))),
            ('Returned hypotheses', len(decoded.hypotheses) if decoded is not None else None),
            ('Local validation status', local_status),
            ('Local rejection reasons', _reasons(getattr(local, 'local_reasons', None)))]),
        _section('Local support measurements', _measurements(local)),
        _section('Diagnostic artifacts', [('Model input retained', input_array is not None),
            ('Full activated fields retained', full is not None), ('Full fields shape (CHW)', full_shape),
            ('Full fields dtype', full_dtype), ('Valid fields retained', valid is not None),
            ('Valid fields shape (CHW)', valid_shape), ('Valid fields dtype', valid_dtype),
            ('Decoder retained', decoded is not None), ('Local validation retained', local is not None)]))


def _hypothesis_sections(grid, observation, hypothesis, identity):
    x, y = hypothesis.x, hypothesis.y
    xe, ye = hypothesis.x_evidence, hypothesis.y_evidence
    applicability = hypothesis_validation(observation, identity.hypothesis_index)
    local = getattr(observation, 'local_validation', None) if applicability.startswith('Evaluated ') else None
    return (
        _section('Hypothesis', [('Observation index (zero-based)', identity.observation_index),
            ('Hypothesis index (zero-based)', identity.hypothesis_index),
            ('Decoder rank (one-based)', identity.hypothesis_index + 1),
            ('Score (not calibrated confidence)', hypothesis.score),
            ('Square preference', hypothesis.square_preference), ('Prior contribution', hypothesis.prior_contribution),
            ('Coordinate space', _observation_space(grid))]),
        _section('Local lattice and equivalent boundaries', [('Pitch X (px)', x.pitch), ('Pitch Y (px)', y.pitch),
            ('Periodic origin X (px modulo pitch)', xe.origin_mod_pitch),
            ('Periodic origin Y (px modulo pitch)', ye.origin_mod_pitch),
            ('Left equivalent boundary (px)', x.origin), ('Right equivalent boundary (px)', x.end),
            ('Top equivalent boundary (px)', y.origin), ('Bottom equivalent boundary (px)', y.end),
            ('Boundary meaning', 'Local equivalent lattice boundaries; physical termination not implied')]),
        _section('Visible occupancy crossings', [('Coordinate space', _observation_space(grid)),
            ('Left crossing (px)', x.visible_start), ('Right crossing (px)', x.visible_end),
            ('Top crossing (px)', y.visible_start), ('Bottom crossing (px)', y.visible_end),
            ('Left censored', x.start_censored), ('Right censored', x.end_censored),
            ('Top censored', y.start_censored), ('Bottom censored', y.end_censored)]),
        _section('Selected frequency evidence', [('X candidate pitch', xe.pitch), ('X coherence', xe.coherence),
            ('X magnitude', xe.magnitude), ('Y candidate pitch', ye.pitch), ('Y coherence', ye.coherence),
            ('Y magnitude', ye.magnitude), ('Interpretation', 'Evidence; not calibrated confidence')]),
        _section('Local validation', [('Applicability', applicability),
            ('Reasons', _reasons(getattr(local, 'local_reasons', None)))]),
        _section('Local support measurements', _measurements(local)))


def _inspector_sections(snapshot, identity):
    """Resolve an ID once and describe only fields present in stored results."""
    if snapshot is None or snapshot.node(identity) is None:
        return ()
    data = snapshot.resolve(identity)
    if identity.kind == 'detection':
        x1, y1, x2, y2 = data.box
        return (_section('Detection', [
            ('Class', data.name), ('Class ID', None),
            ('Detection index (zero-based)', identity.index),
            ('Confidence', data.confidence), ('Coordinate space', 'Screenshot pixels'),
            ('Left (px)', x1), ('Top (px)', y1), ('Right (px)', x2), ('Bottom (px)', y2),
            ('Width (px)', x2 - x1), ('Height (px)', y2 - y1)]),)

    grid = snapshot.grid(identity)
    if grid is None:
        return ()
    validation = grid.validation
    if identity.kind == 'observations':
        return (_section('Observations', [('Count', len(data)), ('Inference mode', grid.diagnostics.get('mode'))]),)
    if identity.kind == 'observation':
        return _observation_sections(grid, data, identity)
    if identity.kind == 'hypothesis':
        return _hypothesis_sections(grid, snapshot.observation(identity), data, identity)
    if identity.kind == 'geometry':
        crop = grid.crop
        x1, y1, x2, y2 = crop.box
        mode = grid.diagnostics.get('mode')
        if mode == 'tiled' and grid.heatmaps is None:
            phase = grid.phase_data
            observations = phase.get('observations') if isinstance(phase, dict) else None
            heatmaps = ('Tile fields retained in observations' if observations and
                        any(getattr(o, 'fields', None) is not None for o in observations) else 'Unavailable')
        elif grid.heatmaps is None:
            heatmaps = 'Unavailable'
        elif mode == 'tiled':
            heatmaps = 'Placeholder; tile fields retained in observations'
        elif mode == 'whole':
            heatmaps = 'Available'
        else:
            heatmaps = 'Stored; interpretation unavailable'
        phase = grid.phase_data
        observations = phase.get('observations') if isinstance(phase, dict) else None
        index = snapshot.detection_index(identity)
        diagnostics = [(key, grid.diagnostics[key]) for key in
                       ('image dimensions', 'crop dimensions', 'model input dimensions',
                        'valid resized region', 'tile layout (columns, rows)', 'tile count',
                        'local valid tiles', 'local valid', 'complete-board validity')
                       if key in grid.diagnostics]
        return (
            _section('Geometry result', [
                ('Originating detection index (zero-based)', index),
                ('Stored detection index (zero-based)', grid.detection_index),
                ('Decoder status', getattr(grid.decoding, 'status', None)),
                ('Validation status', _status(validation)), ('Inference mode (recorded)', mode)]
                + ([('Other run settings', None)] if snapshot.result.geometry_provenance is None else [])),
            _section('Input crop', [('Coordinate space', 'Screenshot pixels'),
                ('Left (px)', x1), ('Top (px)', y1), ('Right (px)', x2), ('Bottom (px)', y2),
                ('Width (px)', x2-x1), ('Height (px)', y2-y1), ('Clipped', crop.clipped)]),
            _section('Stored outputs', [('Fitted lattice', getattr(validation, 'lattice', None) is not None),
                ('Validation', validation is not None), ('Heatmaps', heatmaps),
                ('Global phase data', isinstance(phase, dict) and phase.get('global') is not None),
                ('Observations', len(observations) if observations is not None else None),
                ('Screenshot-coordinate hypotheses', len(grid.hypotheses))]),
            _section('Recorded diagnostics', diagnostics or [('Diagnostics', None)]))
    if identity.kind == 'lattice':
        return (_section('Fitted lattice', [('Coordinate space', _space(grid)),
            ('Origin X (px)', getattr(data, 'origin_x', None)), ('Origin Y (px)', getattr(data, 'origin_y', None)),
            ('Pitch X (px)', getattr(data, 'pitch_x', None)), ('Pitch Y (px)', getattr(data, 'pitch_y', None)),
            ('Residual X (px)', getattr(data, 'residual_x', None)), ('Residual Y (px)', getattr(data, 'residual_y', None)),
            ('Ambiguous', getattr(data, 'ambiguous', None)),
            ('Inlier observations', len(data.inliers) if hasattr(data, 'inliers') else None),
            ('Rejected observations', len(data.rejected) if hasattr(data, 'rejected') else None)]),
            _section('Validation-derived board extent', [('Coordinate space', _space(grid))] + _extent(validation)))
    if identity.kind == 'validation':
        reasons = getattr(data, 'reasons', None)
        reason_text = None if reasons is None else ('; '.join(reasons) if reasons else '(none recorded)')
        return (_section('Board validation', [('Status', _status(data)), ('Reasons', reason_text)]),
                _section('Validation-derived board extent', [('Coordinate space', _space(grid))] + _extent(data)),
                _section('Validation measurements', _measurements(data)))
    return ()


def inspector_sections(snapshot, identity):
    sections = _inspector_sections(snapshot, identity)
    if not sections:
        return sections
    provenance = (snapshot.result.detection_provenance if identity.kind == 'detection'
                  else snapshot.result.geometry_provenance)
    if provenance is None:
        return sections + (_section('Inference provenance', [('Recorded provenance', None)]),)
    pairs = [('Requested device', provenance.requested_device),
             ('Resolved execution device', provenance.resolved_device),
             ('Loaded checkpoint', provenance.checkpoint_path), ('Model load ID', provenance.model_id),
             ('Checkpoint SHA-256', provenance.checkpoint_sha256 or provenance.fingerprint_status)]
    if identity.kind == 'detection':
        pairs += [('Requested YOLO image size', provenance.requested_image_size),
                  ('Effective YOLO image size', provenance.effective_image_size),
                  ('Confidence threshold', provenance.confidence)]
    else:
        pairs += [('Crop padding (px)', provenance.padding), ('Geometry mode', provenance.mode),
                  ('Tile size (px)', provenance.tile), ('Tile overlap (px)', provenance.overlap),
                  ('Whole-mode input size (px)', provenance.whole_input_size)]
    return sections + (_section('Inference provenance', pairs),)


def property_report(sections):
    """Copy the exact values displayed in the property table."""
    return '\n\n'.join(section.title + '\n' + '\n'.join(
        f'{prop.name}: {prop.value}' for prop in section.properties) for section in sections)
