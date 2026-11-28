"""Read-only property presentation over the authoritative recognition snapshot."""
from dataclasses import dataclass
from numbers import Number


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
    if identity.kind == 'geometry':
        crop = grid.crop
        x1, y1, x2, y2 = crop.box
        mode = grid.diagnostics.get('mode')
        if grid.heatmaps is None:
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
             ('Loaded checkpoint', provenance.checkpoint_path), ('Model load ID', provenance.model_id)]
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
