"""Presentation data and viewer selection policy for recognition results."""
from dataclasses import dataclass

from .recognition_state import ResultId


@dataclass(frozen=True)
class ExplorerItem:
    identity: ResultId
    label: str
    summary: str
    children: tuple['ExplorerItem', ...] = ()


def _validation_status(validation):
    if validation is None:
        return 'Unavailable'
    accepted = getattr(validation, 'accepted', None)
    if accepted is True:
        return 'Accepted'
    if accepted is False:
        return 'Rejected'
    return 'Unavailable'


def explorer_items(snapshot):
    """Describe the snapshot's actual nodes, retaining each complete ID."""
    if snapshot is None:
        return ()
    counts = {}

    def describe(node):
        identity = node.id
        data = snapshot.resolve(identity)
        if identity.kind == 'detection':
            counts[data.name] = counts.get(data.name, 0) + 1
            label = f'{data.name.replace("_", " ").title()} {counts[data.name]}'
            summary = f'Confidence {data.confidence:.3f}'
        elif identity.kind == 'geometry':
            label = 'Geometry result'
            if snapshot.detection_index(identity) is None:
                label += ' (unassociated)'
            status = _validation_status(data.validation)
            summary = 'Incomplete · validation unavailable' if data.validation is None else status
            if data.validation is not None and getattr(data.validation, 'geometry', None) is None:
                summary += ' · extent unavailable'
            decoder_status = getattr(data.decoding, 'status', None)
            if decoder_status:
                summary += f' · Decoder: {decoder_status}'
        elif identity.kind == 'lattice':
            label = 'Fitted lattice'
            # Consensus has pitch and origin, but no board dimensions. A global
            # extent, when available, is a separate object and is labeled as such.
            grid = snapshot.grid(identity)
            extent = getattr(grid.validation, 'geometry', None)
            summary = (f'Board extent {extent.rows} × {extent.columns}'
                       if extent is not None else 'Board extent unavailable')
        else:
            label = 'Validation'
            summary = _validation_status(data)
        return ExplorerItem(identity, label, summary, tuple(describe(child) for child in node.children))

    return tuple(describe(node) for node in snapshot.roots)


def viewer_geometry_filter(selection):
    """Return the legacy grid index filter and an exact orphan geometry filter."""
    snapshot = selection.snapshot
    identity = selection.selected
    if snapshot is None or snapshot.node(identity) is None:
        return None, None
    index = selection.detection_index()
    if index is not None:
        detection = selection.detection()
        return (index, None) if detection.name == 'grid' else (None, None)
    return None, selection.grid()
