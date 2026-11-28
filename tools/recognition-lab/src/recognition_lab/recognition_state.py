"""Snapshot-scoped result hierarchy and the shared recognition selection.

IDs use an opaque snapshot token and positional keys, never prediction values.
Every replacement creates a new snapshot: old IDs cannot resolve in it. A geometry
update may transfer selection to the same detection object; geometry-derived
selections fall back to that detection because their underlying data changed.
The hierarchy holds only metadata and references the authoritative result.
"""
from dataclasses import dataclass

from .results import RecognitionResult


@dataclass(frozen=True)
class ResultId:
    snapshot: object
    kind: str
    index: int


@dataclass(frozen=True)
class ResultNode:
    id: ResultId
    children: tuple['ResultNode', ...] = ()


class ResultSnapshot:
    """A read-only index over one result; prediction objects are not copied."""
    def __init__(self, result: RecognitionResult):
        self.result = result
        self._token = object()
        self._nodes = {}
        self._data = {}
        self._detection_indices = {}
        geometry_by_detection = {}
        orphans = []
        for index, grid in enumerate(result.grids):
            children = []
            validation = grid.validation
            lattice = getattr(validation, 'lattice', None) if validation is not None else None
            if lattice is not None:
                children.append(self._node('lattice', index, lattice, grid.detection_index))
            if validation is not None:
                children.append(self._node('validation', index, validation, grid.detection_index))
            node = self._node('geometry', index, grid, grid.detection_index, children)
            if 0 <= grid.detection_index < len(result.detections) and result.detections[grid.detection_index].name == 'grid':
                geometry_by_detection.setdefault(grid.detection_index, []).append(node)
            else:
                orphans.append(node)
        roots = []
        for index, detection in enumerate(result.detections):
            roots.append(self._node('detection', index, detection, index, geometry_by_detection.get(index, ())))
        self.roots = tuple(roots + orphans)

    def _node(self, kind, index, data, detection_index, children=()):
        identity = ResultId(self._token, kind, index)
        node = ResultNode(identity, tuple(children))
        self._nodes[identity] = node
        self._data[identity] = data
        if 0 <= detection_index < len(self.result.detections) and (kind == 'detection' or self.result.detections[detection_index].name == 'grid'):
            self._detection_indices[identity] = detection_index
        return node

    def node(self, identity):
        return self._nodes.get(identity) if isinstance(identity, ResultId) else None

    def resolve(self, identity):
        return self._data.get(identity) if isinstance(identity, ResultId) else None

    def detection_id(self, index):
        identity = ResultId(self._token, 'detection', index)
        return identity if identity in self._nodes else None

    def detection_index(self, identity):
        return self._detection_indices.get(identity) if isinstance(identity, ResultId) else None

    def grid(self, identity):
        if not isinstance(identity, ResultId) or identity not in self._nodes:
            return None
        if identity.kind != 'detection':
            return self.result.grids[identity.index]
        index = self.detection_index(identity)
        if self.result.detections[index].name != 'grid':
            return None
        return next((grid for grid in self.result.grids if grid.detection_index == index), None)


class RecognitionSelection:
    """Owns the current snapshot and selection, independent of any UI widget."""
    def __init__(self):
        self.snapshot = None
        self.selected = None

    def select(self, identity):
        self.selected = identity if self.snapshot is not None and self.snapshot.node(identity) is not None else None
        return self.selected is not None

    def replace(self, result, *, geometry_update=False, fallback_to_first_grid=False):
        old = self.snapshot
        old_index = old.detection_index(self.selected) if old is not None else None
        old_detection = old.result.detections[old_index] if old_index is not None else None
        self.snapshot = ResultSnapshot(result) if result is not None else None
        self.selected = None
        if self.snapshot is None:
            return
        if geometry_update and old_detection is not None:
            matches = [i for i, detection in enumerate(result.detections) if detection is old_detection]
            if len(matches) == 1:
                self.selected = self.snapshot.detection_id(matches[0])
        if self.selected is None and fallback_to_first_grid:
            index = next((i for i, detection in enumerate(result.detections) if detection.name == 'grid'), None)
            if index is not None:
                self.selected = self.snapshot.detection_id(index)

    def resolve(self):
        return self.snapshot.resolve(self.selected) if self.snapshot is not None else None

    def detection_index(self):
        return self.snapshot.detection_index(self.selected) if self.snapshot is not None else None

    def detection(self):
        index = self.detection_index()
        return self.snapshot.result.detections[index] if index is not None else None

    def grid(self):
        return self.snapshot.grid(self.selected) if self.snapshot is not None else None
