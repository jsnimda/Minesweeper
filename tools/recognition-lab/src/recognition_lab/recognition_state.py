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
    observation_index: int | None = None
    hypothesis_index: int | None = None


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
            phase = grid.phase_data if isinstance(grid.phase_data, dict) else {}
            observations = phase.get('observations')
            if observations:
                observation_nodes = []
                for observation_index, observation in enumerate(observations):
                    decoded = getattr(observation, 'decoding', None)
                    hypotheses = getattr(decoded, 'hypotheses', ()) if decoded is not None else ()
                    hypothesis_nodes = [self._node('hypothesis', index, hypothesis, grid.detection_index,
                                                   observation_index=observation_index, hypothesis_index=hypothesis_index)
                                        for hypothesis_index, hypothesis in enumerate(hypotheses)]
                    observation_nodes.append(self._node('observation', index, observation, grid.detection_index,
                                                        hypothesis_nodes, observation_index=observation_index))
                children.append(self._node('observations', index, observations, grid.detection_index, observation_nodes))
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

    def _node(self, kind, index, data, detection_index, children=(), *, observation_index=None, hypothesis_index=None):
        identity = ResultId(self._token, kind, index, observation_index, hypothesis_index)
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

    def observation_id(self, geometry_index, observation_index):
        identity = ResultId(self._token, 'observation', geometry_index, observation_index)
        return identity if identity in self._nodes else None

    def observation(self, identity):
        if self.node(identity) is None or identity.kind not in ('observation', 'hypothesis'):
            return None
        return self.resolve(self.observation_id(identity.index, identity.observation_index))

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

    def select_tile(self, index):
        """Select a tile in the current geometry, rejecting stale or invalid indices."""
        if self.snapshot is None or self.snapshot.node(self.selected) is None:
            return False
        grid = self.grid()
        if grid is None or grid.diagnostics.get('mode') != 'tiled':
            return False
        geometry_index = next((i for i, candidate in enumerate(self.snapshot.result.grids) if candidate is grid), None)
        identity = self.snapshot.observation_id(geometry_index, index)
        return self.select(identity) if identity is not None else False


def hypothesis_validation(observation, index):
    """Only the decoder's first hypothesis is evaluated by the pipeline."""
    if index != 0:
        return 'Not evaluated'
    checked = getattr(observation, 'local_validation', None)
    decoded = getattr(observation, 'decoding', None)
    hypotheses = getattr(decoded, 'hypotheses', ()) if decoded is not None else ()
    if checked is None or not hypotheses or getattr(observation, 'hypothesis', None) is not hypotheses[0]:
        return 'Unavailable'
    accepted = getattr(checked, 'local_accepted', None)
    return 'Evaluated accepted' if accepted is True else 'Evaluated rejected' if accepted is False else 'Unavailable'
