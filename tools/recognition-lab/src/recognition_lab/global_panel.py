"""Read-only global consensus and observation participation panel."""
from numbers import Number

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QHeaderView, QLabel, QScrollArea, QTableView,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .global_diagnostics import observation_row, resolve_global
from .global_validation_diagnostics import GLOBAL_CELL_METRICS
from .local_validation_panel import CellTable


def display(value):
    if value is None: return 'Unavailable'
    if hasattr(value, 'item'): value = value.item()
    return repr(value) if isinstance(value, Number) else str(value)


class ObservationTable(QAbstractTableModel):
    HEADERS = ('Index', 'Decoder', 'Local validation', 'Consensus eligibility', 'Consensus membership',
               'Local pitch X', 'Local pitch Y', 'Local periodic origin X', 'Local periodic origin Y',
               'Individual residual', 'Individual fit weight')

    def __init__(self):
        super().__init__(); self.evidence = None

    def set_evidence(self, evidence):
        self.beginResetModel(); self.evidence = evidence; self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or self.evidence is None or self.evidence.observations is None else len(self.evidence.observations)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole): return None
        row = observation_row(self.evidence, index.row())
        return display(row[index.column()]) if row is not None else 'Unavailable'

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None


class GlobalPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._snapshot = self._grid = None
        self.evidence = None
        layout = QVBoxLayout(self); scroll = QScrollArea(); scroll.setWidgetResizable(True); layout.addWidget(scroll)
        body = QWidget(); content = QVBoxLayout(body); scroll.setWidget(body)
        self.summary = QLabel('No geometry selected'); self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); content.addWidget(self.summary)
        note = QLabel('A fitted periodic lattice does not establish a complete physical board. '
                      'Consensus eligibility and retained membership are separate; excluded observations include locally ineligible ones. '
                      'Fusion weight sums are not observation counts or probabilities.')
        note.setWordWrap(True); content.addWidget(note)
        self.tree = QTreeWidget(); self.tree.setColumnCount(2); self.tree.setHeaderLabels(['Global evidence', 'Retained value'])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setMinimumHeight(200); content.addWidget(self.tree)
        copy = QAction('Copy global value', self.tree); copy.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy.triggered.connect(self.copy_tree); self.tree.addAction(copy)
        label = QLabel('Observations in original order; local pitches and origins are in observation-local pixels. '
                       'Indices are zero-based. Individual residuals and fitting weights were not retained.')
        label.setWordWrap(True); content.addWidget(label)
        self.table = QTableView(); self.table_model = ObservationTable(); self.table.setModel(self.table_model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(145)
        self.table.setMinimumHeight(160); content.addWidget(self.table)
        copy_table = QAction('Copy observation value', self.table); copy_table.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy_table.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy_table.triggered.connect(self.copy_table); self.table.addAction(copy_table)
        cell_label = QLabel('Retained global cell-array values (candidate extent indexing, zero-based). Masks and individual row/column arrays were not retained.')
        cell_label.setWordWrap(True); content.addWidget(cell_label)
        self.cell_choice = QComboBox(); self.cell_choice.addItems(list(GLOBAL_CELL_METRICS)); content.addWidget(self.cell_choice)
        self.cell_table = QTableView(); self.cell_model = CellTable(); self.cell_table.setModel(self.cell_model)
        self.cell_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.cell_table.setMinimumHeight(140); content.addWidget(self.cell_table)
        copy_cell = QAction('Copy global cell value', self.cell_table); copy_cell.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy_cell.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy_cell.triggered.connect(self.copy_cell); self.cell_table.addAction(copy_cell)
        self.cell_choice.currentIndexChanged.connect(self.refresh_cells)

    def copy_tree(self):
        item = self.tree.currentItem()
        if item is not None: QApplication.clipboard().setText(item.text(1))

    def copy_table(self):
        index = self.table.currentIndex()
        if index.isValid(): QApplication.clipboard().setText(str(index.data()))

    def copy_cell(self):
        index = self.cell_table.currentIndex()
        if index.isValid(): QApplication.clipboard().setText(str(index.data()))

    def refresh_cells(self, *args):
        result = self.evidence.result if self.evidence else None
        name = GLOBAL_CELL_METRICS[self.cell_choice.currentText()]
        self.cell_model.set_array(getattr(result, name, None))

    def set_selection(self, selection):
        evidence = resolve_global(selection)
        grid = evidence.grid if evidence else None
        if self._snapshot is selection.snapshot and self._grid is grid: return
        self._snapshot, self._grid, self.evidence = selection.snapshot, grid, evidence
        self.refresh()

    def refresh(self):
        self.tree.clear(); self.table_model.set_evidence(self.evidence); self.refresh_cells()
        evidence = self.evidence
        if evidence is None:
            self.summary.setText('No geometry selected'); return
        result, observations = evidence.result, evidence.observations
        detection = f'detection index {evidence.detection_index}' if evidence.detection_index is not None else 'unassociated geometry'
        n = len(observations) if observations is not None else None
        accepted = sum(getattr(o, 'local_valid', None) is True for o in observations) if observations is not None else None
        local_accepted = sum(getattr(getattr(o, 'local_validation', None), 'local_accepted', None) is True for o in observations) if observations is not None else None
        local_rejected = sum(getattr(getattr(o, 'local_validation', None), 'local_accepted', None) is False for o in observations) if observations is not None else None
        geometry = getattr(result, 'geometry', None)
        dimensions = f'{geometry.rows} × {geometry.columns}' if geometry is not None else 'Unavailable'
        reasons = getattr(result, 'reasons', None)
        reason_text = '; '.join(reasons) if reasons else '(none recorded)' if reasons is not None else 'Unavailable'
        self.summary.setText(f'{detection} · {evidence.mode or "Unavailable"} · observations: {display(n)} · '
                             f'local accepted/rejected: {display(local_accepted)}/{display(local_rejected)} · '
                             f'known locally eligible: {display(accepted)}\nCandidate rows × columns: {dimensions} · '
                             f'complete-board acceptance: {display(getattr(result, "accepted", None))}\nReasons: {reason_text}')
        def section(title, pairs):
            root = QTreeWidgetItem(self.tree, [title, ''])
            for name, value in pairs:
                text = display(value); item = QTreeWidgetItem(root, [name, text]); item.setToolTip(1, text)
            root.setExpanded(title in ('Global lattice (not board extent)', 'Complete-board summary', 'Candidate and accepted geometry'))
        lattice = getattr(result, 'lattice', None)
        space = ('Resized crop pixels (padding removed)' if evidence.mode == 'whole' else
                 'Native crop pixels' if evidence.mode == 'tiled' else 'Unavailable')
        section('Global lattice (not board extent)', [('Coordinate space', space), ('Available', lattice is not None),
                ('Pitch X (px)', getattr(lattice, 'pitch_x', None)), ('Pitch Y (px)', getattr(lattice, 'pitch_y', None)),
                ('Periodic origin X (px)', getattr(lattice, 'origin_x', None)), ('Periodic origin Y (px)', getattr(lattice, 'origin_y', None)),
                ('Max consensus residual X (px)', getattr(lattice, 'residual_x', None)),
                ('Max consensus residual Y (px)', getattr(lattice, 'residual_y', None)),
                ('Ambiguous', getattr(lattice, 'ambiguous', None)),
                ('Inlier indices (zero-based)', getattr(lattice, 'inliers', None)),
                ('Excluded indices (zero-based)', getattr(lattice, 'rejected', None)),
                ('Inlier count', len(lattice.inliers) if getattr(lattice, 'inliers', None) is not None else None),
                ('Excluded count', len(lattice.rejected) if getattr(lattice, 'rejected', None) is not None else None)])
        reasons = getattr(result, 'reasons', None)
        section('Complete-board summary', [('Accepted', getattr(result, 'accepted', None)),
                                           ('Recorded reasons', '; '.join(reasons) or '(none)' if reasons is not None else None)])
        geometry = getattr(result, 'geometry', None)
        xs, ys = getattr(geometry, 'x_edges', None), getattr(geometry, 'y_edges', None)
        section('Candidate and accepted geometry', [('Coordinate space', space),
                ('Candidate extent available', geometry is not None if result is not None else None),
                ('Candidate rows', getattr(geometry, 'rows', None)), ('Candidate columns', getattr(geometry, 'columns', None)),
                ('Candidate left (px)', xs[0] if xs is not None and len(xs) else None),
                ('Candidate right (px)', xs[-1] if xs is not None and len(xs) else None),
                ('Candidate top (px)', ys[0] if ys is not None and len(ys) else None),
                ('Candidate bottom (px)', ys[-1] if ys is not None and len(ys) else None),
                ('Accepted complete-board geometry available', bool(geometry is not None and getattr(result, 'accepted', False)) if result is not None else None),
                ('Validation thresholds / configuration', None)])
        measurements = getattr(result, 'measurements', {}) or {}
        section('Global validation retained measurements', list(measurements.items()) or [('Measurements', None)])
        cell_occ, cell_cov = getattr(result, 'cell_occupancy', None), getattr(result, 'cell_coverage', None)
        section('Global validation arrays', [('Cell occupancy shape (rows, columns)', getattr(cell_occ, 'shape', None)),
                ('Cell coverage shape (rows, columns)', getattr(cell_cov, 'shape', None)),
                ('Observed cell mask', None), ('Supported cell mask', None), ('Phase coverage mask', None),
                ('Per-row support array', None), ('Per-column support array', None)])
        boundaries = getattr(result, 'boundaries', None)
        for name in ('left', 'right', 'top', 'bottom'):
            side = boundaries.get(name) if isinstance(boundaries, dict) else None
            if not isinstance(side, dict):
                section(f'{name.title()} termination', [('Evidence', None)]); continue
            keys = ('count', 'estimate', 'spread_cells', 'samples', 'confidence', 'index', 'snapped', 'residual_cells',
                    'inside_occupancy', 'outside_occupancy', 'inside_coverage', 'outside_coverage', 'exterior_supported_cells')
            pairs = [('confidence (retained evidence weights; not calibrated)' if key == 'confidence' else key,
                      side.get(key)) for key in keys]
            side_reasons = tuple(reason for reason in reasons or () if reason.startswith(name + ' '))
            pairs += [('Recorded side-related reasons', '; '.join(side_reasons) or '(none recorded)'),
                      ('Sample observation identities', None), ('Exterior supported-cell positions', None)]
            section(f'{name.title()} termination (global {space})', pairs)
            for index, candidate in enumerate(side.get('candidates') or ()):
                if isinstance(candidate, dict):
                    section(f'{name.title()} retained boundary candidate {index+1}', list(candidate.items()))
        occ, weights = getattr(result, 'occupancy', None), getattr(result, 'weights', None)
        section('Fusion artifacts', [('Coordinate space', space),
                ('Occupancy shape (height, width)', getattr(occ, 'shape', None)),
                ('Occupancy dtype', str(occ.dtype) if hasattr(occ, 'dtype') else None),
                ('Weight sum shape (height, width)', getattr(weights, 'shape', None)),
                ('Weight dtype', str(weights.dtype) if hasattr(weights, 'dtype') else None),
                ('Coverage', 'Derived from finite fusion weight > 0' if weights is not None else None),
                ('Observation count map', None)])
