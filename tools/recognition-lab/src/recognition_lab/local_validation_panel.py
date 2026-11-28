"""Read-only local validation details for the evaluated hypothesis."""
from numbers import Number

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QHeaderView, QLabel, QScrollArea, QTableView,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .local_validation_diagnostics import METRICS, resolve_validation, side_rows


def display(value):
    if value is None: return 'Unavailable'
    if hasattr(value, 'item'): value = value.item()
    return repr(value) if isinstance(value, Number) else str(value)


class SupportTable(QAbstractTableModel):
    """Virtual rows avoid copying potentially long retained support arrays."""
    def __init__(self):
        super().__init__()
        self.rows = self.columns = None

    def set_arrays(self, rows, columns):
        self.beginResetModel()
        self.rows = rows if isinstance(rows, np.ndarray) and rows.ndim == 1 else None
        self.columns = columns if isinstance(columns, np.ndarray) and columns.ndim == 1 else None
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else max(len(self.rows) if self.rows is not None else 0,
                                               len(self.columns) if self.columns is not None else 0)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 3

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole): return None
        i, c = index.row(), index.column()
        if c == 0: return str(i)
        array = self.rows if c == 1 else self.columns
        return display(array[i]) if array is not None and i < len(array) else 'Unavailable'

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return ('Local index (zero-based)', 'Retained row support', 'Retained column support')[section]
        return None


class CellTable(QAbstractTableModel):
    """Expose retained array values even when spatial mapping is unavailable."""
    def __init__(self):
        super().__init__(); self.values = None

    def set_array(self, values):
        self.beginResetModel()
        self.values = values if isinstance(values, np.ndarray) and values.ndim == 2 else None
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or self.values is None else self.values.shape[0]*self.values.shape[1]

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 3

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self.values is None or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole): return None
        row, col = divmod(index.row(), self.values.shape[1])
        return str(row) if index.column() == 0 else str(col) if index.column() == 1 else display(self.values[row, col])

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return ('Array row', 'Array column', 'Retained value')[section]
        return None


class LocalValidationPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._snapshot = self._identity = self._observation = None
        self._fallback = None
        self.evidence = None
        layout = QVBoxLayout(self); scroll = QScrollArea(); scroll.setWidgetResizable(True); layout.addWidget(scroll)
        body = QWidget(); content = QVBoxLayout(body); scroll.setWidget(body)
        self.summary = QLabel('No observation selected'); self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        content.addWidget(self.summary)
        note = QLabel('Local acceptance and the validator’s complete-extent check are separate from global board validation. '
                      'Threshold settings and unsupported-block locations were not retained. Cell values are sampled at lattice cell centers.')
        note.setWordWrap(True); content.addWidget(note)
        self.tree = QTreeWidget(); self.tree.setColumnCount(2); self.tree.setHeaderLabels(['Evidence', 'Retained value'])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setMinimumHeight(200); content.addWidget(self.tree)
        copy = QAction('Copy validation value', self.tree); copy.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy.triggered.connect(self.copy_tree); self.tree.addAction(copy)
        label = QLabel('Retained row and column support (fractions of observable cells; zero when no centers were observable)')
        label.setWordWrap(True); content.addWidget(label)
        self.table = QTableView(); self.table_model = SupportTable(); self.table.setModel(self.table_model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(160)
        self.table.setMinimumHeight(140); content.addWidget(self.table)
        copy_table = QAction('Copy support value', self.table); copy_table.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy_table.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy_table.triggered.connect(self.copy_table); self.table.addAction(copy_table)
        label = QLabel('Retained cell-array values (zero-based indices; mapping requires a matching hypothesis shape)')
        label.setWordWrap(True); content.addWidget(label)
        self.cell_choice = QComboBox(); self.cell_choice.addItems(list(METRICS))
        content.addWidget(self.cell_choice)
        self.cell_table = QTableView(); self.cell_model = CellTable(); self.cell_table.setModel(self.cell_model)
        self.cell_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.cell_table.setMinimumHeight(140); content.addWidget(self.cell_table)
        copy_cell = QAction('Copy cell value', self.cell_table); copy_cell.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
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
        validation = self.evidence.validation if self.evidence else None
        name = METRICS[self.cell_choice.currentText()][0]
        self.cell_model.set_array(getattr(validation, name, None))

    def set_selection(self, selection, tile):
        evidence = resolve_validation(selection, tile)
        obs = evidence.frequency.observation if evidence else None
        fallback = evidence.frequency.fallback if evidence else None
        if (self._snapshot is selection.snapshot and self._identity == selection.selected
                and self._observation is obs and self._fallback == fallback): return
        self._snapshot, self._identity, self._observation, self._fallback = selection.snapshot, selection.selected, obs, fallback
        self.evidence = evidence
        self.refresh()

    def refresh(self):
        self.tree.clear()
        evidence = self.evidence
        validation = evidence.validation if evidence else None
        if evidence is None:
            self.summary.setText('No observation selected')
        else:
            f = evidence.frequency
            hypothesis_index = (self._identity.hypothesis_index if self._identity.kind == 'hypothesis' else 0) if evidence.local else None
            detection = 'unassociated geometry' if f.detection_index is None else f'detection index {f.detection_index}'
            self.summary.setText(f'{detection} · {f.mode or "Unavailable"} · observation {f.observation_index+1} (one-based) · '
                                 f'hypothesis {hypothesis_index+1 if hypothesis_index is not None else "Unavailable"} (one-based) · '
                                 f'{"selected tile fallback · " if f.fallback else ""}'
                                 f'decoder: {getattr(f.decoder, "status", None) or "Unavailable"} · local: {evidence.status}\n'
                                 f'Local reasons: {"; ".join(validation.local_reasons) or "(none recorded)" if validation is not None else "Unavailable"}')
        self.table_model.set_arrays(getattr(validation, 'row_support', None), getattr(validation, 'column_support', None))
        self.refresh_cells()
        if validation is None:
            QTreeWidgetItem(self.tree, ['Validation evidence', evidence.status if evidence else 'Unavailable'])
            return
        def section(name, pairs):
            parent = QTreeWidgetItem(self.tree, [name, ''])
            for key, value in pairs:
                text = display(value)
                child = QTreeWidgetItem(parent, [str(key), text]); child.setToolTip(1, text)
            parent.setExpanded(name == 'Validation decisions')
        section('Validation decisions', [('Local acceptance', getattr(validation, 'local_accepted', None)),
                ('Local reasons', '; '.join(validation.local_reasons) or '(none)'),
                ('Local validator complete-extent acceptance', getattr(validation, 'accepted', None)),
                ('Complete-extent reasons', '; '.join(validation.reasons) or '(none)')])
        measurements = getattr(validation, 'measurements', {})
        section('Retained scalar measurements', [(key, value) for key, value in measurements.items()
                                                   if value is None or isinstance(value, (str, Number, np.number))])
        section('Retained cell arrays', [(name, getattr(getattr(validation, name, None), 'shape', None)) for name in
                ('cell_occupancy', 'cell_magnitude_x', 'cell_magnitude_y', 'cell_alignment_x', 'cell_alignment_y', 'supported', 'observable')])
        for name, boundary, crossing, censored, stored in side_rows(evidence):
            section(name.title() + ' side (observation-local)', [('Equivalent boundary (px)', boundary),
                    ('Visible occupancy crossing (px)', crossing), ('Decoder censored', censored)] +
                    [('Validator censored (includes unobservable exterior)', stored.get('censored'))] +
                    [(key, stored.get(key)) for key in ('inner_support', 'exterior_occupancy',
                                                         'visible_offset_cells', 'exterior_phase_support')])
