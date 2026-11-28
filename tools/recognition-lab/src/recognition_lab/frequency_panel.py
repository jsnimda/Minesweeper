"""Compact native viewer for retained decoder frequency evidence."""
import math

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QAction, QColor, QKeySequence, QPainter, QPalette, QPen
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QHBoxLayout,
                               QHeaderView, QLabel, QScrollArea, QTableWidget, QTableWidgetItem,
                               QToolTip, QVBoxLayout, QWidget)

from .frequency_diagnostics import (PROFILE_CHOICES, candidate_rows, profile, profile_markers,
                                    resolve_frequency, sample_value)


def _plot_number(value):
    try:
        return float(value) if np.isrealobj(value) else float('nan')
    except (TypeError, ValueError, OverflowError):
        return float('nan')


class EvidencePlot(QWidget):
    """Discrete points only; plotting state is prepared when evidence changes."""
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setMouseTracking(True)
        self.points = (); self.segments = (); self.markers = (); self.values = None
        self.x_range = (0., 1.); self.y_range = (0., 1.)
        self.x_label = ''; self.y_label = ''; self.stride = 1

    def set_data(self, x, y, x_label, y_label, *, selected=None, values=None, markers=()):
        self.x_label, self.y_label = x_label, y_label
        self.values = values
        self.markers = markers
        n = len(y) if x is None else min(len(x), len(y))
        self.stride = max(1, math.ceil(n / 4096))
        points = []
        segments = []
        previous = None
        for i in range(0, n, self.stride):
            try:
                xv, yv = (i + .5 if x is None else float(x[i])), float(y[i])
            except (TypeError, ValueError, OverflowError):
                previous = None; continue
            if math.isfinite(xv) and math.isfinite(yv):
                points.append((xv, yv, bool(selected[i]) if selected is not None else False))
                if values is not None and self.stride == 1 and previous is not None:
                    segments.append((previous, (xv, yv)))
                previous = (xv, yv)
            else:
                previous = None
        self.points = tuple(points)
        self.segments = tuple(segments)
        if self.points:
            # Axis bounds reflect all finite samples, including unplotted points.
            ya = np.asarray(y) if values is not None else np.asarray([_plot_number(v) for v in y])
            xa = None if x is None else np.asarray([_plot_number(v) for v in x])
            finite = np.isfinite(ya[:n]) if xa is None else np.isfinite(xa[:n]) & np.isfinite(ya[:n])
            if finite.any():
                xmin, xmax = ((.5, max(.5, n-.5)) if xa is None else
                              (float(xa[:n][finite].min()), float(xa[:n][finite].max())))
                ymin, ymax = float(ya[:n][finite].min()), float(ya[:n][finite].max())
                if xmin == xmax: xmin, xmax = xmin-.5, xmax+.5
                if ymin == ymax:
                    padding = max(abs(ymin)*.05, .5)
                    ymin, ymax = ymin-padding, ymax+padding
                self.x_range, self.y_range = (xmin, xmax), (ymin, ymax)
        else:
            self.x_range = self.y_range = (0., 1.)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        foreground = self.palette().color(QPalette.ColorRole.Text)
        background = self.palette().color(QPalette.ColorRole.Base)
        highlight = self.palette().color(QPalette.ColorRole.Highlight)
        p.fillRect(self.rect(), background)
        left, top, right, bottom = 65., 16., max(66., self.width()-20.), max(17., self.height()-42.)
        p.setPen(QPen(foreground, 1)); p.drawRect(int(left), int(top), int(right-left), int(bottom-top))
        xmin, xmax = self.x_range; ymin, ymax = self.y_range
        def px(value): return left + (value-xmin)/(xmax-xmin)*(right-left)
        def py(value): return bottom - (value-ymin)/(ymax-ymin)*(bottom-top)
        for kind, value in self.markers:
            if xmin <= value <= xmax:
                color = {'lattice': QColor('#888888'), 'equivalent boundary': QColor('#f09a22'),
                         'visible crossing': QColor('#bb48cc')}[kind]
                pen = QPen(color, 1); pen.setStyle(Qt.PenStyle.DotLine if kind == 'lattice' else Qt.PenStyle.DashLine)
                p.setPen(pen); p.drawLine(QPointF(px(value), top), QPointF(px(value), bottom))
        p.setPen(QPen(highlight, 1))
        for first, second in self.segments:
            p.drawLine(QPointF(px(first[0]), py(first[1])), QPointF(px(second[0]), py(second[1])))
        for xv, yv, chosen in self.points:
            radius = 4. if chosen else 2.
            p.setBrush(highlight if chosen else Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(px(xv), py(yv)), radius, radius)
        p.setPen(QPen(foreground, 1))
        p.drawText(2, int(top+10), f'{ymax:.4g}')
        p.drawText(2, int(bottom), f'{ymin:.4g}')
        p.drawText(2, int((top+bottom)/2), f'{(ymin+ymax)/2:.4g}')
        p.drawText(int(left), int(bottom+16), f'{xmin:.4g}')
        p.drawText(int(max(left, right-62)), int(bottom+16), f'{xmax:.4g}')
        p.drawText(int((left+right)/2), int(bottom+16), f'{(xmin+xmax)/2:.4g}')
        p.drawText(int(left), self.height()-4, f'{self.x_label} · {self.y_label}' +
                   (f' · every {self.stride}th point' if self.stride > 1 else ''))
        if not self.points:
            p.drawText(int(left+8), int(top+20), 'No finite displayed samples')

    def mouseMoveEvent(self, event):
        if self.values is None or not len(self.values): return
        left, right = 65., max(66., self.width()-20.)
        fraction = (event.position().x()-left)/(right-left)
        x = self.x_range[0] + fraction*(self.x_range[1]-self.x_range[0])
        sample = sample_value(self.values, x)
        if sample is not None:
            index, value = sample
            QToolTip.showText(event.globalPosition().toPoint(),
                              f'Pixel center {index+.5!r} px\n{self.y_label}: {value!r}', self)


class FrequencyPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._snapshot = self._observation = self._hypothesis = None
        self._identity = None
        self._fallback = None
        self.evidence = None
        layout = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        body = QWidget(); content = QVBoxLayout(body); scroll.setWidget(body)
        self.summary = QLabel('No observation selected'); self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        content.addWidget(self.summary)
        controls = QHBoxLayout()
        self.axis = QComboBox(); self.axis.addItems(['X', 'Y'])
        self.profile_choice = QComboBox(); self.profile_choice.addItems(PROFILE_CHOICES)
        self.markers = QCheckBox('Hypothesis markers'); self.markers.setChecked(True)
        controls.addWidget(QLabel('Axis')); controls.addWidget(self.axis)
        controls.addWidget(QLabel('Spatial evidence')); controls.addWidget(self.profile_choice)
        controls.addWidget(self.markers); controls.addStretch()
        content.addLayout(controls)
        self.hypothesis = QLabel(); self.hypothesis.setWordWrap(True)
        self.hypothesis.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        content.addWidget(self.hypothesis)
        explanation = QLabel('Candidates retain decoder order (FFT proposal order before refinement). Frequency is 1 / pitch. '
                             'Filled plot point = selected hypothesis evidence. Coherence and magnitude are not calibrated probabilities; '
                             'no per-candidate score or spectrum was retained.')
        explanation.setWordWrap(True); content.addWidget(explanation)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['Rank', 'Pitch (px)', 'Frequency (cycles/px)', 'Origin mod pitch (px)',
                                               'Coherence', 'Magnitude', 'Hypothesis evidence'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(110); self.table.setMaximumHeight(190)
        copy = QAction('Copy candidate value', self.table)
        copy.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy.triggered.connect(self.copy_cell); self.table.addAction(copy)
        content.addWidget(self.table)
        self.candidate_plot = EvidencePlot(); content.addWidget(self.candidate_plot)
        self.profile_plot = EvidencePlot(); content.addWidget(self.profile_plot)
        self.marker_legend = QLabel('Markers: dotted lattice · dashed orange equivalent boundary · dashed purple visible crossing. '
                                   'No marker is a confirmed physical boundary. Signal is the occupancy/magnitude-weighted complex projection; '
                                   'support is the sum of weights; occupancy profile is the mean over active orthogonal positions. '
                                   'Samples are discrete; hover for values.')
        self.marker_legend.setWordWrap(True); content.addWidget(self.marker_legend)
        self.axis.currentIndexChanged.connect(self.refresh)
        self.profile_choice.currentIndexChanged.connect(self.refresh)
        self.markers.toggled.connect(self.refresh)

    def copy_cell(self):
        item = self.table.currentItem()
        if item is not None: QApplication.clipboard().setText(item.text())

    def set_selection(self, selection, tile):
        evidence = resolve_frequency(selection, tile)
        snapshot = selection.snapshot
        obs = evidence.observation if evidence else None
        hypothesis = evidence.hypothesis if evidence else None
        fallback = evidence.fallback if evidence else None
        if (snapshot is self._snapshot and obs is self._observation and hypothesis is self._hypothesis
                and fallback == self._fallback and selection.selected == self._identity):
            return
        self._snapshot, self._observation, self._hypothesis, self._fallback = snapshot, obs, hypothesis, fallback
        self._identity = selection.selected
        self.evidence = evidence
        self.refresh()

    def refresh(self, *args):
        evidence = self.evidence
        axis = self.axis.currentText().lower()
        decoded = evidence.decoder if evidence else None
        if evidence is None:
            self.summary.setText('No observation selected')
        else:
            reasons = getattr(decoded, 'reasons', None)
            reason_text = '; '.join(reasons) if reasons else '(none)' if reasons is not None else 'Unavailable'
            fallback = ' · selected tile fallback' if evidence.fallback else ''
            detection = 'unassociated geometry' if evidence.detection_index is None else f'detection index {evidence.detection_index}'
            box_space = ('resized crop pixels' if evidence.mode == 'whole' else
                         'native crop pixels' if evidence.mode == 'tiled' else 'coordinate space unavailable')
            self.summary.setText(f'{detection} · {evidence.mode or "Unavailable"} · observation {evidence.observation_index+1} (one-based){fallback} · '
                                 f'observation box {getattr(evidence.observation, "box", None)!r} ({box_space}) · '
                                 f'decoder: {getattr(decoded, "status", None) or "Unavailable"} · reasons: {reason_text}')
        h = evidence.hypothesis if evidence else None
        if h is None:
            self.hypothesis.setText('Selected hypothesis evidence: Unavailable')
        else:
            selected = getattr(h, f'{axis}_evidence', None)
            extent = getattr(h, axis)
            self.hypothesis.setText(f'{axis.upper()} selected evidence: pitch {getattr(selected, "pitch", None)!r} px, '
                                    f'origin modulo pitch {getattr(selected, "origin_mod_pitch", None)!r} px, '
                                    f'coherence {getattr(selected, "coherence", None)!r}, magnitude {getattr(selected, "magnitude", None)!r} · '
                                    f'joint score {h.score!r} (mean X/Y coherence + square prior {h.prior_contribution!r}) · '
                                    f'local validation: {evidence.validation or "Unavailable"}. '
                                    f'Crossings start/end: {extent.visible_start!r}, {extent.visible_end!r}; '
                                    f'censored start/end: {extent.start_censored!r}, {extent.end_censored!r}. '
                                    'Not calibrated probabilities.')
        rows = candidate_rows(evidence, axis)
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            fields = (row[0]+1, *row[1:6], 'Selected' if row[6] else '')
            for column, value in enumerate(fields):
                text = 'Unavailable' if value is None else repr(value) if not isinstance(value, str) else value
                item = QTableWidgetItem(text); item.setToolTip(text)
                self.table.setItem(row_index, column, item)
        self.candidate_plot.set_data([row[1] if row[2] is not None else None for row in rows], [row[4] for row in rows],
                                     'Pitch (observation-local px)', 'Candidate coherence (discrete)',
                                     selected=[row[6] for row in rows])
        values = profile(evidence, axis, self.profile_choice.currentText())
        if values is None:
            self.profile_plot.set_data([], [], 'Observation-local pixel centers (px)', self.profile_choice.currentText())
        else:
            markers = profile_markers(evidence, axis, len(values)) if self.markers.isChecked() else ()
            self.profile_plot.set_data(None, values,
                                       'Observation-local pixel centers (px)', self.profile_choice.currentText(),
                                       values=values, markers=markers)
