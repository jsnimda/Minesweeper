from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np
import skia
from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal, QTimer, Slot
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QSplitter, QPushButton, QFileDialog, QComboBox, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QTextEdit, QListWidget, QMessageBox, QInputDialog, QScrollArea, QToolBar)
from .capture import Frame, capture_frame, monitors
from .pipeline import Pipeline, Settings, ROOT
from .rendering import draw
from .results import RecognitionResult


def skia_image(image):
    return skia.Image.fromarray(np.asarray(image.convert('RGBA')).copy())


def surface_pixmap(surface):
    snapshot = surface.makeImageSnapshot()
    pixels = snapshot.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType, alphaType=skia.AlphaType.kPremul_AlphaType)
    h, w = pixels.shape[:2]
    return QPixmap.fromImage(QImage(pixels.data, w, h, pixels.strides[0], QImage.Format.Format_RGBA8888_Premultiplied).copy())


class Canvas(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.scale = 1.0
        self.offset = [0., 0.]
        self.drag = None
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)

    def fit(self):
        if self.main.frame:
            w, h = self.main.frame.image.size
            self.scale = min(self.width()/w, self.height()/h)
            self.offset = [(self.width()-w*self.scale)/2, (self.height()-h*self.scale)/2]
            self.update()

    def paintEvent(self, event):
        if self.width() <= 0 or self.height() <= 0: return
        dpr = self.devicePixelRatioF()
        surface = skia.Surface(max(1, round(self.width()*dpr)), max(1, round(self.height()*dpr)))
        canvas = surface.getCanvas()
        canvas.clear(0xff272727)
        canvas.scale(dpr, dpr)
        if self.main.frame:
            canvas.translate(*self.offset)
            canvas.scale(self.scale, self.scale)
            canvas.drawImage(self.main.image_skia, 0, 0)
            grid = self.main.selected_grid()
            channel = self.main.heatmap.currentIndex()-1
            heat = None
            tile_offset = (0, 0)
            if grid is not None and 0 <= channel < 3 and not grid.phase_data:
                heat = grid.heatmaps[channel]
            elif grid is not None and grid.phase_data:
                mode = self.main.heatmap.currentText()
                phase = grid.phase_data
                if mode == 'Phase occupancy': heat = phase['global'].occupancy
                elif mode == 'Phase coverage': heat = (phase['global'].weights > 0).astype(float)
                elif mode in ('Tile horizontal phase', 'Tile vertical phase'):
                    observations = phase['observations']
                    o = observations[min(self.main.tile_inspect.value(), len(observations)-1)]
                    if o.fields is not None:
                        axis = 0 if mode == 'Tile horizontal phase' else 2
                        # Map phase angle to a scalar hue-like ramp, masked by occupancy.
                        heat = (np.arctan2(o.fields[axis+1], o.fields[axis]) / (2*np.pi) + .5) * o.fields[4]
                        tile_offset = o.box[:2]
            if heat is not None:
                valid = np.isfinite(heat)
                heat = np.nan_to_num(np.clip(heat*255, 0, 255)).astype(np.uint8)
                rgba = np.zeros((*heat.shape, 4), dtype=np.uint8)
                rgba[..., 0] = 255
                rgba[..., 3] = np.where(valid, heat // 2, 0).astype(np.uint8)
                hm = skia.Image.fromarray(rgba)
                tr = grid.heatmap_transform
                left, top = grid.crop.box[:2]
                rect = skia.Rect.MakeXYWH(left+(tile_offset[0]-tr.offset[0])/tr.scale_x, top+(tile_offset[1]-tr.offset[1])/tr.scale_y, heat.shape[1]/tr.scale_x, heat.shape[0]/tr.scale_y)
                canvas.save()
                canvas.clipRect(skia.Rect.MakeLTRB(*grid.crop.box))
                canvas.drawImageRect(hm, rect)
                canvas.restore()
            draw(canvas, self.main.result, self.main.layers(), self.main.selected_detection())
        painter = QPainter(self)
        pixmap = surface_pixmap(surface); pixmap.setDevicePixelRatio(dpr)
        painter.drawPixmap(0, 0, pixmap)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        x, y = event.position().x(), event.position().y()
        self.offset = [x-(x-self.offset[0])*factor, y-(y-self.offset[1])*factor]
        self.scale *= factor
        self.update()

    def mousePressEvent(self, event):
        self.drag = event.position()

    def mouseReleaseEvent(self, event):
        self.drag = None

    def mouseMoveEvent(self, event):
        p = event.position()
        if self.drag is not None:
            self.offset[0] += p.x()-self.drag.x(); self.offset[1] += p.y()-self.drag.y()
            self.drag = p; self.update()
        if self.main.frame:
            self.main.statusBar().showMessage(f'Image: {(p.x()-self.offset[0])/self.scale:.1f}, {(p.y()-self.offset[1])/self.scale:.1f}')


class Overlay(QWidget):
    def __init__(self, main):
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowTransparentForInput | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.main = main
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

    def show_frame(self):
        mapping = self.main.frame.mapping if self.main.frame else None
        if mapping is None:
            raise ValueError('This image has no verified desktop mapping')
        self.setGeometry(round(mapping.origin_x), round(mapping.origin_y), round(mapping.logical_width), round(mapping.logical_height))
        self.setWindowOpacity(self.main.opacity.value())
        self.show(); self.update()

    def paintEvent(self, event):
        mapping = self.main.frame.mapping
        if not mapping or not self.width() or not self.height(): return
        dpr = self.devicePixelRatioF()
        surface = skia.Surface(max(1, round(self.width()*dpr)), max(1, round(self.height()*dpr)))
        canvas = surface.getCanvas(); canvas.clear(skia.ColorTRANSPARENT)
        canvas.scale(dpr*self.width()/mapping.pixel_width, dpr*self.height()/mapping.pixel_height)
        draw(canvas, self.main.result, self.main.overlay_layers())
        pixmap = surface_pixmap(surface); pixmap.setDevicePixelRatio(dpr)
        painter = QPainter(self); painter.drawPixmap(0, 0, pixmap)


class Worker(QThread):
    succeeded = Signal(object, int)
    failed = Signal(str)
    def __init__(self, fn, generation):
        super().__init__(); self.fn = fn; self.generation = generation
    def run(self):
        try: self.succeeded.emit(self.fn(), self.generation)
        except Exception as exc: self.failed.emit(f'{type(exc).__name__}: {exc}')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Minesweeper Recognition Lab')
        self.resize(1450, 850)
        self.pipeline = Pipeline(); self.frame = None; self.result = None; self.image_skia = None
        self.generation = 0; self.worker = None; self.folder_files = []; self.capture_in_progress = False
        self.conflicting_controls = []
        self.overlay = Overlay(self)
        bar = QToolBar(); self.addToolBar(bar)
        for title, fn in [('Open image', self.open_image), ('Open folder', self.open_folder), ('Capture screen', lambda: self.capture(False)), ('Capture region', lambda: self.capture(True)), ('Save screenshot', self.save_image), ('Run YOLO', lambda: self.run('yolo')), ('Run geometry', lambda: self.run('geometry')), ('Run all', lambda: self.run('all')), ('Overlay', self.toggle_overlay), ('Fit', lambda: self.canvas.fit()), ('100%', self.actual_size)]:
            action = bar.addAction(title); action.triggered.connect(fn)
            if title in ('Run YOLO', 'Run geometry', 'Run all'): self.conflicting_controls.append(action)
        splitter = QSplitter(); self.setCentralWidget(splitter)
        left = QWidget(); ll = QVBoxLayout(left)
        self.backend = QComboBox(); self.backend.addItems(['MSS', 'Qt']); self.backend.currentTextChanged.connect(self.refresh_monitors)
        self.monitor = QComboBox(); self.highres = QCheckBox('MSS high resolution on macOS'); self.highres.setChecked(True)
        self.hide_panel = QCheckBox('Hide lab during capture'); self.hide_panel.setChecked(True)
        self.file_list = QListWidget(); self.file_list.currentRowChanged.connect(self.folder_selected)
        self.grid_list = QListWidget(); self.grid_list.currentRowChanged.connect(self.selection_changed)
        for widget in [QLabel('Capture backend'), self.backend, self.monitor, self.highres, self.hide_panel, QLabel('Folder images'), self.file_list, QLabel('Detected grids (select to inspect)'), self.grid_list]: ll.addWidget(widget)
        splitter.addWidget(left)
        self.canvas = Canvas(self); splitter.addWidget(self.canvas)
        panel = QWidget(); form = QFormLayout(panel)
        self.yolo_path = QLineEdit(str(ROOT/'models/yolo.pt')); self.unet_path = QLineEdit(str(ROOT/'models/grid_unet.pt'))
        self.phase_path = QLineEdit(str(Settings().config.parents[1] / 'runs/phase-baseline-v1/best.pt'))
        self.config = QLineEdit(str(Settings().config)); self.validation = QLineEdit(str(Settings().validation))
        for name, field, key in [('YOLO model', self.yolo_path, 'yolo'), ('GridUNet model', self.unet_path, 'unet'), ('PhaseGridUNet model', self.phase_path, 'phase')]:
            row = QWidget(); layout = QHBoxLayout(row); layout.setContentsMargins(0,0,0,0); layout.addWidget(field)
            browse = QPushButton('…'); browse.clicked.connect(lambda checked=False, f=field: self.choose_path(f)); layout.addWidget(browse)
            load = QPushButton('Load'); load.clicked.connect(lambda checked=False, k=key: self.model_action(k, True)); layout.addWidget(load)
            unload = QPushButton('Unload'); unload.clicked.connect(lambda checked=False, k=key: self.model_action(k, False)); layout.addWidget(unload)
            self.conflicting_controls.extend([load, unload])
            form.addRow(name, row)
        self.loaded = QLabel('No models loaded'); self.loaded.setWordWrap(True); form.addRow('Loaded', self.loaded)
        for title, field in [('Training TOML', self.config), ('Validation TOML', self.validation)]:
            row = QWidget(); layout = QHBoxLayout(row); layout.setContentsMargins(0,0,0,0); layout.addWidget(field); b = QPushButton('…'); b.clicked.connect(lambda checked=False, f=field: self.choose_path(f)); layout.addWidget(b); form.addRow(title,row)
        self.device = QComboBox(); self.device.addItems(['auto','cpu','mps','cuda']); form.addRow('Device', self.device)
        self.geometry_backend = QComboBox(); self.geometry_backend.addItems(['gaussian', 'phase']); form.addRow('Geometry backend', self.geometry_backend)
        self.imgsz = self.spin(1280, 32, 8192); form.addRow('YOLO image size', self.imgsz)
        self.conf = QDoubleSpinBox(); self.conf.setRange(0,1); self.conf.setSingleStep(.05); self.conf.setValue(.25); form.addRow('Confidence', self.conf)
        self.padding = self.spin(20,0,2000); form.addRow('Crop padding (px)', self.padding)
        self.mode = QComboBox(); self.mode.addItems(['whole','tiled','native']); form.addRow('U-Net mode', self.mode)
        self.tile = self.spin(384,8,4096); self.overlap = self.spin(96,0,4095); self.batch = self.spin(4,1,128)
        form.addRow('Tile size', self.tile); form.addRow('Tile overlap', self.overlap); form.addRow('Tile batch', self.batch)
        self.mode.currentTextChanged.connect(self.update_tile_controls)
        self.update_tile_controls(self.mode.currentText())
        self.decoder = QComboBox(); self.decoder.addItems(['joint','independent']); form.addRow('Decoder', self.decoder)
        self.heatmap = QComboBox(); self.heatmap.addItems(['None','Centers','Vertical boundaries','Horizontal boundaries', 'Phase occupancy', 'Phase coverage', 'Tile horizontal phase', 'Tile vertical phase']); self.heatmap.currentIndexChanged.connect(self.redraw); form.addRow('Heatmap', self.heatmap)
        self.tile_inspect = self.spin(0, 0, 99999); self.tile_inspect.valueChanged.connect(self.redraw); form.addRow('Phase tile index (zero-based)', self.tile_inspect)
        self.layer_widgets = {}
        for key, label in [('detections','Detections'),('crops','Crop boundaries'),('boundaries','Grid boundaries'),('centers','Cell centers'), ('tiles','Phase tile layout'), ('terminations','Phase observed terminations'), ('cell_support','Phase cell support')]:
            check = QCheckBox(label); check.setChecked(key not in ('tiles', 'terminations', 'cell_support')); check.toggled.connect(self.redraw); self.layer_widgets[key] = check; form.addRow(check)
        self.overlay_layer_widgets = {}
        for key, label in [('detections','Overlay detections'),('crops','Overlay crop boundaries'),('boundaries','Overlay grid boundaries'),('centers','Overlay cell centers')]:
            check = QCheckBox(label); check.setChecked(True); check.toggled.connect(self.redraw); self.overlay_layer_widgets[key] = check; form.addRow(check)
        self.opacity = QDoubleSpinBox(); self.opacity.setRange(.1,1); self.opacity.setSingleStep(.1); self.opacity.setValue(.8); self.opacity.valueChanged.connect(self.change_opacity); form.addRow('Overlay opacity',self.opacity)
        mouse = QPushButton('Inspect mouse position'); mouse.clicked.connect(self.mouse_position); form.addRow(mouse)
        self.diagnostics = QTextEdit(); self.diagnostics.setReadOnly(True); self.diagnostics.setMinimumHeight(250); form.addRow('Diagnostics', self.diagnostics)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel); splitter.addWidget(scroll); splitter.setSizes([250,800,400])
        self.refresh_monitors()

    def update_tile_controls(self, mode):
        for widget in (self.tile, self.overlap, self.batch): widget.setEnabled(mode == 'tiled')

    def spin(self, value, low, high):
        widget = QSpinBox(); widget.setRange(low,high); widget.setValue(value); return widget
    def layers(self): return {k:v.isChecked() for k,v in self.layer_widgets.items()}
    def overlay_layers(self): return {k:v.isChecked() for k,v in self.overlay_layer_widgets.items()}
    def redraw(self, *args): self.canvas.update(); self.overlay.update()
    def change_opacity(self, value): self.overlay.setWindowOpacity(value)
    def selected_detection(self):
        item = self.grid_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None
    def selected_grid(self):
        if not self.result: return None
        return next ( (g for g in self.result.grids if g.detection_index == self.selected_detection()), None)
    def actual_size(self): self.canvas.scale=1. / self.canvas.devicePixelRatioF(); self.canvas.offset=[0.,0.]; self.canvas.update()
    def choose_path(self, field):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose file', field.text())
        if path: field.setText(path)
    def refresh_monitors(self, *args):
        self.monitor.clear()
        try:
            for label, value in monitors(self.backend.currentText()): self.monitor.addItem(label, value)
        except Exception as exc: self.diagnostics.setPlainText(f'Monitor enumeration failed: {exc}')
    def set_frame(self, frame):
        self.generation += 1; self.overlay.hide(); self.frame = frame; self.result = None
        self.image_skia = skia_image(frame.image); self.grid_list.clear(); self.diagnostics.setPlainText(frame.diagnostics); self.canvas.fit()
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open image', '', 'Images (*.png *.jpg *.jpeg *.bmp *.webp)')
        if path: self.load_image(path)
    def load_image(self, path):
        try:
            with Image.open(path) as image: self.set_frame(Frame(image.convert('RGB'), None, f'File: {path}\nDesktop mapping unavailable'))
        except Exception as exc: QMessageBox.warning(self, 'Image error', str(exc))
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Open folder')
        if folder:
            self.folder_files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.webp'))
            self.file_list.clear(); self.file_list.addItems([p.name for p in self.folder_files])
    def folder_selected(self, index):
        if 0 <= index < len(self.folder_files): self.load_image(self.folder_files[index])
    def save_image(self):
        if self.frame:
            path, _ = QFileDialog.getSaveFileName(self, 'Save original image', '', 'PNG (*.png)')
            if path: self.frame.image.save(path)
    def capture(self, region):
        if self.capture_in_progress: return
        monitor = self.monitor.currentData()
        if monitor is None: QMessageBox.warning(self, 'Capture', 'No monitor selected'); return
        rect = None
        if region:
            text, ok = QInputDialog.getText(self, 'Capture region', 'Monitor-local x,y,width,height (logical coordinates)')
            if not ok: return
            try:
                rect = tuple(int(x.strip()) for x in text.split(','))
                if len(rect) != 4 or min(rect) < 0 or min(rect[2:]) <= 0: raise ValueError()
                mw = monitor.geometry().width() if self.backend.currentText() == 'Qt' else monitor['width']
                mh = monitor.geometry().height() if self.backend.currentText() == 'Qt' else monitor['height']
                if rect[0]+rect[2] > mw or rect[1]+rect[3] > mh: raise ValueError()
            except ValueError: QMessageBox.warning(self,'Capture','Enter a rectangle contained within the monitor'); return
        self.capture_in_progress = True
        overlay_visible = self.overlay.isVisible(); panel_visible = self.isVisible()
        self.overlay.hide()
        if self.hide_panel.isChecked(): self.hide()
        backend = self.backend.currentText(); highres = self.highres.isChecked()
        def finish():
            try: self.set_frame(capture_frame(backend, monitor, rect, highres))
            except Exception as exc:
                if overlay_visible: self.overlay.show()
                QMessageBox.warning(self, 'Capture failed', f'{exc}\nCheck screen recording permissions and backend support.')
            finally:
                self.capture_in_progress = False
                if panel_visible: self.show()
        QTimer.singleShot(250, finish)
    def settings(self):
        return Settings(self.device.currentText(), self.imgsz.value(), self.conf.value(), self.padding.value(), self.mode.currentText(), self.tile.value(), self.overlap.value(), self.batch.value(), self.decoder.currentText(), Path(self.config.text()), Path(self.validation.text()), self.geometry_backend.currentText())
    def start_worker(self, fn, done, guard_generation=True):
        if self.worker is not None: return
        self.statusBar().showMessage('Processing…'); self.worker = Worker(fn, self.generation)
        self.worker_done = done; self.guard_generation = guard_generation
        for control in self.conflicting_controls: control.setEnabled(False)
        self.worker.succeeded.connect(self.worker_succeeded)
        self.worker.failed.connect(self.worker_failed)
        self.worker.finished.connect(self.worker_finished); self.worker.start()
    @Slot(object, int)
    def worker_succeeded(self, result, generation):
        if not self.guard_generation or generation == self.generation: self.worker_done(result)
    @Slot(str)
    def worker_failed(self, error):
        QMessageBox.warning(self, 'Processing error', error)
    @Slot()
    def worker_finished(self):
        self.worker = None
        for control in self.conflicting_controls: control.setEnabled(True)
        self.statusBar().showMessage('Ready')
    def model_action(self, key, load):
        if self.worker is not None: return
        if not load:
            getattr(self.pipeline, 'unload_'+key)(); self.update_loaded(); return
        path = {'yolo': self.yolo_path, 'unet': self.unet_path, 'phase': self.phase_path}[key].text(); config = self.config.text()
        def action():
            if key == 'yolo': self.pipeline.load_yolo(path)
            elif key == 'phase': self.pipeline.load_phase(path)
            else: self.pipeline.load_unet(path, config)
        self.start_worker(action, lambda _: self.update_loaded(), guard_generation=False)
    def update_loaded(self): self.loaded.setText(f'YOLO: {self.pipeline.yolo_path or "unloaded"}\nGridUNet: {self.pipeline.unet_path or "unloaded"}\nPhaseGridUNet: {self.pipeline.phase_path or "unloaded"}')
    def run(self, mode):
        if self.worker is not None: return
        if self.frame is None: QMessageBox.warning(self,'Recognition','Open or capture an image first'); return
        image = self.frame.image; settings = self.settings(); previous = self.result
        if mode == 'geometry' and previous is None: QMessageBox.warning(self,'Recognition','Run YOLO first'); return
        def action():
            start = perf_counter()
            result = self.pipeline.detect(image, settings) if mode in ('all','yolo') else previous
            if mode in ('all','geometry'): result = self.pipeline.geometry(image, result, settings)
            result.timings['job wall ms'] = (perf_counter()-start)*1000
            return result
        self.start_worker(action, self.accept_result)
    def accept_result(self, result):
        self.result = result; self.grid_list.clear()
        for i,d in enumerate(result.detections):
            if d.name == 'grid':
                self.grid_list.addItem(f'Grid {i}: {d.confidence:.3f}'); self.grid_list.item(self.grid_list.count()-1).setData(Qt.ItemDataRole.UserRole,i)
        if self.grid_list.count(): self.grid_list.setCurrentRow(0)
        self.show_diagnostics(); self.redraw()
    def selection_changed(self, *args): self.show_diagnostics(); self.redraw()
    def show_diagnostics(self):
        if not self.frame: return
        text = self.frame.diagnostics
        if self.result:
            text += '\n\nDetections:\n' + '\n'.join(f'{d.name} {d.confidence:.3f} {tuple(round(x,1) for x in d.box)}' for d in self.result.detections)
            text += '\n\n' + '\n'.join(f'{k}: {v:.1f}' for k,v in self.result.timings.items())
            grid = self.selected_grid()
            if grid:
                validation_status = grid.validation.status if grid.validation else 'no hypothesis'
                validation_reasons = grid.validation.reasons if grid.validation else ('no hypothesis',)
                text += f'\n\nCrop: {grid.crop.box}; clipped: {grid.crop.clipped}\nDecoder: {grid.decoding.status}\nStrict: {validation_status}\nReasons: {", ".join(validation_reasons) or "none"}\nCompleteness may be uncertain, especially at image boundaries.\n'
                for i,(xs,ys,score) in enumerate(grid.hypotheses):
                    label = 'accepted board' if grid.validation and grid.validation.accepted and i == 0 else 'unvalidated candidate'
                    text += f'\nCandidate {i+1} ({label}): {len(ys)-1} rows × {len(xs)-1} columns; pitch {np.mean(np.diff(xs)):.2f} × {np.mean(np.diff(ys)):.2f}; origin {xs[0]:.2f}, {ys[0]:.2f}; score {score:.4f}'
                text += '\n' + '\n'.join(f'{k}: {v}' for k,v in grid.diagnostics.items())
                text += '\n' + '\n'.join(f'{k}: {v:.1f}' for k,v in grid.timings.items())
        self.diagnostics.setPlainText(text)
    def toggle_overlay(self):
        if self.overlay.isVisible(): self.overlay.hide(); return
        try:
            if not self.result: raise ValueError('Run recognition first')
            self.overlay.show_frame()
        except ValueError as exc: QMessageBox.warning(self,'Overlay',str(exc))
    def mouse_position(self):
        try:
            import pyautogui
            from PySide6.QtGui import QCursor
            p=pyautogui.position(); q=QCursor.pos()
            QMessageBox.information(self,'Coordinate diagnostics',f'PyAutoGUI: {p}\nQt logical desktop: {q.x()}, {q.y()}\nThese coordinate systems can differ, especially on secondary monitors.')
        except Exception as exc: QMessageBox.warning(self,'Mouse diagnostics',str(exc))
    def closeEvent(self, event):
        if self.worker is not None:
            QMessageBox.information(self, 'Processing', 'Wait for the current job to finish before closing.'); event.ignore(); return
        self.overlay.close(); event.accept()
