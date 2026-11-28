import sys
from pathlib import Path
from time import perf_counter
import skia
from PIL import Image
from PySide6.QtCore import QEvent, QSettings, QSize, Qt, QThread, Signal, QTimer, Slot
from PySide6.QtGui import QAction, QActionGroup, QColor, QKeySequence, QPalette, QPainter
from PySide6.QtWidgets import (QApplication, QDialog, QPushButton, QStackedWidget, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QDockWidget, QTabWidget, QToolButton, QFileDialog, QComboBox, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QTextEdit, QListWidget, QTreeWidget, QTreeWidgetItem, QHeaderView, QMessageBox, QInputDialog, QScrollArea, QToolBar, QSizePolicy, QMenu)
from .capture import Frame, capture_frame, monitors
from .pipeline import Pipeline, Settings, ROOT, GEOMETRY_ROOT, device_for
from .provenance import validation_errors, stage_summary, tile_selection
from .recognition_state import RecognitionSelection
from .explorer import explorer_items
from .viewer_selection import hit_detection, image_point, selection_region
from .inspector import inspector_sections, property_report
from .rendering import draw, draw_viewer_visualization, draw_selection, RecognitionStyle
from .viewer import ImageViewer, surface_pixmap
from .preferences import DEFAULTS, PreferencesDialog
from .appearance import apply_appearance
from .icons import icon
from .zoom_text import parse_percent, format_percent
from .viewport import WHEEL_STEP_FACTOR


class ZoomSelector(QComboBox):
    """Editable percentage which leaves the user's active edit untouched."""
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.dirty = False
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.addItems([f'{n}%' for n in (10, 25, 50, 75, 100, 200, 400, 800, 1600, 3200, 6400)])
        self.setMinimumWidth(90); self.setMaximumWidth(120)
        self.setToolTip('Physical magnification; enter a percentage')
        self.lineEdit().textEdited.connect(self._edited)
        self.lineEdit().editingFinished.connect(self.commit)
        self.activated.connect(self._selected)
        viewer.magnification_changed.connect(self.refresh)
        self.refresh(viewer.magnification())

    def _edited(self, text):
        self.dirty = True

    def _selected(self, index):
        self.dirty = True
        self.commit()

    def refresh(self, value):
        if not self.dirty:
            self.setEditText(format_percent(value))

    def commit(self):
        if not self.dirty:
            return
        self.dirty = False
        try:
            self.viewer.set_magnification(parse_percent(self.currentText()))
        except ValueError:
            pass
        self.refresh(self.viewer.magnification())

    def event(self, event):
        if (event.type() == QEvent.Type.ShortcutOverride and event.key() == Qt.Key.Key_Escape
                and not self.view().isVisible()):
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and not self.view().isVisible():
            # Focus-out may emit editingFinished; clear the pending edit first.
            self.dirty = False
            self.refresh(self.viewer.magnification())
            self.viewer.setFocus(Qt.FocusReason.OtherFocusReason)
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        event.ignore()


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
        self.pipeline = Pipeline(); self.frame = None; self.recognition = RecognitionSelection()
        self.generation = 0; self.worker = None; self.folder_files = []; self.capture_in_progress = False
        self.app_settings = QSettings('Minesweeper Recognition Lab', 'Recognition Lab')
        self.preferences = self.read_preferences()
        self.system_style = QApplication.style().objectName()
        self.focus_state = None
        self.folder_open = False
        self.overlay = Overlay(self)
        self._tile_grid = None
        self.canvas = ImageViewer()
        self.canvas.clicked.connect(self.viewer_clicked)
        self.workspace = QStackedWidget()
        self.setCentralWidget(self.workspace)
        self.viewer_container = QWidget()
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0); self.viewer_layout.setSpacing(0)
        self.viewer_toolbar = QToolBar('Viewer')
        self.viewer_toolbar.setMovable(False)
        self.viewer_layout.addWidget(self.viewer_toolbar)
        self.viewer_layout.addWidget(self.canvas, 1)
        self.workspace.addWidget(self.viewer_container)
        self.setDockNestingEnabled(True)

        # Commands are shared by menus, toolbar, and model controls.
        self.actions = {}
        for key, title, fn in [
            ('open_image', 'Open image…', self.open_image),
            ('open_folder', 'Open folder…', self.open_folder),
            ('capture_screen', 'Capture screen', lambda: self.capture(False)),
            ('capture_region', 'Capture region…', lambda: self.capture(True)),
            ('save', 'Save screenshot…', self.save_image),
            ('yolo', 'Run detection', lambda: self.run('yolo')),
            ('geometry', 'Run geometry', lambda: self.run('geometry')),
            ('all', 'Run all', lambda: self.run('all')),
            ('overlay', 'Toggle desktop overlay', self.toggle_overlay),
            ('fit', 'Fit to window', self.canvas.fit),
            ('actual', 'Actual size', self.actual_size),
            ('mouse', 'Inspect mouse position', self.mouse_position),
            ('load_yolo', 'Load YOLO', lambda: self.model_action('yolo', True)),
            ('unload_yolo', 'Unload YOLO', lambda: self.model_action('yolo', False)),
            ('load_phase', 'Load PhaseGridUNet', lambda: self.model_action('phase', True)),
            ('unload_phase', 'Unload PhaseGridUNet', lambda: self.model_action('phase', False)),
            ('preferences', 'Preferences…', self.edit_preferences),
            ('quit', 'Quit', self.close),
            ('reset_layout', 'Reset layout', self.reset_layout),
            ('focus', 'Focus mode', self.toggle_focus),
            ('grid', 'Show grid boundaries', self.toggle_grid),
            ('exit_focus', 'Exit focus mode', self.exit_focus),
        ]:
            action = QAction(title, self)
            if sys.platform == 'darwin':
                action.setIconVisibleInMenu(False)
            action.triggered.connect(fn)
            self.actions[key] = action
        self.actions['focus'].setCheckable(True)
        self.actions['grid'].setCheckable(True)
        self.setup_shortcuts()
        self.setup_viewer_toolbar()
        welcome = QWidget(); welcome.setAutoFillBackground(True)
        welcome_palette = QPalette()
        welcome_palette.setColor(QPalette.ColorRole.Window, QColor('#272727'))
        welcome_palette.setColor(QPalette.ColorRole.WindowText, QColor('#eeeeee'))
        welcome.setPalette(welcome_palette)
        welcome_layout = QVBoxLayout(welcome); welcome_layout.setSpacing(14)
        welcome_layout.addStretch()
        heading = QLabel('Recognition Lab')
        heading_font = heading.font(); heading_font.setPointSize(heading_font.pointSize() + 5); heading_font.setBold(True)
        heading.setFont(heading_font); heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(heading)
        message = QLabel('Open an image or capture a screen to begin inspection.')
        message.setAlignment(Qt.AlignmentFlag.AlignCenter); message.setWordWrap(True); welcome_layout.addWidget(message)
        welcome_buttons = QHBoxLayout(); welcome_buttons.addStretch()
        for key in ('open_image', 'open_folder', 'capture_screen'):
            button = QToolButton(); button.setDefaultAction(self.actions[key])
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly); welcome_buttons.addWidget(button)
        welcome_buttons.addStretch(); welcome_layout.addLayout(welcome_buttons); welcome_layout.addStretch()
        self.workspace.addWidget(welcome); self.workspace.setCurrentWidget(welcome)

        explorer = QWidget(); explorer_layout = QVBoxLayout(explorer)
        explorer_layout.setSpacing(8)
        images_heading = QLabel('Images')
        results_heading = QLabel('Recognition results')
        for heading in (images_heading, results_heading):
            font = heading.font(); font.setBold(True); heading.setFont(font)
        self.file_list = QListWidget(); self.file_list.currentRowChanged.connect(self.folder_selected)
        self.images_empty = QLabel('No folder open.'); self.images_empty.setWordWrap(True)
        self.open_folder_empty = QToolButton(); self.open_folder_empty.setDefaultAction(self.actions['open_folder'])
        self.open_folder_empty.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.result_tree = QTreeWidget()
        self.result_tree.setColumnCount(2); self.result_tree.setHeaderLabels(['Result', 'Summary'])
        self.result_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.result_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.result_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.result_tree.currentItemChanged.connect(self.selection_changed)
        self.result_tree.itemDoubleClicked.connect(self.navigate_result)
        self.results_empty = QLabel('No recognition results available.'); self.results_empty.setWordWrap(True)
        explorer_layout.addWidget(images_heading)
        explorer_layout.addWidget(self.images_empty)
        explorer_layout.addWidget(self.open_folder_empty, 0, Qt.AlignmentFlag.AlignLeft)
        explorer_layout.addWidget(self.file_list, 2)
        explorer_layout.addSpacing(8)
        explorer_layout.addWidget(results_heading)
        explorer_layout.addWidget(self.results_empty)
        explorer_layout.addWidget(self.result_tree, 1)
        explorer_layout.addStretch(1)
        self.explorer_layout = explorer_layout
        self.explorer_dock = self.make_dock('Explorer', explorer, Qt.DockWidgetArea.LeftDockWidgetArea)

        tabs = QTabWidget()
        inspect_panel = QWidget(); inspect_layout = QVBoxLayout(inspect_panel)
        self.inspect_empty = QLabel('Select a recognition result to inspect it.'); self.inspect_empty.setWordWrap(True)
        self.inspect = QTreeWidget(); self.inspect.setColumnCount(2)
        self.inspect.setHeaderLabels(['Property', 'Value'])
        self.inspect.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.inspect.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.inspect.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.inspect.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.inspect.customContextMenuRequested.connect(self.inspect_context_menu)
        copy_value = QAction('Copy property value', self.inspect)
        copy_value.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy_value.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy_value.triggered.connect(self.copy_inspect_value)
        self.inspect.addAction(copy_value)
        self.copy_report_button = QPushButton('Copy property report')
        self.copy_report_button.clicked.connect(self.copy_inspect_report)
        inspect_layout.addWidget(self.inspect_empty); inspect_layout.addWidget(self.inspect, 1)
        inspect_layout.addWidget(self.copy_report_button)
        tabs.addTab(inspect_panel, 'Inspect')
        layers_panel = QWidget(); layers_form = QFormLayout(layers_panel)
        self.image_sampling = QComboBox()
        self.image_sampling.addItem('Nearest neighbor', 'nearest')
        self.image_sampling.addItem('Bilinear', 'smooth')
        self.image_sampling.currentIndexChanged.connect(lambda _: self.canvas.set_sampling(self.image_sampling.currentData()))
        layers_form.addRow('Image sampling', self.image_sampling)
        self.pixel_grid = QCheckBox('Show source pixel grid')
        self.pixel_grid.toggled.connect(self.canvas.set_pixel_grid_visible)
        layers_form.addRow(self.pixel_grid)
        self.toolbar_pixel_grid.toggled.connect(self.pixel_grid.setChecked)
        self.pixel_grid.toggled.connect(self.toolbar_pixel_grid.setChecked)
        self.image_sampling.currentIndexChanged.connect(self.sync_sampling_action)
        self.sync_sampling_action()
        self.heatmap = QComboBox(); self.heatmap.addItems(['None', 'Phase occupancy', 'Phase coverage', 'Tile horizontal phase', 'Tile vertical phase']); self.heatmap.currentIndexChanged.connect(self.redraw); layers_form.addRow('Heatmap', self.heatmap)
        self.tile_inspect = self.spin(0, 0, 99999); self.tile_inspect.valueChanged.connect(self.redraw); layers_form.addRow('Phase tile index (zero-based)', self.tile_inspect)
        self.stroke_mode = QComboBox()
        self.stroke_mode.addItem('Screen space', 'screen')
        self.stroke_mode.addItem('Image space', 'image')
        layers_form.addRow('Recognition stroke mode', self.stroke_mode)
        self.stroke_width = QDoubleSpinBox()
        self.stroke_width.setRange(.5, 8); self.stroke_width.setSingleStep(.5); self.stroke_width.setValue(1)
        self.stroke_width.setSuffix(' physical px')
        layers_form.addRow('Geometry stroke width', self.stroke_width)
        self.vector_opacity = QSpinBox()
        self.vector_opacity.setRange(0, 100); self.vector_opacity.setValue(100); self.vector_opacity.setSuffix('%')
        layers_form.addRow('Recognition vector opacity', self.vector_opacity)
        self.stroke_mode.currentIndexChanged.connect(self.stroke_mode_changed)
        self.stroke_width.valueChanged.connect(self.redraw)
        self.vector_opacity.valueChanged.connect(self.redraw)
        self.layer_widgets = {}
        for key, label in [('detections','Detections'),('crops','Crop boundaries'),('boundaries','Grid boundaries'),('centers','Cell centers'), ('tiles','Phase tile layout'), ('terminations','Phase observed terminations'), ('cell_support','Phase cell support')]:
            check = QCheckBox(label); check.setChecked(key not in ('tiles', 'terminations', 'cell_support')); check.toggled.connect(self.redraw); self.layer_widgets[key] = check; layers_form.addRow(check)
        self.actions['grid'].setChecked(self.layer_widgets['boundaries'].isChecked())
        self.layer_widgets['boundaries'].toggled.connect(self.actions['grid'].setChecked)
        self.overlay_layer_widgets = {}
        for key, label in [('detections','Overlay detections'),('crops','Overlay crop boundaries'),('boundaries','Overlay grid boundaries'),('centers','Overlay cell centers')]:
            check = QCheckBox(label); check.setChecked(True); check.toggled.connect(self.redraw); self.overlay_layer_widgets[key] = check; layers_form.addRow(check)
        self.opacity = QDoubleSpinBox(); self.opacity.setRange(.1,1); self.opacity.setSingleStep(.1); self.opacity.setValue(.8); self.opacity.valueChanged.connect(self.change_opacity); layers_form.addRow('Overlay opacity', self.opacity)
        tabs.addTab(self.scroll_panel(layers_panel), 'Layers')

        panel = QWidget(); form = QFormLayout(panel)
        self.yolo_path = QLineEdit(str(ROOT/'models/yolo.pt'))
        self.phase_path = QLineEdit(str(ROOT / 'models/phase_grid_unet.pt'))
        self.yolo_path.setMinimumWidth(0); self.phase_path.setMinimumWidth(0)
        for name, field, key in [('YOLO model', self.yolo_path, 'yolo'), ('PhaseGridUNet model', self.phase_path, 'phase')]:
            row = QWidget(); layout = QHBoxLayout(row); layout.setContentsMargins(0,0,0,0); layout.addWidget(field)
            browse = QToolButton(); browse.setText('…'); browse.clicked.connect(lambda checked=False, f=field: self.choose_path(f)); layout.addWidget(browse)
            for verb in ('load', 'unload'):
                button = QToolButton(); button.setDefaultAction(self.actions[f'{verb}_{key}']); button.setText(verb.title()); layout.addWidget(button)
            form.addRow(name, row)
        self.loaded = QLabel('No models loaded'); self.loaded.setWordWrap(True); form.addRow('Loaded', self.loaded)
        self.device = QComboBox(); self.device.addItems(['auto','cpu','mps','cuda']); form.addRow('Device', self.device)
        self.imgsz = self.spin(1280, 32, 8192); form.addRow('YOLO image size', self.imgsz)
        self.conf = QDoubleSpinBox(); self.conf.setRange(0,1); self.conf.setSingleStep(.05); self.conf.setValue(.25); form.addRow('Confidence', self.conf)
        self.padding = self.spin(20,0,2000); form.addRow('Crop padding (px)', self.padding)
        self.mode = QComboBox(); self.mode.addItems(['whole','tiled']); form.addRow('Geometry mode', self.mode)
        self.tile = self.spin(384,384,384); self.overlap = self.spin(96,0,383)
        form.addRow('Tile size', self.tile); form.addRow('Tile overlap', self.overlap)
        self.settings_status = QLabel(); self.settings_status.setWordWrap(True)
        self.settings_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow('Completed runs', self.settings_status)
        self.mode.currentTextChanged.connect(self.update_tile_controls)
        self.update_tile_controls(self.mode.currentText())
        self.backend = QComboBox(); self.backend.addItems(['MSS', 'Qt']); self.backend.currentTextChanged.connect(self.refresh_monitors)
        self.monitor = QComboBox(); self.highres = QCheckBox('MSS high resolution on macOS'); self.highres.setChecked(True)
        self.hide_panel = QCheckBox('Hide lab during capture'); self.hide_panel.setChecked(True)
        form.addRow('Capture backend', self.backend); form.addRow('Monitor', self.monitor)
        form.addRow(self.highres); form.addRow(self.hide_panel)
        tabs.addTab(self.scroll_panel(panel), 'Settings')
        self.inspector_dock = self.make_dock('Inspector', tabs, Qt.DockWidgetArea.RightDockWidgetArea)

        diagnostic_tabs = QTabWidget()
        self.diagnostics = QTextEdit(); self.diagnostics.setReadOnly(True)
        self.performance = QTextEdit(); self.performance.setReadOnly(True)
        self.diagnostics.setPlaceholderText('No image diagnostics available.')
        self.performance.setPlaceholderText('No timing information available.')
        diagnostic_tabs.addTab(self.diagnostics, 'Diagnostics'); diagnostic_tabs.addTab(self.performance, 'Performance')
        self.diagnostics_dock = self.make_dock('Diagnostics', diagnostic_tabs, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.set_default_dock_sizes()

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        for key in ('open_image', 'open_folder', 'capture_screen', 'capture_region', 'save'): file_menu.addAction(self.actions[key])
        file_menu.addSeparator(); file_menu.addAction(self.actions['preferences']); file_menu.addAction(self.actions['quit'])
        view_menu = menu_bar.addMenu('View')
        for key in ('fit', 'actual', 'grid', 'overlay', 'focus', 'reset_layout'): view_menu.addAction(self.actions[key])
        view_menu.addSeparator()
        for dock in (self.explorer_dock, self.inspector_dock, self.diagnostics_dock): view_menu.addAction(dock.toggleViewAction())
        recognition_menu = menu_bar.addMenu('Recognition')
        for key in ('yolo', 'geometry', 'all'): recognition_menu.addAction(self.actions[key])
        models_menu = menu_bar.addMenu('Models')
        for key in ('load_yolo', 'unload_yolo', 'load_phase', 'unload_phase'): models_menu.addAction(self.actions[key])
        tools_menu = menu_bar.addMenu('Tools'); tools_menu.addAction(self.actions['mouse'])
        self.bar = QToolBar('Main', self); self.bar.setObjectName('mainToolbar'); self.addToolBar(self.bar)
        self.bar.setIconSize(QSize(18, 18))
        for key, label in [('open_image', 'Open image'), ('open_folder', 'Open folder'),
                           ('capture_screen', 'Capture screen'), ('capture_region', 'Capture region'),
                           ('yolo', 'Detect'), ('geometry', 'Geometry'), ('all', 'Run all'),
                           ('fit', 'Fit'), ('actual', 'Actual size'), ('focus', 'Focus'),
                           ('preferences', 'Preferences')]:
            self.actions[key].setIconText(label)
        for group in (('open_image', 'open_folder', 'capture_screen', 'capture_region'),
                      ('yolo', 'geometry', 'all'), ('preferences',)):
            if self.bar.actions(): self.bar.addSeparator()
            for key in group: self.bar.addAction(self.actions[key])

        self.image_status = QLabel('Image: —'); self.zoom_status = QLabel('Zoom: —')
        self.cursor_status = QLabel('Cursor: —'); self.device_status = QLabel('Device setting: auto')
        for label in (self.image_status, self.zoom_status, self.cursor_status, self.device_status): self.statusBar().addPermanentWidget(label)
        self.canvas.magnification_changed.connect(self.update_zoom_status)
        self.canvas.cursor_changed.connect(self.update_cursor_status)
        self.canvas.cursor_left.connect(lambda: self.cursor_status.setText('Cursor: —'))
        self.device.currentTextChanged.connect(lambda value: self.device_status.setText(f'Device setting: {value}'))
        self.statusBar().showMessage('Ready')
        self.refresh_monitors()
        self.apply_preferences()
        self.restore_workspace()
        for widget in (self.device, self.mode):
            widget.currentIndexChanged.connect(self.pending_settings_changed)
        for widget in (self.imgsz, self.conf, self.padding, self.tile, self.overlap):
            widget.valueChanged.connect(self.pending_settings_changed)
        for widget in (self.yolo_path, self.phase_path):
            widget.textChanged.connect(self.pending_settings_changed)
        self.update_empty_states()
        self.update_action_states()
        self.update_inspector()
        self.redraw()
        QApplication.instance().paletteChanged.connect(self.update_action_icons)
        QApplication.instance().installEventFilter(self)

    def make_dock(self, title, widget, area):
        dock = QDockWidget(title, self); dock.setObjectName(title.lower() + 'Dock')
        dock.setWidget(widget); self.addDockWidget(area, dock)
        return dock

    def scroll_panel(self, panel):
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel)
        return scroll

    def update_cursor_status(self, inspection):
        if inspection.pixel is None:
            detail = 'Pixel: — | RGB: —'
        else:
            x, y = inspection.pixel
            r, g, b, _ = inspection.rgba
            detail = f'Pixel: {x}, {y} | RGB: {r}, {g}, {b}'
        self.cursor_status.setText(f'Cursor: {inspection.x:.3f}, {inspection.y:.3f} | {detail}')

    def update_zoom_status(self, magnification):
        self.zoom_status.setText(f'Zoom: {format_percent(magnification)}')

    def setup_viewer_toolbar(self):
        bar = self.viewer_toolbar
        bar.setIconSize(QSize(16, 16))
        self.zoom_out = QAction('Zoom out', self); self.zoom_out.setToolTip('Zoom out at viewport center')
        self.zoom_out.triggered.connect(lambda: self.canvas.zoom_by(1 / WHEEL_STEP_FACTOR))
        bar.addAction(self.zoom_out)
        self.zoom_selector = ZoomSelector(self.canvas); bar.addWidget(self.zoom_selector)
        self.zoom_in = QAction('Zoom in', self); self.zoom_in.setToolTip('Zoom in at viewport center')
        self.zoom_in.triggered.connect(lambda: self.canvas.zoom_by(WHEEL_STEP_FACTOR))
        bar.addAction(self.zoom_in)
        bar.addSeparator()
        bar.addAction(self.actions['fit'])
        self.actions['actual'].setIconText('1:1')
        bar.addAction(self.actions['actual'])
        bar.widgetForAction(self.actions['actual']).setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        bar.addSeparator()
        self.sampling_action = QAction('Nearest neighbor', self)
        self.sampling_action.setToolTip('Source image sampling')
        sampling = QMenu(bar)
        self.sampling_group = QActionGroup(self); self.sampling_group.setExclusive(True)
        self.sampling_nearest = sampling.addAction('Nearest neighbor')
        self.sampling_bilinear = sampling.addAction('Bilinear')
        for index, action in enumerate((self.sampling_nearest, self.sampling_bilinear)):
            action.setCheckable(True)
            self.sampling_group.addAction(action)
            action.triggered.connect(lambda checked=False, i=index: self.image_sampling.setCurrentIndex(i))
        self.sampling_action.setMenu(sampling)
        bar.addAction(self.sampling_action)
        bar.widgetForAction(self.sampling_action).setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.toolbar_pixel_grid = QAction('Show source pixel grid', self)
        self.toolbar_pixel_grid.setCheckable(True)
        self.toolbar_pixel_grid.setToolTip('Show source pixel grid at 800% magnification or greater')
        bar.addAction(self.toolbar_pixel_grid)
        self.minimap_toggle = QAction('Show minimap', self)
        self.minimap_toggle.setCheckable(True); self.minimap_toggle.setChecked(True)
        self.minimap_toggle.setToolTip('Show minimap when the image is not fully visible')
        self.minimap_toggle.toggled.connect(self.canvas.set_minimap_enabled)
        bar.addAction(self.minimap_toggle)
        spacer = QWidget(); spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        bar.addAction(self.actions['focus'])

    def sync_sampling_action(self, *args):
        index = self.image_sampling.currentIndex()
        self.sampling_nearest.setChecked(index == 0)
        self.sampling_bilinear.setChecked(index == 1)
        self.sampling_action.setText(self.image_sampling.currentText())

    def setup_shortcuts(self):
        for key, shortcut in [('open_image', QKeySequence.StandardKey.Open),
                              ('save', QKeySequence.StandardKey.Save),
                              ('preferences', QKeySequence.StandardKey.Preferences),
                              ('quit', QKeySequence.StandardKey.Quit)]:
            self.actions[key].setShortcuts(shortcut)
        self.actions['open_folder'].setShortcut(QKeySequence('Ctrl+Shift+O'))
        self.actions['focus'].setShortcut(QKeySequence('Ctrl+Shift+F'))
        for key, shortcut in [('fit', 'F'), ('actual', '1'), ('grid', 'G'), ('exit_focus', 'Escape')]:
            action = self.actions[key]
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
            self.canvas.addAction(action)

    def read_preferences(self):
        values = {}
        for key, default in DEFAULTS.items():
            try:
                value = self.app_settings.value('preferences/' + key, default, type=type(default))
            except (TypeError, ValueError):
                value = default
            if key == 'appearance' and value not in ('system', 'light', 'dark'): value = default
            if key == 'toolbar' and value not in ('both', 'icons', 'text'): value = default
            values[key] = value
        return values

    def edit_preferences(self):
        dialog = PreferencesDialog(self.preferences, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.preferences = dialog.values()
            for key, value in self.preferences.items(): self.app_settings.setValue('preferences/' + key, value)
            self.apply_preferences()

    def apply_preferences(self):
        apply_appearance(self.preferences['appearance'], self.system_style)
        styles = {'both': Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
                  'icons': Qt.ToolButtonStyle.ToolButtonIconOnly, 'text': Qt.ToolButtonStyle.ToolButtonTextOnly}
        self.bar.setToolButtonStyle(styles[self.preferences['toolbar']])
        self.update_action_icons()

    def update_action_icons(self, *args):
        names = {'open_image': 'folder-open', 'open_folder': 'folder', 'capture_screen': 'monitor',
                 'capture_region': 'scan-line', 'yolo': 'scan', 'geometry': 'grid-3x3',
                 'all': 'play', 'fit': 'maximize', 'actual': 'scaling',
                 'focus': 'panels-top-left', 'preferences': 'settings', 'save': 'save'}
        palette = QApplication.palette()
        self.minimap_toggle.setIcon(icon('map', palette))
        self.zoom_out.setIcon(icon('zoom-out', palette))
        self.zoom_in.setIcon(icon('zoom-in', palette))
        self.toolbar_pixel_grid.setIcon(icon('grid-3x3', palette))
        for key, name in names.items():
            self.actions[key].setIcon(icon(name, palette))
        self.update_action_states()

    def update_command_tips(self, prerequisites=None):
        descriptions = {'open_image': 'Open an image file', 'open_folder': 'Browse images in a folder',
                        'capture_screen': 'Capture the selected screen', 'capture_region': 'Capture a region of the selected screen',
                        'yolo': 'Run grid detection', 'geometry': 'Run geometry on detected grids',
                        'all': 'Run detection and geometry', 'fit': 'Fit the image to the workspace',
                        'actual': 'View the image at actual size', 'focus': 'Show or hide the workspace panels',
                        'preferences': 'Change application preferences', 'save': 'Save the original image'}
        for key, description in descriptions.items():
            missing = prerequisites.get(key, []) if prerequisites else []
            tip = description + ('. Requires: ' + ', '.join(missing) if missing else '')
            action = self.actions[key]
            shortcut = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            action.setToolTip(tip + (f' ({shortcut})' if shortcut else ''))
            action.setStatusTip(tip)

    def eventFilter(self, watched, event):
        if (event.type() == QEvent.Type.ToolTip and not self.preferences['tooltips']
                and isinstance(watched, (QToolButton, QPushButton)) and self.isAncestorOf(watched)):
            return True
        return super().eventFilter(watched, event)

    def restore_workspace(self):
        if not self.preferences['restore_workspace']: return
        geometry = self.app_settings.value('workspace/geometry')
        state = self.app_settings.value('workspace/state')
        if geometry is not None:
            try: self.restoreGeometry(geometry)
            except (TypeError, ValueError): pass
        if state is not None:
            try:
                if not self.restoreState(state, 1): self.reset_layout()
            except (TypeError, ValueError): self.reset_layout()

    def save_workspace(self):
        # Focus mode hides docks temporarily; save its captured normal state instead.
        self.app_settings.setValue('workspace/geometry', self.saveGeometry())
        self.app_settings.setValue('workspace/state', self.focus_state if self.focus_state is not None else self.saveState(1))
        self.app_settings.sync()

    def reset_layout(self):
        if self.focus_state is not None: self.exit_focus()
        for dock, area in ((self.explorer_dock, Qt.DockWidgetArea.LeftDockWidgetArea),
                           (self.inspector_dock, Qt.DockWidgetArea.RightDockWidgetArea),
                           (self.diagnostics_dock, Qt.DockWidgetArea.BottomDockWidgetArea)):
            dock.setFloating(False); self.removeDockWidget(dock); self.addDockWidget(area, dock); dock.show()
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.removeToolBar(self.bar); self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.bar); self.bar.show()
        self.set_default_dock_sizes()

    def set_default_dock_sizes(self):
        self.resizeDocks([self.explorer_dock, self.inspector_dock], [250, 320], Qt.Orientation.Horizontal)
        self.resizeDocks([self.diagnostics_dock], [125], Qt.Orientation.Vertical)

    def toggle_focus(self, checked):
        if checked:
            self.focus_state = self.saveState(1)
            self.bar.show()
            docks = (self.explorer_dock, self.inspector_dock, self.diagnostics_dock)
            self.focus_dock_actions_enabled = [dock.toggleViewAction().isEnabled() for dock in docks]
            for dock in docks:
                dock.hide()
                dock.toggleViewAction().setEnabled(False)
        else:
            self.exit_focus()
        self.update_action_states()

    def exit_focus(self):
        if self.focus_state is None: return
        state, self.focus_state = self.focus_state, None
        restored = self.restoreState(state, 1)
        for dock, enabled in zip((self.explorer_dock, self.inspector_dock, self.diagnostics_dock),
                                 self.focus_dock_actions_enabled):
            dock.toggleViewAction().setEnabled(enabled)
        if not restored:
            self.reset_layout()
        self.actions['focus'].setChecked(False)
        self.update_action_states()

    def toggle_grid(self, checked):
        self.layer_widgets['boundaries'].setChecked(checked)

    def update_empty_states(self):
        self.images_empty.setText('No supported images in this folder.' if self.folder_open else 'No folder open.')
        self.images_empty.setVisible(not self.folder_files)
        self.open_folder_empty.setVisible(not self.folder_files)
        self.file_list.setVisible(bool(self.folder_files))
        self.results_empty.setText('No detections or geometry results.' if self.result is not None else 'No recognition results available.')
        self.results_empty.setVisible(self.result_tree.topLevelItemCount() == 0)
        self.result_tree.setVisible(self.result_tree.topLevelItemCount() != 0)
        self.explorer_layout.setStretch(self.explorer_layout.count() - 1,
                                        0 if self.folder_files or self.result_tree.topLevelItemCount() else 1)

    def update_action_states(self):
        busy = self.worker is not None
        capturing = self.capture_in_progress
        image = self.frame is not None
        result = self.result is not None
        grids = result and any(d.name == 'grid' for d in self.result.detections)
        mapped = image and self.frame.mapping is not None
        for key in ('open_image', 'open_folder', 'capture_screen', 'capture_region'):
            self.actions[key].setEnabled(not capturing)
        self.actions['save'].setEnabled(image and not capturing)
        yolo_loaded = self.pipeline.yolo is not None
        phase_loaded = self.pipeline.phase is not None
        idle = not busy and not capturing
        settings = self.settings()
        device_errors = []
        if settings.device != 'auto':
            try: device_for(settings.device)
            except (RuntimeError, ValueError) as exc: device_errors.append(str(exc))
        errors = {key: list(validation_errors(settings, stage)) + device_errors
                  for key, stage in (('yolo', 'detection'), ('geometry', 'geometry'), ('all', 'all'))}
        self.actions['yolo'].setEnabled(image and yolo_loaded and idle and not errors['yolo'])
        self.actions['geometry'].setEnabled(image and bool(grids) and phase_loaded and idle and not errors['geometry'])
        self.actions['all'].setEnabled(image and yolo_loaded and phase_loaded and idle and not errors['all'])
        common = ([] if image else ['an image']) + ([] if idle else ['the current operation to finish'])
        self.update_command_tips({
            'yolo': common + ([] if yolo_loaded else ['the YOLO model']) + errors['yolo'],
            'geometry': common + ([] if grids else ['detected grids']) + ([] if phase_loaded else ['the PhaseGridUNet model']) + errors['geometry'],
            'all': common + ([] if yolo_loaded else ['the YOLO model']) + ([] if phase_loaded else ['the PhaseGridUNet model']) + errors['all'],
        })
        self.update_settings_status(settings)
        self.actions['overlay'].setEnabled(not capturing and ((result and mapped) or self.overlay.isVisible()))
        self.file_list.setEnabled(not capturing)
        for key in ('fit', 'actual', 'grid'): self.actions[key].setEnabled(image)
        for widget in (self.zoom_out, self.zoom_in, self.zoom_selector): widget.setEnabled(image)
        self.sampling_action.setEnabled(image); self.toolbar_pixel_grid.setEnabled(image)
        for key in ('load_yolo', 'load_phase'): self.actions[key].setEnabled(not busy and not capturing)
        self.actions['unload_yolo'].setEnabled(not busy and not capturing and self.pipeline.yolo is not None)
        self.actions['unload_phase'].setEnabled(not busy and not capturing and self.pipeline.phase is not None)
        self.actions['exit_focus'].setEnabled(self.focus_state is not None)

    def pending_settings_changed(self, *args):
        self.update_action_states()

    def update_settings_status(self, settings):
        result = self.result
        detection = result.detection_provenance if result is not None else None
        geometry = result.geometry_provenance if result is not None else None
        self.settings_status.setText('Pending configuration: controls above\n\n' +
            stage_summary('Detection', settings, detection, self.pipeline.yolo_path,
                          self.pipeline.yolo_id, self.yolo_path.text()) + '\n\n' +
            stage_summary('Geometry', settings, geometry, self.pipeline.phase_path,
                          self.pipeline.phase_id, self.phase_path.text()))

    def update_tile_controls(self, mode):
        for widget in (self.tile, self.overlap): widget.setEnabled(mode == 'tiled')

    def spin(self, value, low, high):
        widget = QSpinBox(); widget.setRange(low,high); widget.setValue(value); return widget
    def layers(self): return {k:v.isChecked() for k,v in self.layer_widgets.items()}
    def overlay_layers(self): return {k:v.isChecked() for k,v in self.overlay_layer_widgets.items()}
    def stroke_mode_changed(self, *args):
        screen = self.stroke_mode.currentData() == 'screen'
        self.stroke_width.setSuffix(' physical px' if screen else ' source px')
        self.stroke_width.blockSignals(True)
        self.stroke_width.setValue(1 if screen else 2)
        self.stroke_width.blockSignals(False)
        self.redraw()

    def redraw(self, *args):
        result = self.result
        grid = self.selected_grid()
        layers = self.layers()
        bounds = selection_region(self.recognition.snapshot, self.recognition.selected, layers)
        phase = grid.phase_data if grid is not None else None
        self.heatmap.setEnabled(bool(phase))
        limits, index = tile_selection(grid, self._tile_grid, self.tile_inspect.value())
        self.tile_inspect.blockSignals(True)
        try:
            self.tile_inspect.setSpecialValueText('Unavailable' if limits is None else '')
            self.tile_inspect.setRange(*(limits if limits is not None else (0, 0)))
            self.tile_inspect.setValue(index)
            self._tile_grid = grid
        finally:
            self.tile_inspect.blockSignals(False)
        self.tile_inspect.setEnabled(limits is not None and self.heatmap.currentText().startswith('Tile '))
        mode = self.heatmap.currentText()
        tile = self.tile_inspect.value()
        style = RecognitionStyle(self.stroke_mode.currentData(), self.stroke_width.value(), self.vector_opacity.value() / 100)
        def visualize(canvas, magnification, dpr):
            draw_viewer_visualization(canvas, result, layers, None, grid, mode, tile, style, magnification, dpr)
            draw_selection(canvas, bounds, magnification, dpr, style.opacity)
        self.canvas.set_visualization(visualize)
        self.overlay.update()
    def change_opacity(self, value): self.overlay.setWindowOpacity(value)
    @property
    def result(self):
        return self.recognition.snapshot.result if self.recognition.snapshot is not None else None

    def selected_detection(self):
        return self.recognition.detection_index()
    def selected_grid(self):
        return self.recognition.grid()
    def actual_size(self): self.canvas.actual_size()
    def choose_path(self, field):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose file', field.text())
        if path: field.setText(path)
    def refresh_monitors(self, *args):
        self.monitor.clear()
        try:
            for label, value in monitors(self.backend.currentText()): self.monitor.addItem(label, value)
        except Exception as exc: self.diagnostics.setPlainText(f'Monitor enumeration failed: {exc}')
    def set_frame(self, frame):
        self.generation += 1; self.overlay.hide(); self.frame = frame; self.recognition.replace(None)
        self.canvas.set_image(frame.image)
        self.rebuild_result_tree(); self.show_diagnostics(); self.redraw()
        self.workspace.setCurrentWidget(self.viewer_container); self.canvas.fit()
        self.image_status.setText(f'Image: {frame.image.width} × {frame.image.height}'); self.cursor_status.setText('Cursor: —')
        self.update_empty_states(); self.update_action_states()
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
            self.folder_open = True
            self.file_list.clear(); self.file_list.addItems([p.name for p in self.folder_files]); self.update_empty_states()
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
        self.update_action_states()
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
                self.update_action_states()
        QTimer.singleShot(250, finish)
    def settings(self):
        return Settings(self.device.currentText(), self.imgsz.value(), self.conf.value(), self.padding.value(), self.mode.currentText(), self.tile.value(), self.overlap.value())
    def start_worker(self, fn, done, guard_generation=True):
        if self.worker is not None: return
        self.statusBar().showMessage('Processing…'); self.worker = Worker(fn, self.generation)
        self.worker_done = done; self.guard_generation = guard_generation
        self.update_action_states()
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
        self.update_action_states()
        self.statusBar().showMessage('Ready')
    def model_action(self, key, load):
        if self.worker is not None: return
        if not load:
            getattr(self.pipeline, 'unload_'+key)(); self.update_loaded(); return
        path = {'yolo': self.yolo_path, 'phase': self.phase_path}[key].text()
        def action():
            if key == 'yolo': self.pipeline.load_yolo(path)
            else: self.pipeline.load_phase(path)
        self.start_worker(action, lambda _: self.update_loaded(), guard_generation=False)
    def update_loaded(self):
        self.loaded.setText(f'YOLO: {self.pipeline.yolo_path or "unloaded"}\nPhaseGridUNet: {self.pipeline.phase_path or "unloaded"}')
        self.update_action_states()
    def run(self, mode):
        if self.worker is not None: return
        if self.frame is None: QMessageBox.warning(self,'Recognition','Open or capture an image first'); return
        image = self.frame.image; settings = self.settings(); previous = self.result
        if mode == 'geometry' and (previous is None or not any(d.name == 'grid' for d in previous.detections)):
            QMessageBox.warning(self,'Recognition','Run detection and obtain a grid first'); return
        stage = {'yolo': 'detection', 'geometry': 'geometry', 'all': 'all'}[mode]
        errors = list(validation_errors(settings, stage))
        try: device_for(settings.device)
        except (RuntimeError, ValueError) as exc: errors.append(str(exc))
        if errors:
            QMessageBox.warning(self, 'Invalid settings', '\n'.join(errors)); return
        def action():
            start = perf_counter()
            result = self.pipeline.detect(image, settings) if mode in ('all','yolo') else previous
            if mode in ('all','geometry'): result = self.pipeline.geometry(image, result, settings)
            result.timings['job wall ms'] = (perf_counter()-start)*1000
            return result
        self.start_worker(action, lambda result: self.accept_result(result, geometry_update=mode == 'geometry'))
    def accept_result(self, result, geometry_update=False):
        self.canvas.cancel_pointer()
        self.recognition.replace(result, geometry_update=geometry_update, fallback_to_first_grid=True)
        self.rebuild_result_tree()
        self.show_diagnostics(); self.redraw(); self.update_empty_states(); self.update_action_states()

    def rebuild_result_tree(self):
        self.result_tree.blockSignals(True)
        try:
            self.result_tree.clear()
            selected_item = None
            def add_item(parent, data):
                nonlocal selected_item
                item = QTreeWidgetItem(parent, [data.label, data.summary])
                item.setData(0, Qt.ItemDataRole.UserRole, data.identity)
                item.setToolTip(0, data.label); item.setToolTip(1, data.summary)
                if data.identity == self.recognition.selected:
                    selected_item = item
                for child in data.children:
                    add_item(item, child)
            for data in explorer_items(self.recognition.snapshot):
                add_item(self.result_tree, data)
            self.result_tree.expandAll()
            if selected_item is not None:
                self.result_tree.setCurrentItem(selected_item)
                self.result_tree.scrollToItem(selected_item)
        finally:
            self.result_tree.blockSignals(False)

    def selection_changed(self, current, previous):
        identity = current.data(0, Qt.ItemDataRole.UserRole) if current is not None else None
        self.recognition.select(identity)
        self.show_diagnostics(); self.redraw(); self.update_empty_states()

    def viewer_clicked(self, point):
        # Resolve on release against the current snapshot, never an ID saved at press.
        if image_point(self.canvas.transform, point, self.canvas.image_size) is None:
            return
        identity = hit_detection(self.recognition.snapshot, self.canvas.transform, point,
                                 self.canvas.image_size, self.layers().get('detections') and self.vector_opacity.value() > 0)
        self.recognition.select(identity)
        self.sync_result_tree_selection()
        self.show_diagnostics(); self.redraw(); self.update_empty_states()

    def sync_result_tree_selection(self):
        def find(parent):
            for index in range(parent.childCount()):
                item = parent.child(index)
                if item.data(0, Qt.ItemDataRole.UserRole) == self.recognition.selected:
                    return item
                found = find(item)
                if found is not None:
                    return found
            return None
        item = find(self.result_tree.invisibleRootItem()) if self.recognition.selected is not None else None
        self.result_tree.blockSignals(True)
        try:
            self.result_tree.setCurrentItem(item)
            if item is None:
                self.result_tree.clearSelection()
            else:
                parent = item.parent()
                while parent is not None:
                    parent.setExpanded(True)
                    parent = parent.parent()
                self.result_tree.scrollToItem(item)
        finally:
            self.result_tree.blockSignals(False)

    def navigate_result(self, item, column):
        identity = item.data(0, Qt.ItemDataRole.UserRole)
        bounds = selection_region(self.recognition.snapshot, identity)
        if bounds is not None:
            self.canvas.fit_region(bounds)

    def update_inspector(self):
        sections = inspector_sections(self.recognition.snapshot, self.recognition.selected)
        self.inspect.clear()
        for section in sections:
            heading = QTreeWidgetItem(self.inspect, [section.title, ''])
            font = heading.font(0); font.setBold(True); heading.setFont(0, font)
            for prop in section.properties:
                item = QTreeWidgetItem(heading, [prop.name, prop.value])
                item.setData(1, Qt.ItemDataRole.UserRole, prop.value)
                item.setToolTip(0, prop.name); item.setToolTip(1, prop.value)
        self.inspect.expandAll()
        self.inspect.setVisible(bool(sections)); self.inspect_empty.setVisible(not sections)
        self.copy_report_button.setEnabled(bool(sections))

    def copy_inspect_value(self):
        item = self.inspect.currentItem()
        value = item.data(1, Qt.ItemDataRole.UserRole) if item is not None else None
        if value is not None:
            QApplication.clipboard().setText(value)

    def copy_inspect_report(self):
        sections = inspector_sections(self.recognition.snapshot, self.recognition.selected)
        if sections:
            QApplication.clipboard().setText(property_report(sections))

    def inspect_context_menu(self, position):
        item = self.inspect.itemAt(position)
        if item is None or item.data(1, Qt.ItemDataRole.UserRole) is None:
            return
        self.inspect.setCurrentItem(item)
        menu = QMenu(self.inspect)
        menu.addAction('Copy property value', self.copy_inspect_value)
        menu.exec(self.inspect.viewport().mapToGlobal(position))

    def show_diagnostics(self):
        self.update_inspector()
        if not self.frame: return
        details = self.frame.diagnostics
        timing = ''
        if self.result:
            timing = '\n'.join(f'{k}: {v:.1f}' for k,v in self.result.timings.items())
            grid = self.selected_grid()
            if grid:
                details += '\n\nSelected grid:\n' + '\n'.join(f'{k}: {v}' for k,v in grid.diagnostics.items())
                timing += '\n\nSelected grid:\n' + '\n'.join(f'{k}: {v:.1f}' for k,v in grid.timings.items())
        self.diagnostics.setPlainText(details)
        self.performance.setPlainText(timing)
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
        if self.capture_in_progress:
            QMessageBox.information(self, 'Capture', 'Wait for the current capture to finish before closing.'); event.ignore(); return
        if self.worker is not None:
            QMessageBox.information(self, 'Processing', 'Wait for the current job to finish before closing.'); event.ignore(); return
        self.save_workspace()
        QApplication.instance().removeEventFilter(self)
        self.overlay.close(); event.accept()
