"""Application appearance and workspace preferences."""
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QVBoxLayout)


DEFAULTS = {'appearance': 'system', 'toolbar': 'both',
            'restore_workspace': True, 'tooltips': True}


class PreferencesDialog(QDialog):
    def __init__(self, preferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Preferences')
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.appearance = QComboBox()
        for label, value in [('System', 'system'), ('Light', 'light'), ('Dark', 'dark')]:
            self.appearance.addItem(label, value)
        self.toolbar = QComboBox()
        for label, value in [('Icons and text', 'both'), ('Icons only', 'icons'), ('Text only', 'text')]:
            self.toolbar.addItem(label, value)
        self.restore = QCheckBox('Restore window and dock arrangement on startup')
        self.tooltips = QCheckBox('Show command tooltips')
        form.addRow('Appearance', self.appearance)
        form.addRow('Toolbar', self.toolbar)
        form.addRow(self.restore)
        form.addRow(self.tooltips)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel |
                                   QDialogButtonBox.StandardButton.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self.restore_defaults)
        layout.addWidget(buttons)
        self.set_preferences(preferences)

    def set_preferences(self, preferences):
        self.appearance.setCurrentIndex(self.appearance.findData(preferences['appearance']))
        self.toolbar.setCurrentIndex(self.toolbar.findData(preferences['toolbar']))
        self.restore.setChecked(preferences['restore_workspace'])
        self.tooltips.setChecked(preferences['tooltips'])

    def restore_defaults(self):
        self.set_preferences(DEFAULTS)

    def values(self):
        return {'appearance': self.appearance.currentData(), 'toolbar': self.toolbar.currentData(),
                'restore_workspace': self.restore.isChecked(), 'tooltips': self.tooltips.isChecked()}
