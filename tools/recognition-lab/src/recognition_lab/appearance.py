"""Native system appearance and controlled explicit Qt palettes."""
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def apply_appearance(appearance, system_style):
    app = QApplication.instance()
    if appearance == 'system':
        # Clear our palette override and let Qt follow the platform color scheme.
        unset_scheme = getattr(app.styleHints(), 'unsetColorScheme', None)
        if unset_scheme is not None:
            unset_scheme()
        app.setStyle(system_style)
        app.setPalette(QPalette())
        return

    # Fusion has predictable palette support across platforms, unlike some native styles.
    app.setStyle('Fusion')
    palette = app.style().standardPalette()
    if appearance == 'dark':
        colors = {'Window': '#353535', 'WindowText': '#f0f0f0', 'Base': '#232323',
                  'AlternateBase': '#353535', 'Text': '#f0f0f0', 'PlaceholderText': '#aaaaaa',
                  'Button': '#454545', 'ButtonText': '#f0f0f0', 'BrightText': '#ffffff',
                  'Highlight': '#3678b8', 'HighlightedText': '#ffffff',
                  'ToolTipBase': '#353535', 'ToolTipText': '#ffffff'}
        disabled = '#888888'
    else:
        colors = {'Window': '#efefef', 'WindowText': '#202020', 'Base': '#ffffff',
                  'AlternateBase': '#f3f3f3', 'Text': '#202020', 'PlaceholderText': '#737373',
                  'Button': '#efefef', 'ButtonText': '#202020', 'BrightText': '#ffffff',
                  'Highlight': '#3678b8', 'HighlightedText': '#ffffff',
                  'ToolTipBase': '#ffffff', 'ToolTipText': '#202020'}
        disabled = '#858585'
    for role, color in colors.items():
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText, QPalette.ColorRole.PlaceholderText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(disabled))
    app.setPalette(palette)
