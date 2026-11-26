"""Physical boundaries and native layout of desktop shell components."""

from ..scaling import pixels, validate_scale


DOCK_ICON_COUNT = 6


def dock_layout(size, desktop, scale=1.0):
    """Resolve the dock and its icon slots in physical pixels.

    One icon is retained in a screen-clipped dock if even a single padded icon
    cannot fit. Otherwise each slot keeps its resolved native dimensions.
    """
    scale = validate_scale(scale)
    width, height = size
    logical_height = desktop.get('dock_height', 74)
    ratio = logical_height / 74
    dock_height = pixels(logical_height, scale)
    icon = pixels(42 * ratio, scale)
    gap = pixels(12 * ratio, scale)
    padding = pixels(16 * ratio, scale)
    count = min(DOCK_ICON_COUNT, max(0, (width - 2 * padding + gap) // (icon + gap)))
    if count:
        dock_width = 2 * padding + count * icon + (count - 1) * gap
    else:
        count = 1
        dock_width = width
    left = (width - dock_width) // 2
    bottom = height - pixels(4, scale)
    top = bottom - dock_height
    first_icon = left + padding if dock_width >= 2 * padding + icon else left + (dock_width - icon) // 2
    icon_top = top + (dock_height - icon) // 2
    icons = [(first_icon + i * (icon + gap), icon_top,
              first_icon + i * (icon + gap) + icon, icon_top + icon)
             for i in range(count)]
    return {'box': (left, top, left + dock_width, bottom), 'icons': icons,
            'icon_size': icon, 'gap': gap, 'padding': padding, 'ratio': ratio}


def component_geometry(size, desktop, scale=1.0):
    scale = validate_scale(scale)
    width, height = size
    shell = desktop['shell']
    if shell == 'taskbar':
        if not desktop.get('show_taskbar', True):
            return {}
        extent = pixels(36 if desktop['style'] == 'classic' else 48, scale)
        return {'taskbar': (0, height - extent, width, height)}
    if shell == 'top_panel':
        if not desktop.get('show_top_panel', True):
            return {}
        return {'top_panel': (0, 0, width, pixels(36, scale))}
    if shell == 'dock':
        geometry = {}
        if desktop.get('show_status_bar', True):
            geometry['status_bar'] = (0, 0, width, pixels(32, scale))
        if desktop.get('show_dock', True):
            geometry['dock'] = dock_layout(size, desktop, scale)['box']
        return geometry
    raise ValueError(f'Unknown shell: {shell}')
