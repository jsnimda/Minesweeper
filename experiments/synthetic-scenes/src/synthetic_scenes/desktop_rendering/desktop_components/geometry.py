"""Physical boundaries of the existing desktop shell components."""

from ..scaling import pixels, validate_scale


def component_geometry(size, desktop, scale=1.0):
    scale = validate_scale(scale)
    width, height = size
    shell = desktop["shell"]
    if shell == "taskbar":
        extent = pixels(36 if desktop["style"] == "classic" else 48, scale)
        return {"taskbar": (0, height - extent, width, height)}
    if shell == "top_panel":
        extent = pixels(36, scale)
        return {"top_panel": (0, 0, width, extent)}
    if shell == "dock":
        menu = pixels(32, scale)
        dock_width = pixels(6 * 42 + 5 * 12 + 2 * 16, scale)
        dock_height = pixels(74, scale)
        bottom = height - pixels(16, scale)
        left = (width - dock_width) // 2
        return {"status_bar": (0, 0, width, menu),
                "dock": (left, bottom - dock_height, left + dock_width, bottom)}
    raise ValueError(f"Unknown shell: {shell}")
