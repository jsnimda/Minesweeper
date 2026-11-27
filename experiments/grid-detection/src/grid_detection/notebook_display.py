"""Small HTML tables for notebook inspection of saved reports."""
from html import escape
from IPython.display import HTML, display


def show_table(title, columns, rows):
    def cell(value):
        if value is None:
            return "unavailable"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)
    head = "".join(f"<th>{escape(str(c))}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{escape(cell(v))}</td>" for v in row) + "</tr>" for row in rows)
    display(HTML(f"<strong>{escape(title)}</strong><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"))


def show_mapping(title, mapping):
    rows = []
    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, path + [str(key)])
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + [str(index)])
        else:
            rows.append((" / ".join(path), value))
    visit(mapping, [])
    show_table(title, ("Field", "Value"), rows)
