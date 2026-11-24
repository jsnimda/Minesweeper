"""Portable prediction exports and bounded evaluation preview panels."""

from dataclasses import dataclass
import json
from pathlib import Path
from textwrap import wrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .evaluation_metrics import BoardDecoding, BoardEstimate, BoardTruth, SIDES
from .geometry import Geometry
from .preprocessing import ImageTransform
from .validation import ValidationResult


@dataclass(frozen=True, eq=False)
class SampleRecord:
    index: int
    sample_id: str
    board_id: str
    game_state_id: str | None
    style: str
    geometry: Geometry
    transform: ImageTransform
    source_crop: tuple[float, ...] | None
    preprocessing_ms: float

    def coordinates(self) -> dict:
        transform = self.transform
        return {
            "input_size": transform.canvas_size,
            "source_size": transform.source_size,
            "resized_size": transform.resized_size,
            "padding_offset_xy": transform.offset,
            "scale_xy": (transform.scale_x, transform.scale_y),
            "source_scene_crop_xyxy": self.source_crop,
        }


def _crop_axis(values, transform: ImageTransform, *, horizontal: bool):
    index = 0 if horizontal else 1
    scale = transform.scale_x if horizontal else transform.scale_y
    return (values - transform.offset[index]) / scale


def _scene_axis(values, sample: SampleRecord, *, horizontal: bool):
    x0, y0, x1, y1 = sample.source_crop
    width, height = sample.transform.source_size
    return x0 + values * (x1 - x0) / width if horizontal else y0 + values * (y1 - y0) / height


def hypothesis_record(estimate: BoardEstimate, sample: SampleRecord) -> dict:
    transform = sample.transform
    x, y = estimate.x_edges, estimate.y_edges
    crop_x = _crop_axis(x, transform, horizontal=True)
    crop_y = _crop_axis(y, transform, horizontal=False)

    def coordinates(x_edges, y_edges):
        return {
            "ideal_x_edges": x_edges.tolist(),
            "ideal_y_edges": y_edges.tolist(),
            "center_x": ((x_edges[1:] + x_edges[:-1]) / 2).tolist(),
            "center_y": ((y_edges[1:] + y_edges[:-1]) / 2).tolist(),
        }

    edges = {}
    for index, (side, edge) in enumerate(zip(SIDES, estimate.outer_edges, strict=True)):
        if edge is None:
            edges[side] = {"state": "unknown"}
            continue
        horizontal = not index % 2
        crop_position = _crop_axis(edge.position, transform, horizontal=horizontal)
        edges[side] = {
            "state": "censored" if edge.censored else "estimated",
            "input_observation": edge.position,
            "crop_observation": crop_position,
            "strength": edge.strength,
        }
        if sample.source_crop is not None:
            edges[side]["scene_observation"] = _scene_axis(crop_position, sample, horizontal=horizontal)

    record = {
        "rows": estimate.rows, "columns": estimate.columns, "score": estimate.score,
        "input": coordinates(x, y), "source_crop": coordinates(crop_x, crop_y), "outer_edges": edges,
    }
    if sample.source_crop is not None:
        record["source_scene"] = coordinates(
            _scene_axis(crop_x, sample, horizontal=True), _scene_axis(crop_y, sample, horizontal=False),
        )
    return record


def export_prediction(
    folder: Path, sample: SampleRecord, truth: BoardTruth, probabilities: np.ndarray,
    decodings: dict[str, BoardDecoding], *, save_heatmaps: bool, save_hypotheses: bool,
    validations: dict[str, ValidationResult] | None = None, observation_complete: bool | None = None,
) -> dict:
    stem = f"{sample.index:06d}"
    record = {
        "index": sample.index,
        "sample_id": sample.sample_id,
        "board_id": sample.board_id,
        "game_state_id": sample.game_state_id,
        "style": sample.style,
        "coordinates": sample.coordinates(),
        "truth_input": {
            "rows": truth.rows, "columns": truth.columns, "pitch_x": truth.pitch_x, "pitch_y": truth.pitch_y,
            "origin_x": truth.origin_x, "origin_y": truth.origin_y, "visible_extent_xyxy": truth.visible_extent,
            "center_x": truth.centers[0, :, 0].tolist(), "center_y": truth.centers[:, 0, 1].tolist(),
        },
    }
    if observation_complete is not None:
        record["observation_complete"] = observation_complete
    if validations:
        record["validation"] = {method: result.to_dict() for method, result in validations.items()}
    if save_heatmaps:
        filename = f"{stem}.npz"
        np.savez_compressed(folder / filename, probabilities=np.asarray(probabilities, dtype=np.float32))
        record["probabilities"] = filename
    if save_hypotheses:
        filename = f"{stem}.json"
        hypotheses = {
            method: {
                "status": decoding.status, "axis_status_xy": decoding.axis_status,
                "hypotheses": [hypothesis_record(estimate, sample) for estimate in decoding.hypotheses],
            }
            for method, decoding in decodings.items()
        }
        for method, result in (validations or {}).items():
            hypotheses[method]["validation"] = result.to_dict(include_support=True)
        (folder / filename).write_text(
            json.dumps(hypotheses, allow_nan=False, indent=2) + "\n", encoding="utf-8",
        )
        record["hypotheses"] = filename
    return record


@dataclass(frozen=True, eq=False)
class _Preview:
    sample: SampleRecord
    image: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray
    decodings: dict[str, BoardDecoding]
    validations: dict[str, ValidationResult]
    outcome: str


class PreviewSelection:
    """Keep examples of distinct outcomes before filling spare slots by style."""

    def __init__(self, limit: int):
        self.limit = limit
        self.examples: dict[tuple[str, str], _Preview] = {}
        self.outcomes: set[str] = set()

    def consider(
        self, sample: SampleRecord, image: np.ndarray, targets: np.ndarray, probabilities: np.ndarray,
        decodings: dict[str, BoardDecoding], outcome: str, *, validations: dict[str, ValidationResult] | None = None,
    ) -> None:
        key = (outcome, sample.style)
        if not self.limit or key in self.examples or (outcome in self.outcomes and len(self.examples) >= self.limit):
            return
        self.examples[key] = _Preview(
            sample, image.copy(), targets.copy(), probabilities.copy(), decodings, validations or {}, outcome,
        )
        self.outcomes.add(outcome)

    def write(self, folder: Path) -> list[str]:
        if not self.examples:
            return []
        folder.mkdir()
        # Prefer distinct failures, success, then further examples by style.
        ordered = sorted(self.examples.values(), key=lambda entry: entry.outcome in ("correct", "heatmaps_only"))
        first = {}
        extras = []
        for entry in ordered:
            if entry.outcome in first:
                extras.append(entry)
            else:
                first[entry.outcome] = entry
        files = []
        for entry in [*first.values(), *extras][:self.limit]:
            filename = f"{entry.sample.index:06d}-{entry.outcome}.png"
            _render_preview(entry).save(folder / filename)
            files.append(filename)
        return files


def _draw_grid(image: Image.Image, x_edges, y_edges, *, color) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for x in x_edges:
        draw.line((x - 0.5, y_edges[0] - 0.5, x - 0.5, y_edges[-1] - 0.5), fill=color)
    for y in y_edges:
        draw.line((x_edges[0] - 0.5, y - 0.5, x_edges[-1] - 0.5, y - 0.5), fill=color)
    return overlay


def _draw_prediction(image: Image.Image, estimate: BoardEstimate) -> Image.Image:
    color = (44, 215, 255, 225)
    overlay = _draw_grid(image, estimate.x_edges, estimate.y_edges, color=color)
    draw = ImageDraw.Draw(overlay, "RGBA")
    left, right = estimate.x_edges[[0, -1]]
    top, bottom = estimate.y_edges[[0, -1]]
    for i, edge in enumerate(estimate.outer_edges):
        if edge is None:
            continue
        position = edge.position - 0.5
        if edge.censored:
            x, y = ((left + right) / 2, position) if i % 2 else (position, (top + bottom) / 2)
            draw.rectangle((x - 2, y - 2, x + 2, y + 2), fill="#ffa347")
        else:
            points = (left, position, right, position) if i % 2 else (position, top, position, bottom)
            draw.line(points, fill=(122, 255, 144, 230), width=2)
    return overlay


def draw_validation_support(image: Image.Image, estimate: BoardEstimate, result: ValidationResult) -> Image.Image:
    """Overlay supported centers and segments on a canvas in input coordinates."""
    overlay = image.copy()
    if result.centers is None:
        return overlay
    draw = ImageDraw.Draw(overlay, "RGBA")
    x_edges, y_edges = estimate.x_edges - 0.5, estimate.y_edges - 0.5
    supported, missing = (67, 216, 201, 180), (255, 91, 104, 245)
    for row, col in np.ndindex(result.vertical.supported.shape):
        x = x_edges[col + 1]
        ok = result.vertical.supported[row, col]
        draw.line((x, y_edges[row], x, y_edges[row + 1]), fill=supported if ok else missing, width=1 if ok else 3)
    for row, col in np.ndindex(result.horizontal.supported.shape):
        y = y_edges[row + 1]
        ok = result.horizontal.supported[row, col]
        draw.line((x_edges[col], y, x_edges[col + 1], y), fill=supported if ok else missing, width=1 if ok else 3)
    for side, segments in result.outer.items():
        offset = result.measurements["outer_edges"][side]["position"] - 0.5
        for i, ok in enumerate(segments.supported):
            if side in ("left", "right"):
                line = (offset, y_edges[i], offset, y_edges[i + 1])
            else:
                line = (x_edges[i], offset, x_edges[i + 1], offset)
            draw.line(line, fill=supported if ok else missing, width=2 if ok else 3)
    for row, col in np.ndindex(result.centers.supported.shape):
        x, y = (x_edges[col] + x_edges[col + 1]) / 2, (y_edges[row] + y_edges[row + 1]) / 2
        ok = result.centers.supported[row, col]
        radius = 2 if ok else 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=supported if ok else missing)
    return overlay


def _validation_panel(image, decoding, result) -> Image.Image:
    metrics = result.measurements
    lines = []
    if metrics:
        coverage = []
        for name in ("center", "vertical", "horizontal"):
            value = metrics[f"{name}_coverage"]
            coverage.append("n/a" if value is None else f"{value:.0%}")
        lines.extend([
            "Center / vertical / horizontal",
            " / ".join(coverage),
            f"Largest missing cluster {metrics['largest_missing_center_cluster']}",
            f"Confirmed outer edges {metrics['confirmed_outer_edges']}/4",
        ])
    if result.reasons:
        reasons = ", ".join(reason.replace("_", " ") for reason in result.reasons)
        lines.extend(wrap(reasons, width=39))
    height = max(180, 16 + 15 * len(lines))
    panel = Image.new("RGB", (464, height), "#132335")
    if decoding.hypotheses:
        overlay = draw_validation_support(image, decoding.hypotheses[0], result)
        preview = ImageOps.contain(overlay, (198, 180))
        panel.paste(preview, ((198 - preview.width) // 2, (height - preview.height) // 2))
    draw = ImageDraw.Draw(panel)
    draw.multiline_text((207, 4), "\n".join(lines), fill="#c9dce9", font=ImageFont.load_default(size=12), spacing=2)
    return panel


def _render_preview(entry: _Preview) -> Image.Image:
    image = Image.fromarray(np.rint(entry.image.transpose(1, 2, 0) * 255).astype(np.uint8))
    geometry = entry.sample.geometry
    truth = _draw_grid(image, geometry.x_edges, geometry.y_edges, color=(205, 129, 255, 210))
    overlays = [("Model input", image), (f"Annotated {geometry.rows} x {geometry.columns}", truth)]
    for method, decoding in entry.decodings.items():
        best = decoding.hypotheses[0] if decoding.hypotheses else None
        title = f"{method}: {best.rows} x {best.columns} / {decoding.status}" if best else f"{method}: {decoding.status}"
        overlays.append((title, _draw_prediction(image, best) if best else image))

    rows = [overlays]
    validation_row = []
    if entry.validations:
        validation_row = [
            (f"{method} strict {result.status}", _validation_panel(image, entry.decodings[method], result))
            for method, result in entry.validations.items()
        ]
        rows.append(validation_row)
    for label, heatmaps in (("Target", entry.targets), ("Predicted", entry.probabilities)):
        panels = []
        for channel, name in zip(heatmaps, ("centers", "verticals", "horizontals"), strict=True):
            pixels = np.rint(channel * 255).astype(np.uint8)
            panels.append((f"{label} {name}", Image.fromarray(pixels).convert("RGB")))
        rows.append(panels)

    font = ImageFont.load_default(size=13)
    width, row_height, header = 1008, 210, 48
    heights = [
        max(row_height, max(panel.height + 24 for _, panel in panels)) if panels is validation_row else row_height
        for panels in rows
    ]
    canvas = Image.new("RGB", (width, header + sum(heights)), "#132335")
    draw = ImageDraw.Draw(canvas)
    title = f"{entry.sample.sample_id}  |  {entry.sample.style}  |  {entry.outcome}"
    draw.text((8, 8), title, fill="white", font=font)
    legend = "Purple annotations   Cyan lattice   Green visible edge   Orange censored endpoint   Red unsupported evidence"
    draw.text((8, 26), legend, fill="#b7d0e2", font=font)
    y = header
    for panels, height in zip(rows, heights, strict=True):
        panel_width = width // len(panels)
        for column, (label, panel) in enumerate(panels):
            x = column * panel_width
            draw.text((x + 8, y + 3), label, fill="white", font=font)
            preview = ImageOps.contain(panel, (panel_width - 12, height - 24))
            canvas.paste(preview, (x + (panel_width - preview.width) // 2, y + 21))
        y += height
    return canvas
