"""Load paired synthetic-scenes geometry PNGs and JSON annotations."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL.Image import DecompressionBombError
import torch
from torch.utils.data import Dataset

from .geometry import FloatArray, Geometry, Size, coordinate_array
from .heatmaps import DEFAULT_SIGMA, geometry_heatmaps, validate_sigma
from .preprocessing import (
    DEFAULT_CANVAS_SIZE,
    ImageTransform,
    load_rgb_image,
    preprocess_image,
    transform_geometry,
    validate_canvas_size,
)


@dataclass(frozen=True, eq=False)
class GeometrySample:
    """A prepared sample with metadata for inspection and coordinate recovery."""

    image: torch.Tensor
    heatmaps: torch.Tensor
    image_path: Path
    board_id: str
    style: str
    game_state_id: str | None
    source_crop: FloatArray
    source_geometry: Geometry
    geometry: Geometry
    transform: ImageTransform

    @property
    def sample_id(self) -> str:
        return self.image_path.stem


class GeometryDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read one train or val split from an exported dataset's geometry directory.

    Indexing returns two batchable tensors. Use load_sample for the geometry
    and identity of an individual board.
    """

    def __init__(
        self,
        geometry_root: str | Path,
        split: str,
        *,
        canvas_size: Size = DEFAULT_CANVAS_SIZE,
        sigma: float = DEFAULT_SIGMA,
        include_outer: bool = True,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self._initialize(Path(geometry_root) / split, canvas_size, sigma, include_outer)

    @classmethod
    def from_split_directory(
        cls, folder: str | Path, *, canvas_size: Size = DEFAULT_CANVAS_SIZE,
        sigma: float = DEFAULT_SIGMA, include_outer: bool = True,
    ) -> "GeometryDataset":
        """Read a named evaluation directory with the same exported PNG/JSON schema."""
        dataset = cls.__new__(cls)
        dataset._initialize(Path(folder), canvas_size, sigma, include_outer)
        return dataset

    def _initialize(self, folder: Path, canvas_size: Size, sigma: float, include_outer: bool) -> None:
        self.canvas_size = validate_canvas_size(canvas_size)
        self.sigma = validate_sigma(sigma)
        self.include_outer = include_outer
        if not folder.is_dir():
            raise FileNotFoundError(f"Geometry split directory does not exist: {folder}")

        images = {path.stem: path for path in folder.glob("*.png")}
        annotations = {path.stem: path for path in folder.glob("*.json")}
        if images.keys() != annotations.keys():
            missing_json = sorted(images.keys() - annotations.keys())
            missing_png = sorted(annotations.keys() - images.keys())
            raise ValueError(
                f"Unpaired geometry files in {folder}. "
                f"Missing JSON for {missing_json}; missing PNG for {missing_png}"
            )
        if not images:
            raise ValueError(f"No geometry samples in {folder}")
        self._pairs = [(images[stem], annotations[stem]) for stem in sorted(images)]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.load_sample(index)
        return sample.image, sample.heatmaps

    def load_sample(self, index: int) -> GeometrySample:
        image_path, annotation_path = self._pairs[index]
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if not isinstance(annotation, dict):
                raise ValueError("annotation must be a JSON object")
            source_geometry = Geometry(
                size=annotation["size"],
                rows=annotation["rows"],
                columns=annotation["columns"],
                grid_box=annotation["grid_box"],
                x_edges=annotation["x_edges"],
                y_edges=annotation["y_edges"],
                cell_boxes=annotation["cell_boxes"],
                centers=annotation["centers"],
            )
            board_id = _annotation_text(annotation, "board_id")
            style = _annotation_text(annotation, "style")
            game_state_id = (
                _annotation_text(annotation, "game_state_id")
                if "game_state_id" in annotation else None
            )
            source_crop = coordinate_array(annotation["source_crop"], name="source_crop")
            if source_crop.shape != (4,):
                raise ValueError("source_crop must be an xyxy box")
            left, top, right, bottom = source_crop
            if not (0 <= left < right and 0 <= top < bottom):
                raise ValueError("source_crop must have nonnegative origin and positive extents")

            image = load_rgb_image(image_path)
            if image.size != source_geometry.size:
                raise ValueError(f"PNG size {image.size} does not match JSON size {source_geometry.size}")
            canvas, transform = preprocess_image(image, self.canvas_size)
            geometry = transform_geometry(source_geometry, transform)
        except KeyError as exc:
            raise ValueError(f"{annotation_path}: missing required field {exc.args[0]!r}") from exc
        except (OSError, ValueError, DecompressionBombError) as exc:
            raise ValueError(f"{annotation_path} (image {image_path}): {exc}") from exc

        pixels = np.asarray(canvas)
        chw = np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32)
        chw /= 255.0
        heatmaps = geometry_heatmaps(geometry, sigma=self.sigma, include_outer=self.include_outer)
        return GeometrySample(
            image=torch.from_numpy(chw),
            heatmaps=torch.from_numpy(heatmaps),
            image_path=image_path,
            board_id=board_id,
            style=style,
            game_state_id=game_state_id,
            source_crop=source_crop,
            source_geometry=source_geometry,
            geometry=geometry,
            transform=transform,
        )


def _annotation_text(annotation: dict, name: str) -> str:
    value = annotation[name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value
