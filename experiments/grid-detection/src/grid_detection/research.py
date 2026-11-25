"""Paths and visualization helpers shared by the detector notebooks."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"


def dataset_yaml(name: str) -> Path:
    """Select a finished dataset produced by the detector generator."""
    root = DATA_DIR / name
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    yaml_path = root / "data.yaml"
    if config["status"] != "complete" or not yaml_path.is_file():
        raise ValueError(f"Expected a complete YOLO dataset at {root}")
    return yaml_path


def show_yolo_examples(root: Path, split: str, names: tuple[str, ...], count: int = 3):
    """Preview exported labels, including an empty label file if sampled."""
    images = sorted((root / "images" / split).glob("*.png"))
    if not images:
        return
    count = min(count, len(images))
    step = len(images) / count
    fig, axes = plt.subplots(1, count, figsize=(6 * count, 4), squeeze=False)
    for index, ax in enumerate(axes[0]):
        path = images[int(index * step)]
        with Image.open(path) as image:
            ax.imshow(image)
            width, height = image.size
        labels = root / "labels" / split / f"{path.stem}.txt"
        for label in labels.read_text(encoding="utf-8").splitlines():
            class_id, *coordinates = label.split()
            cx, cy, w, h = map(float, coordinates)
            x = (cx - w / 2) * width
            y = (cy - h / 2) * height
            color = f"C{class_id}"
            ax.add_patch(Rectangle((x, y), w * width, h * height, fill=False, edgecolor=color))
            ax.text(x, y, names[int(class_id)], color="white", backgroundcolor=color, fontsize=8)
        ax.set_title(f"{split} / {path.name}")
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def grid_crops(result, image: Image.Image, margin: int):
    """Yield pixel-aligned crops with original screenshot coordinates."""
    if margin < 0:
        raise ValueError("Crop margin must be nonnegative")
    width, height = image.size
    for box in result.boxes:
        if result.names[int(box.cls.item())] != "grid":
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        crop_xyxy = (
            max(0, math.floor(x1) - margin),
            max(0, math.floor(y1) - margin),
            min(width, math.ceil(x2) + margin),
            min(height, math.ceil(y2) + margin),
        )
        yield {
            "grid_xyxy": (x1, y1, x2, y2),
            "confidence": box.conf.item(),
            "crop_xyxy": crop_xyxy,
            "crop_origin": crop_xyxy[:2],
            "image": image.crop(crop_xyxy),
        }
