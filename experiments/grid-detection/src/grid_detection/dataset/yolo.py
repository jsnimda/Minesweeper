
"""Convert synthetic screenshot annotations into a YOLO dataset."""

import json
import random
import shutil
from pathlib import Path


CLASS_NAMES = (
    "game_region",
    "grid",
    "mine_counter",
    "reset_button",
    "timer",
)

CLASS_IDS = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


def _to_yolo_box(
    box: list[int] | tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert an xyxy pixel box into normalized YOLO xywh."""
    if len(box) != 4:
        raise ValueError(f"Invalid bounding box: {box}")

    x0, y0, x1, y1 = box

    if not (
        0 <= x0 < x1 <= image_width
        and 0 <= y0 < y1 <= image_height
    ):
        raise ValueError(f"Out-of-bounds bounding box: {box}")

    return (
        (x0 + x1) / (2 * image_width),
        (y0 + y1) / (2 * image_height),
        (x1 - x0) / image_width,
        (y1 - y0) / image_height,
    )


def _annotation_lines(record: dict) -> list[str]:
    """Convert every supported annotation into a YOLO label."""
    width = record["width"]
    height = record["height"]

    lines = []

    # Annotations are grouped by source window, so multiple
    # Minesweeper games remain separate instances.
    for window_boxes in record["annotations"]:
        for name in CLASS_NAMES:
            if name not in window_boxes:
                continue

            cx, cy, w, h = _to_yolo_box(
                window_boxes[name],
                width,
                height,
            )

            lines.append(
                f"{CLASS_IDS[name]} "
                f"{cx:.8f} {cy:.8f} "
                f"{w:.8f} {h:.8f}"
            )

    return lines


def export_yolo_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 42,
) -> dict:
    """Export a five-class YOLO detection dataset.

    Split by screenshot, keeping all game instances and their
    annotations in the same train/val/test partition.

    The source dataset is not modified.
    """
    if (
        val_ratio < 0
        or test_ratio < 0
        or val_ratio + test_ratio >= 1
    ):
        raise ValueError("Invalid validation/test ratios")

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    source_manifest = json.loads(
        (source_dir / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    samples = list(source_manifest["accepted"])

    if not samples:
        raise ValueError("Source dataset contains no samples")

    # Never silently overwrite an existing exported dataset.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}"
        )

    rng = random.Random(split_seed)
    rng.shuffle(samples)

    total = len(samples)
    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)
    n_train = total - n_val - n_test

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    for split in splits:
        (output_dir / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )
        (output_dir / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

    exported = []

    for split, entries in splits.items():
        for sample in entries:
            image_path = source_dir / sample["image"]
            annotation_path = (
                source_dir / sample["annotation"]
            )

            record = json.loads(
                annotation_path.read_text(encoding="utf-8")
            )

            if record["seed"] != sample["seed"]:
                raise ValueError(
                    f"Seed mismatch in {annotation_path}"
                )

            if record["image"] != sample["image"]:
                raise ValueError(
                    f"Image mismatch in {annotation_path}"
                )

            label_lines = _annotation_lines(record)

            image_name = image_path.name
            label_name = image_path.stem + ".txt"

            target_image = (
                output_dir / "images" / split / image_name
            )
            target_label = (
                output_dir / "labels" / split / label_name
            )

            shutil.copy2(image_path, target_image)

            # Empty files are valid negative YOLO examples.
            target_label.write_text(
                "\n".join(label_lines)
                + ("\n" if label_lines else ""),
                encoding="utf-8",
            )

            exported.append({
                "seed": sample["seed"],
                "split": split,
                "image": f"images/{split}/{image_name}",
                "label": f"labels/{split}/{label_name}",
                "object_count": len(label_lines),
            })

    # JSON-quoted strings are valid YAML scalars.
    # Use an absolute path so Ultralytics resolves it reliably.
    yaml_lines = [
        f"path: {json.dumps(str(output_dir.resolve()))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]

    for class_id, name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {class_id}: {name}")

    (output_dir / "data.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )

    export_manifest = {
        "source_dir": str(source_dir.resolve()),
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "class_names": list(CLASS_NAMES),
        "counts": {
            name: len(entries)
            for name, entries in splits.items()
        },
        "samples": exported,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(export_manifest, indent=2),
        encoding="utf-8",
    )

    return export_manifest
