"""Visualize saved detection samples and annotations."""
import random
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from grid_detection.dataset.yolo import YOLO_CLASSES

def visualize_samples(records, destination, seed=42):
    rng = random.Random(seed)
    selected = {}
    def choose(label, record):
        if record is None:
            print(f"Unavailable: {label}")
        else:
            selected.setdefault(record["image"], {"record": record, "reasons": []})["reasons"].append(label)

    def box_areas(record):
        size = record.get("final_size")
        if not size or not size[0] or not size[1]: return []
        return [(b["final_grid_box"][2]-b["final_grid_box"][0]) *
                (b["final_grid_box"][3]-b["final_grid_box"][1]) / (size[0]*size[1])
                for b in record.get("boards", []) if b.get("final_grid_box")]

    def fitted(record):
        return any(b.get("sampled_cell_size") is not None and b.get("resolved_cell_size") is not None
                   and b["sampled_cell_size"] > b["resolved_cell_size"] for b in record.get("boards", []))

    for record in rng.sample(records, min(3, len(records))): choose("random", record)
    for mode in ("desktop", "game", "grid"):
        choose(mode, next((r for r in records if r.get("scene_mode") == mode), None))
    choose("negative", next((r for r in records if not r.get("board_ids")), None))
    choose("multi-board", next((r for r in records if len(r.get("board_ids", [])) > 1), None))
    choose("fitted board", next((r for r in records if fitted(r)), None))
    scaled = [r for r in records if (r.get("desktop") or {}).get("screen", {}).get("scale") is not None]
    choose("highest desktop scale", max(scaled, key=lambda r: r["desktop"]["screen"]["scale"]) if scaled else None)
    with_boxes = [r for r in records if box_areas(r)]
    choose("smallest grid box", min(with_boxes, key=lambda r: min(box_areas(r))) if with_boxes else None)
    choose("largest grid box", max(with_boxes, key=lambda r: max(box_areas(r))) if with_boxes else None)

    items = list(selected.values())
    for start in range(0, len(items), 4):
        batch = items[start:start+4]
        fig, axes = plt.subplots(1, len(batch), figsize=(5*len(batch), 4), squeeze=False)
        for ax, item in zip(axes.flat, batch):
            record = item["record"]
            with Image.open(destination / record["image"]) as image:
                ax.imshow(image)
                width, height = image.size
            classes_shown = set()
            for line in (destination / record["label"]).read_text().splitlines():
                class_id, x, y, w, h = map(float, line.split())
                class_id = int(class_id)
                left, top = (x-w/2)*width, (y-h/2)*height
                color = f"C{class_id}"
                ax.add_patch(Rectangle((left, top), w*width, h*height, fill=False, edgecolor=color, linewidth=1.5))
                classes_shown.add(class_id)
            if classes_shown:
                from matplotlib.patches import Patch
                ax.legend(handles=[Patch(facecolor="none", edgecolor=f"C{i}", label=YOLO_CLASSES[i])
                                   for i in sorted(classes_shown)], fontsize=6, loc="lower right")
            scale = (record.get("desktop") or {}).get("screen", {}).get("scale")
            scale_text = f" | desktop scale={scale:g}" if scale is not None else ""
            ax.set_title(f"{', '.join(item['reasons'])}\n{record['split']}/{Path(record['image']).name} | {record.get('scene_mode')} | boards={len(record.get('board_ids', []))}{scale_text}", fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        plt.show()
