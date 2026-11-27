"""Visual diagnostics for saved detection runs."""
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from ultralytics import YOLO
from grid_detection.dataset.yolo import YOLO_CLASSES

def visualize_predictions(saved_records, saved_root, saved_args, best_path, device, DIAGNOSTIC_SPLIT, CONFIDENCE, MATCH_IOU, FAILURE_SCAN_LIMIT, RANDOM_EXAMPLES):
    if not 0 <= CONFIDENCE <= 1 or not 0 <= MATCH_IOU <= 1 or FAILURE_SCAN_LIMIT < 0:
        raise ValueError("Invalid diagnostic thresholds or scan limit")
    if not best_path.is_file():
        raise FileNotFoundError(best_path)
    examples = [r for r in saved_records if r["split"] == DIAGNOSTIC_SPLIT]
    rng = random.Random(42)
    shuffled = rng.sample(examples, len(examples))
    scan = shuffled[:FAILURE_SCAN_LIMIT]
    print(f"Failure selection scans {len(scan)} of {len(examples)} {DIAGNOSTIC_SPLIT} images (seed 42). Confidence={CONFIDENCE}; matching IoU={MATCH_IOU}.")

    def labels_for(record):
        rows = []
        for line in (saved_root / record["label"]).read_text().splitlines():
            cls, x, y, w, h = map(float, line.split())
            rows.append((int(cls), (x-w/2, y-h/2, x+w/2, y+h/2)))
        return rows

    def iou(a, b):
        intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
        return intersection / union if union > 0 else 0.0

    def failures(gt, pred):
        pairs = sorted(((iou(gbox, pbox), gi, pi) for gi, (gc, gbox) in enumerate(gt)
                        for pi, (pc, pbox, _) in enumerate(pred) if gc == pc), reverse=True)
        matched_g, matched_p = set(), set()
        for overlap, gi, pi in pairs:
            if overlap >= MATCH_IOU and gi not in matched_g and pi not in matched_p:
                matched_g.add(gi)
                matched_p.add(pi)
        return len(gt)-len(matched_g), len(pred)-len(matched_p)

    predictor = YOLO(str(best_path))
    prediction_cache = {}
    def predictions_for(record):
        key = record["image"]
        if key not in prediction_cache:
            result = predictor.predict(source=str(saved_root / key), imgsz=saved_args["imgsz"], device=device,
                                       conf=CONFIDENCE, verbose=False)[0]
            prediction_cache[key] = [(int(c), tuple(coords), float(score)) for c, coords, score in
                                     zip(result.boxes.cls.cpu().tolist(), result.boxes.xyxyn.cpu().tolist(), result.boxes.conf.cpu().tolist())]
        return prediction_cache[key]

    selected = {}
    def select(record, reason):
        if record is not None:
            selected.setdefault(record["image"], {"record": record, "reasons": []})["reasons"].append(reason)
    for record in shuffled[:RANDOM_EXAMPLES]:
        select(record, "random")
    select(next((r for r in shuffled if not r["board_ids"]), None), "negative")
    select(next((r for r in shuffled if len(r["board_ids"]) > 1), None), "multi-board")
    small = []
    for record in examples:
        areas = [(b[2]-b[0])*(b[3]-b[1]) for cls, b in labels_for(record) if YOLO_CLASSES[cls] == "grid"]
        if areas:
            small.append((min(areas), record["image"], record))
    if small:
        select(min(small, key=lambda item: (item[0], item[1]))[2], "small grid (relative area)")
    missed = false_positive = None
    for record in scan:
        miss_count, false_count = failures(labels_for(record), predictions_for(record))
        if miss_count and missed is None:
            missed = record
        if false_count and false_positive is None:
            false_positive = record
    select(missed, "missed detection")
    select(false_positive, "false positive")
    if missed is None: print("No missed detection found in the scanned subset.")
    if false_positive is None: print("No false positive found in the scanned subset.")
    if not selected: print("No images available for visualization.")

    colors = [f"C{i}" for i in range(len(YOLO_CLASSES))]
    for item in selected.values():
        record = item["record"]
        with Image.open(saved_root / record["image"]) as image:
            image = image.convert("RGB")
            width, height = image.size
        ground_truth = [(cls, box, None) for cls, box in labels_for(record)]
        predicted = predictions_for(record)
        miss_count, false_count = failures(labels_for(record), predicted)
        print(f"{record['image']}: ground truth={len(ground_truth)}, predictions={len(predicted)}, matched={len(ground_truth)-miss_count}, missed={miss_count}, false positives={false_count}")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, boxes, title in zip(axes, (ground_truth, predicted), ("Ground truth", "Predictions")):
            ax.imshow(image)
            label_positions = []
            for cls, (x1, y1, x2, y2), score in sorted(boxes, key=lambda b: b[2] if b[2] is not None else 0, reverse=True):
                color = colors[cls]
                ax.add_patch(Rectangle((x1*width, y1*height), (x2-x1)*width, (y2-y1)*height,
                                       fill=False, edgecolor=color, linewidth=2))
                position = (x1*width, y1*height)
                if score is not None and all(abs(position[0]-x) > width*0.06 or abs(position[1]-y) > height*0.025 for x, y in label_positions):
                    ax.text(*position, f"{score:.2f}", color="white", backgroundcolor=color, fontsize=6)
                    label_positions.append(position)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{record['image']} — {', '.join(item['reasons'])}")
        fig.legend(handles=[Patch(color=colors[i], label=name) for i, name in enumerate(YOLO_CLASSES)],
                   loc="lower center", ncol=len(YOLO_CLASSES), fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        plt.show()
