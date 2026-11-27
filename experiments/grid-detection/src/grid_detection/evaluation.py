import hashlib
import json
from pathlib import Path

import ultralytics
from ultralytics import YOLO

from grid_detection.dataset.yolo import YOLO_CLASSES


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(paths, root):
    return [[str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(paths) if p.is_file()]


def read_completed(destination, expected, required):
    hint = (f"Preserve the existing directory by moving it elsewhere, then rerun: {destination}")
    try:
        metadata = json.loads((destination / "metadata.json").read_text())
        result = json.loads((destination / "report.json").read_text())
        if (not isinstance(metadata, dict) or metadata.get("settings") != expected
                or metadata.get("report_sha256") != file_hash(destination / "report.json")
                or not isinstance(result, dict) or not all(k in result for k in required)):
            raise ValueError("metadata or report does not match")
        return result
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Incomplete, legacy, or incompatible evaluation at {destination}. {hint}") from error


def evaluate_run(run, *, device):
    """Evaluate or reuse both saved splits for a loaded training run."""
    best_path, saved_root, saved_yaml, saved_args = run.best_path, run.dataset_root, run.data_yaml, run.args
    if not best_path.is_file():
        raise FileNotFoundError(best_path)
    evaluation_root = run.directory / "evaluation"
    reports = {}
    for split in ("val", "test"):
        destination = evaluation_root / split
        expected = {"schema": 1, "checkpoint_sha256": file_hash(best_path),
                    "dataset_root": str(saved_root), "dataset_config_sha256": file_hash(saved_root / "config.json"),
                    "manifest_sha256": file_hash(saved_root / "manifest.jsonl"),
                    "data_yaml_sha256": file_hash(saved_yaml),
                    "inventory": inventory(list((saved_root / "images" / split).glob("*.png")) +
                                           list((saved_root / "labels" / split).glob("*.txt")), saved_root),
                    "split": split, "imgsz": saved_args["imgsz"], "batch": saved_args["batch"],
                    "workers": saved_args["workers"], "device": device, "plots": True,
                    "ultralytics": ultralytics.__version__}
        if destination.exists():
            report = read_completed(destination, expected, ("aggregate", "per_class", "counts", "timing_ms_per_image"))
            print(f"Loaded evaluation: {destination}")
        else:
            destination.mkdir(parents=True, exist_ok=False)
            print(f"Evaluation destination: {destination}")
            metrics = YOLO(str(best_path)).val(data=str(saved_yaml), split=split, imgsz=saved_args["imgsz"],
                                batch=saved_args["batch"], device=device, workers=saved_args["workers"],
                                project=str(evaluation_root), name=split, exist_ok=True, plots=True)
            box = metrics.box
            classes = {}
            for i, cls in enumerate(box.ap_class_index):
                precision, recall, ap50, ap = box.class_result(i)
                classes[YOLO_CLASSES[int(cls)]] = dict(precision=float(precision), recall=float(recall),
                    ap50=float(ap50), ap75=float(box.all_ap[i, 5]), ap50_95=float(ap))
            report = {"aggregate": dict(precision=float(box.mp), recall=float(box.mr), map50=float(box.map50),
                                        map75=float(box.map75), map50_95=float(box.map)),
                      "per_class": classes,
                      "counts": {"images": len(list((saved_root / "images" / split).glob("*.png"))),
                                 "instances_per_class": [int(x) for x in metrics.nt_per_class],
                                 "images_per_class": [int(x) for x in metrics.nt_per_image]},
                      "timing_ms_per_image": {k: float(v) for k, v in metrics.speed.items()}}
            (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            (destination / "metadata.json").write_text(json.dumps({"settings": expected, "report_sha256": file_hash(destination / "report.json")}, indent=2) + "\n")
        reports[split] = report
    return reports
