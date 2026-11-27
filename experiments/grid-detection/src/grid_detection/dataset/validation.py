"""Validation of saved grid-detection datasets."""
import json
import math
from itertools import combinations
from pathlib import Path

import yaml

from grid_detection.dataset.yolo import YOLO_CLASSES

SPLITS = ("train", "val", "test")
SUPPORTED_REVISION = 7


def validate_saved_dataset(root):
    """Check saved dataset identity and class compatibility without scanning images."""
    root = Path(root).resolve()
    for name in ("config.json", "data.yaml", "manifest.jsonl", "statistics.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    config = json.loads((root / "config.json").read_text())
    for key, expected in {"experiment": "grid-detection", "generator_revision": SUPPORTED_REVISION,
                          "status": "complete", "classes": list(YOLO_CLASSES)}.items():
        if config.get(key) != expected:
            raise ValueError(f"{root / 'config.json'}: {key} expected {expected!r}, got {config.get(key)!r}")
    if tuple(YOLO_CLASSES) != ("game_region", "grid", "mine_counter", "reset_button", "timer"):
        raise ValueError("Unexpected production class order")
    data = yaml.safe_load((root / "data.yaml").read_text())
    if not isinstance(data, dict) or data.get("names") != dict(enumerate(YOLO_CLASSES)):
        raise ValueError("Unexpected data.yaml class order")
    return config


def validate_descriptor_path(root):
    """Ensure YOLO will use this dataset and its expected split directories."""
    root = Path(root).resolve()
    data = yaml.safe_load((root / "data.yaml").read_text())
    path = data.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute() or Path(path).resolve() != root:
        raise ValueError(f"data.yaml must contain the local absolute dataset path {root}; if transferred, update its path manually before continuing")
    for split in SPLITS:
        if data.get(split) != f"images/{split}":
            raise ValueError(f"Invalid {split} path in data.yaml")


def validate_training_dataset(root):
    """Check dataset integrity, annotations, statistics and YOLO paths before training."""
    root = Path(root).resolve()
    config = validate_saved_dataset(root)
    validate_descriptor_path(root)
    splits = SPLITS
    stems, seen, identities = {}, {}, {}
    for split in splits:
        images = {p.stem for p in (root / "images" / split).glob("*.png")}
        labels = {p.stem for p in (root / "labels" / split).glob("*.txt")}
        count = config.get(split)
        if type(count) is not int or count < 0 or images != labels or len(images) != count:
            raise ValueError(f"Invalid {split} image/label pairing or saved count")
        stems[split], seen[split], identities[split] = images, set(), set()
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
    for record in records:
        split = record.get("split")
        if split not in splits:
            raise ValueError(f"Invalid manifest split: {split!r}")
        image, label = root / record["image"], root / record["label"]
        stem = image.stem
        if (image != root / "images" / split / f"{stem}.png" or label != root / "labels" / split / f"{stem}.txt"
                or stem not in stems[split] or stem in seen[split]):
            raise ValueError(f"Invalid or duplicate manifest entry: {record}")
        seen[split].add(stem)
        identities[split].update(record["board_ids"])
        lines = label.read_text().splitlines()
        if not record["board_ids"] and lines:
            raise ValueError(f"Negative scene has labels: {label}")
        for row in lines:
            fields = row.split()
            if len(fields) != 5 or fields[0] not in {str(i) for i in range(len(YOLO_CLASSES))}:
                raise ValueError(f"Invalid YOLO annotation: {label}: {row}")
            x, y, w, h = map(float, fields[1:])
            if (not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0
                    or x-w/2 < -1e-7 or y-h/2 < -1e-7 or x+w/2 > 1+1e-7 or y+h/2 > 1+1e-7):
                raise ValueError(f"Invalid YOLO box: {label}: {row}")
    for split in splits:
        if seen[split] != stems[split]:
            raise ValueError(f"Incomplete manifest coverage: {split}")
    for a, b in combinations(splits, 2):
        if identities[a] & identities[b]:
            raise ValueError(f"Board identities overlap: {a} / {b}")
    statistics = json.loads((root / "statistics.json").read_text())
    unavailable = []
    def optional_check(label, value):
        if value is None:
            unavailable.append(label)
        if value is False:
            raise ValueError(f"Integrity failure: {label}")

    for split in splits:
        actual = len(seen[split])
        accepted = statistics.get("accepted", {}).get("splits", {}).get(split, {}).get("totals", {}).get("images")
        optional_check(f"{split} accepted images", accepted == actual if accepted is not None else None)
        legacy = statistics.get(split, {}).get("counts", {})
        optional_check(f"{split} statistics scenes", legacy["scenes"] == actual if "scenes" in legacy else None)
        attempt = statistics.get("attempts", {}).get("splits", {}).get(split)
        if attempt and all(k in attempt for k in ("attempted", "accepted", "rejected")):
            ok = attempt["attempted"] == attempt["accepted"] + attempt["rejected"] and attempt["accepted"] == actual
            if "retries" in legacy: ok = ok and legacy["retries"] == attempt["rejected"]
            optional_check(f"{split} attempt counts", ok)
        else:
            optional_check(f"{split} attempt counts", None)
    accepted_all = statistics.get("accepted", {}).get("overall", {}).get("totals", {}).get("images")
    optional_check("overall accepted images", accepted_all == len(records) if accepted_all is not None else None)
    attempt_all = statistics.get("attempts", {}).get("overall")
    if attempt_all and all(k in attempt_all for k in ("attempted", "accepted", "rejected")):
        optional_check("overall attempt counts", attempt_all["attempted"] == attempt_all["accepted"] + attempt_all["rejected"] and attempt_all["accepted"] == len(records))
    else:
        optional_check("overall attempt counts", None)
    print("Dataset validation passed." + (" Unavailable checks: " + ", ".join(unavailable) if unavailable else ""))
    return config, records
