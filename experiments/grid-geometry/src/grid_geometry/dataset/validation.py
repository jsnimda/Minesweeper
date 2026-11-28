"""Validate completed PhaseGrid datasets for inspection and training."""
import json
import math
from pathlib import Path

from .loader import inspect_dataset

SPLITS = ("train", "val", "test")


def validate_saved_dataset(root):
    """Check dataset identity without scanning samples."""
    root = Path(root)
    for name in ("config.json", "manifest.jsonl", "statistics.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    metadata = json.loads((root / "config.json").read_text())
    for key, expected in {"experiment": "grid-geometry", "generator": "phase-grid-v1",
                          "generator_revision": 2, "status": "complete"}.items():
        if metadata.get(key) != expected or (key == "generator_revision" and type(metadata.get(key)) is not int):
            raise ValueError(f"Dataset {key}: expected {expected!r}, got {metadata.get(key)!r}")
    return metadata


def validate_training_dataset(root):
    """Inspect samples and check manifest metadata and saved generation statistics."""
    destination = Path(root)
    metadata = validate_saved_dataset(destination)
    if tuple(metadata["input_size"]) != (384, 384):
        raise ValueError("This workflow requires the model-compatible 384 x 384 input size")
    verification = inspect_dataset(destination)
    if any(verification["split_counts"][split] == 0 for split in SPLITS):
        raise ValueError("The complete workflow requires nonempty train, val and test splits")
    statistics = json.loads((destination / "statistics.json").read_text())
    with (destination / "manifest.jsonl").open() as manifest:
        records = [json.loads(line) for line in manifest if line.strip()]
    splits = SPLITS
    unavailable = []
    def optional_check(label, value):
        if value is None:
            unavailable.append(label)
        if value is False:
            raise ValueError(f"Integrity failure: {label}")

    for record in records:
        split, name = record["split"], record["sample_id"]
        if split not in splits or Path(name).name != name:
            raise ValueError(f"Integrity failure: invalid manifest identity {split}/{name}")
        saved = json.loads((destination / split / f"{name}.json").read_text())
        if saved != record:
            raise ValueError(f"Integrity failure: manifest metadata differs for {split}/{name}")
    for split in splits:
        actual = sum(r.get("split") == split for r in records)
        accepted = statistics.get("accepted", {}).get("splits", {}).get(split, {}).get("totals", {}).get("images")
        optional_check(f"{split} accepted images", accepted == actual if accepted is not None else None)
        legacy = statistics.get("splits", {}).get(split, {})
        optional_check(f"{split} statistics samples", legacy["samples"] == actual if "samples" in legacy else None)
        attempt = statistics.get("attempts", {}).get("splits", {}).get(split)
        if attempt and all(k in attempt for k in ("attempted", "accepted", "rejected")):
            ok = attempt["attempted"] == attempt["accepted"] + attempt["rejected"] and attempt["accepted"] == actual
            if "retries" in legacy: ok = ok and sum(legacy["retries"].values()) == attempt["rejected"]
            optional_check(f"{split} source attempts", ok)
        else:
            optional_check(f"{split} source attempts", None)
    accepted_all = statistics.get("accepted", {}).get("overall", {}).get("totals", {}).get("images")
    optional_check("overall accepted images", accepted_all == len(records) if accepted_all is not None else None)
    attempt_all = statistics.get("attempts", {}).get("overall")
    if attempt_all and all(k in attempt_all for k in ("attempted", "accepted", "rejected")):
        optional_check("overall source attempts", attempt_all["attempted"] == attempt_all["accepted"] + attempt_all["rejected"] and attempt_all["accepted"] == len(records))
    else:
        optional_check("overall source attempts", None)

    positives = [r for r in records if r.get("board_id") is not None]
    pitch_checked = category_checked = 0
    for r in positives:
        requested, x, y, aspect = (r.get(k) for k in ("requested_pitch", "effective_pitch_x", "effective_pitch_y", "anisotropy_ratio"))
        if all(v is not None for v in (requested, x, y, aspect)):
            pitch_checked += 1
            if not (math.isclose(x, requested, rel_tol=1e-7, abs_tol=1e-7) and math.isclose(y, requested * aspect, rel_tol=1e-7, abs_tol=1e-7)):
                raise ValueError(f"Integrity failure: requested/realized pitch for {r['split']}/{r['sample_id']}")
        if r.get("requested_category") is not None and r.get("category") is not None:
            category_checked += 1
            if r["requested_category"] != r["category"]:
                raise ValueError(f"Integrity failure: requested/realized crop category for {r['split']}/{r['sample_id']}")
            crop, original = r.get("crop"), r.get("original_geometry")
            if crop and original and original.get("visible_box"):
                left, top, right, bottom = original["visible_box"]
                sides = sum((crop[0] < left < crop[2], crop[0] < right < crop[2],
                             crop[1] < top < crop[3], crop[1] < bottom < crop[3]))
                if sides != {"complete": 4, "interior": 0, "side": 1, "corner": 2}.get(r["category"]):
                    raise ValueError(f"Integrity failure: crop geometry for {r['split']}/{r['sample_id']}")
    if not pitch_checked: unavailable.append("requested/realized pitch")
    if not category_checked: unavailable.append("requested/realized crop category")
    budget = metadata.get("standalone_source_budget")
    standalone = [r for r in records if r.get("scene_mode") == "game" and r.get("source_size")]
    if budget is not None:
        violations = [r for r in standalone if max(r["source_size"]) > budget["max_dimension"] or r["source_size"][0] * r["source_size"][1] > budget["max_pixels"]]
        optional_check(f"standalone source budget ({len(standalone)} checked)", not violations if standalone else None)
    else:
        unavailable.append("standalone source budget")
    print("Dataset validation passed." + (" Unavailable checks: " + ", ".join(unavailable) if unavailable else ""))
    return verification
