import hashlib
import json
from pathlib import Path

from grid_geometry.phase.evaluate import evaluate
from grid_geometry.phase.train import dataset_identity
from grid_geometry.training import select_device


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


def evaluate_run(run, *, device, limit=None, seed=42):
    """Evaluate or reuse both saved splits for a loaded training run."""
    best_path, saved_root = run.best_path, run.dataset_root
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")
    evaluation_root = run.directory / "evaluation"
    reports = {}
    for split in ("val", "test"):
        destination = evaluation_root / split
        expected = {"schema": 2, "checkpoint_sha256": file_hash(best_path),
                    "dataset_root": str(saved_root), "dataset_identity": dataset_identity(saved_root),
                    "inventory": inventory([p for p in (saved_root / split).glob("*") if p.suffix in (".png", ".json")], saved_root),
                    "split": split, "limit": limit, "seed": seed,
                    "device": str(select_device(device)), "global_geometry": True,
                    "evaluator_sha256": file_hash(Path(__file__).with_name("evaluate.py")),
                    "metrics_sha256": file_hash(Path(__file__).with_name("metrics.py"))}
        if destination.exists():
            report = read_completed(destination, expected, ("records", "sample_ids", "target_metrics", "geometry", "global_geometry_single_crop"))
            if (report.get("split") != split or not isinstance(report["records"], list)
                    or not isinstance(report["sample_ids"], list)
                    or not all(k in report.get("target_metrics", {}).get("all", {}).get("occupancy_confusion", {}) for k in ("tp", "fp", "fn", "tn"))
                    or len(report["records"]) != len(report["sample_ids"])
                    or any(not isinstance(r, dict) or "global" not in r or "local_accepted" not in r["global"] for r in report["records"])):
                raise ValueError(f"Incomplete evaluation report at {destination}; preserve it by moving the directory elsewhere before rerunning")
            print(f"Loaded evaluation: {destination}")
        else:
            destination.mkdir(parents=True, exist_ok=False)
            print(f"Evaluation destination: {destination}")
            report = evaluate(best_path, saved_root, split=split, device=device,
                              limit=limit, seed=seed,
                              output=destination / "report.json", global_geometry=True)
            (destination / "metadata.json").write_text(json.dumps({"settings": expected, "report_sha256": file_hash(destination / "report.json")}, indent=2) + "\n")
        reports[split] = report
    return reports
