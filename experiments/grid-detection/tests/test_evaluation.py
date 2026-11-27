import json
from types import SimpleNamespace

import pytest

from grid_detection import evaluation


@pytest.fixture
def evaluation_run(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    for name in ("config.json", "manifest.jsonl", "data.yaml"):
        (root / name).write_text("dataset")
    for split in ("val", "test"):
        for kind in ("images", "labels"):
            (root / kind / split).mkdir(parents=True)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    run = SimpleNamespace(directory=tmp_path / "run", best_path=checkpoint,
                          dataset_root=root, data_yaml=root / "data.yaml",
                          args={"imgsz": 640, "batch": 2, "workers": 0})
    box = SimpleNamespace(ap_class_index=[], mp=.5, mr=.6, map50=.7, map75=.6, map=.5)
    metrics = SimpleNamespace(box=box, nt_per_class=[], nt_per_image=[], speed={"inference": 1})
    calls = []
    def model(path):
        calls.append(path)
        return SimpleNamespace(val=lambda **kwargs: metrics)
    monkeypatch.setattr(evaluation, "YOLO", model)
    return run, calls


def test_reuse_and_reject_changed_identity(evaluation_run):
    run, calls = evaluation_run
    first = evaluation.evaluate_run(run, device="cpu")
    assert len(calls) == 2
    assert evaluation.evaluate_run(run, device="cpu") == first
    assert len(calls) == 2
    run.best_path.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="incompatible"):
        evaluation.evaluate_run(run, device="cpu")
    assert len(calls) == 2


@pytest.mark.parametrize("damage", ["missing_metadata", "missing_report", "changed_report", "missing_field"])
def test_reject_incomplete_or_corrupt_report(evaluation_run, damage):
    run, calls = evaluation_run
    evaluation.evaluate_run(run, device="cpu")
    directory = run.directory / "evaluation" / "val"
    if damage == "missing_metadata":
        (directory / "metadata.json").unlink()
    elif damage == "missing_report":
        (directory / "report.json").unlink()
    elif damage == "changed_report":
        (directory / "report.json").write_text('{}')
    else:
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text())
        del report["aggregate"]
        report_path.write_text(json.dumps(report))
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["report_sha256"] = evaluation.file_hash(report_path)
        metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incomplete|Incomplete"):
        evaluation.evaluate_run(run, device="cpu")
    assert len(calls) == 2
