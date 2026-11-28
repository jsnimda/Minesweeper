import json
from types import SimpleNamespace

import pytest

from grid_geometry.phase import evaluation_runs


@pytest.fixture
def evaluation_run(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    for split in ("val", "test"):
        (root / split).mkdir(parents=True)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    run = SimpleNamespace(directory=tmp_path / "run", best_path=checkpoint, dataset_root=root)
    calls = []
    def evaluate(checkpoint, root, *, split, output, **kwargs):
        calls.append(split)
        report = {"report_version": evaluation_runs.REPORT_VERSION, "split": split, "records": [{"global": {"local_accepted": False}}], "sample_ids": ["one"],
                  "target_metrics": {"all": {"occupancy_confusion": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}}}, "geometry": {}, "global_geometry_single_crop": {}}
        output.write_text(json.dumps(report))
        return report
    monkeypatch.setattr(evaluation_runs, "evaluate", evaluate)
    monkeypatch.setattr(evaluation_runs, "dataset_identity", lambda root: {"dataset": "identity"})
    monkeypatch.setattr(evaluation_runs, "select_device", lambda device: device)
    return run, calls


def test_reuse_and_reject_changed_identity(evaluation_run):
    run, calls = evaluation_run
    first = evaluation_runs.evaluate_run(run, device="cpu")
    assert calls == ["val", "test"]
    assert evaluation_runs.evaluate_run(run, device="cpu") == first
    assert calls == ["val", "test"]
    with pytest.raises(ValueError, match="incompatible"):
        evaluation_runs.evaluate_run(run, device="cpu", seed=43)
    assert calls == ["val", "test"]


def test_previous_evaluator_fingerprint_is_rejected(evaluation_run):
    run, calls = evaluation_run
    evaluation_runs.evaluate_run(run, device="cpu")
    path = run.directory / "evaluation" / "val" / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["settings"]["evaluator_sha256"] = "previous implementation"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incompatible"):
        evaluation_runs.evaluate_run(run, device="cpu")
    assert calls == ["val", "test"]


@pytest.mark.parametrize("damage", ["missing_metadata", "missing_report", "changed_report", "missing_global"])
def test_reject_incomplete_or_corrupt_report(evaluation_run, damage):
    run, calls = evaluation_run
    evaluation_runs.evaluate_run(run, device="cpu")
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
        del report["records"][0]["global"]
        report_path.write_text(json.dumps(report))
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["report_sha256"] = evaluation_runs.file_hash(report_path)
        metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incomplete|Incomplete"):
        evaluation_runs.evaluate_run(run, device="cpu")
    assert calls == ["val", "test"]


def test_legacy_schema_is_preserved_and_rejected(evaluation_run):
    run, calls = evaluation_run
    evaluation_runs.evaluate_run(run, device="cpu")
    path = run.directory / "evaluation" / "val" / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["settings"]["schema"] = 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="moving it elsewhere"):
        evaluation_runs.evaluate_run(run, device="cpu")
    assert json.loads(path.read_text())["settings"]["schema"] == 1
    assert calls == ["val", "test"]


def test_outdated_report_version_is_preserved_and_rejected(evaluation_run):
    run, calls = evaluation_run
    evaluation_runs.evaluate_run(run, device="cpu")
    directory = run.directory / "evaluation" / "val"
    path = directory / "report.json"
    report = json.loads(path.read_text())
    report.pop("report_version")
    path.write_text(json.dumps(report))
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["report_sha256"] = evaluation_runs.file_hash(path)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Incomplete"):
        evaluation_runs.evaluate_run(run, device="cpu")
    assert "report_version" not in json.loads(path.read_text())
    assert calls == ["val", "test"]
