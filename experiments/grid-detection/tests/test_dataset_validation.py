import json

import pytest
import yaml

from grid_detection.dataset.validation import (
    validate_descriptor_path,
    validate_saved_dataset,
    validate_training_dataset,
)
from grid_detection.dataset.yolo import YOLO_CLASSES


@pytest.fixture
def dataset(tmp_path):
    config = dict(experiment="grid-detection", generator_revision=7, status="complete",
                  classes=list(YOLO_CLASSES), train=1, val=1, test=1)
    (tmp_path / "config.json").write_text(json.dumps(config))
    descriptor = dict(path=str(tmp_path), names=dict(enumerate(YOLO_CLASSES)),
                      **{s: f"images/{s}" for s in ("train", "val", "test")})
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(descriptor))
    records = []
    for split in ("train", "val", "test"):
        for kind, suffix in (("images", "png"), ("labels", "txt")):
            directory = tmp_path / kind / split
            directory.mkdir(parents=True)
            (directory / f"scene.{suffix}").write_text("" if kind == "images" else "0 0.5 0.5 0.2 0.2\n")
        records.append(dict(split=split, image=f"images/{split}/scene.png",
                            label=f"labels/{split}/scene.txt", board_ids=[split]))
    (tmp_path / "manifest.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in records))
    (tmp_path / "statistics.json").write_text(json.dumps({"accepted": {"overall": {"totals": {"images": 3}}}}))
    return tmp_path


def test_valid_dataset_and_lightweight_check(dataset):
    assert validate_saved_dataset(dataset)["status"] == "complete"
    assert len(validate_training_dataset(dataset)[1]) == 3
    (dataset / "labels/train/scene.txt").unlink()
    assert validate_saved_dataset(dataset)["train"] == 1
    with pytest.raises(ValueError, match="pairing"):
        validate_training_dataset(dataset)


@pytest.mark.parametrize("key,value", [("experiment", "other"), ("generator_revision", 6),
                                       ("status", "generating"), ("classes", [])])
def test_identity_rejected(dataset, key, value):
    path = dataset / "config.json"
    config = json.loads(path.read_text())
    config[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=key):
        validate_saved_dataset(dataset)


def test_required_files_and_descriptor_compatibility(dataset):
    (dataset / "statistics.json").unlink()
    with pytest.raises(FileNotFoundError):
        validate_saved_dataset(dataset)
    (dataset / "statistics.json").write_text('{}')
    path = dataset / "data.yaml"
    descriptor = yaml.safe_load(path.read_text())
    descriptor["path"] = "/old/location"
    path.write_text(yaml.safe_dump(descriptor))
    validate_saved_dataset(dataset)
    with pytest.raises(ValueError, match="absolute dataset path"):
        validate_descriptor_path(dataset)
    descriptor["names"][0] = "wrong"
    path.write_text(yaml.safe_dump(descriptor))
    with pytest.raises(ValueError, match="class order"):
        validate_saved_dataset(dataset)


@pytest.mark.parametrize("damage,match", [("annotation", "YOLO box"), ("negative", "Negative scene"),
                                           ("coverage", "manifest coverage"), ("identity", "identities overlap"),
                                           ("statistics", "accepted images")])
def test_comprehensive_checks(dataset, damage, match):
    manifest = dataset / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    if damage == "annotation":
        (dataset / "labels/train/scene.txt").write_text("0 0.9 0.5 0.4 0.2\n")
    elif damage == "negative":
        records[0]["board_ids"] = []
    elif damage == "coverage":
        records.pop()
    elif damage == "identity":
        records[1]["board_ids"] = records[0]["board_ids"]
    else:
        (dataset / "statistics.json").write_text('{"accepted": {"overall": {"totals": {"images": 2}}}}')
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
    with pytest.raises(ValueError, match=match):
        validate_training_dataset(dataset)
