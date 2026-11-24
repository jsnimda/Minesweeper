import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from grid_geometry.dataset import GeometryDataset


def write_sample(root, split, name, annotation):
    folder = root / split
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (31, 19), (255, 127, 0)).save(folder / f"{name}.png")
    (folder / f"{name}.json").write_text(json.dumps(annotation), encoding="utf-8")
    return folder


def test_loads_both_splits_and_default_loader_batches(tmp_path, annotation):
    write_sample(tmp_path, "train", "b", {**annotation, "board_id": "other"})
    write_sample(tmp_path, "train", "a", annotation)
    write_sample(tmp_path, "val", "v", {**annotation, "board_id": "validation"})
    dataset = GeometryDataset(tmp_path, "train", canvas_size=(80, 64))
    validation = GeometryDataset(tmp_path, "val")
    images, targets = next(iter(DataLoader(dataset, batch_size=2)))

    assert len(dataset) == 2
    assert validation[0][0].shape == (3, 384, 384)
    assert images.shape == targets.shape == (2, 3, 64, 80)
    assert images.dtype == targets.dtype == torch.float32
    assert images.is_contiguous()
    assert images.min() == 0
    assert images.max() == 1
    torch.testing.assert_close(images[0, :, 20, 15], torch.tensor([1, 127 / 255, 0]))

    sample = dataset.load_sample(0)
    assert sample.sample_id == "a"
    assert sample.board_id == annotation["board_id"]
    assert sample.game_state_id == annotation["game_state_id"]
    assert sample.style == annotation["style"]
    np.testing.assert_array_equal(sample.source_crop, annotation["source_crop"])
    np.testing.assert_allclose(sample.transform.to_source(sample.geometry.centers), annotation["centers"])
    torch.testing.assert_close(images[0], sample.image)
    torch.testing.assert_close(targets[0], sample.heatmaps)


def test_uses_json_when_precomputed_arrays_are_present(tmp_path, annotation):
    annotation.pop("game_state_id")
    folder = write_sample(tmp_path, "train", "a", annotation)
    (folder / "a.npz").write_bytes(b"the dataset does not read this file")
    sample = GeometryDataset(tmp_path, "train", include_outer=False).load_sample(0)

    assert sample.game_state_id is None
    assert sample.heatmaps.max() > 0


@pytest.mark.parametrize("extension", ["png", "json"])
def test_reports_orphaned_files(tmp_path, annotation, extension):
    folder = write_sample(tmp_path, "train", "a", annotation)
    (folder / f"a.{extension}").unlink()

    with pytest.raises(ValueError, match=rf"(?i)missing {extension} for \['a'\]"):
        GeometryDataset(tmp_path, "train")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("size", [32, 19], "does not match JSON size"),
        ("style", None, "style must be"),
        ("board_id", "", "board_id must be"),
        ("source_crop", [3.25, 4.125, 34.0, float("inf")], "source_crop must contain finite"),
        ("cell_boxes", [], "cell_boxes must have shape"),
    ],
)
def test_annotates_invalid_sample_errors_with_path(tmp_path, annotation, field, value, error):
    write_sample(tmp_path, "train", "invalid", {**annotation, field: value})
    with pytest.raises(ValueError, match=f"invalid.json.*{error}"):
        GeometryDataset(tmp_path, "train")[0]


def test_rejects_a_malformed_json_file(tmp_path, annotation):
    folder = write_sample(tmp_path, "train", "invalid", annotation)
    (folder / "invalid.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid.json.*Expecting"):
        GeometryDataset(tmp_path, "train")[0]


def test_rejects_missing_annotation_field(tmp_path, annotation):
    annotation.pop("x_edges")
    write_sample(tmp_path, "train", "invalid", annotation)
    with pytest.raises(ValueError, match="invalid.json.*missing required field 'x_edges'"):
        GeometryDataset(tmp_path, "train")[0]


def test_rejects_an_invalid_png(tmp_path, annotation):
    folder = write_sample(tmp_path, "train", "invalid", annotation)
    (folder / "invalid.png").write_bytes(b"not a PNG")

    with pytest.raises(ValueError, match="invalid.png"):
        GeometryDataset(tmp_path, "train")[0]
