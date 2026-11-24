from pathlib import Path
import tomllib

import pytest

from grid_geometry.training_config import load_config, resolved_toml


def test_baseline_uses_provisional_defaults():
    path = Path(__file__).resolve().parents[1] / "configs" / "baseline.toml"
    config = load_config(path)
    assert (config.dataset.width, config.dataset.height) == (384, 384)
    assert config.dataset.sigma == 2
    assert config.dataset.include_outer is True
    assert config.training.optimizer == "adam"
    assert config.training.learning_rate == 0.001
    assert config.training.batch_size == 8
    assert config.training.epochs == 30
    assert config.training.seed == 42
    assert config.training.loss_weight == 10
    assert config.run.resume is False


def test_resolves_paths_relative_to_config_and_round_trips(tmp_path):
    folder = tmp_path / "configs"
    folder.mkdir()
    path = folder / "run.toml"
    path.write_text(
        '[dataset]\nroot = "../data set/geometry"\nwidth = 64\nheight = 48\ninclude_outer = false\n'
        '[training]\nworkers = 2\nloss_weight = 0\n'
        '[run]\ndirectory = "../runs/example"\narchive_every = 3\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.dataset.root == tmp_path.resolve() / "data set" / "geometry"
    assert config.run.directory == tmp_path.resolve() / "runs" / "example"
    assert config.training.workers == 2
    raw = tomllib.loads(resolved_toml(config, device="cpu"))
    assert raw["training"]["device"] == "cpu"
    assert raw["dataset"]["include_outer"] is False
    assert raw["run"]["archive_every"] == 3


@pytest.mark.parametrize(("section", "setting", "message"), [
    ("dataset", "width = 63", "divisible by eight"),
    ("dataset", "sigma = nan", "sigma"),
    ("dataset", 'include_outer = "yes"', "boolean"),
    ("dataset", 'validation_split = "train"', "distinct"),
    ("training", "batch_size = 0", "batch_size"),
    ("training", "workers = -1", "workers"),
    ("training", "epochs = true", "epochs"),
    ("training", "seed = 4294967296", "seed"),
    ("training", "loss_weight = -1", "loss_weight"),
    ("training", "learning_rate = inf", "learning_rate"),
    ("training", 'optimizer = "sgd"', "optimizer"),
    ("training", 'device = "gpu"', "device"),
    ("training", "mixed_precision = 1", "boolean"),
    ("training", "epohcs = 2", "unknown"),
    ("run", "archive_every = -1", "archive_every"),
])
def test_invalid_config_fails_clearly(tmp_path, section, setting, message):
    sections = {"dataset": 'root = "data"', "training": "", "run": 'directory = "runs"'}
    sections[section] += "\n" + setting
    path = tmp_path / "invalid.toml"
    path.write_text("\n".join(f"[{key}]\n{value}" for key, value in sections.items()), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)
