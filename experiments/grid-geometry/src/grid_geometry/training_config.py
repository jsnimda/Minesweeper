"""TOML settings for GridUNet training."""

from dataclasses import dataclass, fields, replace
import json
import math
from pathlib import Path
import tomllib

from .heatmaps import validate_sigma
from .preprocessing import validate_canvas_size


DEVICES = ("auto", "cpu", "mps", "cuda")
OPTIMIZERS = ("adam", "adamw")


def _integer(name: str, value: object, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _number(name: str, value: object, *, zero_allowed: bool = False) -> None:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (value == 0 and not zero_allowed):
        threshold = "nonnegative" if zero_allowed else "positive"
        raise ValueError(f"{name} must be finite and {threshold}")


def _boolean(name: str, value: object) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    train_split: str = "train"
    validation_split: str = "val"
    width: int = 384
    height: int = 384
    sigma: float = 2.0
    include_outer: bool = True

    def __post_init__(self) -> None:
        if self.train_split not in ("train", "val") or self.validation_split not in ("train", "val"):
            raise ValueError("dataset splits must be 'train' or 'val'")
        if self.train_split == self.validation_split:
            raise ValueError("training and validation splits must be distinct")
        validate_canvas_size((self.width, self.height))
        validate_sigma(self.sigma)
        _boolean("include_outer", self.include_outer)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    workers: int = 0
    optimizer: str = "adam"
    learning_rate: float = 0.001
    loss_weight: float = 10.0
    epochs: int = 30
    seed: int = 42
    device: str = "auto"
    mixed_precision: bool = False
    log_every_batches: int = 25

    def __post_init__(self) -> None:
        _integer("batch_size", self.batch_size, 1)
        _integer("workers", self.workers, 0)
        _integer("epochs", self.epochs, 1)
        _integer("seed", self.seed, 0)
        if self.seed >= 2**32:
            raise ValueError("seed must be less than 2**32")
        _integer("log_every_batches", self.log_every_batches, 0)
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {', '.join(OPTIMIZERS)}")
        _number("learning_rate", self.learning_rate)
        _number("loss_weight", self.loss_weight, zero_allowed=True)
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {', '.join(DEVICES)}")
        _boolean("mixed_precision", self.mixed_precision)


@dataclass(frozen=True)
class RunConfig:
    directory: Path
    resume: bool = False
    archive_every: int = 0

    def __post_init__(self) -> None:
        _boolean("resume", self.resume)
        _integer("archive_every", self.archive_every, 0)


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig
    training: TrainingConfig
    run: RunConfig


def _table(schema, raw: dict, name: str) -> dict:
    table = raw.get(name, {})
    if not isinstance(table, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    unknown = table.keys() - {field.name for field in fields(schema)}
    if unknown:
        raise ValueError(f"unknown [{name}] fields: {', '.join(sorted(unknown))}")
    return dict(table)


def _config_path(value: object, relative_to: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty path")
    return (relative_to / Path(value).expanduser()).resolve()


def load_config(path: Path) -> Config:
    path = path.resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    unknown = raw.keys() - {"dataset", "training", "run"}
    if unknown:
        raise ValueError(f"unknown top-level configuration fields: {', '.join(sorted(unknown))}")

    dataset = _table(DatasetConfig, raw, "dataset")
    run = _table(RunConfig, raw, "run")
    dataset["root"] = _config_path(dataset.get("root"), path.parent, "dataset.root")
    run["directory"] = _config_path(run.get("directory"), path.parent, "run.directory")
    return Config(
        DatasetConfig(**dataset),
        TrainingConfig(**_table(TrainingConfig, raw, "training")),
        RunConfig(**run),
    )


def resolved_toml(config: Config, *, device: str) -> str:
    """Serialize the selected configuration with resolved paths and device."""
    sections = (
        ("dataset", config.dataset),
        ("training", replace(config.training, device=device)),
        ("run", config.run),
    )
    lines: list[str] = []
    for name, section in sections:
        lines.append(f"[{name}]")
        for field in fields(section):
            value = getattr(section, field.name)
            if isinstance(value, (str, Path)):
                literal = json.dumps(str(value), ensure_ascii=False)
            elif isinstance(value, bool):
                literal = str(value).lower()
            else:
                literal = repr(value)
            lines.append(f"{field.name} = {literal}")
        lines.append("")
    return "\n".join(lines)
