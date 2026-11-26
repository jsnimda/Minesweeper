"""Defaults and validation for phase dataset generation."""
from dataclasses import dataclass, field
from collections.abc import Mapping
import math

PITCH_BINS = ((6., 11., .10), (12., 19., .25), (20., 31., .35), (32., 47., .20), (48., 64., .10))
CONTEXT_WEIGHTS = {'procedural': 70, 'desktop': 30}
CATEGORIES = {'complete': .35, 'interior': .25, 'side': .20, 'corner': .20}
STYLES = ('classic', 'windows7', 'microsoft', 'google')
THEMES = {'classic': ('default', 'blue', 'green', 'warm'), 'windows7': ('blue', 'green'),
          'microsoft': ('default', 'blue', 'warm'), 'google': ('default', 'blue', 'warm')}
NEUTRAL_SCALE = {1.0: 1}


def scale_choices(weights):
    """Validate a scale distribution and return its positive-weight choices."""
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError('scale distribution must be a nonempty mapping')
    choices = {}
    for scale, weight in weights.items():
        try:
            valid_scale = type(scale) in (int, float) and math.isfinite(scale) and scale > 0
            valid_weight = type(weight) in (int, float) and math.isfinite(weight) and weight >= 0
        except OverflowError:
            valid_scale = valid_weight = False
        if not valid_scale or not valid_weight:
            raise ValueError('scale factors and weights must be finite; scales must be positive and weights nonnegative')
        if weight > 0:
            choices[scale] = weight
    try:
        valid_total = bool(choices) and math.isfinite(sum(choices.values()))
    except OverflowError:
        valid_total = False
    if not valid_total:
        raise ValueError('scale distribution must have a finite positive total weight')
    return choices


@dataclass(frozen=True)
class GenerationConfig:
    train: int = 40
    val: int = 12
    test: int = 12
    seed: int = 42
    workers: int = 1
    png_compression: int = 1
    input_size: tuple[int, int] = (384, 384)
    pitch_bins: tuple = PITCH_BINS
    categories: dict = field(default_factory=lambda: dict(CATEGORIES))
    negative_fraction: float = .15
    anisotropic_fraction: float = .10
    anisotropic_max: float = .05
    transition_ratio: float = .25
    padding_example_fraction: float = .05
    padding_max_fraction: float = .20
    context_weights: dict = field(default_factory=lambda: dict(CONTEXT_WEIGHTS))
    coverage_exploration_fraction: float = .12
    coverage_preference_exponent: float = 1.5
    desktop_fraction: float = .8
    max_attempts: int = 30
    progress: bool = True
    game_zoom: dict = field(default_factory=lambda: dict(NEUTRAL_SCALE))
    display_scale: dict = field(default_factory=lambda: dict(NEUTRAL_SCALE))
    standalone_source_budget: dict | None = None

    def __post_init__(self):
        scale_choices(self.game_zoom)
        scale_choices(self.display_scale)
        budget = self.standalone_source_budget
        if budget is not None and (
            not isinstance(budget, dict)
            or set(budget) != {'max_dimension', 'max_pixels'}
            or any(type(value) is not int or value <= 0 for value in budget.values())
        ):
            raise ValueError('standalone source budget must contain positive integer max_dimension and max_pixels')
        if any(type(n) is not int or n < 0 for n in (self.train, self.val, self.test, self.seed)):
            raise ValueError('counts and seed must be nonnegative integers')
        if type(self.workers) is not int or self.workers < 1 or type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError('workers and max_attempts must be positive integers')
        if type(self.png_compression) is not int or not 0 <= self.png_compression <= 9:
            raise ValueError('png_compression must be in 0..9')
        if len(self.input_size) != 2 or any(type(v) is not int or v <= 0 or v % 8 for v in self.input_size):
            raise ValueError('input dimensions must be positive multiples of eight')
        if not (0 <= self.negative_fraction <= 1 and 0 <= self.anisotropic_fraction <= 1
                and 0 <= self.anisotropic_max <= .2 and math.isfinite(self.transition_ratio) and self.transition_ratio > 0
                and 0 <= self.padding_example_fraction <= 1 and 0 < self.padding_max_fraction <= .4
                and 0 <= self.coverage_exploration_fraction <= 1
                and 0 <= self.desktop_fraction <= 1 and math.isfinite(self.coverage_preference_exponent)
                and 0 <= self.coverage_preference_exponent <= 5):
            raise ValueError('invalid probability, anisotropy or transition ratio')
        if set(self.context_weights) != set(CONTEXT_WEIGHTS) or any(not math.isfinite(v) or v < 0 for v in self.context_weights.values()) or sum(self.context_weights.values()) <= 0:
            raise ValueError('invalid context weights')
        if set(self.categories) != set(CATEGORIES) or any(not math.isfinite(v) or v < 0 for v in self.categories.values()) or sum(self.categories.values()) <= 0:
            raise ValueError('invalid category weights')
        if not self.pitch_bins or any(len(b) != 3 or not all(math.isfinite(v) for v in b) or b[0] <= 0 or b[1] < b[0] or b[2] < 0 for b in self.pitch_bins) or sum(b[2] for b in self.pitch_bins) <= 0:
            raise ValueError('invalid pitch bins')


