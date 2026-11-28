"""Plan independent native-resolution evaluation scenes without rendering.

The scene specifications are inputs to synthetic_scenes.render.render_scene.
Requested cell size is not a ground-truth pitch: style geometry, zoom and
possible desktop fitting determine actual rendered geometry. Crop intentions
are descriptive and must be resolved against renderer annotations later.
"""
from dataclasses import dataclass
import hashlib
import random

from .sampler import _game, _scene

CATEGORIES = ('large_complete', 'small_complete', 'partial', 'hard_negative', 'negative')
DEFAULT_COUNTS = {
    'smoke': (4, 2, 2, 1, 1),
    'development': (40, 20, 20, 15, 5),
    'test': (40, 20, 20, 15, 5),
}
REFERENCE_TILE = 384
REFERENCE_OVERLAP = 96


@dataclass(frozen=True)
class PlanConfig:
    split: str
    seed: int = 42
    counts: dict[str, int] | None = None

    def __post_init__(self):
        if self.split not in DEFAULT_COUNTS:
            raise ValueError('split must be smoke, development or test')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if self.counts is not None:
            if not isinstance(self.counts, dict) or set(self.counts) != set(CATEGORIES):
                raise ValueError('counts must specify exactly the five categories')
            if any(type(v) is not int or v < 0 for v in self.counts.values()):
                raise ValueError('category counts must be nonnegative integers')
            if not sum(self.counts.values()):
                raise ValueError('at least one sample is required')
            # Avoid retaining a caller-owned mutable dictionary.
            object.__setattr__(self, 'counts', dict(self.counts))


def plan(config: PlanConfig):
    """Return ordered JSON-compatible scene plans with per-sample random streams.

    IDs include split, seed and category-local index; changes in other category
    counts do not perturb existing plans. Unique planned IDs and seeds do not
    prove unique rendered board identities. Later generation must compare the
    renderer's board_id with other splits and available training provenance.
    """
    counts = config.counts if config.counts is not None else dict(zip(CATEGORIES, DEFAULT_COUNTS[config.split]))
    plans = []
    for category in CATEGORIES:
        for index in range(counts[category]):
            identity = f'tiled-eval-v1:{config.split}:{config.seed}:{category}:{index}'
            digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()
            rng = random.Random(int(digest, 16))
            negative = category in ('negative', 'hard_negative')
            game = None
            if not negative:
                # Reuse the phase sampler's appearance, themes, game states,
                # and board-seed policy; only the intended size is replaced.
                game, progress = _game(rng, 'complete', 16., (384, 384))
                if category in ('large_complete', 'partial'):
                    rows, columns = rng.randint(40, 80), rng.randint(40, 100)
                    cell_size = rng.randint(16, 22)
                else:
                    rows, columns = rng.randint(5, 14), rng.randint(5, 14)
                    cell_size = rng.randint(16, 24)
                density = game['mine_count'] / (game['rows'] * game['columns'])
                game.update(rows=rows, columns=columns,
                            mine_count=max(1, min(rows * columns - 1, round(rows * columns * density))))
                game['appearance'].update(cell_size=cell_size, zoom=1.0)
                # Large and partial games are standalone to avoid desktop fit.
                # Small games may be standalone or composed on a desktop.
                scene = _scene(rng, game, desktop_fraction=0. if category != 'small_complete' else .8)
            else:
                progress = None
                scene = _scene(rng, None, negative_kind=category)
            crop = {'margin_pitches': rng.choice((.35, .6, 1., 2.)),
                    'clipping': rng.choice(('one_side', 'adjacent_sides', 'opposite_sides', 'interior'))
                    if category == 'partial' else None}
            plans.append({'sample_id': f'{config.split}-{category}-{index:04d}-{digest[:16]}',
                          'scene_identity': identity, 'planning_seed_sha256': digest,
                          'split': config.split, 'category': category, 'category_index': index,
                          'scene': scene, 'game_progress_category': progress, 'crop_intent': crop,
                          'reference_tile': REFERENCE_TILE, 'reference_overlap': REFERENCE_OVERLAP})
    return plans
