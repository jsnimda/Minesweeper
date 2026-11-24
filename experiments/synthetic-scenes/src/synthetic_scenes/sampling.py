"""Explicit first-pass sampling policies. Ratios are experimental defaults."""
import random

from .scene import GameSpec


THEMES = {'classic': ('default','blue','green','warm'),
          'windows7': ('blue','green'), 'microsoft': ('default','blue','warm'),
          'google': ('default','blue','warm')}


def sample_game(rng, board_seed, *, targeted=False, target_number=None, small=False):
    style = rng.choice(tuple(THEMES))
    theme = THEMES[style][0] if rng.random() < .65 else rng.choice(THEMES[style])
    rows, columns, mines = (9,9,10) if small else rng.choice(((9,9,10),(16,16,40),(16,30,99)))
    status = rng.choices(('new','active','won','lost'), (12,76,6,6))[0]
    if targeted:
        status = 'active'
        target_number = target_number if target_number is not None else rng.randint(5,8)
    progress = rng.uniform(.12,.88) if status != 'new' else 0.
    appearance = {'style': style, 'theme': theme, 'cell_size': rng.randint(20,40),
                  'font_family': rng.choice(('Arial','Tahoma','Verdana')),
                  'font_scale': rng.uniform(.65,.77), 'bold': rng.random() < .9}
    if style == 'google':
        appearance['boundary_ratio'] = rng.choice((0., .06, .08, .10))
    if style == 'windows7':
        appearance['mine_variant'] = rng.choice(('classic','flower'))
    return GameSpec(rows=rows, columns=columns, mine_count=mines, status=status,
                    progress=progress, seed=board_seed,
                    elapsed_seconds=0 if status=='new' else rng.randint(1,600),
                    mode='targeted' if targeted else 'normal', target_number=target_number,
                    appearance=appearance)


def sample_scene(seed, *, targeted_probability=.25, negative_probability=.08,
                 mode=None, max_games=3):
    """Return specs and a mode. Unique board IDs derive from unique nonnegative scene seeds."""
    if seed < 0 or not 1 <= max_games <= 8:
        raise ValueError('Expected nonnegative seed and 1..8 max_games')
    if not 0 <= targeted_probability <= 1 or not 0 <= negative_probability <= 1:
        raise ValueError('Probabilities must be in [0,1]')
    rng = random.Random(seed)
    if mode is None:
        mode = rng.choices(('game','desktop','grid'), (.60,.30,.10))[0]
    if mode == 'desktop' and rng.random() < negative_probability:
        return [], mode
    count = rng.randint(1,max_games) if mode == 'desktop' else 1
    specs = [sample_game(rng, seed*16+i, targeted=rng.random()<targeted_probability,
                         small=count>1) for i in range(count)]
    return specs, mode
