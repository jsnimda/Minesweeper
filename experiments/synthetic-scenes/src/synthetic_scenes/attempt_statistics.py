"""Compact counters for logical attempts; domain-specific observation stays with callers."""
from bisect import bisect_right
from collections import Counter
from .dataset_statistics import BINS, area_bucket

NOT_APPLICABLE = '<not_applicable>'
CELL_EDGES = (12, 20, 33, 49, 65)


def cell_bucket(value):
    if value is None:
        return None
    if value < CELL_EDGES[0]: return 'underflow <12'
    if value >= CELL_EDGES[-1]: return 'overflow >=65'
    index = bisect_right(CELL_EDGES, value) - 1
    return f'{CELL_EDGES[index]}-{CELL_EDGES[index + 1] - 1}'


def fixed_bucket(value, edges):
    if value is None: return None
    if value < edges[0]: return 'underflow'
    if value > edges[-1]: return 'overflow'
    index = min(len(edges)-2, bisect_right(edges, value)-1)
    return f'{edges[index]:g}-{edges[index+1]:g}'


def board_factors(game):
    appearance = game.get('appearance', {})
    rows, columns = game.get('rows'), game.get('columns')
    return {'game_style': appearance.get('style'), 'game_zoom': appearance.get('zoom'),
            'sampled_cell_size': cell_bucket(appearance.get('cell_size')),
            'board_area': area_bucket(rows, columns), 'rows': rows, 'columns': columns}


def _key(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


class AttemptCounters:
    """Only counters are retained; missing and inapplicable factors stay distinct."""
    def __init__(self):
        self.outcomes = Counter()
        self.reasons = Counter()
        self.availability = {}
        self.buckets = {}
        self.board_availability = {}
        self.boards = {}
        self.samples = Counter()
        self.objectives = {}
        self.objective_values = {}
        self.fitting = Counter()

    @staticmethod
    def _observe(mapping, availability, fields, outcome, reason):
        for name, value in fields.items():
            availability.setdefault(name, Counter())
            if value is None:
                availability[name]['unavailable'] += 1
            elif value == NOT_APPLICABLE:
                availability[name]['inapplicable'] += 1
            else:
                availability[name]['available'] += 1
                bucket = mapping.setdefault(name, {}).setdefault(_key(value), Counter())
                bucket['attempted'] += 1
                bucket[outcome] += 1
                if reason is not None: bucket['reason:' + reason] += 1

    def observe(self, fields, boards, outcome, reason=None, fitting=None):
        self.outcomes['attempted'] += 1
        self.outcomes[outcome] += 1
        if reason is not None: self.reasons[reason] += 1
        self._observe(self.buckets, self.availability, fields, outcome, reason)
        for board in boards:
            self.outcomes['board_observations'] += 1
            self._observe(self.boards, self.board_availability, board, outcome, reason)
        if fitting:
            for name, value in fitting.items():
                self.fitting[name + ('_yes' if value else '_no')] += 1

    def finish_sample(self, attempts, objective=None):
        self.samples[str(attempts)] += 1
        if objective is None: return
        for name in ('polarity', 'objective_type', 'negative_kind', 'category', 'pitch_bin', 'anisotropy_selected'):
            value = objective.get(name)
            if value in (None, NOT_APPLICABLE): continue
            counter = self.objectives.setdefault(name, {}).setdefault(_key(value), Counter())
            counter['selected'] += 1
            counter['successful'] += 1
            counter['source_attempts'] += attempts
            counter['rejected_source_attempts'] += attempts - 1
            counter['samples_requiring_retry'] += attempts > 1
            counter['attempts:' + str(attempts)] += 1
        for name, edges in (('desired_pitch', BINS['pitch']), ('anisotropy_ratio', BINS['scale'])):
            value = objective.get(name)
            if value is not None and value != NOT_APPLICABLE:
                self.objective_values.setdefault(name, Counter())[fixed_bucket(value, edges)] += 1

    def merge(self, other):
        self.outcomes.update(other.outcomes); self.reasons.update(other.reasons)
        self.samples.update(other.samples); self.fitting.update(other.fitting)
        for name in ('availability', 'board_availability', 'objective_values'):
            destination = getattr(self, name)
            for key, counter in getattr(other, name).items(): destination.setdefault(key, Counter()).update(counter)
        for name in ('buckets', 'boards', 'objectives'):
            destination = getattr(self, name)
            for field, buckets in getattr(other, name).items():
                for value, counter in buckets.items():
                    destination.setdefault(field, {}).setdefault(value, Counter()).update(counter)

    @staticmethod
    def _buckets(mapping, objective=False, board=False):
        result = {}
        for field, buckets in sorted(mapping.items()):
            result[field] = {}
            for value, counter in sorted(buckets.items()):
                if objective:
                    selected = counter['selected']
                    attempts = {k[9:]: v for k, v in counter.items() if k.startswith('attempts:')}
                    result[field][value] = {'selected': selected, 'successful': counter['successful'],
                        'source_attempts': counter['source_attempts'], 'rejected_source_attempts': counter['rejected_source_attempts'],
                        'samples_requiring_retry': counter['samples_requiring_retry'],
                        'attempts_per_sample': dict(sorted(attempts.items(), key=lambda item: int(item[0]))),
                        'mean_attempts': counter['source_attempts']/selected if selected else None}
                else:
                    count = counter['attempted']
                    rate_name = 'containing_attempt_rejection_fraction' if board else 'rejection_rate'
                    result[field][value] = {'attempted': count, 'accepted': counter['accepted'],
                        'rejected': counter['rejected'], rate_name: counter['rejected']/count if count else None,
                        'rejection_reasons': {k[7:]: v for k, v in sorted(counter.items()) if k.startswith('reason:')}}
        return result

    def result(self):
        samples = sum(self.samples.values())
        total = sum(int(k)*v for k, v in self.samples.items())
        return {'attempted': self.outcomes['attempted'], 'accepted': self.outcomes['accepted'],
                'rejected': self.outcomes['rejected'], 'rejection_reasons': dict(sorted(self.reasons.items())),
                'board_observations': self.outcomes['board_observations'],
                'attempts_per_sample': {'successful_samples': samples, 'first_attempt': self.samples['1'],
                    'retried': samples-self.samples['1'], 'total_attempts': total,
                    'rejected_attempts': total-samples, 'mean': total/samples if samples else None,
                    'max': max((int(k) for k in self.samples), default=None),
                    'counts': dict(sorted(self.samples.items(), key=lambda item: int(item[0])))},
                'availability': {k: {x: v[x] for x in ('available', 'unavailable', 'inapplicable')} for k, v in sorted(self.availability.items())},
                'by': self._buckets(self.buckets),
                'board_availability': {k: {x: v[x] for x in ('available', 'unavailable', 'inapplicable')} for k, v in sorted(self.board_availability.items())},
                'boards_by_containing_attempt_outcome': self._buckets(self.boards, board=True),
                'objectives': {'by': self._buckets(self.objectives, objective=True),
                    'distributions': {name: dict(sorted(counter.items())) for name, counter in sorted(self.objective_values.items())}},
                'fitting_observations': dict(sorted(self.fitting.items()))}


def summaries(overall, per_split, splits, *, unit):
    return {'conventions': {'logical_attempt': unit,
        'outcomes': 'one accepted or retryable rejection per attempt; fitting is not rejection',
        'conditional_denominator': 'available observations in that bucket; missing and inapplicable are separate',
        'boards': 'board observations inherit containing attempt outcome; not individual board failure probabilities',
        'geometry_objectives': 'one objective per successful sample; objective source attempts are attempt-weighted',
        'board_area': '0-100, 101-300, 301-700, 701-900, >900 cells',
        'cell_size': 'inclusive 12-19, 20-32, 33-48, 49-64; separate underflow/overflow',
        'screen_dimension_edges': BINS['dimension'],
        'objective_histograms': {'desired_pitch_edges': BINS['pitch'], 'anisotropy_ratio_edges': BINS['scale'],
                                 'intervals': '[left,right), final right inclusive'}},
        'overall': overall.result(),
        'splits': {split: per_split.get(split, AttemptCounters()).result() for split in splits}}
