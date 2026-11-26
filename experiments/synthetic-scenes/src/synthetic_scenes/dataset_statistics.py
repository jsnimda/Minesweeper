"""Small accepted-sample statistics helpers; no sampling or rendering."""
from bisect import bisect_right
from collections import Counter, defaultdict
import math

# Fixed, shared edges; bins are [left,right), except the final right edge is inclusive.
BINS = {
    'fraction': (0., .001, .005, .01, .025, .05, .1, .2, .35, .5, .75, 1.),
    'dimension': (0, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1600, 2048, 4096, 8192),
    'pixels': (0, 10000, 100000, 500000, 1000000, 3000000, 6000000, 12000000, 25000000, 100000000),
    'board_area': (0, 25, 81, 100, 256, 480, 700, 900, 1500, 3000, 6000, 10000),
    'pitch': (0, 6, 12, 20, 32, 48, 64, 96, 128, 256, 512),
    'scale': (0., .125, .25, .5, .75, 1., 1.25, 1.5, 2., 4., 8., 16.),
    'reduction': (0, 1, 2, 4, 8, 16, 32, 64, 128),
    'error': (0., 1e-12, 1e-9, 1e-6, .001, .01, .1, 1., 10.),
    'photometric': (0., .5, .9, .95, .975, 1., 1.025, 1.05, 1.1, 1.5, 2.),
}


def numeric_summary(values, edges):
    """Linear interpolation at (n-1)*p; missing/non-finite values excluded."""
    data = sorted(float(v) for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v))
    counts = [0] * (len(edges) - 1)
    underflow = overflow = 0
    for value in data:
        if value < edges[0]: underflow += 1
        elif value > edges[-1]: overflow += 1
        else: counts[min(len(counts)-1, bisect_right(edges, value)-1)] += 1
    def quantile(p):
        position = (len(data)-1)*p
        low = int(position); high = min(low+1, len(data)-1)
        fraction = position-low
        return data[low] * (1-fraction) + data[high] * fraction
    return {'count': len(data), 'min': data[0] if data else None, 'max': data[-1] if data else None,
            'mean': sum(value/len(data) for value in data) if data else None,
            'median': quantile(.5) if data else None, 'p05': quantile(.05) if data else None,
            'p95': quantile(.95) if data else None,
            'histogram': {'edges': list(edges), 'counts': counts, 'underflow': underflow, 'overflow': overflow}}


def area_bucket(rows, columns):
    if rows is None or columns is None: return None
    area = rows * columns
    return '0-100' if area <= 100 else '101-300' if area <= 300 else '301-700' if area <= 700 else '701-900' if area <= 900 else '>900'


class Summary:
    """Accumulate compact observations, not records or rendered images."""
    def __init__(self):
        self.totals = Counter()
        self.categories = defaultdict(Counter)
        self.values = defaultdict(list)
        self.kinds = {}
        self.fit = defaultdict(lambda: {'applicable': 0, 'budget_applicable': 0, 'placement_applicable': 0,
                                        'budget_fitted': 0, 'placement_fitted': 0, 'any_fitted': 0, 'reductions': []})
        self.fit[('overall', 'all')]
        for prefix in ('source_desktops', 'fill_desktops'):
            for name in ('shell', 'style', 'color_scheme', 'resolution_group', 'display_scale', 'cleanliness',
                         'resolution', 'show_taskbar', 'show_dock', 'show_status_bar', 'show_top_panel',
                         'application_count', 'application_type', 'wallpaper_category', 'wallpaper_subtype', 'wallpaper_placement'):
                self.cat(prefix + '_' + name, None)
            self.num(prefix + '_visible_dock_height', None, 'dimension')
        for name in ('style', 'theme', 'status', 'rows', 'columns', 'mine_count', 'cell_width', 'cell_height',
                     'sampled_cell_size', 'requested_cell_size', 'resolved_cell_size', 'requested_zoom', 'requested_display_scale'):
            self.cat('board_' + name, None)
        self.num('board_area_cells', None, 'board_area')
        self.num('board_mine_density', None, 'fraction')

    def cat(self, key, value):
        counter = self.categories[key]
        if value is not None:
            if isinstance(value, bool): value = 'true' if value else 'false'
            counter[str(value)] += 1

    def num(self, key, value, kind):
        self.kinds[key] = kind
        if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
            self.values[key].append(value)

    def desktop(self, desktop, prefix):
        if not desktop: return
        self.totals[prefix] += 1
        appearance = desktop.get('appearance') or {}
        screen = desktop.get('screen') or {}
        shell = appearance.get('shell')
        for name, value in (('shell', shell), ('style', appearance.get('style')),
                            ('color_scheme', appearance.get('color_scheme')),
                            ('resolution_group', desktop.get('resolution_group')),
                            ('display_scale', screen.get('scale')), ('cleanliness', desktop.get('cleanliness'))):
            self.cat(prefix + '_' + name, value)
        if screen.get('width') is not None and screen.get('height') is not None:
            self.cat(prefix + '_resolution', f"{screen['width']}x{screen['height']}")
        relevant = {'taskbar': ('show_taskbar',), 'dock': ('show_dock', 'show_status_bar'),
                    'top_panel': ('show_top_panel',)}.get(shell, ())
        for name in relevant:
            self.cat(prefix + '_' + name, appearance.get(name))
        if shell == 'dock' and appearance.get('show_dock') is True:
            self.num(prefix + '_visible_dock_height', appearance.get('dock_height'), 'dimension')
        apps = desktop.get('applications')
        if apps is not None:
            self.cat(prefix + '_application_count', len(apps))
            for app in apps: self.cat(prefix + '_application_type', app)
        wallpaper = desktop.get('wallpaper') or {}
        for name in ('category', 'subtype', 'placement'):
            self.cat(prefix + '_wallpaper_' + name, wallpaper.get(name))

    def board(self, board, mode, resolution_group=None):
        self.totals['boards'] += 1
        for name in ('style', 'theme', 'status', 'rows', 'columns', 'mine_count', 'cell_width', 'cell_height', 'sampled_cell_size',
                     'requested_cell_size', 'resolved_cell_size', 'requested_zoom', 'requested_display_scale'):
            self.cat('board_' + name, board.get(name))
        rows, cols = board.get('rows'), board.get('columns')
        if rows is not None and cols is not None:
            self.num('board_area_cells', rows*cols, 'board_area')
            if board.get('mine_count') is not None and rows*cols:
                self.num('board_mine_density', board['mine_count']/(rows*cols), 'fraction')
        for name, kind in (('requested_effective_scale', 'scale'), ('pitch_x', 'pitch'), ('pitch_y', 'pitch')):
            self.num('board_' + name, board.get(name), kind)
        sampled, requested, resolved = (board.get(k) for k in ('sampled_cell_size', 'requested_cell_size', 'resolved_cell_size'))
        if (any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                for v in (sampled, requested, resolved)) or mode not in ('game', 'grid', 'desktop')):
            return
        budget = mode in ('game', 'grid') and sampled > requested
        placement = mode == 'desktop' and requested > resolved
        any_fit = sampled > resolved
        for group, value in (('overall', 'all'), ('mode', mode), ('zoom', board.get('requested_zoom')),
                             ('display_scale', board.get('requested_display_scale')),
                             ('board_area', area_bucket(rows, cols)), ('resolution_group', resolution_group)):
            if value is None: continue
            item = self.fit[(group, str(value))]
            item['budget_applicable'] += mode in ('game', 'grid')
            item['placement_applicable'] += mode == 'desktop'
            item['applicable'] += 1; item['budget_fitted'] += budget; item['placement_fitted'] += placement
            item['any_fitted'] += any_fit; item['reductions'].append(sampled-resolved)
        if mode in ('game', 'grid'):
            self.totals['budget_applicable_boards'] += 1; self.totals['budget_fitted_boards'] += budget
            self.num('budget_reduction', sampled-requested, 'reduction')
            self.num('budget_relative_reduction', (sampled-requested)/sampled, 'fraction')
        if mode == 'desktop':
            self.totals['placement_applicable_boards'] += 1; self.totals['placement_fitted_boards'] += placement
            self.num('placement_reduction', requested-resolved, 'reduction')
            self.num('placement_relative_reduction', (requested-resolved)/requested, 'fraction')
        self.num('total_reduction', sampled-resolved, 'reduction')

    def result(self):
        fitting = defaultdict(dict)
        for (group, value), data in sorted(self.fit.items()):
            count = data['applicable']; any_fit = data['any_fitted']
            fitting[group][value] = {'applicable_boards': count, 'budget_fitted': data['budget_fitted'],
                'budget_applicable_boards': data['budget_applicable'],
                'placement_applicable_boards': data['placement_applicable'],
                'placement_fitted': data['placement_fitted'], 'any_fitted': any_fit, 'not_fitted': count-any_fit,
                'budget_fitting_rate': data['budget_fitted']/data['budget_applicable'] if data['budget_applicable'] else None,
                'placement_fitting_rate': data['placement_fitted']/data['placement_applicable'] if data['placement_applicable'] else None,
                'any_fitted_rate': any_fit/count if count else None,
                'total_reduction': numeric_summary(data['reductions'], BINS['reduction'])}
        totals = dict(sorted(self.totals.items()))
        for name in ('images', 'positive_images', 'negative_images', 'boards', 'source_desktops', 'fill_desktops',
                     'standalone_images', 'padded_images', 'budget_applicable_boards', 'budget_fitted_boards',
                     'placement_applicable_boards', 'placement_fitted_boards'):
            totals.setdefault(name, 0)
        for kind in ('budget', 'placement'):
            denominator = totals[kind + '_applicable_boards']
            totals[kind + '_fitting_rate'] = totals[kind + '_fitted_boards']/denominator if denominator else None
        classified = totals['positive_images'] + totals['negative_images']
        totals['classification_applicable_images'] = classified
        totals['negative_image_rate'] = totals['negative_images']/classified if classified else None
        padded_count = totals.get('padding_observations', 0)
        totals['padding_observations'] = padded_count
        totals['padded_image_rate'] = totals['padded_images']/padded_count if padded_count else None
        return {'totals': totals,
                'categorical': {key: {'count': sum(counter.values()), 'counts': dict(sorted(counter.items()))}
                                for key, counter in sorted(self.categories.items())},
                'numerical': {key: numeric_summary(self.values[key], BINS[kind]) for key, kind in sorted(self.kinds.items())},
                'fitting': dict(fitting)}


CONVENTIONS = {'percentiles': 'linear interpolation at (n-1)*p',
    'histograms': '[left,right), final right inclusive; underflow and overflow separate',
    'missing_values': 'excluded; category count is applicable observations',
    'board_area_buckets': '0-100, 101-300, 301-700, 701-900, >900 cells',
    'fitting': 'sampled-requested budget, requested-resolved placement; accepted boards only'}
