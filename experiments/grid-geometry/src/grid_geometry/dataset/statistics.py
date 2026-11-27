"""Accepted geometry statistics from existing and extended sample metadata."""
import math

from synthetic_scenes.dataset_statistics import CONVENTIONS, Summary

NEUTRAL_TOLERANCE = 1e-9


class AcceptedStatistics:
    def __init__(self, standalone_source_budget=None):
        self.standalone_source_budget = standalone_source_budget
        self.overall = self._new()
        self.splits = {}

    @staticmethod
    def _new():
        result = Summary()
        for name in ('source_mode', 'category', 'requested_positive_category', 'realized_positive_category',
                     'requested_pitch_bin', 'generated_fill', 'context_type', 'fill_procedural_subtype',
                     'anisotropy_selected', 'resampling_x', 'resampling_y', 'padding_side',
                     'augmentation_resampling', 'augmentation_blur', 'standalone_source_budget'):
            result.cat(name, None)
        for name, kind in (
            ('source_width', 'dimension'), ('source_height', 'dimension'), ('source_pixels', 'pixels'),
            ('desired_pitch', 'pitch'), ('final_pitch_x', 'pitch'), ('final_pitch_y', 'pitch'),
            ('pitch_error_x', 'error'), ('pitch_error_y', 'error'), ('anisotropy_ratio', 'scale'),
            ('resampling_scale_x', 'scale'), ('resampling_scale_y', 'scale'),
            ('padding_extent', 'dimension'), ('padding_extent_fraction', 'fraction'),
            ('known_fraction', 'fraction'), ('padding_fraction', 'fraction'),
            ('grid_bbox_image_fraction', 'fraction'), ('grid_coverage_fraction', 'fraction'), ('phase_supervision_fraction', 'fraction'),
            ('grid_occupancy_fraction', 'fraction'), ('augmentation_brightness', 'photometric'),
            ('augmentation_contrast', 'photometric'), ('board_pitch_x', 'pitch'), ('board_pitch_y', 'pitch'),
            ('board_requested_effective_scale', 'scale'), ('total_reduction', 'reduction'),
            ('budget_reduction', 'reduction'), ('budget_relative_reduction', 'fraction'),
            ('placement_reduction', 'reduction'), ('placement_relative_reduction', 'fraction')):
            result.num(name, None, kind)
        return result

    def add(self, record):
        required = ('sample_id', 'split', 'board_id', 'scene_mode', 'source_size', 'final_size',
                    'geometry', 'transform', 'padding_type', 'padding_side', 'padding_extent',
                    'padding_fraction', 'augmentation')
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f'missing required geometry record fields: {missing}')
        if not isinstance(record['sample_id'], str) or not record['sample_id']:
            raise ValueError('invalid geometry sample identity')
        if record['split'] not in ('train', 'val', 'test') or record['scene_mode'] not in ('game', 'desktop'):
            raise ValueError('invalid geometry split or scene mode')
        for name in ('source_size', 'final_size'):
            size = record[name]
            if not isinstance(size, (tuple, list)) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
                raise ValueError(f'invalid geometry {name}')
        transform = record['transform']
        if not isinstance(transform, dict) or any(type(transform.get(k)) not in (int, float) or not math.isfinite(transform[k]) or transform[k] <= 0 for k in ('scale_x', 'scale_y')):
            raise ValueError('invalid geometry transform')
        augmentation = record['augmentation']
        if not isinstance(augmentation, dict) or any(k not in augmentation for k in ('resampling', 'blur', 'brightness', 'contrast')):
            raise ValueError('invalid geometry augmentation metadata')
        if (type(record['padding_extent']) is not int or record['padding_extent'] < 0
                or type(record['padding_fraction']) not in (int, float)
                or not math.isfinite(record['padding_fraction']) or not 0 <= record['padding_fraction'] <= 1):
            raise ValueError('invalid geometry padding observation')
        if record['padding_type'] is not None and record['padding_side'] not in ('left', 'right', 'top', 'bottom'):
            raise ValueError('invalid geometry padding side')
        board_id = record['board_id']
        if board_id is not None:
            if not isinstance(board_id, str) or not board_id:
                raise ValueError('invalid geometry board identity')
            fields = ('board_dimensions', 'game_specification', 'appearance', 'original_geometry',
                      'sampled_cell_size', 'requested_cell_size', 'resolved_cell_size',
                      'requested_zoom', 'requested_display_scale', 'requested_effective_scale', 'style', 'theme',
                      'requested_pitch', 'effective_pitch_x', 'effective_pitch_y', 'anisotropy_ratio')
            if any(record.get(k) is None for k in fields):
                raise ValueError('missing positive geometry board metadata')
            dimensions = record['board_dimensions']
            if not isinstance(dimensions, (tuple, list)) or len(dimensions) != 2 or any(type(v) is not int or v <= 0 for v in dimensions):
                raise ValueError('invalid positive geometry board dimensions')
            for name, keys in (('game_specification', ('mine_count', 'status')), ('appearance', ('cell_width', 'cell_height')),
                               ('original_geometry', ('pitch_x', 'pitch_y')), ('geometry', ('visible_box',))):
                value = record.get(name)
                if not isinstance(value, dict) or any(value.get(k) is None for k in keys):
                    raise ValueError(f'invalid positive geometry {name}')
            for name in ('geometry', 'original_geometry'):
                value = record[name]
                for key in ('origin_x', 'origin_y', 'pitch_x', 'pitch_y'):
                    item = value.get(key)
                    if (type(item) not in (int, float) or not math.isfinite(item)
                            or (key.startswith('pitch_') and item <= 0)):
                        raise ValueError(f'invalid positive geometry {name} {key}')
            for key in ('sampled_cell_size', 'requested_cell_size', 'resolved_cell_size',
                        'requested_zoom', 'requested_display_scale', 'requested_effective_scale',
                        'requested_pitch', 'effective_pitch_x', 'effective_pitch_y', 'anisotropy_ratio'):
                value = record[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'invalid positive geometry {key}')
            box = record['geometry']['visible_box']
            if (not isinstance(box, (tuple, list)) or len(box) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError('invalid positive geometry visible box')
        elif record['geometry'] is not None:
            raise ValueError('negative geometry record contains board geometry')
        split = record['split']
        if split not in self.splits: self.splits[split] = self._new()
        for summary in (self.overall, self.splits[split]):
            self._add(summary, record, self.standalone_source_budget)

    @staticmethod
    def _add(summary, record, budget):
        summary.totals['images'] += 1
        board_id = record['board_id']
        summary.totals['positive_images' if board_id else 'negative_images'] += 1
        mode = record['scene_mode']
        if mode == 'game': summary.totals['standalone_images'] += 1
        summary.cat('source_mode', mode)
        summary.cat('category', record.get('category'))
        if board_id:
            summary.cat('requested_positive_category', record.get('requested_category'))
            summary.cat('realized_positive_category', record.get('category'))
            summary.cat('requested_pitch_bin', record.get('requested_pitch_bin'))
        source_size = record['source_size']
        if source_size:
            summary.num('source_width', source_size[0], 'dimension')
            summary.num('source_height', source_size[1], 'dimension')
            summary.num('source_pixels', source_size[0]*source_size[1], 'pixels')
            if mode == 'game' and budget is not None:
                within = (max(source_size) <= budget['max_dimension']
                          and source_size[0]*source_size[1] <= budget['max_pixels'])
                summary.cat('standalone_source_budget', 'within' if within else 'exceeds')
        summary.desktop(record.get('source_desktop'), 'source_desktops')
        summary.desktop(record.get('fill_desktop'), 'fill_desktops')
        background = record.get('background_visible')
        if background is not None:
            summary.cat('generated_fill', record.get('context_type') if background else 'none')
        summary.cat('context_type', record.get('context_type'))
        summary.cat('fill_procedural_subtype', record.get('context_subtype'))
        if board_id:
            game = record['game_specification']
            dimensions = record['board_dimensions']
            appearance = record['appearance']
            original = record['original_geometry']
            board = {key: record.get(key) for key in ('sampled_cell_size', 'requested_cell_size', 'resolved_cell_size',
                       'requested_zoom', 'requested_display_scale', 'requested_effective_scale', 'style', 'theme')}
            board.update(rows=dimensions[0], columns=dimensions[1], mine_count=game.get('mine_count'),
                         status=game.get('status'), pitch_x=original.get('pitch_x'), pitch_y=original.get('pitch_y'),
                         cell_width=appearance.get('cell_width'), cell_height=appearance.get('cell_height'))
            summary.board(board, mode)
            geometry, final_size = record['geometry'], record['final_size']
            if geometry and final_size:
                left, top, right, bottom = geometry['visible_box']
                width, height = final_size
                intersection = max(0., min(width, right)-max(0., left)) * max(0., min(height, bottom)-max(0., top))
                summary.num('grid_bbox_image_fraction', intersection/(width*height), 'fraction')
            desired, x, y = (record.get(k) for k in ('requested_pitch', 'effective_pitch_x', 'effective_pitch_y'))
            aspect = record.get('anisotropy_ratio')
            for name, value, kind in (('desired_pitch', desired, 'pitch'), ('final_pitch_x', x, 'pitch'),
                ('final_pitch_y', y, 'pitch'), ('anisotropy_ratio', aspect, 'scale')):
                summary.num(name, value, kind)
            summary.cat('anisotropy_selected', record.get('anisotropy_selected'))
            if desired is not None and x is not None: summary.num('pitch_error_x', abs(x-desired), 'error')
            if desired is not None and y is not None and aspect is not None:
                summary.num('pitch_error_y', abs(y-desired*aspect), 'error')
        transform = record['transform']
        for axis in ('x', 'y'):
            value = transform.get('scale_' + axis)
            summary.num('resampling_scale_' + axis, value, 'scale')
            if value is not None:
                state = 'neutral' if abs(value-1.) <= NEUTRAL_TOLERANCE else 'upsampling' if value > 1 else 'downsampling'
                summary.cat('resampling_' + axis, state)
        padding = record['padding_type']
        summary.totals['padding_observations'] += 1
        if padding is not None:
            summary.totals['padded_images'] += 1
            summary.cat('padding_side', record.get('padding_side'))
            extent, size, side = record.get('padding_extent'), record.get('final_size'), record.get('padding_side')
            summary.num('padding_extent', extent, 'dimension')
            if extent is not None and size and side:
                summary.num('padding_extent_fraction', extent/size[0 if side in ('left', 'right') else 1], 'fraction')
        fraction = record['padding_fraction']
        summary.num('padding_fraction', fraction, 'fraction')
        if fraction is not None: summary.num('known_fraction', 1-fraction, 'fraction')
        for name in ('grid_coverage_fraction', 'phase_supervision_fraction', 'grid_occupancy_fraction'):
            summary.num(name, record.get(name), 'fraction')
        augmentation = record['augmentation']
        for name in ('resampling', 'blur'): summary.cat('augmentation_' + name, augmentation.get(name))
        for name in ('brightness', 'contrast'): summary.num('augmentation_' + name, augmentation.get(name), 'photometric')

    def result(self, splits):
        requested = dict(splits) if isinstance(splits, dict) else {split: None for split in splits}
        return {'conventions': {**CONVENTIONS, 'resampling_neutral_tolerance': NEUTRAL_TOLERANCE,
            'coverage': 'bbox fraction is continuous clipped area / final image area; grid/phase coverage is pixel-center count restricted to known pixels / image pixels; occupancy is soft target mean',
            'padding_extent_fraction': 'fraction of corresponding final image dimension',
            'crop_category': 'accepted category passed the existing edge-count validation'},
            'requested_images': requested,
            'overall': self.overall.result(),
            'splits': {split: self.splits.get(split, self._new()).result() for split in requested}}
