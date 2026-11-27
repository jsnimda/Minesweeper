"""Accepted detection statistics from compact manifest records."""
import math
from synthetic_scenes.dataset_statistics import CONVENTIONS, Summary


class AcceptedStatistics:
    def __init__(self):
        self.overall = self._new()
        self.splits = {}

    @staticmethod
    def _new():
        summary = Summary()
        for name in ('scene_mode', 'board_count_per_image', 'augmentation_state', 'augmentation_blur',
                     'augmentation_resampling', 'augmentation_jpeg_quality'):
            summary.cat(name, None)
        for name, kind in (
            ('native_width', 'dimension'), ('native_height', 'dimension'),
            ('final_width', 'dimension'), ('final_height', 'dimension'),
            ('grid_normalized_width', 'fraction'), ('grid_normalized_height', 'fraction'),
            ('grid_normalized_area', 'fraction'), ('augmentation_resize_factor', 'scale'),
            ('augmentation_scale_x', 'scale'), ('augmentation_scale_y', 'scale'),
            ('augmentation_brightness', 'photometric'), ('augmentation_contrast', 'photometric'),
            ('board_pitch_x', 'pitch'), ('board_pitch_y', 'pitch'),
            ('board_requested_effective_scale', 'scale'), ('total_reduction', 'reduction'),
            ('budget_reduction', 'reduction'), ('budget_relative_reduction', 'fraction'),
            ('placement_reduction', 'reduction'), ('placement_relative_reduction', 'fraction')):
            summary.num(name, None, kind)
        return summary

    def add(self, record):
        required = ('split', 'boards', 'board_count', 'scene_mode', 'native_size', 'final_size', 'augmentation_enabled')
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f'missing required detection record fields: {missing}')
        if record['split'] not in ('train', 'val', 'test') or record['scene_mode'] not in ('game', 'grid', 'desktop'):
            raise ValueError('invalid detection split or scene mode')
        boards, count = record['boards'], record['board_count']
        if not isinstance(boards, list) or type(count) is not int or count < 0 or count != len(boards):
            raise ValueError('invalid detection boards or board count')
        for name in ('native_size', 'final_size'):
            size = record[name]
            if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
                raise ValueError(f'invalid detection {name}')
        if type(record['augmentation_enabled']) is not bool:
            raise ValueError('invalid detection augmentation_enabled')
        if record['augmentation_enabled'] and not isinstance(record.get('augmentation'), dict):
            raise ValueError('enabled detection augmentation requires metadata')
        for board in boards:
            box = board.get('final_grid_box') if isinstance(board, dict) else None
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError('invalid detection final grid box')
        split = record['split']
        if split not in self.splits: self.splits[split] = self._new()
        for summary in (self.overall, self.splits[split]): self._add(summary, record)

    @staticmethod
    def _add(summary, record):
        boards = record['boards']
        count = record['board_count']
        mode = record['scene_mode']
        summary.totals['images'] += 1
        if count is not None:
            summary.totals['positive_images' if count else 'negative_images'] += 1
            summary.cat('board_count_per_image', count)
        if mode in ('game', 'grid'): summary.totals['standalone_images'] += 1
        summary.cat('scene_mode', mode)
        desktop = record.get('desktop')
        summary.desktop(desktop, 'source_desktops')
        for name in ('native', 'final'):
            size = record.get(name + '_size')
            if size:
                summary.num(name + '_width', size[0], 'dimension')
                summary.num(name + '_height', size[1], 'dimension')
        augmentation = record.get('augmentation')
        enabled = record['augmentation_enabled']
        if enabled is not None: summary.cat('augmentation_state', 'enabled' if enabled else 'disabled')
        if augmentation is not None:
            for field in ('blur', 'resampling'):
                summary.cat('augmentation_' + field, augmentation.get(field))
            summary.cat('augmentation_jpeg_quality', augmentation.get('jpeg_quality') if augmentation.get('jpeg_quality') is not None else 'none')
            for field, kind in (('resize_factor', 'scale'), ('brightness', 'photometric'), ('contrast', 'photometric')):
                summary.num('augmentation_' + field, augmentation.get(field), kind)
            if augmentation.get('actual_scale'):
                summary.num('augmentation_scale_x', augmentation['actual_scale'][0], 'scale')
                summary.num('augmentation_scale_y', augmentation['actual_scale'][1], 'scale')
        for board in boards:
            summary.board(board, mode, (desktop or {}).get('resolution_group'))
            box, size = board.get('final_grid_box'), record.get('final_size')
            if box and size and size[0] > 0 and size[1] > 0:
                w, h = (box[2]-box[0])/size[0], (box[3]-box[1])/size[1]
                summary.num('grid_normalized_width', w, 'fraction')
                summary.num('grid_normalized_height', h, 'fraction')
                summary.num('grid_normalized_area', w*h, 'fraction')

    def result(self, splits):
        requested = dict(splits) if isinstance(splits, dict) else {split: None for split in splits}
        return {'conventions': {**CONVENTIONS, 'grid_size': 'final half-open grid box / final image dimensions'},
                'requested_images': requested,
                'overall': self.overall.result(),
                'splits': {split: self.splits.get(split, self._new()).result() for split in requested}}
