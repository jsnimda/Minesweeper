"""Accepted detection statistics from compact manifest records.

Older records may lack fields: missing measurements are excluded, never inferred.
"""
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
        split = record.get('split', 'unknown')
        if split not in self.splits: self.splits[split] = self._new()
        for summary in (self.overall, self.splits[split]): self._add(summary, record)

    @staticmethod
    def _add(summary, record):
        boards = record.get('boards') or []
        count = record.get('board_count')
        if count is None and 'board_ids' in record: count = len(record['board_ids'])
        mode = record.get('scene_mode')
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
        enabled = record.get('augmentation_enabled')
        if enabled is None and augmentation is not None: enabled = True
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
