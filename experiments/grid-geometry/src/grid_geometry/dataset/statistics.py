"""Accepted geometry statistics from existing and extended sample metadata."""
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
        split = record.get('split', 'unknown')
        if split not in self.splits: self.splits[split] = self._new()
        for summary in (self.overall, self.splits[split]):
            self._add(summary, record, self.standalone_source_budget)

    @staticmethod
    def _add(summary, record, budget):
        summary.totals['images'] += 1
        board_id = record.get('board_id')
        if 'board_id' in record:
            summary.totals['positive_images' if board_id else 'negative_images'] += 1
        mode = record.get('scene_mode')
        if mode == 'game': summary.totals['standalone_images'] += 1
        summary.cat('source_mode', mode)
        summary.cat('category', record.get('category'))
        if board_id:
            summary.cat('requested_positive_category', record.get('requested_category'))
            summary.cat('realized_positive_category', record.get('category'))
            summary.cat('requested_pitch_bin', record.get('requested_pitch_bin'))
        source_size = record.get('source_size')
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
            game = record.get('game_specification') or {}
            dimensions = record.get('board_dimensions') or (None, None)
            appearance = record.get('appearance') or {}
            original = record.get('original_geometry') or {}
            board = {key: record.get(key) for key in ('sampled_cell_size', 'requested_cell_size', 'resolved_cell_size',
                       'requested_zoom', 'requested_display_scale', 'requested_effective_scale', 'style', 'theme')}
            board.update(rows=dimensions[0], columns=dimensions[1], mine_count=game.get('mine_count'),
                         status=game.get('status'), pitch_x=original.get('pitch_x'), pitch_y=original.get('pitch_y'),
                         cell_width=appearance.get('cell_width'), cell_height=appearance.get('cell_height'))
            summary.board(board, mode)
            geometry, final_size = record.get('geometry'), record.get('final_size')
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
        transform = record.get('transform') or {}
        for axis in ('x', 'y'):
            value = transform.get('scale_' + axis)
            summary.num('resampling_scale_' + axis, value, 'scale')
            if value is not None:
                state = 'neutral' if abs(value-1.) <= NEUTRAL_TOLERANCE else 'upsampling' if value > 1 else 'downsampling'
                summary.cat('resampling_' + axis, state)
        padding = record.get('padding_type')
        if 'padding_type' in record: summary.totals['padding_observations'] += 1
        if padding is not None:
            summary.totals['padded_images'] += 1
            summary.cat('padding_side', record.get('padding_side'))
            extent, size, side = record.get('padding_extent'), record.get('final_size'), record.get('padding_side')
            summary.num('padding_extent', extent, 'dimension')
            if extent is not None and size and side:
                summary.num('padding_extent_fraction', extent/size[0 if side in ('left', 'right') else 1], 'fraction')
        fraction = record.get('padding_fraction')
        summary.num('padding_fraction', fraction, 'fraction')
        if fraction is not None: summary.num('known_fraction', 1-fraction, 'fraction')
        for name in ('grid_coverage_fraction', 'phase_supervision_fraction', 'grid_occupancy_fraction'):
            summary.num(name, record.get(name), 'fraction')
        augmentation = record.get('augmentation') or {}
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
