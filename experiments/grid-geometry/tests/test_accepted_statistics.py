"""Pure accepted geometry aggregation and lifecycle contracts."""
import json
import pytest
from grid_geometry.dataset.statistics import AcceptedStatistics
from grid_geometry.dataset import generator
from grid_geometry.dataset.sampling_config import GenerationConfig


def test_geometry_pitch_context_padding_and_resampling():
    stats = AcceptedStatistics()
    source = {'screen': {'width': 1600, 'height': 1000, 'scale': 1.5},
              'appearance': {'shell': 'taskbar', 'style': 'modern', 'color_scheme': 'light', 'show_taskbar': True},
              'applications': ['browser'], 'wallpaper': {'category': 'procedural', 'subtype': 'solid'}}
    fill = {'screen': {'width': 384, 'height': 384, 'scale': 1.},
            'appearance': {'shell': 'dock', 'style': 'classic', 'color_scheme': 'dark', 'show_dock': False,
                           'show_status_bar': False}, 'applications': ['sudoku']}
    stats.add({'split': 'train', 'board_id': 'a', 'scene_mode': 'desktop', 'source_desktop': source,
        'fill_desktop': fill, 'source_size': [1600, 1000], 'final_size': [384, 384],
        'requested_category': 'side', 'category': 'side', 'requested_pitch_bin': '20-31',
        'requested_pitch': 24., 'effective_pitch_x': 24., 'effective_pitch_y': 25.2,
        'anisotropy_ratio': 1.05, 'anisotropy_selected': True,
        'board_dimensions': [30, 20], 'game_specification': {'mine_count': 80, 'status': 'active'},
        'sampled_cell_size': 32, 'requested_cell_size': 32, 'resolved_cell_size': 24,
        'requested_zoom': 1.25, 'requested_display_scale': 1.5, 'requested_effective_scale': 1.875,
        'original_geometry': {'pitch_x': 50, 'pitch_y': 50}, 'transform': {'scale_x': .48, 'scale_y': 1.},
        'background_visible': True, 'context_type': 'desktop', 'padding_type': 'black',
        'padding_side': 'left', 'padding_extent': 38, 'padding_fraction': 38/384,
        'grid_coverage_fraction': .5, 'phase_supervision_fraction': .5, 'grid_occupancy_fraction': .51,
        'augmentation': {'resampling': 'BILINEAR', 'blur': .25, 'brightness': 1., 'contrast': 1.}})
    stats.add({'split': 'val', 'board_id': None, 'scene_mode': 'desktop', 'source_desktop': source,
        'source_size': [1600, 1000], 'category': 'hard_negative', 'background_visible': False,
        'context_type': 'source_desktop', 'padding_type': None, 'padding_fraction': 0.,
        'transform': {'scale_x': 2., 'scale_y': 1.}})
    result = stats.result({'train': 1, 'val': 1, 'test': 0})
    overall = result['overall']; totals = overall['totals']
    assert totals['images'] == 2 and totals['boards'] == 1 and totals['negative_images'] == 1
    assert totals['source_desktops'] == 2 and totals['fill_desktops'] == 1
    assert totals['placement_fitted_boards'] == 1 and totals['padded_image_rate'] == .5
    assert overall['numerical']['pitch_error_x']['max'] == 0
    assert overall['numerical']['pitch_error_y']['max'] == pytest.approx(0., abs=1e-12)
    assert overall['numerical']['source_pixels']['mean'] == 1600000
    assert overall['numerical']['padding_extent_fraction']['mean'] == pytest.approx(38/384)
    assert overall['numerical']['known_fraction']['mean'] == pytest.approx((1 + 1-38/384)/2)
    assert overall['categorical']['resampling_x']['counts'] == {'downsampling': 1, 'upsampling': 1}
    assert overall['categorical']['resampling_y']['counts'] == {'neutral': 2}
    assert overall['categorical']['generated_fill']['counts'] == {'desktop': 1, 'none': 1}
    assert result['splits']['test']['numerical']['desired_pitch']['count'] == 0


def test_statistics_failure_keeps_incomplete(monkeypatch, tmp_path):
    import synthetic_scenes.provenance as provenance
    monkeypatch.setattr(provenance, 'run_provenance', lambda *args: {'source': {'commit': 'kept'}})
    def fail(*args, **kwargs): raise RuntimeError('statistics failed')
    monkeypatch.setattr(generator.AcceptedStatistics, 'result', fail)
    with pytest.raises(RuntimeError, match='statistics failed'):
        generator.generate_dataset(tmp_path / 'dataset', GenerationConfig(train=0, val=0, test=0, progress=False))
    config = json.loads((tmp_path / 'dataset' / 'config.json').read_text())
    assert config['status'] == 'incomplete' and config['provenance']['source']['commit'] == 'kept'
