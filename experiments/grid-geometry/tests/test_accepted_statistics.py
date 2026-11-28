"""Pure accepted geometry aggregation and lifecycle contracts."""
import json
import pytest
from grid_geometry.dataset.statistics import AcceptedStatistics
from grid_geometry.dataset import generator
from grid_geometry.dataset.sampling_config import GenerationConfig


def test_geometry_pitch_context_and_resampling():
    stats = AcceptedStatistics()
    source = {'screen': {'width': 1600, 'height': 1000, 'scale': 1.5},
              'appearance': {'shell': 'taskbar', 'style': 'modern', 'color_scheme': 'light', 'show_taskbar': True},
              'applications': ['browser'], 'wallpaper': {'category': 'procedural', 'subtype': 'solid'}}
    fill = {'screen': {'width': 384, 'height': 384, 'scale': 1.},
            'appearance': {'shell': 'dock', 'style': 'classic', 'color_scheme': 'dark', 'show_dock': False,
                           'show_status_bar': False}, 'applications': ['sudoku']}
    stats.add({'sample_id': 'a', 'split': 'train', 'board_id': 'a', 'scene_mode': 'desktop', 'source_desktop': source,
        'fill_desktop': fill, 'source_size': [1600, 1000], 'final_size': [384, 384],
        'requested_category': 'side', 'category': 'side', 'requested_pitch_bin': '20-31',
        'requested_pitch': 24., 'effective_pitch_x': 24., 'effective_pitch_y': 25.2,
        'anisotropy_ratio': 1.05, 'anisotropy_selected': True,
        'board_dimensions': [30, 20], 'game_specification': {'mine_count': 80, 'status': 'active'},
        'sampled_cell_size': 32, 'requested_cell_size': 32, 'resolved_cell_size': 24,
        'requested_zoom': 1.25, 'requested_display_scale': 1.5, 'requested_effective_scale': 1.875,
        'style': 'classic', 'theme': 'default', 'appearance': {'cell_width': 50, 'cell_height': 50},
        'geometry': {'origin_x': 0, 'origin_y': 0, 'pitch_x': 24., 'pitch_y': 25.2, 'visible_box': [0, 0, 384, 384]},
        'original_geometry': {'origin_x': 0, 'origin_y': 0, 'pitch_x': 50, 'pitch_y': 50}, 'transform': {'scale_x': .48, 'scale_y': 1.},
        'background_visible': True, 'context_type': 'desktop',
        'grid_coverage_fraction': .5, 'phase_supervision_fraction': .5, 'grid_occupancy_fraction': .51,
        'augmentation': {'resampling': 'BILINEAR', 'blur': .25, 'brightness': 1., 'contrast': 1.}})
    stats.add({'sample_id': 'b', 'split': 'val', 'board_id': None, 'scene_mode': 'desktop', 'source_desktop': source,
        'source_size': [1600, 1000], 'final_size': [384, 384], 'geometry': None, 'category': 'hard_negative', 'background_visible': False,
        'context_type': 'source_desktop',
        'augmentation': {'resampling': 'BILINEAR', 'blur': 0., 'brightness': 1., 'contrast': 1.},
        'transform': {'scale_x': 2., 'scale_y': 1.}})
    result = stats.result({'train': 1, 'val': 1, 'test': 0})
    overall = result['overall']; totals = overall['totals']
    assert totals['images'] == 2 and totals['boards'] == 1 and totals['negative_images'] == 1
    assert totals['source_desktops'] == 2 and totals['fill_desktops'] == 1
    assert totals['placement_fitted_boards'] == 1
    assert {'padded_images', 'padding_observations', 'padded_image_rate'}.isdisjoint(totals)
    assert overall['numerical']['pitch_error_x']['max'] == 0
    assert overall['numerical']['pitch_error_y']['max'] == pytest.approx(0., abs=1e-12)
    assert overall['numerical']['source_pixels']['mean'] == 1600000
    assert overall['categorical']['resampling_x']['counts'] == {'downsampling': 1, 'upsampling': 1}
    assert overall['categorical']['resampling_y']['counts'] == {'neutral': 2}
    assert overall['categorical']['generated_fill']['counts'] == {'desktop': 1, 'none': 1}
    assert result['splits']['test']['numerical']['desired_pitch']['count'] == 0


@pytest.mark.parametrize('field', ['sample_id', 'split', 'board_id', 'scene_mode', 'source_size', 'final_size',
                                    'geometry', 'transform', 'augmentation'])
def test_missing_required_record_field(field):
    record = {'sample_id': 'a', 'split': 'train', 'board_id': None, 'scene_mode': 'game',
              'source_size': [32, 32], 'final_size': [32, 32], 'geometry': None,
              'transform': {'scale_x': 1., 'scale_y': 1.},
              'augmentation': {'resampling': 'BILINEAR', 'blur': 0., 'brightness': 1., 'contrast': 1.}}
    del record[field]
    with pytest.raises(ValueError, match='missing required geometry record fields'):
        AcceptedStatistics().add(record)


def test_positive_requires_board_metadata():
    record = {'sample_id': 'a', 'split': 'train', 'board_id': 'board', 'scene_mode': 'game',
              'source_size': [32, 32], 'final_size': [32, 32], 'geometry': {},
              'transform': {'scale_x': 1., 'scale_y': 1.},
              'augmentation': {'resampling': 'BILINEAR', 'blur': 0., 'brightness': 1., 'contrast': 1.}}
    with pytest.raises(ValueError, match='positive geometry board metadata'):
        AcceptedStatistics().add(record)


def test_statistics_failure_keeps_incomplete(monkeypatch, tmp_path):
    import synthetic_scenes.provenance as provenance
    monkeypatch.setattr(provenance, 'run_provenance', lambda *args: {'source': {'commit': 'kept'}})
    def fail(*args, **kwargs): raise RuntimeError('statistics failed')
    monkeypatch.setattr(generator.AcceptedStatistics, 'result', fail)
    with pytest.raises(RuntimeError, match='statistics failed'):
        generator.generate_dataset(tmp_path / 'dataset', GenerationConfig(train=0, val=0, test=0, progress=False))
    config = json.loads((tmp_path / 'dataset' / 'config.json').read_text())
    assert config['status'] == 'incomplete' and config['provenance']['source']['commit'] == 'kept'
