"""Pure aggregation of compact accepted detection records."""
import pytest
from grid_detection.dataset.statistics import AcceptedStatistics


def test_multiboard_normalized_geometry_and_split_aggregation():
    stats = AcceptedStatistics()
    base = {'style': 'classic', 'theme': 'default', 'status': 'new', 'rows': 9, 'columns': 9,
            'mine_count': 10, 'sampled_cell_size': 20, 'requested_cell_size': 20,
            'resolved_cell_size': 20, 'requested_zoom': 1., 'requested_display_scale': 1.,
            'requested_effective_scale': 1., 'pitch_x': 20, 'pitch_y': 20}
    stats.add({'split': 'train', 'scene_mode': 'game', 'board_ids': ['a'], 'board_count': 1,
        'native_size': [100, 100], 'final_size': [200, 100], 'augmentation_enabled': False,
        'boards': [{**base, 'final_grid_box': [20, 10, 60, 60]}]})
    desktop = {'screen': {'width': 800, 'height': 600, 'scale': 1.5}, 'resolution_group': 'retro',
        'appearance': {'shell': 'dock', 'style': 'classic', 'color_scheme': 'dark',
                       'show_dock': False, 'show_status_bar': True},
        'applications': ['browser', 'browser'], 'wallpaper': {'category': 'procedural', 'subtype': 'solid'}}
    stats.add({'split': 'val', 'scene_mode': 'desktop', 'board_ids': ['b', 'c'], 'board_count': 2,
        'desktop': desktop, 'native_size': [100, 100], 'final_size': [100, 100], 'augmentation_enabled': True,
        'boards': [{**base, 'final_grid_box': [0, 0, 50, 50]}, {**base, 'final_grid_box': [0, 0, 100, 100]}],
        'augmentation': {'resize_factor': 1., 'actual_scale': [1., 1.], 'resampling': 'LANCZOS',
                         'jpeg_quality': None, 'blur': 0., 'brightness': 1., 'contrast': 1.}})
    stats.add({'split': 'val', 'scene_mode': 'desktop', 'board_ids': [], 'board_count': 0,
               'desktop': desktop, 'boards': [], 'native_size': [100, 100], 'final_size': [100, 100], 'augmentation_enabled': False})
    result = stats.result({'train': 1, 'val': 2, 'test': 0})
    overall = result['overall']; totals = overall['totals']
    assert totals['images'] == 3 and totals['boards'] == 3 and totals['negative_images'] == 1
    assert totals['source_desktops'] == 2 and totals['negative_image_rate'] == pytest.approx(1/3)
    assert overall['numerical']['grid_normalized_area']['mean'] == pytest.approx((.1 + .25 + 1)/3)
    assert overall['categorical']['source_desktops_application_type']['counts']['browser'] == 4
    assert overall['categorical']['source_desktops_show_dock']['counts']['false'] == 2
    assert overall['categorical']['augmentation_state']['counts'] == {'disabled': 2, 'enabled': 1}
    assert result['splits']['test']['totals']['images'] == 0
    assert result['splits']['test']['numerical']['grid_normalized_area']['mean'] is None
    assert result['splits']['train']['numerical']['grid_normalized_area']['histogram']['edges'] == overall['numerical']['grid_normalized_area']['histogram']['edges']


@pytest.mark.parametrize('field', ['split', 'boards', 'board_count', 'scene_mode', 'native_size', 'final_size', 'augmentation_enabled'])
def test_missing_required_record_field(field):
    record = {'split': 'train', 'boards': [], 'board_count': 0, 'scene_mode': 'game',
              'native_size': [100, 100], 'final_size': [100, 100], 'augmentation_enabled': False}
    del record[field]
    with pytest.raises(ValueError, match='missing required detection record fields'):
        AcceptedStatistics().add(record)


def test_invalid_board_box_rejected():
    record = {'split': 'train', 'boards': [{'final_grid_box': [0, 0, 0, 3]}], 'board_count': 1,
              'scene_mode': 'game', 'native_size': [100, 100], 'final_size': [100, 100], 'augmentation_enabled': False}
    with pytest.raises(ValueError, match='final grid box'):
        AcceptedStatistics().add(record)


def test_statistics_failure_does_not_complete_config(monkeypatch, tmp_path):
    import json
    from grid_detection.dataset import generator
    monkeypatch.setattr(generator, 'run_provenance', lambda *args: {'source': {'commit': 'unchanged'}})
    def fail(*args, **kwargs):
        raise RuntimeError('statistics failed')
    monkeypatch.setattr(generator.AcceptedStatistics, 'result', fail)
    with pytest.raises(RuntimeError, match='statistics failed'):
        generator.generate_dataset(tmp_path / 'dataset', train=0, val=0, progress=False)
    config = json.loads((tmp_path / 'dataset' / 'config.json').read_text())
    assert config['status'] == 'incomplete' and config['provenance']['source']['commit'] == 'unchanged'


def test_sampling_metadata_does_not_change_seeded_scene(monkeypatch):
    from grid_detection.dataset import sampler
    config = sampler.SAMPLING_CONFIG
    monkeypatch.setitem(config, 'scene_mode', {'desktop': 1})
    monkeypatch.setitem(config, 'game_count', {0: 1})
    monkeypatch.setitem(config, 'wallpaper_type', {'procedural': 1})
    metadata = {}
    scene = sampler.sample_scene(9, sampling_metadata=metadata)
    assert scene == sampler.sample_scene(9)
    assert metadata['resolution_group'] in config['resolution']
