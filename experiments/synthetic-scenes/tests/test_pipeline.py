import json
import random

import numpy as np
import pytest
from PIL import Image

from synthetic_scenes.coordinates import intersects, transform_board
from synthetic_scenes.scene import GameSpec, plan_windows
from synthetic_scenes.sampling import sample_scene
from synthetic_scenes.augmentation import augment_scene
from synthetic_scenes.exporters import (geometry_sample, geometry_heatmaps, cell_sample,
                                        select_targets, yolo_lines, export_yolo)


def example():
    board = {'board_id':'test','style':'microsoft','rows':1,'columns':2,
             'boxes':{'grid':(3,4,24,14)},'cell_boxes':[[(3,4,13,14),(14,4,24,14)]],
             'x_edges':[3,13.5,24],'y_edges':[4,14],'cell_states':[[9,1]]}
    return Image.new('RGB',(30,20),'white'), {'size':(30,20),'boards':[board]}


def test_plan():
    specs = [GameSpec(seed=i,appearance={'style':style,'cell_size':40})
             for i,style in enumerate(('classic','windows7','google'))]
    plan = plan_windows(specs,(1000,700),random.Random(42))
    assert len(plan)==3
    for i,item in enumerate(plan):
        x,y,r,b = item['box']
        assert 0<=x<r<=1000 and 0<=y<b<=660
        assert not any(intersects(item['box'],p['box']) for p in plan[:i])
    assert plan==plan_windows(specs,(1000,700),random.Random(42))
    with pytest.raises(ValueError):
        plan_windows([GameSpec(rows=100,columns=100)],(240,200),random.Random(1))


def test_plan_reaches_configured_minimum_on_small_desktop():
    appearance = {'style': 'classic', 'cell_size': 40}
    spec = GameSpec(rows=8, appearance=appearance)
    for attempts in (1, 60):
        plan = plan_windows([spec], (240, 220), random.Random(1),
                            min_cell_size=12, attempts=attempts)
        assert plan[0]['cell_size'] == 12
        _, top, _, bottom = plan[0]['box']
        assert (top, bottom) == (0, 180)
    with pytest.raises(ValueError, match='Could not fit all games'):
        plan_windows([GameSpec(appearance=appearance)], (240, 220), random.Random(1))


def test_transform_and_augment():
    image,meta=example()
    result,new=augment_scene(image,meta,scale=.63,blur=.2,jpeg_quality=90)
    assert result.size==(19,13)
    assert new['boards'][0]['boxes']['grid']==pytest.approx((3*19/30,4*13/20,24*19/30,14*13/20))
    assert meta['boards'][0]['x_edges']==[3,13.5,24]


def test_geometry():
    image,meta=example()
    crop,target=geometry_sample(image,meta['boards'][0],margin=0,output_size=(42,20))
    assert crop.size==(42,20)
    assert target['x_edges']==[0,21,42]
    maps=geometry_heatmaps(target)
    assert maps.shape==(3,20,42)
    assert maps.dtype==np.float32 and maps.max()<=1
    assert maps[0,9,9]>.8
    assert geometry_heatmaps(target,include_outer=False)[2].max()==0


@pytest.mark.parametrize('sigma', [0, -1, float('inf'), float('-inf'), float('nan')])
def test_invalid_heatmap_sigma(tmp_path, sigma):
    from synthetic_scenes.dataset import generate_dataset

    image, meta = example()
    _, target = geometry_sample(image, meta['boards'][0])
    with pytest.raises(ValueError, match='finite and positive'):
        geometry_heatmaps(target, sigma=sigma)
    root = tmp_path/'invalid'
    with pytest.raises(ValueError, match='finite and positive'):
        generate_dataset(root, heatmap_sigma=sigma)
    assert not root.exists()


@pytest.mark.parametrize('sigma', [5e-324, 1e308])
def test_extreme_finite_heatmap_sigma(sigma):
    target = {
        'size': (1, 1), 'centers': [[(.5, .5)]],
        'x_edges': [0, 1], 'y_edges': [0, 1], 'grid_box': (0, 0, 1, 1),
    }
    maps = geometry_heatmaps(target, sigma=sigma)
    assert np.isfinite(maps).all()
    assert maps[0, 0, 0] == 1


def test_cells():
    image,meta=example()
    sample=cell_sample(image,meta['boards'][0],cell_size=12,jitter=0)
    assert sample['images'].shape==(2,12,12,3)
    assert sample['labels'].tolist()==[0,2]
    assert sample['surface_labels'].tolist()==[0,1]
    assert sample['content_labels'].tolist()==[0,1]
    indices=select_targets([0,2,-1],10,rng=random.Random(2))
    assert sorted(indices.tolist())==[0,1]


def test_select_targets_stops_when_positive_weights_are_exhausted():
    labels = [0, 1, 2, 1, -1]
    indices = select_targets(labels, 5, rng=random.Random(2), class_weights=[0, 1, 0])
    assert sorted(indices.tolist()) == [1, 3]
    assert select_targets(labels, 5, class_weights=[0, 0, 0]).size == 0


def test_yolo(tmp_path):
    image,meta=example()
    parts=yolo_lines(meta)[0].split()
    assert parts[0]=='1'
    assert [float(x) for x in parts[1:]]==pytest.approx([.45,.45,.7,.5])
    export_yolo(tmp_path,'train','negative',image,{'size':image.size,'boards':[]})
    assert (tmp_path/'labels/train/negative.txt').read_text()==''


def test_sampling():
    seen=set()
    for seed in range(100):
        specs,mode=sample_scene(seed)
        for spec in specs:
            assert spec.seed not in seen
            seen.add(spec.seed)
            if spec.mode=='targeted':
                game=spec.generate()
                assert game['revealed'][game['target_position']]


def test_dataset_without_rasterizer(tmp_path, monkeypatch):
    """Exercise disk output and split IDs using a tiny stand-in screenshot."""
    from synthetic_scenes import dataset

    def fake_render(specs, *, size, seed, mode, wallpaper_path):
        image, meta = example()
        if seed == 85 * 10**9:  # First validation attempt duplicates the first train minefield.
            seed = 84 * 10**9
        for i, board in enumerate(meta['boards']):
            board.update(board_id=f'board-{seed*16+i}',renderer='microsoft_hd',
                         theme='default',style_parameters={},status='active',
                         generation_mode='normal',target_number=None)
        return image,meta

    monkeypatch.setattr(dataset,'render_scene',fake_render)
    root=tmp_path/'data'
    stats=dataset.generate_dataset(root,train_count=2,val_count=1,cell_size=8,
                                  geometry_size=(32,24),heatmap_sigma=1,augment=False)
    assert stats['train']['counts']['scenes']==2
    assert stats['val']['counts']['minefield_overlap_retries'] == 1
    train=np.load(root/'cells/train/000000_00.npz')
    assert train['images'].shape==(2,8,8,3)
    assert np.load(root/'geometry/val/000000_00.npz')['heatmaps'].shape==(3,24,32)
    def ids(split):
        return {board for line in (root/f'scenes/{split}/manifest.jsonl').read_text().splitlines()
                for board in json.loads(line)['board_ids']}
    assert not ids('train') & ids('val')
    assert json.loads((root/'config.json').read_text())['status']=='complete'
    with pytest.raises(FileExistsError):
        dataset.generate_dataset(root)
