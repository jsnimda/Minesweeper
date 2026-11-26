"""Focused Phase D infrastructure checks; all fixtures are synthetic and small."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from grid_geometry.phase.dataset import PhaseDataset
from grid_geometry.phase.evaluate import geometry_record, summarize
from grid_geometry.phase.metrics import TargetMetrics
from grid_geometry.phase.train import Config, TrainingDataset, run_epoch, train


def dataset(root):
    (root / 'config.json').write_text(json.dumps({'status':'complete','generator':'phase-grid-v1','input_size':[32,32], 'transition_ratio':.25}))
    (root / 'manifest.jsonl').write_text('')
    for split in ('train','val','test'):
        folder = root / split; folder.mkdir()
        for name, positive in [('a',True),('b',False)]:
            Image.fromarray(np.zeros((32,32,3),dtype=np.uint8)).save(folder / (name+'.png'))
            geometry = {'origin_x':4., 'origin_y':4., 'pitch_x':8., 'pitch_y':8., 'visible_box':[4.,4.,28.,28.]} if positive else None
            (folder / (name+'.json')).write_text(json.dumps({'sample_id':name,'split':split,'board_id':'board' if positive else None,'geometry':geometry,'known_box':[4,0,32,32]}))


def test_dataset_masks_and_negative(tmp_path):
    dataset(tmp_path); ds = PhaseDataset(tmp_path,'train')
    a, b = ds.load_sample(0), ds.load_sample(1)
    assert not a.occupancy_mask[:, :4].any() and a.occupancy_mask[:, 4:].all()
    assert a.phase_mask.any() and not b.phase_mask.any()
    assert b.occupancy_mask.any() and not b.targets.any()


def test_metrics_circular_aggregation_and_empty_phase():
    m = TargetMetrics(); target = torch.zeros(2,5,1,1); target[:,0] = -1; target[:,1] = .01; target[:,2] = 1
    logits = torch.zeros_like(target); logits[:,0] = -5; logits[:,1] = -.01; logits[:,2] = 5
    pm = torch.tensor([[[True]],[[False]]]); om = torch.ones_like(pm)
    m.update(logits[:1],target[:1],pm[:1],om[:1],torch.tensor([True])); m.update(logits[1:],target[1:],pm[1:],om[1:],torch.tensor([False]))
    result = m.compute()
    assert result['all']['phase_mae_radians_x'] < .1
    assert result['all']['phase_pixels'] == 1 and result['negative']['phase_loss'] is None
    assert result['all']['occupancy_pixels'] == 2


def test_positive_without_phase_supervision():
    metrics = TargetMetrics()
    logits = torch.zeros(1, 5, 1, 1)
    metrics.update(logits, torch.zeros_like(logits), torch.zeros(1, 1, 1, dtype=torch.bool), torch.ones(1, 1, 1, dtype=torch.bool), torch.tensor([True]))
    result = metrics.compute()
    assert result['positive']['samples'] == 1
    assert result['positive']['phase_loss'] is None
    assert result['positive']['occupancy_loss'] is not None


def test_gaussian_model_remains_three_channel():
    from grid_geometry.model import GridUNet
    assert GridUNet().segmentation_head.out_channels == 3


def test_geometry_categorization():
    result = SimpleNamespace(status='insufficient_evidence', hypotheses=(), reasons=('no support',))
    records = [geometry_record({'sample_id':'a','geometry':{},'complete':False},result,None), geometry_record({'sample_id':'b','geometry':None},result,None)]
    summary = summarize(records)
    assert summary['false_rejection_positive_rate'] == 1 and summary['false_acceptance_negative_rate'] == 0
    assert summary['correct_among_accepted_positive'] is None


def test_cpu_training_and_resume(tmp_path):
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    config = Config(str(root), str(tmp_path/'run'), batch_size=2, epochs=1, device='cpu')
    history = train(config)
    assert len(history) == 1 and (tmp_path/'run'/'best.pt').exists()
    from dataclasses import replace
    resumed = train(replace(config, epochs=2), resume=True)
    assert len(resumed) == 2
    checkpoint = torch.load(tmp_path/'run'/'last.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 2 and checkpoint['optimizer']['state']
