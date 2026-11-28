"""Focused Phase D infrastructure checks; all fixtures are synthetic and small."""
import json
import math
from pathlib import Path
import tomllib
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from grid_geometry.dataset.loader import PhaseDataset
from grid_geometry.phase.evaluate import geometry_record, summarize
from grid_geometry.phase.metrics import TargetMetrics
from grid_geometry.phase.train import Config, TrainingDataset, run_epoch, train
import grid_geometry.phase.train as trainer


def dataset(root):
    (root / 'config.json').write_text(json.dumps({'status':'complete','generator':'phase-grid-v1','generator_revision':1,'input_size':[32,32], 'transition_ratio':.25}))
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


def test_geometry_categorization():
    result = SimpleNamespace(status='insufficient_evidence', hypotheses=(), reasons=('no support',))
    records = [geometry_record({'sample_id':'a','geometry':{'origin_x': 0, 'origin_y': 0, 'pitch_x': 8, 'pitch_y': 8, 'visible_box': [0, 0, 24, 24]}, 'board_dimensions': [3, 3], 'known_box': [0, 0, 24, 24], 'complete':False},result,None), geometry_record({'sample_id':'b','geometry':None},result,None)]
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


def test_global_summary_separates_complete_partial_and_negative():
    from grid_geometry.phase.evaluate import summarize_global
    records = [
        {'positive': True, 'complete': True, 'global': {'accepted': True, 'correct': True, 'rows_correct': True, 'columns_correct': True}},
        {'positive': True, 'complete': False, 'global': {'accepted': False, 'correct': None}},
        {'positive': False, 'complete': None, 'global': {'accepted': False, 'correct': None}},
    ]
    summary = summarize_global(records)
    assert summary['complete_correct_accepted'] == 1
    assert summary['complete_with_geometry'] == 1
    assert summary['partial_samples'] == 1
    assert summary['negative_accepted'] == 0


def fake_epochs(monkeypatch, values):
    values = iter(values)
    def run(model, loader, device, weight, optimizer=None, scaler=None, amp=False, *, epoch=None):
        if optimizer is not None:
            optimizer.step()
            value = 1.0
        else:
            value = next(values)
        return {'metrics': {'all': {'loss': value, 'phase_loss': value, 'occupancy_loss': value}},
                'seconds': 0., 'successful_updates': int(optimizer is not None), 'skipped_updates': 0}
    monkeypatch.setattr(trainer, 'run_epoch', run)


def test_early_stopping_ties_and_resume(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    config = Config(str(root), str(tmp_path / 'run'), epochs=10, patience=2, device='cpu')
    fake_epochs(monkeypatch, [3., 2., 2., 2.])
    history = train(config)
    last = torch.load(tmp_path / 'run' / 'last.pt', map_location='cpu', weights_only=False)
    best = torch.load(tmp_path / 'run' / 'best.pt', map_location='cpu', weights_only=False)
    assert len(history) == last['epoch'] == 4 and best['epoch'] == last['best_epoch'] == 2
    assert last['epochs_without_improvement'] == 2 and last['stopping_reason'] == 'early_stopping'
    summary = json.loads((tmp_path / 'run' / 'summary.json').read_text())
    assert summary['stopping_reason'] == 'early_stopping' and summary['epochs_without_improvement'] == 2
    assert train(config, resume=True) == history
    assert 'already stopped' in capsys.readouterr().out


def test_improvement_resets_patience_for_selected_component(tmp_path, monkeypatch):
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    config = Config(str(root), str(tmp_path / 'run'), epochs=6, patience=2,
                    selection_metric='phase_loss', device='cpu')
    fake_epochs(monkeypatch, [3., 3., 2., 2., 2.])
    history = train(config)
    assert [h['epochs_without_improvement'] for h in history] == [0, 1, 0, 1, 2]
    assert history[-1]['best_epoch'] == 3 and history[-1]['stopping_reason'] == 'early_stopping'


def test_epoch_extension_resume_counter_and_learning_rate(tmp_path, monkeypatch, capsys):
    from dataclasses import replace
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    config = Config(str(root), str(tmp_path / 'run'), epochs=2, patience=2, device='cpu', scheduler='step', step_size=1, gamma=.5)
    fake_epochs(monkeypatch, [1., 1.])
    history = train(config)
    assert [h['learning_rate'] for h in history] == pytest.approx([.001, .0005])
    assert history[-1]['epochs_without_improvement'] == 1
    assert train(config, resume=True) == history
    assert 'already completed' in capsys.readouterr().out
    fake_epochs(monkeypatch, [1.])
    history = train(replace(config, epochs=5), resume=True)
    assert len(history) == 3 and history[-1]['stopping_reason'] == 'early_stopping'
    assert history[-1]['learning_rate'] == pytest.approx(.00025)


def test_training_scheduler_defaults_and_alternatives():
    config = Config('data', 'run')
    assert (config.scheduler, config.epochs, config.learning_rate,
            config.cosine_t_max, config.cosine_eta_min) == ('cosine', 50, .001, 50, .00001)
    assert Config('data', 'run', scheduler='none').scheduler == 'none'
    step = Config('data', 'run', scheduler='step')
    assert (step.scheduler, step.step_size, step.gamma) == ('step', 10, .5)


def test_standalone_development_schedule():
    path = Path(__file__).resolve().parents[1] / 'configs' / 'phase_baseline.toml'
    with path.open('rb') as stream:
        config = Config(**tomllib.load(stream))
    assert (config.epochs, config.scheduler, config.cosine_t_max,
            config.learning_rate, config.cosine_eta_min, config.batch_size) == (50, 'cosine', 50, .001, .00001, 8)


def test_cosine_configuration_validation():
    from dataclasses import replace
    config = Config('data', 'run', scheduler='cosine')
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            replace(config, cosine_t_max=value)
    for value in (-1., float('nan'), float('inf'), .002):
        with pytest.raises(ValueError):
            replace(config, cosine_eta_min=value)
    assert replace(config, cosine_eta_min=0.).cosine_eta_min == 0.
    assert replace(config, cosine_eta_min=config.learning_rate).cosine_eta_min == config.learning_rate


def test_cosine_sequence_resume_and_horizon(tmp_path, monkeypatch):
    from dataclasses import replace
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    out = tmp_path / 'run'
    config = Config(str(root), str(out), epochs=2, device='cpu', scheduler='cosine',
                    cosine_t_max=4, cosine_eta_min=.00001)
    fake_epochs(monkeypatch, [1., 1.])
    history = train(config)
    fake_epochs(monkeypatch, [1., 1.])
    history = train(replace(config, epochs=4), resume=True)
    state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    assert state['scheduler']['last_epoch'] == 4
    assert state['optimizer']['param_groups'][0]['lr'] == pytest.approx(.00001)
    fake_epochs(monkeypatch, [1., 1.])
    history = train(replace(config, epochs=6), resume=True)
    expected = [.00001 + (.001 - .00001) * (1 + math.cos(math.pi * i / 4)) / 2 for i in range(4)]
    assert [row['learning_rate'] for row in history] == pytest.approx(expected + [.00001, .00001])
    state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    assert state['scheduler']['last_epoch'] == 4
    assert state['optimizer']['param_groups'][0]['lr'] == pytest.approx(.00001)


@pytest.mark.parametrize('changes', [
    {'scheduler': 'none'}, {'scheduler': 'step'}, {'cosine_t_max': 5},
    {'cosine_eta_min': .00002},
])
def test_cosine_resume_rejects_incompatible_configuration(tmp_path, monkeypatch, changes):
    from dataclasses import replace
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    config = Config(str(root), str(tmp_path / 'run'), epochs=1, device='cpu', scheduler='cosine')
    fake_epochs(monkeypatch, [1.]); train(config)
    with pytest.raises(ValueError, match='resume configuration differs'):
        train(replace(config, epochs=2, **changes), resume=True)


@pytest.mark.parametrize('scheduler', ['none', 'step'])
def test_resume_old_config_without_cosine_fields(tmp_path, monkeypatch, scheduler):
    from dataclasses import replace
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    out = tmp_path / 'run'
    config = Config(str(root), str(out), epochs=1, device='cpu', scheduler=scheduler, step_size=1)
    fake_epochs(monkeypatch, [1.]); train(config)
    state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    state['config'].pop('cosine_t_max'); state['config'].pop('cosine_eta_min')
    torch.save(state, out / 'last.pt')
    with pytest.raises(ValueError, match='resume configuration differs'):
        train(replace(config, epochs=2, learning_rate=.002), resume=True)
    with pytest.raises(ValueError, match='resume configuration differs'):
        train(replace(config, epochs=2, cosine_t_max=10), resume=True)
    with pytest.raises(ValueError, match='resume configuration differs'):
        train(replace(config, epochs=2, scheduler='cosine'), resume=True)
    fake_epochs(monkeypatch, [1.])
    history = train(replace(config, epochs=2), resume=True)
    assert history[-1]['learning_rate'] == pytest.approx(.0005 if scheduler == 'step' else .001)


def test_unsupported_resume_version_and_sidecar_recovery(tmp_path, monkeypatch):
    from dataclasses import replace
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    out = tmp_path / 'run'
    config = Config(str(root), str(out), epochs=1, device='cpu')
    fake_epochs(monkeypatch, [1.]); train(config)
    state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    for version in (None, 0, 2):
        invalid = dict(state)
        if version is None:
            invalid.pop('checkpoint_version')
        else:
            invalid['checkpoint_version'] = version
        torch.save(invalid, out / 'last.pt')
        with pytest.raises(ValueError, match='unsupported phase checkpoint version'):
            train(replace(config, epochs=2), resume=True)
    torch.save(state, out / 'last.pt')
    (out / 'best.pt').unlink(); (out / 'history.json').unlink()
    fake_epochs(monkeypatch, [2.])
    history = train(replace(config, epochs=2), resume=True)
    assert len(history) == 2 and json.loads((out / 'history.json').read_text()) == history
    assert torch.load(out / 'best.pt', map_location='cpu', weights_only=False)['epoch'] == 1
    (out / 'best.pt').unlink()
    with pytest.raises(ValueError, match='best.pt is missing or inconsistent'):
        train(replace(config, epochs=3), resume=True)
    (out / 'best.pt').write_bytes(b'corrupt checkpoint')
    with pytest.raises(ValueError, match='best.pt is missing or inconsistent'):
        train(replace(config, epochs=3), resume=True)


def test_fingerprint_and_dimensions_and_patience(tmp_path, monkeypatch):
    from dataclasses import replace
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    config = Config(str(root), str(tmp_path / 'run'), epochs=1, device='cpu')
    fake_epochs(monkeypatch, [1.]); train(config)
    (root / 'manifest.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='inventory changed'):
        train(replace(config, epochs=2), resume=True)
    settings = json.loads((root / 'config.json').read_text()); settings['input_size'] = [31, 32]
    (root / 'config.json').write_text(json.dumps(settings))
    with pytest.raises(ValueError, match='multiples of 8'):
        train(replace(config, output_dir=str(tmp_path / 'other')))
    for patience in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='patience'):
            replace(config, patience=patience)


def test_epoch_metrics_weight_pixels_across_short_batches():
    class Fixed(torch.nn.Module):
        def forward(self, image): return image
    logits = torch.zeros(3, 5, 1, 2)
    target = torch.zeros_like(logits)
    target[0, :4] = 1.; target[1, :4] = .5
    pm = torch.tensor([[[1, 1]], [[1, 0]], [[0, 0]]], dtype=torch.bool)
    om = torch.tensor([[[1, 1]], [[1, 0]], [[1, 1]]], dtype=torch.bool)
    loader = DataLoader(list(zip(logits, target, pm, om, [True, True, False])), batch_size=2)
    result = run_epoch(Fixed(), loader, torch.device('cpu'), 2.)['metrics']['all']
    assert result['phase_pixels'] == 3 and result['occupancy_pixels'] == 5
    assert result['phase_loss'] == pytest.approx((2 + .25) / 3)
    assert result['occupancy_loss'] == pytest.approx(np.log(2))
    assert result['loss'] == pytest.approx((2 + .25) / 3 + 2 * np.log(2))


def test_nonfinite_batch_logits_loss_and_gradients(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.))
        def forward(self, image): return image * self.weight
    image = torch.zeros(1, 5, 1, 1); mask = torch.ones(1, 1, 1, dtype=torch.bool)
    def loader(x): return [(x, torch.zeros_like(x), mask, mask, torch.tensor([True]))]
    model = Model(); device = torch.device('cpu')
    invalid = image.clone(); invalid[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='train epoch 3, batch 1, device cpu'):
        run_epoch(model, loader(invalid), device, 1., torch.optim.SGD(model.parameters(), .1), epoch=3)
    with torch.no_grad(): model.weight.fill_(float('inf'))
    with pytest.raises(FloatingPointError, match='non-finite logits.*validation epoch 2'):
        run_epoch(model, loader(image), device, 1., epoch=2)
    with torch.no_grad(): model.weight.fill_(1.)
    original = trainer.phase_loss
    monkeypatch.setattr(trainer, 'phase_loss', lambda *args: SimpleNamespace(total=model.weight * float('nan')))
    with pytest.raises(FloatingPointError, match='non-finite loss'):
        run_epoch(model, loader(image), device, 1.)
    monkeypatch.setattr(trainer, 'phase_loss', original)
    hook = model.weight.register_hook(lambda grad: grad * float('inf'))
    with pytest.raises(FloatingPointError, match='non-finite gradients'):
        run_epoch(model, loader(image), device, 1., torch.optim.SGD(model.parameters(), .1))
    hook.remove()


def test_cpu_amp_fallback_and_scaler_transition(tmp_path, monkeypatch):
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    config = Config(str(root), str(tmp_path / 'run'), epochs=1, device='cpu', amp=True)
    fake_epochs(monkeypatch, [1.])
    with pytest.warns(UserWarning, match='using full precision'):
        history = train(config)
    assert history[0]['amp_requested'] and not history[0]['amp_effective']
    state = torch.load(tmp_path / 'run' / 'last.pt', map_location='cpu', weights_only=False)
    state['scaler'] = {'scale': 1024.}
    torch.save(state, tmp_path / 'run' / 'last.pt')
    with pytest.warns(UserWarning) as captured:
        train(config, resume=True)
    assert any('scaler state is not active' in str(w.message) for w in captured)


def test_scaler_restore_decisions():
    class Scaler:
        def __init__(self): self.loaded = None
        def load_state_dict(self, state): self.loaded = state
    scaler = Scaler()
    with pytest.warns(UserWarning, match='initializing a new scaler'):
        trainer._restore_scaler(scaler, {}, True)
    assert scaler.loaded is None
    saved = {'scale': 512.}
    trainer._restore_scaler(scaler, saved, True)
    assert scaler.loaded == saved
    with pytest.warns(UserWarning, match='not active'):
        trainer._restore_scaler(scaler, saved, False)


def test_mps_amp_falls_back(tmp_path, monkeypatch):
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    monkeypatch.setattr(trainer, 'select_device', lambda requested: torch.device('mps'))
    monkeypatch.setattr(trainer.PhaseGridUNet, 'to', lambda self, device: self)
    fake_epochs(monkeypatch, [1.])
    with pytest.warns(UserWarning, match='using full precision'):
        history = train(Config(str(root), str(tmp_path / 'run'), epochs=1, device='mps', amp=True))
    assert not history[0]['amp_effective']


def test_failed_epoch_preserves_last_checkpoint(tmp_path, monkeypatch):
    from dataclasses import replace
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    config = Config(str(root), str(tmp_path / 'run'), epochs=1, device='cpu')
    fake_epochs(monkeypatch, [1.]); train(config)
    before = (tmp_path / 'run' / 'last.pt').read_bytes()
    def fail(*args, **kwargs): raise FloatingPointError('failed batch')
    monkeypatch.setattr(trainer, 'run_epoch', fail)
    with pytest.raises(FloatingPointError, match='failed batch'):
        train(replace(config, epochs=2), resume=True)
    assert (tmp_path / 'run' / 'last.pt').read_bytes() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_amp_overflow_skips_and_recovers():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.conv = torch.nn.Conv2d(5, 5, 1)
        def forward(self, x): return self.conv(x)
    device = torch.device('cuda'); model = Model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), .01)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.)
    image = torch.ones(1, 5, 1, 1); target = torch.zeros_like(image)
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    batches = [(image, target, mask, mask, torch.tensor([True]))] * 2
    calls = []
    def overflow_once(grad):
        calls.append(1)
        return torch.full_like(grad, float('inf')) if len(calls) == 1 else grad
    handle = model.conv.weight.register_hook(overflow_once)
    result = run_epoch(model, batches, device, 1., optimizer, scaler, True, epoch=1)
    handle.remove()
    assert result['successful_updates'] == 1 and result['skipped_updates'] == 1

    handle = model.conv.weight.register_hook(lambda grad: torch.full_like(grad, float('inf')))
    with pytest.raises(FloatingPointError, match='no successful optimizer updates'):
        run_epoch(model, batches, device, 1., optimizer, scaler, True, epoch=2)
    handle.remove()


def test_performance_sample_counts_and_progress(monkeypatch):
    class Fixed(torch.nn.Module):
        def forward(self, image): return image
    class Progress:
        calls = []
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.disabled = kwargs['disable']
            self.postfixes = []
            self.calls.append(self)
        def __iter__(self): return iter(self.iterable)
        def set_postfix(self, **kwargs): self.postfixes.append(kwargs)
    monkeypatch.setattr(trainer, 'tqdm', Progress)
    image = torch.zeros(3, 5, 1, 1)
    mask = torch.ones(3, 1, 1, dtype=torch.bool)
    loader = DataLoader(list(zip(image, image, mask, mask, [True] * 3)), batch_size=2)
    for enabled in (False, True):
        result = run_epoch(Fixed(), loader, torch.device('cpu'), 1., progress=enabled)
        assert result['samples'] == 3 and result['seconds'] >= 0
        if result['seconds'] > 0:
            assert result['samples_per_second'] == pytest.approx(3 / result['seconds'])
        assert Progress.calls[-1].disabled is not enabled
        assert len(Progress.calls[-1].postfixes) == (2 if enabled else 0)


def test_summary_resume_recovery_and_legacy_performance(tmp_path, monkeypatch):
    from dataclasses import replace
    root = tmp_path / 'data'; root.mkdir(); dataset(root)
    out = tmp_path / 'run'
    config = Config(str(root), str(out), epochs=1, device='cpu')
    fake_epochs(monkeypatch, [1.]); first = train(config)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['completed_epochs'] == summary['epoch_limit'] == 1
    assert summary['stopping_reason'] == 'epoch_limit' and summary['best_epoch'] == 1
    assert summary['successful_updates'] == 1 and summary['total_training_seconds'] >= 0
    assert first[0]['performance']['epoch_seconds'] >= 0
    (out / 'summary.json').unlink()
    train(config, resume=True)
    assert json.loads((out / 'summary.json').read_text()) == summary
    fake_epochs(monkeypatch, [2.]); history = train(replace(config, epochs=2), resume=True)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['successful_updates'] == 2
    assert summary['total_training_seconds'] == pytest.approx(sum(r['performance']['epoch_seconds'] for r in history))
    state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    state['history'][0].pop('performance')
    torch.save(state, out / 'last.pt')
    (out / 'summary.json').unlink()
    train(replace(config, epochs=2), resume=True)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['total_training_seconds'] is None
    assert summary['aggregate_samples_per_second']['train'] is None
