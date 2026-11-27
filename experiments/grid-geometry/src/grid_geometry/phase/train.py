"""Train and resume PhaseGridUNet with explicit experiment state.

Each batch optimizes separately normalized phase and occupancy losses. Epoch
metrics instead combine each component's sums and valid-pixel counts across
the whole split, then reconstruct the total; they are not mean batch losses.
Split timing covers loader iteration through metric computation; epoch timing
spans loader construction through scheduler stepping, excluding checkpoint and
sidecar writes. Elapsed time sums completed epoch intervals across sessions.
Accelerator operations are not synchronized for timing. CUDA memory counters
are allocator snapshots and process-lifetime peaks, not total device usage.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
import tomllib
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..dataset.loader import PhaseDataset
from .loss import phase_loss
from .metrics import TargetMetrics
from .model import MODEL_ID, REPRESENTATION_ID, PhaseGridUNet, load_phase_weights
from ..training import select_device

CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class Config:
    dataset_root: str
    output_dir: str
    batch_size: int = 8
    workers: int = 0
    epochs: int = 30
    optimizer: str = 'adamw'
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    scheduler: str = 'none'
    step_size: int = 10
    gamma: float = 0.5
    seed: int = 42
    device: str = 'auto'
    amp: bool = False
    occupancy_weight: float = 1.0
    selection_metric: str = 'loss'
    patience: int | None = None

    def __post_init__(self):
        if any(type(v) is not int or v < low for v, low in ((self.batch_size, 1), (self.workers, 0), (self.epochs, 1), (self.step_size, 1), (self.seed, 0))) or self.seed >= 2**32:
            raise ValueError('invalid integer training setting')
        if self.optimizer not in ('adam', 'adamw') or self.scheduler not in ('none', 'step') or self.device not in ('auto', 'cpu', 'cuda', 'mps'):
            raise ValueError('invalid optimizer, scheduler or device')
        if any(not math.isfinite(v) for v in (self.learning_rate, self.weight_decay, self.gamma, self.occupancy_weight)) or self.learning_rate <= 0 or self.weight_decay < 0 or not 0 < self.gamma <= 1 or self.occupancy_weight < 0:
            raise ValueError('invalid numeric training setting')
        if type(self.amp) is not bool or self.selection_metric not in ('loss', 'occupancy_loss', 'phase_loss'):
            raise ValueError('invalid amp or selection metric')
        if self.patience is not None and (type(self.patience) is not int or self.patience < 1):
            raise ValueError('patience must be None or a positive integer')


def dataset_identity(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in [root / 'config.json', root / 'manifest.jsonl']:
        digest.update(path.read_bytes())
    for split in ('train', 'val'):
        for path in sorted((root / split).glob('*')):
            if path.suffix in ('.png', '.json'):
                stat = path.stat()
                digest.update(f'{split}/{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


class TrainingDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.load_sample(index)
        return sample.image, sample.targets, sample.phase_mask, sample.occupancy_mask, sample.metadata.get('geometry') is not None


def run_epoch(model, loader, device, weight, optimizer=None, scaler=None, amp=False, *, epoch=None, progress=None):
    """Optimize independently normalized batches; report pixel-weighted epoch metrics."""
    training = optimizer is not None
    split = 'train' if training else 'validation'
    model.train(training)
    metrics = TargetMetrics(weight)
    start = time.perf_counter()
    successful, skipped, samples = 0, 0, 0
    enabled = sys.stderr.isatty() if progress is None else progress
    batches = tqdm(loader, desc=f'{split} epoch {epoch or "?"}', unit='batch', leave=False, disable=not enabled)
    for i, (image, target, pm, om, positive) in enumerate(batches):
        context = f'{split} epoch {epoch if epoch is not None else "?"}, batch {i+1}, device {device}'
        image, target, pm, om = (v.to(device) for v in (image, target, pm, om))
        if not torch.isfinite(image).all() or not torch.isfinite(target).all():
            raise ValueError(f'non-finite inputs or targets: {context}')
        if not om.flatten(1).any(1).all() or (pm & ~om).any():
            raise ValueError(f'invalid supervision masks: {context}')
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                logits = model(image)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f'non-finite logits: {context}')
            loss = phase_loss(logits, target, pm, om, weight)
            if not torch.isfinite(loss.total):
                raise FloatingPointError(f'non-finite loss: {context}')
            if training:
                if amp:
                    scaler.scale(loss.total).backward()
                    # The public optimizer hook detects an actual step, even if the
                    # scale is unchanged or grows after GradScaler.update().
                    stepped = []
                    handle = optimizer.register_step_post_hook(lambda *args: stepped.append(True))
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    finally:
                        handle.remove()
                    if stepped:
                        successful += 1
                    else:
                        skipped += 1
                else:
                    loss.total.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError(f'non-finite gradients: {context}')
                    optimizer.step()
                    successful += 1
        metrics.update(logits, target, pm, om, positive.to(device))
        samples += image.shape[0]
        if enabled:
            elapsed = time.perf_counter() - start
            batches.set_postfix(batch_loss=f'{loss.total.item():.4f}', samples_per_s=f'{samples / elapsed:.1f}' if elapsed > 0 else 'n/a', refresh=False)
    if training and not successful:
        raise FloatingPointError(f'no successful optimizer updates: {split} epoch {epoch}, device {device}; skipped {skipped}')
    result = metrics.compute()
    seconds = time.perf_counter()-start
    return {'metrics': result, 'seconds': seconds, 'samples': samples,
            'samples_per_second': samples / seconds if seconds > 0 else None,
            'successful_updates': successful, 'skipped_updates': skipped}


def _history_stopping_state(history, metric):
    """Recover only values supported by a complete, ordered legacy history."""
    best, best_epoch, counter = float('inf'), None, 0
    for expected, row in enumerate(history, 1):
        if row.get('epoch') != expected:
            raise ValueError('cannot recover early-stopping state from incomplete history')
        value = row['validation']['metrics']['all'][metric]
        if value is None or not math.isfinite(value):
            raise ValueError('cannot recover early-stopping state from invalid history')
        if value < best:
            best, best_epoch, counter = value, expected, 0
        else:
            counter += 1
    return best, best_epoch, counter


def _save_checkpoint(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def _write_history(out, history):
    temporary = out / 'history.tmp'
    temporary.write_text(json.dumps(history, indent=2, allow_nan=False))
    temporary.replace(out / 'history.json')


def _device_memory(device):
    if device.type != 'cuda':
        return None
    return {'allocated_bytes': torch.cuda.memory_allocated(device),
            'reserved_bytes': torch.cuda.memory_reserved(device),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}


def _summary(state):
    history, config = state['history'], state['config']
    def complete_sum(values):
        return sum(values) if all(v is not None for v in values) else None
    duration = complete_sum([row.get('performance', {}).get('epoch_seconds') for row in history])
    rates = {}
    for split in ('train', 'validation'):
        samples = complete_sum([row[split].get('samples') for row in history])
        seconds = complete_sum([row[split].get('seconds') for row in history])
        rates[split] = samples / seconds if samples is not None and seconds is not None and seconds > 0 else None
    final = history[-1] if history else None
    return {'completed_epochs': state['epoch'], 'epoch_limit': config['epochs'],
            'best_epoch': state.get('best_epoch'), 'selection_metric': config['selection_metric'],
            'best_value': state['best'], 'epochs_without_improvement': state.get('epochs_without_improvement'),
            'patience': config.get('patience'), 'stopping_reason': state.get('stopping_reason'),
            'final_training_metrics': final['train']['metrics'] if final else None,
            'final_validation_metrics': final['validation']['metrics'] if final else None,
            'total_training_seconds': duration, 'aggregate_samples_per_second': rates,
            'device_requested': config['device'], 'device_effective': state.get('environment', {}).get('device'),
            'amp_requested': state.get('amp_requested', config.get('amp')),
            'amp_effective': state.get('amp_effective'),
            'successful_updates': complete_sum([row['train'].get('successful_updates') for row in history]),
            'skipped_updates': complete_sum([row['train'].get('skipped_updates') for row in history]),
            'device_memory': state.get('device_memory')}


def _write_summary(out, state):
    temporary = out / 'summary.tmp'
    temporary.write_text(json.dumps(_summary(state), indent=2, allow_nan=False))
    temporary.replace(out / 'summary.json')


def _epoch_report(row, config, improved):
    def losses(split):
        values = row[split]['metrics']['all']
        return '/'.join(f'{values[k]:.4f}' if values[k] is not None else 'n/a'
                        for k in ('loss', 'phase_loss', 'occupancy_loss'))
    def seconds(value):
        return f'{value:.1f}s' if value is not None else 'unavailable'
    def rate(split):
        value = row[split].get('samples_per_second')
        return f'{value:.1f}' if value is not None else 'n/a'
    perf = row['performance']
    print(f"Epoch {row['epoch']}/{config.epochs} {'improved' if improved else 'no improvement'}: "
          f"loss total/phase/occupancy train {losses('train')} val {losses('validation')}; lr {row['learning_rate']:.3g}\n"
          f"  train {seconds(row['train']['seconds'])} ({rate('train')} samples/s), "
          f"val {seconds(row['validation']['seconds'])} ({rate('validation')} samples/s); "
          f"epoch {seconds(perf['epoch_seconds'])}, elapsed {seconds(perf['elapsed_seconds'])}, remaining {seconds(perf['estimated_remaining_seconds'])}\n"
          f"  best epoch {row['best_epoch']} {config.selection_metric}={row['best']:.6f}; "
          f"patience {row['epochs_without_improvement']}/{config.patience if config.patience is not None else 'none'}; "
          f"updates {row['train']['successful_updates']} successful, {row['train']['skipped_updates']} skipped", flush=True)


def _recover_sidecars(out, state, best_epoch):
    """Rebuild history and verify best; only last's own weights can repair best."""
    best_path = out / 'best.pt'
    if best_epoch == state['epoch']:
        _save_checkpoint(state, best_path)
    else:
        try:
            best_state = torch.load(best_path, map_location='cpu', weights_only=False)
        except Exception as exc:
            raise ValueError('best.pt is missing or inconsistent; last.pt does not contain the best weights') from exc
        if (not isinstance(best_state, dict) or best_state.get('model_id') != MODEL_ID or
                best_state.get('representation_id') != REPRESENTATION_ID or
                best_state.get('epoch') != best_epoch or
                best_state.get('best') != state['best'] or 'model' not in best_state):
            raise ValueError('best.pt is missing or inconsistent; last.pt does not contain the best weights')
        weights = best_state['model']
        if (not isinstance(weights, dict) or weights.keys() != state['model'].keys() or
                any(not isinstance(weights[k], torch.Tensor) or weights[k].shape != v.shape
                    for k, v in state['model'].items())):
            raise ValueError('best.pt is missing or inconsistent; last.pt does not contain the best weights')
    _write_history(out, state['history'])
    _write_summary(out, state)


def _restore_scaler(scaler, saved_scaler, effective_amp):
    if effective_amp and saved_scaler:
        scaler.load_state_dict(saved_scaler)
    elif effective_amp:
        warnings.warn('No enabled CUDA scaler state in checkpoint; initializing a new scaler', UserWarning)
    elif saved_scaler:
        warnings.warn('CUDA scaler state is not active on this device; it will be reinitialized if AMP is resumed later', UserWarning)


def train(config, resume=False):
    device = select_device(config.device)
    effective_amp = config.amp and device.type == 'cuda'
    if config.amp and not effective_amp:
        warnings.warn(f'AMP requested on {device}; using full precision', UserWarning)
    root, out = Path(config.dataset_root), Path(config.output_dir)
    datasets = [TrainingDataset(PhaseDataset(root, split, require_nonempty=True)) for split in ('train', 'val')]
    if any(d % 8 for d in datasets[0].dataset.size):
        raise ValueError(f'PhaseGridUNet input dimensions must be multiples of 8: {datasets[0].dataset.size}')
    identity = dataset_identity(root)
    out.mkdir(parents=True, exist_ok=True)
    last = out / 'last.pt'
    if not resume and any(out.iterdir()):
        raise FileExistsError(f'run directory is not empty: {out}')
    if resume and not last.is_file():
        raise FileNotFoundError(f'no checkpoint to resume: {last}')
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    model = PhaseGridUNet().to(device)
    opt = (torch.optim.Adam if config.optimizer == 'adam' else torch.optim.AdamW)(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, config.step_size, gamma=config.gamma) if config.scheduler == 'step' else None
    scaler = torch.amp.GradScaler('cuda', enabled=effective_amp)
    print(f'Dataset: {root}\nOutput: {out}', flush=True)
    print(f'Samples: train {len(datasets[0])}, validation {len(datasets[1])}; input {datasets[0].dataset.size}; parameters {sum(p.numel() for p in model.parameters()):,}\n'
          f'Device: requested {config.device}, resolved {device}; AMP: requested {config.amp}, effective {effective_amp}\n'
          f'Batch size {config.batch_size}, workers {config.workers}, epochs {config.epochs}; optimizer {config.optimizer}, lr {config.learning_rate:g}, weight decay {config.weight_decay:g}, scheduler {config.scheduler}'
          + (f' (step size {config.step_size}, gamma {config.gamma:g})' if scheduler else '')
          + f'; selection {config.selection_metric}, patience {config.patience}', flush=True)
    history, best, best_epoch, counter, start_epoch = [], float('inf'), None, 0, 0
    if resume:
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
        version = ckpt.get('checkpoint_version', 0)
        if type(version) is not int or version not in (0, CHECKPOINT_VERSION):
            raise ValueError(f'unsupported phase checkpoint version: {version}')
        old = dict(ckpt['config']); new = asdict(config)
        if version == 0:
            old.setdefault('patience', None)
        for key in new.keys() - {'epochs', 'device', 'output_dir'}:
            if key not in old or old[key] != new[key]:
                raise ValueError(f'resume configuration differs: {key}')
        if ckpt['dataset_identity'] != identity:
            raise ValueError('dataset inventory changed since checkpoint')
        load_phase_weights(model, ckpt); opt.load_state_dict(ckpt['optimizer'])
        if scheduler: scheduler.load_state_dict(ckpt['scheduler'])
        history, best, start_epoch = ckpt['history'], ckpt['best'], ckpt['epoch']
        recovered_best, recovered_epoch, recovered_counter = _history_stopping_state(history, config.selection_metric)
        if len(history) != start_epoch or recovered_best != best or recovered_epoch is None:
            raise ValueError('checkpoint history and best metric are inconsistent')
        best_epoch = ckpt.get('best_epoch', recovered_epoch)
        counter = ckpt.get('epochs_without_improvement', recovered_counter)
        if best_epoch != recovered_epoch or counter != recovered_counter:
            raise ValueError('checkpoint early-stopping state is inconsistent with history')
        recovered_state = dict(ckpt, best_epoch=best_epoch, epochs_without_improvement=counter)
        if 'stopping_reason' not in recovered_state:
            recovered_state['stopping_reason'] = history[-1].get('stopping_reason')
        _recover_sidecars(out, recovered_state, best_epoch)
        _restore_scaler(scaler, ckpt.get('scaler'), effective_amp)
        random.setstate(ckpt['python_rng']); np.random.set_state(ckpt['numpy_rng']); torch.set_rng_state(ckpt['torch_rng'].cpu())
        if device.type == 'cuda' and ckpt['cuda_rng'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in ckpt['cuda_rng']])
        if ckpt.get('stopping_reason') == 'early_stopping' or (config.patience is not None and counter >= config.patience):
            print(f'Run already stopped by early stopping at epoch {start_epoch}; best epoch {best_epoch}', flush=True)
            return history
        if start_epoch >= config.epochs:
            print(f'Run already completed {start_epoch} epochs; requested limit is {config.epochs}', flush=True)
            return history
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.perf_counter()
        loaders = [DataLoader(ds, batch_size=config.batch_size, shuffle=idx == 0, num_workers=config.workers, worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(config.seed + epoch * 2 + idx)) for idx, ds in enumerate(datasets)]
        learning_rate = opt.param_groups[0]['lr']
        tr = run_epoch(model, loaders[0], device, config.occupancy_weight, opt, scaler, effective_amp, epoch=epoch+1)
        val = run_epoch(model, loaders[1], device, config.occupancy_weight, amp=effective_amp, epoch=epoch+1)
        value = val['metrics']['all'][config.selection_metric]
        if value is None or not math.isfinite(value):
            raise ValueError(f'undefined or non-finite selection metric: validation epoch {epoch+1}, device {device}')
        improved = value < best
        if improved:
            best, best_epoch, counter = value, epoch+1, 0
        else:
            counter += 1
        stopped = config.patience is not None and counter >= config.patience
        reason = 'early_stopping' if stopped else ('epoch_limit' if epoch+1 == config.epochs else None)
        if scheduler: scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        previous = [row.get('performance', {}).get('epoch_seconds') for row in history]
        elapsed = sum(previous) + epoch_seconds if all(v is not None for v in previous) else None
        remaining = elapsed / (epoch + 1) * (config.epochs - epoch - 1) if elapsed is not None else None
        history.append({'epoch': epoch+1, 'train': tr, 'validation': val, 'learning_rate': learning_rate,
                        'best_epoch': best_epoch, 'best': best, 'epochs_without_improvement': counter,
                        'stopping_reason': reason, 'amp_requested': config.amp, 'amp_effective': effective_amp,
                        'performance': {'epoch_seconds': epoch_seconds, 'elapsed_seconds': elapsed,
                                        'estimated_remaining_seconds': remaining}})
        state = {'checkpoint_version': CHECKPOINT_VERSION, 'model_id': MODEL_ID, 'representation_id': REPRESENTATION_ID, 'model': {k:v.detach().cpu() for k,v in model.state_dict().items()}, 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict() if scheduler else None, 'scaler': scaler.state_dict(), 'epoch': epoch+1, 'best': best, 'best_epoch': best_epoch, 'epochs_without_improvement': counter, 'stopping_reason': reason, 'amp_requested': config.amp, 'amp_effective': effective_amp, 'history': history, 'config': asdict(config), 'dataset_identity': identity, 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None, 'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform(), 'device': str(device)}}
        state['device_memory'] = _device_memory(device)
        _save_checkpoint(state, last)
        if improved:
            _save_checkpoint(state, out / 'best.pt')
        _write_history(out, history)
        _write_summary(out, state)
        _epoch_report(history[-1], config, improved)
        if state['device_memory'] is not None:
            memory = state['device_memory']
            print(f"  CUDA allocator MiB: allocated {memory['allocated_bytes']/2**20:.1f}, reserved {memory['reserved_bytes']/2**20:.1f}, peak allocated {memory['peak_allocated_bytes']/2**20:.1f}, peak reserved {memory['peak_reserved_bytes']/2**20:.1f}", flush=True)
        if stopped:
            print(f'Early stopping after {counter} epochs without improvement; best epoch {best_epoch}', flush=True)
            break
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    path = Path(args.config).resolve()
    with path.open('rb') as f: values = tomllib.load(f)
    for key in ('dataset_root', 'output_dir'):
        values[key] = str((path.parent / values[key]).resolve())
    train(Config(**values), resume=args.resume)


if __name__ == '__main__': main()
