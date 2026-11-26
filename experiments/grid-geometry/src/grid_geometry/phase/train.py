"""Train and resume PhaseGridUNet with explicit experiment state."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
import tomllib

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PhaseDataset
from .loss import phase_loss
from .metrics import TargetMetrics
from .model import MODEL_ID, REPRESENTATION_ID, PhaseGridUNet, load_phase_weights
from ..training import select_device


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

    def __post_init__(self):
        if any(type(v) is not int or v < low for v, low in ((self.batch_size, 1), (self.workers, 0), (self.epochs, 1), (self.step_size, 1), (self.seed, 0))) or self.seed >= 2**32:
            raise ValueError('invalid integer training setting')
        if self.optimizer not in ('adam', 'adamw') or self.scheduler not in ('none', 'step') or self.device not in ('auto', 'cpu', 'cuda', 'mps'):
            raise ValueError('invalid optimizer, scheduler or device')
        if any(not math.isfinite(v) for v in (self.learning_rate, self.weight_decay, self.gamma, self.occupancy_weight)) or self.learning_rate <= 0 or self.weight_decay < 0 or not 0 < self.gamma <= 1 or self.occupancy_weight < 0:
            raise ValueError('invalid numeric training setting')
        if type(self.amp) is not bool or self.selection_metric not in ('loss', 'occupancy_loss', 'phase_loss'):
            raise ValueError('invalid amp or selection metric')


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


def run_epoch(model, loader, device, weight, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    metrics = TargetMetrics(weight)
    start = time.perf_counter()
    for i, (image, target, pm, om, positive) in enumerate(loader):
        image, target, pm, om = (v.to(device) for v in (image, target, pm, om))
        if not torch.isfinite(image).all() or not torch.isfinite(target).all() or not om.flatten(1).any(1).all() or (pm & ~om).any():
            raise ValueError(f'invalid batch {i}: values or supervision masks')
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16 if device.type in ('cuda', 'mps') else torch.bfloat16):
                logits = model(image)
                loss = phase_loss(logits, target, pm, om, weight)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss.total):
                raise FloatingPointError(f'non-finite predictions or loss in batch {i}')
            if training:
                scaler.scale(loss.total).backward()
                scaler.unscale_(optimizer)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError(f'non-finite gradients in batch {i}')
                scaler.step(optimizer); scaler.update()
        metrics.update(logits, target, pm, om, positive.to(device))
        if (i + 1) % 25 == 0:
            print(f"{'train' if training else 'val'} batch {i+1}/{len(loader)}", flush=True)
    return {'metrics': metrics.compute(), 'seconds': time.perf_counter()-start}


def train(config, resume=False):
    device = select_device(config.device)
    if config.amp and device.type not in ('cuda', 'cpu', 'mps'):
        raise ValueError('AMP unavailable for selected device')
    root, out = Path(config.dataset_root), Path(config.output_dir)
    datasets = [TrainingDataset(PhaseDataset(root, split, require_nonempty=True)) for split in ('train', 'val')]
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
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp and device.type == 'cuda')
    history, best, start_epoch = [], float('inf'), 0
    if resume:
        ckpt = torch.load(last, map_location=device, weights_only=False)
        old = ckpt['config']; new = asdict(config)
        for key in new.keys() - {'epochs', 'device', 'output_dir'}:
            if old[key] != new[key]:
                raise ValueError(f'resume configuration differs: {key}')
        if ckpt['dataset_identity'] != identity:
            raise ValueError('dataset inventory changed since checkpoint')
        load_phase_weights(model, ckpt); opt.load_state_dict(ckpt['optimizer'])
        if scheduler: scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        history, best, start_epoch = ckpt['history'], ckpt['best'], ckpt['epoch']
        random.setstate(ckpt['python_rng']); np.random.set_state(ckpt['numpy_rng']); torch.set_rng_state(ckpt['torch_rng'].cpu())
        if device.type == 'cuda' and ckpt['cuda_rng'] is not None: torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    for epoch in range(start_epoch, config.epochs):
        loaders = [DataLoader(ds, batch_size=config.batch_size, shuffle=idx == 0, num_workers=config.workers, worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(config.seed + epoch * 2 + idx)) for idx, ds in enumerate(datasets)]
        tr = run_epoch(model, loaders[0], device, config.occupancy_weight, opt, scaler, config.amp)
        val = run_epoch(model, loaders[1], device, config.occupancy_weight, amp=config.amp)
        value = val['metrics']['all'][config.selection_metric]
        if value is None or not math.isfinite(value): raise ValueError('selection metric is undefined or non-finite')
        improved = value < best
        if improved: best = value
        if scheduler: scheduler.step()
        history.append({'epoch': epoch+1, 'train': tr, 'validation': val, 'learning_rate': opt.param_groups[0]['lr']})
        state = {'model_id': MODEL_ID, 'representation_id': REPRESENTATION_ID, 'model': {k:v.detach().cpu() for k,v in model.state_dict().items()}, 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict() if scheduler else None, 'scaler': scaler.state_dict(), 'epoch': epoch+1, 'best': best, 'history': history, 'config': asdict(config), 'dataset_identity': identity, 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None, 'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform(), 'device': str(device)}}
        torch.save(state, out / 'last.tmp'); (out / 'last.tmp').replace(last)
        if improved:
            torch.save(state, out / 'best.tmp'); (out / 'best.tmp').replace(out / 'best.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False))
        print(f'epoch {epoch+1}: validation {config.selection_metric}={value:.6f}', flush=True)
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
