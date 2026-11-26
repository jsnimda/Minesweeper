"""Load compact phase dataset samples and generate dense targets on demand."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .targets import PhaseGeometry, phase_targets


@dataclass(frozen=True)
class PhaseSample:
    image: torch.Tensor
    targets: torch.Tensor
    phase_mask: torch.Tensor
    occupancy_mask: torch.Tensor
    metadata: dict


class PhaseDataset(Dataset):
    """Read a split written by dataset.generator; indexing is batchable."""

    def __init__(self, root, split, *, transition_ratio=None, require_nonempty=False):
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        self.root = Path(root)
        self.split = split
        config = json.loads((self.root / 'config.json').read_text(encoding='utf-8'))
        if config.get('status') != 'complete':
            raise ValueError('dataset generation is incomplete')
        if config.get('generator') not in (None, 'phase-grid-v1'):
            raise ValueError('incompatible phase dataset generator')
        self.size = tuple(config['input_size'])
        if len(self.size) != 2 or any(type(v) is not int or v <= 0 for v in self.size):
            raise ValueError('invalid dataset input_size')
        self.transition_ratio = config['transition_ratio'] if transition_ratio is None else transition_ratio
        directory = self.root / split
        images = {p.stem for p in directory.glob('*.png')}
        annotations = {p.stem for p in directory.glob('*.json')}
        if images != annotations:
            raise ValueError(f'unpaired images and metadata in {directory}')
        self.names = sorted(images)
        if not directory.is_dir() or (require_nonempty and not self.names):
            raise ValueError(f'empty or missing required split: {directory}')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        sample = self.load_sample(index)
        return sample.image, sample.targets, sample.phase_mask, sample.occupancy_mask

    def load_sample(self, index):
        name = self.names[index]
        directory = self.root / self.split
        metadata = json.loads((directory / f'{name}.json').read_text(encoding='utf-8'))
        if metadata['sample_id'] != name or metadata['split'] != self.split:
            raise ValueError(f'invalid identity in {name}.json')
        with Image.open(directory / f'{name}.png') as image:
            if image.size != self.size:
                raise ValueError(f'invalid image dimensions for {name}')
            rgb = np.asarray(image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1).copy() / 255.
        known = np.zeros((self.size[1], self.size[0]), dtype=bool)
        left, top, right, bottom = metadata['known_box']
        if not (0 <= left < right <= self.size[0] and 0 <= top < bottom <= self.size[1]):
            raise ValueError(f'invalid known_box in {name}.json')
        # Coordinates are continuous: only pixel centers inside the source are known.
        x = np.arange(self.size[0]) + .5
        y = np.arange(self.size[1]) + .5
        known[:] = (x[None, :] >= left) & (x[None, :] < right) & (y[:, None] >= top) & (y[:, None] < bottom)
        data = metadata.get('geometry')
        if (data is None) != (metadata.get('board_id') is None):
            raise ValueError(f'inconsistent geometry and board identity in {name}.json')
        geometry = PhaseGeometry(**data) if data is not None else None
        targets, phase_mask, occupancy_mask = phase_targets(self.size, geometry,
            transition_ratio=self.transition_ratio, known_mask=known)
        return PhaseSample(torch.from_numpy(rgb), torch.from_numpy(targets),
                           torch.from_numpy(phase_mask), torch.from_numpy(occupancy_mask), metadata)


def inspect_dataset(root, *, check_targets=True):
    """Stream file, metadata, tensor and cross-split identity integrity checks."""
    root = Path(root)
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    ids = {}
    counts = {}
    expected_manifest = set()
    for split in ('train', 'val', 'test'):
        dataset = PhaseDataset(root, split)
        counts[split] = len(dataset)
        if len(dataset) != config[split]:
            raise ValueError(f'incorrect {split} sample count')
        ids[split] = set()
        for index, name in enumerate(dataset.names):
            metadata = json.loads((root / split / f'{name}.json').read_text(encoding='utf-8'))
            if metadata['split'] != split or metadata['sample_id'] != name:
                raise ValueError(f'incorrect sample identity: {split}/{name}')
            expected_manifest.add((split, name))
            board_id = metadata.get('board_id')
            if board_id is not None:
                if metadata.get('geometry') is None:
                    raise ValueError(f'positive sample lacks geometry: {split}/{name}')
                ids[split].add(board_id)
            elif metadata.get('geometry') is not None:
                raise ValueError(f'negative sample contains geometry: {split}/{name}')
            if check_targets:
                sample = dataset.load_sample(index)
                width, height = dataset.size
                if sample.image.shape != (3, height, width) or sample.targets.shape != (5, height, width):
                    raise ValueError(f'invalid tensor shape: {split}/{name}')
                if not torch.isfinite(sample.targets).all() or sample.targets[:4].abs().max() > 1.00001 or sample.targets[4].min() < 0 or sample.targets[4].max() > 1:
                    raise ValueError(f'invalid target values: {split}/{name}')
                if sample.phase_mask.shape != (height, width) or sample.occupancy_mask.shape != (height, width):
                    raise ValueError(f'invalid mask shape: {split}/{name}')
                if (sample.phase_mask & ~sample.occupancy_mask).any() or (board_id is None and sample.phase_mask.any()):
                    raise ValueError(f'inconsistent supervision masks: {split}/{name}')
    actual_manifest = []
    with (root / 'manifest.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            actual_manifest.append((record['split'], record['sample_id']))
    if len(actual_manifest) != len(expected_manifest) or set(actual_manifest) != expected_manifest:
        raise ValueError('manifest and sample files differ')
    overlap = {f'{a}:{b}': len(ids[a] & ids[b]) for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    if any(overlap.values()):
        raise ValueError(f'cross-split board identity overlap: {overlap}')
    return {'split_counts': counts, 'unique_board_ids': {k: len(v) for k, v in ids.items()}, 'board_identity_overlap': overlap}
