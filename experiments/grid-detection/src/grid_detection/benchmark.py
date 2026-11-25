"""Benchmark the production per-scene pipeline without retaining images.

Modes are 'render', 'encode' (PNG in memory), and 'export' (temporary dataset).
Reported logical RGB throughput is not physical memory bandwidth; encoded
throughput is not an SSD bandwidth measurement.
"""
from pathlib import Path
import tempfile
from time import perf_counter

from .dataset.generator import _validate_parameters, generate_dataset, prepare_assets, run_splits


def benchmark(*, mode='render', scenes=20, seed=42, workers=1,
              png_compression=1, augment=True, max_attempts=20, progress=True):
    if mode not in ('render', 'encode', 'export'):
        raise ValueError('mode must be render, encode or export')
    _validate_parameters(scenes, 0, 0, seed, workers, png_compression, max_attempts)
    assets = prepare_assets() if scenes else {}  # Excluded from timing.
    with tempfile.TemporaryDirectory(prefix='yolo-benchmark-') as temp:
        start = perf_counter()
        if mode == 'export':
            stats = generate_dataset(Path(temp) / 'dataset', train=scenes, val=0, test=0,
                seed=seed, workers=workers, png_compression=png_compression,
                augment=augment, max_attempts=max_attempts, progress=progress)
        else:
            stats = run_splits([('train', scenes)], seed=seed, workers=workers,
                png_compression=png_compression, augment=augment, max_attempts=max_attempts,
                scene_details=False, progress=progress, mode=mode, assets=assets)
        elapsed = perf_counter() - start
    counts = stats['train']['counts']
    pixels = counts.get('pixels', 0)
    raw = pixels * 3
    encoded = counts.get('encoded_bytes', 0)
    result = {'mode': mode, 'accepted_scenes': counts.get('scenes', 0),
        'seconds': elapsed, 'images_per_second': counts.get('scenes', 0) / elapsed,
        'pixels': pixels, 'megapixels_per_second': pixels / elapsed / 1_000_000,
        'logical_rgb_bytes': raw, 'logical_rgb_MiB_per_second': raw / elapsed / 2**20,
        'workers': workers, 'png_compression': png_compression,
        'retries': counts.get('retries', 0),
        'placement_retries': counts.get('placement_retries', 0),
        'minefield_overlap_retries': counts.get('minefield_overlap_retries', 0),
        'assets_prepared_before_timing': True, 'process_startup_included': workers > 1 and scenes > 0}
    if mode != 'render':
        result.update(encoded_bytes=encoded, encoded_MiB_per_second=encoded / elapsed / 2**20,
                      compression_ratio=raw / encoded if encoded else None)
    return result
