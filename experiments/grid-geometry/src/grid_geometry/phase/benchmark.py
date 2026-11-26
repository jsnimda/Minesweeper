"""Synthetic numerical benchmark: python -m grid_geometry.phase.benchmark.

Target construction occurs outside the timed region. No trained model is needed.
"""
import argparse
import json
from time import perf_counter

import numpy as np

from .decoder import DecoderSettings, decode_phase
from ..dataset.targets import PhaseGeometry, phase_targets
from .validation import validate_phase


CASES = (
    ("small_384", 9, 9, 25., (384, 384), "clean"),
    ("medium_384", 16, 16, 18., (384, 384), "clean"),
    ("expert_384", 16, 30, 10., (384, 384), "clean"),
    ("small_corrupt", 9, 9, 25., (384, 384), "corrupt"),
    ("expert_incomplete", 16, 30, 14., (384, 384), "incomplete"),
    ("large_native", 60, 100, 20., (2047, 1240), "clean"),
    ("large_corrupt", 60, 100, 20., (2047, 1240), "corrupt"),
)


def _case(rows, columns, pitch, size, kind, seed):
    width, height = size
    # Center complete boards. The incomplete case places the left boundary
    # outside the image without changing the underlying lattice.
    left = (width - columns * pitch) / 2
    top = (height - rows * pitch) / 2
    if kind == "incomplete":
        left = -3 * pitch
    geometry = PhaseGeometry(left, top, pitch, pitch,
                             (left, top, left + columns * pitch, top + rows * pitch))
    fields, _, _ = phase_targets(size, geometry)
    if kind == "corrupt":
        rng = np.random.default_rng(seed)
        fields[:4] = np.clip(fields[:4] + rng.normal(0, .08, fields[:4].shape), -1, 1)
        fields[4, height // 3:height // 3 + max(1, int(pitch)), width // 3:width // 3 + max(1, int(pitch))] = 0
    return fields


def run_benchmark(*, warmup=2, repeats=10, seed=42):
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    report = {}
    stages = ("signal_extraction", "frequency_proposals", "pitch_refinement", "joint_candidates", "extent_estimation", "total", "strict_validation")
    for name, rows, columns, pitch, size, kind in CASES:
        fields = _case(rows, columns, pitch, size, kind, seed)
        measurements = {stage: [] for stage in stages}
        outcomes = []
        for index in range(warmup + repeats):
            result = decode_phase(fields, DecoderSettings())
            started = perf_counter()
            validation = validate_phase(fields, result.hypotheses[0], decoder_status=result.status) if result.hypotheses else None
            elapsed = (perf_counter() - started) * 1000
            if index >= warmup:
                for stage in stages[:-1]:
                    measurements[stage].append(result.timings_ms.get(stage, 0.))
                measurements['strict_validation'].append(elapsed)
                top = result.hypotheses[0] if result.hypotheses else None
                outcomes.append({"status": result.status, "rows": top.rows if top else None,
                                 "columns": top.columns if top else None,
                                 "dimensions_correct": bool(top and top.rows == rows and top.columns == columns),
                                 "accepted": validation.accepted if validation else False,
                                 "reasons": list(validation.reasons) if validation else list(result.reasons)})
        report[name] = {"expected_rows": rows, "expected_columns": columns, "size": size, "kind": kind,
                        "timings_ms": {stage: {"median": float(np.median(samples)), "p95": float(np.percentile(samples, 95))}
                                       for stage, samples in measurements.items()},
                        "outcomes": outcomes}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(warmup=args.warmup, repeats=args.repeats, seed=args.seed), indent=2))


if __name__ == '__main__':
    main()
