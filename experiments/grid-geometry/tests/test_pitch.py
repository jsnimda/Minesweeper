import numpy as np
import pytest

from grid_geometry.pitch import estimate_pitch, phase_alignment
from grid_geometry.signals import AxisSignals, project_heatmaps


def gaussian_signal(length, positions, amplitudes, sigma):
    coordinates = np.arange(length) + 0.5
    distance = coordinates[:, None] - positions
    return np.max(amplitudes * np.exp(-0.5 * (distance / sigma)**2), axis=1)


@pytest.mark.parametrize(("pitch", "count"), [(7.45, 26), (12.61, 11), (27.37, 5), (41.85, 7)])
def test_fractional_period_and_half_cell_phase_on_clean_signals(pitch, count):
    start = 13.27
    length = int(start + pitch * count + 23)
    centers = start + (np.arange(count) + 0.5) * pitch
    boundaries = start + np.arange(count + 1) * pitch
    signals = AxisSignals(
        gaussian_signal(length, centers, 0.34, 1.45),
        gaussian_signal(length, boundaries, 0.82, 1.7),
    )

    result = estimate_pitch(signals)
    best = result.candidates[0]
    assert best.pitch == pytest.approx(pitch, abs=0.08)
    assert best.phase.offset_cycles == pytest.approx(0.5, abs=0.015)
    assert best.phase.score > 0.95
    assert set(best.sources) == {"autocorrelation", "peak_pairs"}
    assert result.centers.peak_heights.max() < 0.35
    assert result.boundaries.peak_heights.max() > 0.8
    assert result.centers.autocorrelation[0] == pytest.approx(1)
    assert len(result.candidates) > 1


@pytest.mark.parametrize("seed", [17, 23, 91, 204, 671, 954])
def test_missing_peaks_spurious_peaks_noise_and_fractional_errors(seed):
    rng = np.random.default_rng(seed)
    pitch = rng.uniform(9, 23)
    count = rng.integers(10, 17)
    origin = rng.uniform(6, 38)
    length = int(origin + count * pitch + 40)

    def corrupted(offset):
        positions = origin + (np.arange(count) + offset) * pitch
        positions = np.delete(positions, [1, 3, 6])
        positions += rng.normal(0, 0.3, len(positions))
        positions = np.r_[positions, origin + pitch * 2.7, origin + pitch * 7.27]
        amplitudes = np.r_[rng.uniform(0.4, 0.8, len(positions) - 2), 0.43, 0.34]
        signal = gaussian_signal(length, positions, amplitudes, 1.25)
        return signal + rng.uniform(0, 0.03, length)

    candidates = estimate_pitch(AxisSignals(corrupted(0.5), corrupted(0))).candidates
    assert any(abs(candidate.pitch - pitch) <= 0.45 for candidate in candidates[:3])


def test_resolves_half_pitch_when_phase_is_observable_but_keeps_harmonics():
    pitch = 18.4
    start = 26.93
    centers = start + (np.arange(12) + 0.5) * pitch
    boundaries = start + np.arange(13) * pitch
    result = estimate_pitch(AxisSignals(
        gaussian_signal(282, centers, 0.6, 1.6),
        gaussian_signal(282, boundaries, 0.6, 1.6),
    ), min_pitch=8, max_pitch=40)

    assert result.candidates[0].pitch == pytest.approx(pitch, abs=0.08)
    assert any(abs(candidate.pitch - pitch / 2) < 0.2 for candidate in result.candidates)
    assert any(abs(candidate.pitch - 2 * pitch) < 0.2 for candidate in result.candidates)
    half = phase_alignment(result.centers, result.boundaries, pitch / 2)
    double = phase_alignment(result.centers, result.boundaries, 2 * pitch)
    assert half.score < 0.01
    assert half.offset_cycles < 0.03 or half.offset_cycles > 0.97
    assert double.center_concentration < 0.12


def test_boundary_only_alternating_loss_leaves_a_double_pitch_ambiguity():
    pitch = 13.6
    positions = 10.1 + np.arange(0, 16, 2) * pitch
    peaks = gaussian_signal(256, positions, 0.7, 1.5)
    result = estimate_pitch(AxisSignals(np.zeros(256), peaks), min_pitch=10, max_pitch=40)

    assert result.candidates[0].pitch == pytest.approx(2 * pitch, abs=0.1)
    assert result.candidates[0].phase.offset_cycles is None
    assert any(abs(candidate.pitch - pitch) < 0.15 for candidate in result.candidates)


def test_peak_pairs_bridge_large_gaps_without_autocorrelation_support():
    signal = gaussian_signal(90, np.array([10.6, 50.4]), 0.8, 1.5)
    result = estimate_pitch(AxisSignals(signal, np.zeros_like(signal)), min_pitch=18, max_pitch=22)

    assert result.candidates[0].pitch == pytest.approx(19.9, abs=0.05)
    assert result.candidates[0].sources == ("peak_pairs",)


def test_padding_and_truncated_outer_peaks_do_not_change_recovered_pitch():
    pitch = 16.73
    centers = 0.2 + (np.arange(8) + 0.5) * pitch
    boundaries = 0.2 + np.arange(9) * pitch
    c = gaussian_signal(134, centers, 0.4, 1.7)
    b = gaussian_signal(134, boundaries, np.r_[0.18, np.full(7, 0.82), 0.22], 1.7)
    tight = estimate_pitch(AxisSignals(c, b))
    padded = estimate_pitch(AxisSignals(np.pad(c, (141, 319)), np.pad(b, (141, 319))))

    for result in (tight, padded):
        assert result.candidates[0].pitch == pytest.approx(pitch, abs=0.08)
        lags = result.boundaries.lags
        assert lags[-1] < 0.61 * (result.boundaries.support[1] - result.boundaries.support[0])
    assert padded.boundaries.support[1] - padded.boundaries.support[0] < 150


@pytest.mark.parametrize("signals", [
    AxisSignals(np.zeros(64), np.ones(64)),
    AxisSignals(np.array([0.2, 0.3]), np.array([0.4, 0.1])),
    AxisSignals(np.r_[np.zeros(20), 1, np.zeros(20)], np.r_[np.zeros(19), 1, np.zeros(21)]),
])
def test_insufficient_evidence_has_no_candidates(signals):
    assert estimate_pitch(signals).candidates == ()


@pytest.mark.parametrize("signals", [
    AxisSignals(np.zeros((1, 3)), np.zeros((1, 3))),
    AxisSignals(np.zeros(3), np.zeros(4)),
    AxisSignals(np.empty(0), np.empty(0)),
    AxisSignals(np.array([np.nan, 0]), np.zeros(2)),
])
def test_invalid_signals(signals):
    with pytest.raises(ValueError):
        estimate_pitch(signals)


def test_invalid_pitch_constraints_and_candidate_limit():
    signals = AxisSignals(np.zeros(16), np.zeros(16))
    with pytest.raises(ValueError, match="pitch bounds"):
        estimate_pitch(signals, min_pitch=9, max_pitch=7)
    with pytest.raises(ValueError, match="max_candidates"):
        estimate_pitch(signals, max_candidates=0)


@pytest.mark.parametrize("resolution", [256, 384, 512])
def test_real_exported_targets(diagnostic_root, resolution):
    from grid_geometry.dataset import GeometryDataset

    dataset = GeometryDataset(diagnostic_root, "train", canvas_size=(resolution, resolution))
    for index in range(len(dataset)):
        sample = dataset.load_sample(index)
        signals = project_heatmaps(sample.heatmaps.numpy())
        for axis, centers in (
            (signals.x, sample.geometry.centers[0, :, 0]),
            (signals.y, sample.geometry.centers[:, 0, 1]),
        ):
            true_pitch = np.median(np.diff(centers))
            result = estimate_pitch(axis)
            assert result.candidates[0].pitch == pytest.approx(true_pitch, abs=0.20)
