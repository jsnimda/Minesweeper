import numpy as np
import pytest

from grid_geometry.signals import project_heatmaps


@pytest.mark.parametrize("method", ["mean", "max", "rms", "top_k"])
def test_projections_use_the_correct_heatmap_channel_and_axis(method):
    heatmaps = np.array([
        [[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]],
        [[0.2, 0.3, 0.4], [0.4, 0.5, 0.6]],
        [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]],
    ])

    result = project_heatmaps(heatmaps, method=method, top_k_fraction=0.5)

    def expected(short_signal):
        if method == "rms":
            return np.sqrt(np.square(short_signal).mean())
        if method == "max":
            return max(short_signal)
        if method == "top_k":
            count = (len(short_signal) + 1) // 2
            return np.sort(short_signal)[-count:].mean()
        return short_signal.mean()

    for projection, slices in (
        (result.x.centers, heatmaps[0].T),
        (result.x.boundaries, heatmaps[1].T),
        (result.y.centers, heatmaps[0]),
        (result.y.boundaries, heatmaps[2]),
    ):
        assert projection.shape == (len(slices),)
        np.testing.assert_allclose(projection, [expected(signal) for signal in slices])


def test_projection_retains_absolute_strength_and_all_axis_top_k_equals_mean():
    heatmaps = np.full((3, 5, 8), 0.12)
    mean = project_heatmaps(heatmaps)
    top_k = project_heatmaps(heatmaps, method="top_k", top_k_fraction=1)

    np.testing.assert_allclose(mean.x.centers, 0.12)
    np.testing.assert_allclose(top_k.x.centers, mean.x.centers)
    np.testing.assert_allclose(top_k.y.boundaries, mean.y.boundaries)


@pytest.mark.parametrize("shape", [(1, 3), (2, 4, 5), (3, 0, 5), (1, 3, 4, 5)])
def test_rejects_invalid_heatmap_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        project_heatmaps(np.zeros(shape))


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("inf"), float("nan"), "0.4"])
def test_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError, match="probabilities"):
        project_heatmaps(np.full((3, 2, 3), probability))


def test_rejects_invalid_projection_settings():
    heatmaps = np.zeros((3, 2, 3))
    with pytest.raises(ValueError, match="method"):
        project_heatmaps(heatmaps, method="median")
    with pytest.raises(ValueError, match="top_k_fraction"):
        project_heatmaps(heatmaps, method="top_k", top_k_fraction=float("nan"))
