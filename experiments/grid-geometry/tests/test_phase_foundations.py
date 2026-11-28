"""Focused checks for the phase representation, to be run manually."""
import numpy as np
import pytest
import torch

from grid_geometry.phase import PhaseGeometry, PhaseGridUNet, phase_predictions, phase_targets
from grid_geometry.phase.decoder import directional_signal, frequency_candidates
from grid_geometry.phase.loss import phase_loss
from grid_geometry.phase.model import load_phase_weights


def test_model_shape_count_and_activation():
    model = PhaseGridUNet()
    assert sum(p.numel() for p in model.parameters()) == 487365
    logits = model(torch.zeros(1, 3, 24, 40))
    assert logits.shape == (1, 5, 24, 40)
    predictions = phase_predictions(logits)
    assert torch.all(predictions[:, :4].abs() <= 1)
    assert torch.all((predictions[:, 4] >= 0) & (predictions[:, 4] <= 1))
    with pytest.raises(ValueError):
        load_phase_weights(model, {"model": model.state_dict()})


def test_fractional_phase_and_pitch_normalized_occupancy():
    geometry = PhaseGeometry(.5, .5, 8., 12., (.5, .5, 24.5, 24.5))
    target, phase_mask, occupancy_mask = phase_targets((32, 32), geometry)
    assert target[0, 1, 0] == pytest.approx(1)
    assert target[0, 1, 4] == pytest.approx(-1)
    assert target[4, 4, 0] == pytest.approx(.5)
    assert target[4, 6, 4] == pytest.approx(1)
    assert target[4, 6, 28] == 0
    assert phase_mask[4, 4] and not phase_mask[4, 28]
    assert occupancy_mask.all()


def test_crop_masks_and_transform():
    original = PhaseGeometry(-20., -30., 10.5, 12.5, (-20., -30., 100., 100.))
    known = np.ones((16, 16), bool)
    known[:2] = False
    visible = np.ones_like(known)
    visible[5:7, 5:7] = False
    target, mask, occ_mask = phase_targets((16, 16), original, known_mask=known, visible_mask=visible)
    assert np.all(target[4] == 1)
    unmasked, _, _ = phase_targets((16, 16), original)
    np.testing.assert_array_equal(target, unmasked)
    np.testing.assert_array_equal(mask, known & visible)
    np.testing.assert_array_equal(occ_mask, known)
    assert not mask[5, 5] and occ_mask[5, 5] and not occ_mask[0, 0]
    negative, negative_phase, negative_occupancy = phase_targets((16, 16), None, known_mask=known)
    assert not negative.any() and not negative_phase.any()
    np.testing.assert_array_equal(negative_occupancy, known)
    transformed = original.transformed(2, 3, 4, 5)
    assert transformed.origin_x == -36 and transformed.pitch_y == 37.5
    negative, mask, _ = phase_targets((16, 16), None)
    assert not negative.any() and not mask.any()


def test_empty_phase_loss_has_finite_gradients():
    logits = torch.zeros((2, 5, 8, 8), requires_grad=True)
    target = torch.zeros_like(logits)
    mask = torch.zeros((2, 8, 8), dtype=torch.bool)
    loss = phase_loss(logits, target, mask, ~mask)
    loss.total.backward()
    assert loss.phase.item() == 0
    assert torch.isfinite(logits.grad).all()


def test_complex_pitch_proposal():
    geometry = PhaseGeometry(3.25, 5.5, 12.3, 17.1, (3.25, 5.5, 200., 120.))
    target, _, _ = phase_targets((220, 140), geometry)
    signal, support = directional_signal(target, "x")
    candidates = frequency_candidates(signal, support)
    assert candidates and abs(candidates[0].pitch - geometry.pitch_x) < .1
