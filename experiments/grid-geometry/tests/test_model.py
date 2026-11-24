import os
from pathlib import Path

import pytest
import torch

from grid_geometry.model import GridUNet


def test_trainable_parameter_count():
    model = GridUNet()
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert trainable_count == 487_331


@pytest.mark.parametrize(
    ("batch_size", "height", "width"),
    [(1, 8, 8), (2, 32, 48), (1, 120, 160), (1, 64, 32)],
)
def test_forward_shape(batch_size, height, width):
    model = GridUNet().eval()
    inputs = torch.zeros(batch_size, 3, height, width)

    with torch.inference_mode():
        logits = model(inputs)

    assert logits.shape == (batch_size, 3, height, width)
    assert torch.isfinite(logits).all()


def test_cpu_forward_backward():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = GridUNet().train()

    generator = torch.Generator().manual_seed(42)
    inputs = torch.rand((1, 3, 24, 32), generator=generator)
    targets = torch.rand(inputs.shape, generator=generator)

    predictions = model(inputs).sigmoid()
    weights = 1.0 + 10.0 * targets
    loss = (weights * (predictions - targets).square()).mean()
    loss.backward()

    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(model.enc1.layers[0].weight.grad) > 0
    assert torch.count_nonzero(model.segmentation_head.weight.grad) > 0


def test_legacy_checkpoint_loads_strictly_on_cpu():
    checkpoint = os.environ.get("GRID_GEOMETRY_LEGACY_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set GRID_GEOMETRY_LEGACY_CHECKPOINT to the original Wine V2 checkpoint")

    state_dict = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    model = GridUNet()
    result = model.load_state_dict(state_dict, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
