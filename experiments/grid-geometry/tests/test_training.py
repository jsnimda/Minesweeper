import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from grid_geometry.training import make_optimizer, run_epoch, select_device, weighted_mse
from grid_geometry.training_config import TrainingConfig


def test_weighted_mse_and_channels_against_manual_values():
    probabilities = torch.tensor([[0.50, 0.25, 0.75], [0.25, 0.50, 0.50]])
    logits = torch.logit(probabilities)[:, :, None, None].requires_grad_()
    targets = torch.tensor([[1., 0., 0.5], [0., 1., 0.]])[:, :, None, None]
    loss = weighted_mse(logits, targets)

    expected = torch.tensor([
        (11 * 0.5**2 + 0.25**2) / 2,
        (0.25**2 + 11 * 0.5**2) / 2,
        (6 * 0.25**2 + 0.50**2) / 2,
    ])
    torch.testing.assert_close(loss.channels, expected)
    torch.testing.assert_close(loss.total, expected.mean())
    loss.total.backward()
    assert torch.isfinite(logits.grad).all()


def test_fp16_inputs_use_fp32_for_sigmoid_and_reduction():
    logits = torch.tensor([80, -80, -1], dtype=torch.float16).reshape(1, 3, 1, 1).requires_grad_()
    targets = torch.tensor([0, 1, 0.5], dtype=torch.float16).reshape(1, 3, 1, 1)
    result = weighted_mse(logits, targets, weight=5)
    assert result.channels.dtype == torch.float32
    torch.testing.assert_close(result.channels[:2], torch.tensor([1., 6.]))
    result.total.backward()
    assert torch.isfinite(logits.grad).all()


class Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(3))
        self.calls = []

    def forward(self, images):
        self.calls.append((self.training, torch.is_grad_enabled()))
        return images + self.bias[None, :, None, None]


def test_epochs_set_mode_and_validation_aggregates_uneven_batches_by_sample():
    model = Recorder()
    targets = torch.tensor([[1., 1., 1.], [1., 0., 0.], [0., 0., 0.]])[:, :, None, None]
    loader = DataLoader(TensorDataset(torch.zeros_like(targets), targets), batch_size=2)
    device = torch.device("cpu")

    model.train()
    validation = run_epoch(model, loader, device=device)
    assert model.calls == [(False, False), (False, False)]
    assert validation.samples == 3
    assert validation.channels == pytest.approx((5.75 / 3, 3.25 / 3, 3.25 / 3))
    assert validation.loss == pytest.approx(12.25 / 9)
    assert model.bias.grad is None

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    training = run_epoch(model, loader, device=device, optimizer=optimizer)
    assert model.calls[-2:] == [(True, True), (True, True)]
    assert training.samples == 3
    assert model.bias.detach().abs().sum() > 0
    saved = model.bias.detach().clone()
    run_epoch(model, loader, device=device)
    torch.testing.assert_close(saved, model.bias.detach(), atol=0, rtol=0)


def test_device_requests_and_auto_order(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    assert select_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="CUDA was requested"):
        select_device("cuda")
    with pytest.raises(ValueError, match="MPS was requested"):
        select_device("mps")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device("auto").type == "mps"
    assert select_device("mps").type == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert select_device("auto") == select_device("cuda") == torch.device("cuda:1")
    assert select_device("cpu").type == "cpu"


@pytest.mark.parametrize(("name", "expected"), [("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW)])
def test_selects_optimizer_and_learning_rate(name, expected):
    config = TrainingConfig(optimizer=name, learning_rate=0.0007)
    optimizer = make_optimizer(Recorder(), config)
    assert type(optimizer) is expected
    assert optimizer.param_groups[0]["lr"] == 0.0007
