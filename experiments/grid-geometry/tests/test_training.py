import pytest
import torch

from grid_geometry.training import select_device


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
