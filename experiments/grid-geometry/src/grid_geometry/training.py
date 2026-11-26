"""Shared device selection for training and evaluation."""

import torch

DEVICES = ("auto", "cpu", "mps", "cuda")


def select_device(requested: str) -> torch.device:
    if requested not in DEVICES:
        raise ValueError(f"unknown training device {requested!r}")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but PyTorch reports no available CUDA device")
        return torch.device("cuda", torch.cuda.current_device())
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but PyTorch reports it unavailable")
    return torch.device(requested)
