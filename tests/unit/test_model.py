"""Unit tests for FedShieldNet architecture constraints."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.models.network import FedShieldNet, assert_no_batchnorm, create_model


def test_forward_pass_shape() -> None:
    model = create_model(input_dim=784, num_classes=10)
    x = torch.randn(16, 784)
    y = model(x)
    assert y.shape == (16, 10)


def test_rejects_wrong_input_dim() -> None:
    model = create_model(input_dim=784, num_classes=10)
    with pytest.raises(ValueError, match="Expected last dim"):
        model(torch.randn(4, 100))


def test_no_batchnorm_layers() -> None:
    model = create_model()
    assert_no_batchnorm(model)
    for module in model.modules():
        assert not isinstance(
            module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        )


def test_assert_no_batchnorm_detects_violation() -> None:
    class BadNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 8)
            self.bn = nn.BatchNorm1d(8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bn(self.fc(x))

    with pytest.raises(AssertionError, match="BatchNorm"):
        assert_no_batchnorm(BadNet())


def test_parameter_count_deterministic() -> None:
    a = create_model()
    b = create_model()
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    assert list(a.state_dict().keys()) == list(b.state_dict().keys())


def test_layernorm_present() -> None:
    model = FedShieldNet()
    assert any(isinstance(m, nn.LayerNorm) for m in model.modules())
    assert not any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
