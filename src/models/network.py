"""DP-SGD compatible neural network for federated training.

Constraint: BatchNorm is forbidden under Opacus (leaks batch statistics).
This MLP uses LayerNorm (allowed alternative to GroupNorm for rank-2 tensors).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearNormBlock(nn.Module):
    """Linear → LayerNorm → ReLU → Dropout (DP-compatible, no BatchNorm)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.fc(x))))


class FedShieldNet(nn.Module):
    """Compact MLP for tabular or flattened image features.

    Architecture is static across all clients; weight shapes must match exactly.
    """

    def __init__(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
        hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 0:
            raise ValueError("input_dim and num_classes must be positive")

        blocks: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            blocks.append(LinearNormBlock(prev, hidden))
            prev = hidden
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Linear(prev, num_classes)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {x.size(-1)}. "
                "Static model architecture alignment required."
            )
        return self.classifier(self.features(x))


def assert_no_batchnorm(model: nn.Module) -> None:
    """Raise if any BatchNorm layer is present (privacy constraint)."""
    for name, module in model.named_modules():
        if isinstance(
            module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        ):
            raise AssertionError(
                f"BatchNorm detected at '{name}'. Use GroupNorm/LayerNorm for DP-SGD."
            )


def create_model(input_dim: int = 784, num_classes: int = 10) -> FedShieldNet:
    """Factory used by server and all clients — guarantees identical architecture."""
    model = FedShieldNet(input_dim=input_dim, num_classes=num_classes)
    assert_no_batchnorm(model)
    return model
