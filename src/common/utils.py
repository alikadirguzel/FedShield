"""PyTorch parameter conversion helpers for Flower NumPyClient."""

from __future__ import annotations

from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model weights as a list of NumPy arrays (Flower format).

    Always copy so callers can compare snapshots without aliasing live tensors.
    """
    return [val.detach().cpu().numpy().copy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load Flower NumPy parameters into a PyTorch model (strict shape check)."""
    keys = list(model.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError(
            f"Parameter count mismatch: model has {len(keys)} tensors, "
            f"received {len(parameters)}. All clients must share the same architecture."
        )

    state_dict = OrderedDict()
    for key, array in zip(keys, parameters):
        expected = model.state_dict()[key].shape
        if tuple(array.shape) != tuple(expected):
            raise ValueError(
                f"Shape mismatch for '{key}': expected {expected}, got {array.shape}. "
                "Static model architecture alignment is required."
            )
        state_dict[key] = torch.tensor(array)

    model.load_state_dict(state_dict, strict=True)


def parameters_nbytes(parameters: List[np.ndarray]) -> int:
    """Total byte size of a parameter list (for transfer audits)."""
    return int(sum(arr.nbytes for arr in parameters))
