"""Unit tests for client partition loading (named and isolated layouts)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.client.data_loader import load_client_partition


def test_load_named_partition_files(tmp_path: Path) -> None:
    x = np.ones((20, 8), dtype=np.float32)
    y = np.zeros(20, dtype=np.int64)
    np.save(tmp_path / "client-alpha_x.npy", x)
    np.save(tmp_path / "client-alpha_y.npy", y)
    train, val = load_client_partition("client-alpha", tmp_path, val_ratio=0.25)
    assert len(train) + len(val) == 20


def test_load_isolated_volume_layout(tmp_path: Path) -> None:
    x = np.ones((16, 4), dtype=np.float32)
    y = np.arange(16, dtype=np.int64)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", y)
    train, val = load_client_partition("client-beta", tmp_path, val_ratio=0.25)
    assert len(train) + len(val) == 16


def test_missing_partition_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing partition"):
        load_client_partition("client-alpha", tmp_path)
