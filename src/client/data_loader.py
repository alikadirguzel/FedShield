"""Non-IID local data loader — raw samples never leave the client container."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.logger import setup_logger

logger = setup_logger("fedshield.data_loader")


class LocalClientDataset(Dataset):
    """In-memory feature/label dataset owned exclusively by one client."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if len(features) != len(labels):
            raise ValueError("features and labels length mismatch")
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def load_client_partition(
    client_id: str,
    data_dir: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Load Non-IID partition for client_id from processed data directory.

    Expected files: `{data_dir}/{client_id}_x.npy`, `{client_id}_y.npy`
    """
    root = Path(data_dir)
    x_path = root / f"{client_id}_x.npy"
    y_path = root / f"{client_id}_y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing partition for {client_id} under {root}. "
            "Run scripts/generate_non_iid_data.py first."
        )

    features = np.load(x_path)
    labels = np.load(y_path)
    logger.info(
        "Loaded local partition",
        extra={
            "client_id": client_id,
            "n_samples": int(len(labels)),
            "n_features": int(features.shape[1]) if features.ndim > 1 else 1,
            "n_classes_present": int(len(np.unique(labels))),
        },
    )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    split = int(len(indices) * (1.0 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    train_ds = LocalClientDataset(features[train_idx], labels[train_idx])
    val_ds = LocalClientDataset(features[val_idx], labels[val_idx])
    return train_ds, val_ds


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader that stays inside the client process (no network export)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def class_distribution(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Return class histogram for Non-IID verification."""
    hist = np.zeros(num_classes, dtype=np.int64)
    for cls, count in zip(*np.unique(labels, return_counts=True)):
        hist[int(cls)] = int(count)
    return hist
