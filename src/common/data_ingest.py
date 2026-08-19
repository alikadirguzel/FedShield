"""Load hospital-local CSV/NPY partitions without sending them anywhere."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

APP_UID = 10001
APP_GID = 10001


def partition_paths(dest_dir: Path) -> tuple[Path, Path]:
    root = Path(dest_dir)
    return root / "x.npy", root / "y.npy"


def write_partition(dest_dir: Path, features: np.ndarray, labels: np.ndarray) -> None:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if len(x) != len(y):
        raise ValueError("features and labels length mismatch")
    if x.ndim != 2:
        raise ValueError("features must be a 2-D array [n_samples, n_features]")
    x_path, y_path = partition_paths(dest)
    np.save(x_path, x)
    np.save(y_path, y)
    _chown_tree(dest)


def load_csv_xy(csv_path: Path, input_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """CSV: feature columns then a final integer label column."""
    raw = np.genfromtxt(csv_path, delimiter=",")
    if raw.ndim == 1:
        raw = np.atleast_2d(raw)
    if raw.size == 0 or np.isnan(raw).all():
        raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        if raw.ndim == 1:
            raw = np.atleast_2d(raw)
    if np.isnan(raw[0]).all() and raw.shape[0] > 1:
        raw = raw[1:]
    if np.isnan(raw).any():
        raise ValueError("CSV contains non-numeric cells (use a header row plus numeric feature/label columns)")
    if raw.shape[1] < 2:
        raise ValueError("CSV needs feature columns plus a label column")
    features = raw[:, :-1].astype(np.float32)
    labels = raw[:, -1].astype(np.int64)
    if features.shape[1] != int(input_dim):
        raise ValueError(
            f"CSV has {features.shape[1]} feature columns; this job expects {input_dim}. "
            "All hospitals must share the same schema."
        )
    return features, labels


def summarize_partition(dest_dir: Path, num_classes: int = 10) -> dict[str, Any]:
    """Return counts and hashes only — never the raw rows."""
    x_path, y_path = partition_paths(Path(dest_dir))
    if not x_path.exists() or not y_path.exists():
        return {"present": False, "n_samples": 0}
    features = np.load(x_path, mmap_mode="r")
    labels = np.load(y_path)
    hist = np.bincount(labels, minlength=int(num_classes)).tolist()
    digest = hashlib.sha256()
    digest.update(x_path.read_bytes())
    digest.update(y_path.read_bytes())
    return {
        "present": True,
        "n_samples": int(len(labels)),
        "n_features": int(features.shape[1]) if features.ndim > 1 else 1,
        "class_hist": hist,
        "sha256": digest.hexdigest()[:16],
        "files": ["x.npy", "y.npy"],
    }


def _chown_tree(path: Path, uid: int = APP_UID, gid: int = APP_GID) -> None:
    try:
        import os

        os.chown(path, uid, gid)
        if path.is_dir():
            for child in path.rglob("*"):
                os.chown(child, uid, gid)
    except (OSError, AttributeError):
        pass
