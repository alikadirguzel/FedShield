"""Load hospital-local CSV/NPY partitions without sending them anywhere."""

from __future__ import annotations

import hashlib
import io
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


def _integer_labels(label_col: np.ndarray) -> np.ndarray:
    """Require whole, non-negative class ids. Never truncate 1.9 to 1."""
    values = np.asarray(label_col, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Etiket sütunu son sütunda tam sayı olmalı.")
    nearest = np.rint(values)
    if float(np.max(np.abs(values - nearest))) > 1e-6:
        raise ValueError(
            "Etiket sütunu tam sayı olmalı (0, 1, 2, …). "
            "Ondalıklı etiketler kesilip kaydedilmez."
        )
    labels = nearest.astype(np.int64)
    if np.any(labels < 0):
        raise ValueError("Etiketler negatif olamaz.")
    return labels


def _class_histogram(labels: np.ndarray, num_classes: int) -> tuple[list[int], int]:
    """Count labels in ``[0, num_classes)`` without sizing the histogram by max(label)."""
    n_classes = int(num_classes)
    if n_classes < 1:
        raise ValueError("num_classes must be positive")
    hist = [0] * n_classes
    flat = np.asarray(labels).reshape(-1)
    if flat.size == 0:
        return hist, 0
    invalid = 0
    if np.issubdtype(flat.dtype, np.integer):
        usable = flat.astype(np.int64, copy=False)
    else:
        values = flat.astype(np.float64, copy=False)
        finite = np.isfinite(values)
        nearest = np.rint(values)
        whole = finite & (np.abs(values - nearest) <= 1e-6)
        invalid = int((~whole).sum())
        usable = nearest[whole].astype(np.int64) if np.any(whole) else np.empty(0, dtype=np.int64)
    if usable.size == 0:
        return hist, invalid
    in_range = (usable >= 0) & (usable < n_classes)
    invalid += int((~in_range).sum())
    if np.any(in_range):
        hist = np.bincount(usable[in_range], minlength=n_classes).tolist()
    return hist, invalid


def load_csv_xy(
    csv_path: Path,
    input_dim: int,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CSV: numeric feature columns then a final integer label column.

    Accepts an optional header row and comma or semicolon delimiters.
    Fractional labels are rejected. When ``num_classes`` is set, labels must
    fall in ``[0, num_classes)``.
    """
    text = Path(csv_path).read_text(encoding="utf-8-sig", errors="replace")
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    delimiter = ";" if first.count(";") > first.count(",") else ","
    raw = np.genfromtxt(csv_path, delimiter=delimiter, encoding="utf-8-sig")
    if raw.ndim == 1:
        raw = np.atleast_2d(raw)
    if raw.size == 0 or not np.isfinite(raw).any():
        raw = np.genfromtxt(csv_path, delimiter=delimiter, skip_header=1, encoding="utf-8-sig")
        if raw.ndim == 1:
            raw = np.atleast_2d(raw)
    if raw.size == 0 or not np.isfinite(raw).any():
        raise ValueError("CSV okunamadı. Sayısal sütunlar ve son sütunda etiket olmalı.")
    if raw.shape[0] > 1 and np.isnan(raw[0]).all():
        raw = raw[1:]
    finite_rows = np.isfinite(raw).all(axis=1)
    raw = raw[finite_rows]
    if raw.size == 0:
        raise ValueError("CSV'de sayısal satır yok (başlık dışında metin/Excel formatı olabilir).")
    if raw.shape[1] < 2:
        raise ValueError("CSV'de en az bir özellik sütunu ve bir etiket sütunu olmalı.")
    features = raw[:, :-1].astype(np.float32)
    labels = _integer_labels(raw[:, -1])
    if num_classes is not None:
        n_classes = int(num_classes)
        if n_classes < 1:
            raise ValueError("num_classes must be positive")
        if np.any(labels >= n_classes):
            raise ValueError(
                f"Etiketler 0-{n_classes - 1} aralığında olmalı; "
                f"dosyada {int(labels.max())} var."
            )
    expected = int(input_dim)
    if features.shape[1] != expected:
        raise ValueError(
            f"CSV'de {features.shape[1]} özellik sütunu var; bu model {expected} bekliyor "
            f"(son sütun etiket). Rastgele bir Excel dosyası yüklemeyin — karttaki "
            f"'örnek CSV indir' ile doğru şemayı alın."
        )
    return features, labels


def partition_to_csv(dest_dir: Path) -> str:
    """Serialize an existing npy partition to CSV text (features..., label)."""
    x_path, y_path = partition_paths(Path(dest_dir))
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Bu hastanede henüz x.npy/y.npy yok")
    features = np.load(x_path)
    labels = np.load(y_path).reshape(-1, 1)
    stacked = np.hstack([features.astype(np.float32), labels.astype(np.float32)])
    buf = io.StringIO()
    np.savetxt(buf, stacked, delimiter=",", fmt="%.6g")
    return buf.getvalue()


def summarize_partition(dest_dir: Path, num_classes: int = 10) -> dict[str, Any]:
    """Return counts and hashes only — never the raw rows."""
    x_path, y_path = partition_paths(Path(dest_dir))
    if not x_path.exists() or not y_path.exists():
        return {"present": False, "n_samples": 0}
    features = np.load(x_path, mmap_mode="r")
    labels = np.load(y_path)
    hist, out_of_range = _class_histogram(labels, num_classes)
    digest = hashlib.sha256()
    digest.update(x_path.read_bytes())
    digest.update(y_path.read_bytes())
    return {
        "present": True,
        "n_samples": int(len(labels)),
        "n_features": int(features.shape[1]) if features.ndim > 1 else 1,
        "class_hist": hist,
        "out_of_range": out_of_range,
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
