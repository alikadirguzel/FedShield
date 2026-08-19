#!/usr/bin/env python3
"""Generate isolated hospital partitions (never placed on the FL network).

Default profile ``hospitals``: five distinct class mixes (like five hospitals
seeing different patient demographics) from one shared task/schema.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

APP_UID = 10001
APP_GID = 10001


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    """Return index arrays per client with Dirichlet(label) Non-IID skew."""
    rng = np.random.default_rng(seed)
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = class_indices[c]
        rng.shuffle(idx)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cuts)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    return [np.array(sorted(indices), dtype=np.int64) for indices in client_indices]


def hospital_mix_partition(
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    seed: int,
    primary_fraction: float = 0.75,
) -> list[np.ndarray]:
    """Each hospital is dominated by a disjoint pair of classes, plus a little overlap."""
    rng = np.random.default_rng(seed)
    pairs = [(i % num_classes, (i + 1) % num_classes) for i in range(0, num_clients * 2, 2)]
    assigned: list[list[int]] = [[] for _ in range(num_clients)]
    leftover: list[int] = []
    for cls in range(num_classes):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        owners = [i for i, pair in enumerate(pairs[:num_clients]) if cls in pair]
        if not owners:
            leftover.extend(idx.tolist())
            continue
        cut = int(len(idx) * primary_fraction)
        primary, rest = idx[:cut], idx[cut:]
        splits = np.array_split(primary, len(owners))
        for owner, split in zip(owners, splits):
            assigned[owner].extend(split.tolist())
        leftover.extend(rest.tolist())
    rng.shuffle(leftover)
    for i, sample in enumerate(leftover):
        assigned[i % num_clients].append(int(sample))
    return [np.array(sorted(set(items)), dtype=np.int64) for items in assigned]


def synthesize_dataset(
    n_samples: int,
    input_dim: int,
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple linear-separable-ish synthetic classification dataset."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(input_dim, num_classes))
    features = rng.normal(size=(n_samples, input_dim)).astype(np.float32)
    logits = features @ weights
    labels = logits.argmax(axis=1).astype(np.int64)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    return features.astype(np.float32), labels


def _chown_tree(path: Path, uid: int = APP_UID, gid: int = APP_GID) -> None:
    try:
        os.chown(path, uid, gid)
        if path.is_dir():
            for child in path.rglob("*"):
                os.chown(child, uid, gid)
    except (OSError, AttributeError):
        pass


def write_partition(dest_dir: Path, features: np.ndarray, labels: np.ndarray, prefix: str | None) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if prefix:
        np.save(dest_dir / f"{prefix}_x.npy", features)
        np.save(dest_dir / f"{prefix}_y.npy", labels)
    else:
        np.save(dest_dir / "x.npy", features)
        np.save(dest_dir / "y.npy", labels)
    _chown_tree(dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Non-IID FedShield partitions")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument(
        "--profile",
        choices=("hospitals", "dirichlet"),
        default="hospitals",
        help="hospitals: 5 distinct class mixes; dirichlet: classic label skew",
    )
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration (lower = more Non-IID)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--isolated-root",
        type=Path,
        default=None,
        help="Write {root}/{client_id}/x.npy (Docker volumes). Skips host data/ paths.",
    )
    parser.add_argument(
        "--client-ids",
        nargs="+",
        default=["hospital-a", "hospital-b", "hospital-c", "hospital-d", "hospital-e"],
    )
    args = parser.parse_args()

    if len(args.client_ids) != args.num_clients:
        raise SystemExit("client-ids length must equal num-clients")

    features, labels = synthesize_dataset(
        args.n_samples, args.input_dim, args.num_classes, args.seed
    )
    if args.profile == "hospitals":
        partitions = hospital_mix_partition(
            labels, args.num_clients, args.num_classes, args.seed
        )
    else:
        partitions = dirichlet_partition(
            labels, args.num_clients, args.num_classes, args.alpha, args.seed
        )

    if args.isolated_root is None:
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.raw_dir / "features.npy", features)
        np.save(args.raw_dir / "labels.npy", labels)

    for client_id, indices in zip(args.client_ids, partitions):
        x = features[indices]
        y = labels[indices]
        if args.isolated_root is not None:
            write_partition(args.isolated_root / client_id, x, y, prefix=None)
        else:
            write_partition(args.out_dir, x, y, prefix=client_id)
        hist = np.bincount(y, minlength=args.num_classes)
        print(f"{client_id}: n={len(y)} class_hist={hist.tolist()}")

    dest = args.isolated_root or args.out_dir
    print(f"Wrote Non-IID partitions to {dest}")


if __name__ == "__main__":
    main()
