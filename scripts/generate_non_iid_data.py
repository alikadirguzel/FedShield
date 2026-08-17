#!/usr/bin/env python3
"""Generate synthetic Non-IID client partitions (Dirichlet label skew).

Raw samples are written only under data/ — never transferred over the FL network.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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
        # Ensure no empty slice due to rounding
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cuts)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    return [np.array(sorted(indices), dtype=np.int64) for indices in client_indices]


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
    # Mild feature scaling
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    return features.astype(np.float32), labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Non-IID FedShield partitions")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration (lower = more Non-IID)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--client-ids",
        nargs="+",
        default=["client-alpha", "client-beta", "client-gamma"],
    )
    args = parser.parse_args()

    if len(args.client_ids) != args.num_clients:
        raise SystemExit("client-ids length must equal num-clients")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features, labels = synthesize_dataset(
        args.n_samples, args.input_dim, args.num_classes, args.seed
    )
    np.save(args.raw_dir / "features.npy", features)
    np.save(args.raw_dir / "labels.npy", labels)

    partitions = dirichlet_partition(
        labels, args.num_clients, args.num_classes, args.alpha, args.seed
    )

    for client_id, indices in zip(args.client_ids, partitions):
        x = features[indices]
        y = labels[indices]
        np.save(args.out_dir / f"{client_id}_x.npy", x)
        np.save(args.out_dir / f"{client_id}_y.npy", y)
        hist = np.bincount(y, minlength=args.num_classes)
        print(f"{client_id}: n={len(y)} class_hist={hist.tolist()}")

    print(f"Wrote Non-IID partitions to {args.out_dir}")


if __name__ == "__main__":
    main()
