"""Hospital ingest, mix partitions, isolation proofs, DP-off training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.client.dp_trainer import DPTrainConfig, DPTrainer
from src.common.data_ingest import load_csv_xy, summarize_partition, write_partition
from src.common.tenants import hospital_ids
from src.console.isolation import collect_proof
from src.models.network import create_model

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))
import generate_non_iid_data as gen  # noqa: E402


def test_five_hospital_ids() -> None:
    ids = hospital_ids()
    assert ids == [
        "hospital-a",
        "hospital-b",
        "hospital-c",
        "hospital-d",
        "hospital-e",
    ]


def test_hospital_mix_gives_distinct_dominant_classes() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=2000)
    parts = gen.hospital_mix_partition(labels, 5, 10, seed=0)
    assert len(parts) == 5
    modes = []
    for idx in parts:
        hist = np.bincount(labels[idx], minlength=10)
        modes.append(int(hist.argmax()))
    assert len(set(modes)) >= 4


def test_csv_ingest_and_summary_hides_rows(tmp_path: Path) -> None:
    features = np.random.default_rng(1).normal(size=(12, 8)).astype(np.float32)
    labels = np.arange(12, dtype=np.int64) % 3
    csv_path = tmp_path / "h.csv"
    stacked = np.column_stack([features, labels])
    np.savetxt(csv_path, stacked, delimiter=",")
    x, y = load_csv_xy(csv_path, input_dim=8)
    dest = tmp_path / "hospital-a"
    write_partition(dest, x, y)
    summary = summarize_partition(dest, num_classes=3)
    assert summary["present"] is True
    assert summary["n_samples"] == 12
    assert "sha256" in summary
    dumped = json.dumps(summary)
    assert "0.123" not in dumped


def test_csv_rejects_fractional_labels(tmp_path: Path) -> None:
    path = tmp_path / "frac.csv"
    path.write_text("1,2,1.9\n3,4,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tam sayı"):
        load_csv_xy(path, input_dim=2)


def test_csv_rejects_negative_and_out_of_range_labels(tmp_path: Path) -> None:
    negative = tmp_path / "neg.csv"
    negative.write_text("1,2,-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="negatif"):
        load_csv_xy(negative, input_dim=2)

    high = tmp_path / "high.csv"
    high.write_text("1,2,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0-9"):
        load_csv_xy(high, input_dim=2, num_classes=10)


def test_csv_accepts_integer_valued_float_labels(tmp_path: Path) -> None:
    path = tmp_path / "ok.csv"
    path.write_text("1,2,1.0\n3,4,0\n", encoding="utf-8")
    _, labels = load_csv_xy(path, input_dim=2, num_classes=10)
    assert labels.tolist() == [1, 0]


def test_summary_survives_out_of_range_labels(tmp_path: Path) -> None:
    dest = tmp_path / "hospital-a"
    write_partition(
        dest,
        np.ones((3, 2), dtype=np.float32),
        np.array([0, -1, 10**9], dtype=np.int64),
    )
    summary = summarize_partition(dest, num_classes=10)
    assert summary["n_samples"] == 3
    assert summary["class_hist"] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert summary["out_of_range"] == 2
    assert "1000000000" not in json.dumps(summary["class_hist"])


def test_csv_rejects_wrong_feature_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("1,2,3\n4,5,6\n", encoding="utf-8")
    try:
        load_csv_xy(path, input_dim=784)
    except ValueError as exc:
        assert "784" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_proof_without_docker_reports_unavailable() -> None:
    result = collect_proof(None)
    assert result["available"] is False
    assert result["checks"] == []
    assert "hospitals" in result


def test_dp_disabled_keeps_epsilon_zero() -> None:
    model = create_model()
    cfg = DPTrainConfig(
        epochs=1,
        batch_size=32,
        learning_rate=0.05,
        epsilon_max=8.0,
        num_rounds=1,
        dp_enabled=False,
        calibrate_to_target=False,
    )
    trainer = DPTrainer(model, cfg, n_samples=64, device="cpu")
    x = torch.randn(64, 784)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    result = trainer.train(loader)
    assert result.epsilon == 0.0
    assert result.noise_multiplier == 0.0
    assert result.steps_taken > 0
