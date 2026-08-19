"""Hospital ingest, mix partitions, isolation proofs, DP-off training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.client.dp_trainer import DPTrainConfig, DPTrainer
from src.common.data_ingest import load_csv_xy, summarize_partition, write_partition
from src.common.tenants import hospital_ids
from src.console.isolation import (
    no_shared_client_networks,
    parse_egress_audit_lines,
    unique_data_sources,
)
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


def test_unique_mounts_and_networks() -> None:
    mounts = unique_data_sources(
        {
            "hospital-a": ["/data/hospitals/hospital-a"],
            "hospital-b": ["/data/hospitals/hospital-b"],
        }
    )
    assert mounts["ok"] is True
    shared = unique_data_sources(
        {"hospital-a": ["/same"], "hospital-b": ["/same"]}
    )
    assert shared["ok"] is False
    nets = no_shared_client_networks(
        {
            "hospital-a": {"fl-hospital-a"},
            "hospital-b": {"fl-hospital-b"},
        }
    )
    assert nets["ok"] is True
    leak = no_shared_client_networks(
        {
            "hospital-a": {"fl-fabric"},
            "hospital-b": {"fl-fabric"},
        }
    )
    assert leak["ok"] is False


def test_egress_audit_parser_requires_zero_raw_samples() -> None:
    line = json.dumps(
        {
            "message": "Egress audit — raw samples stay local",
            "client_id": "hospital-a",
            "raw_samples_sent": 0,
            "payload_kind": "model_weights",
            "nbytes": 1024,
            "n_tensors": 8,
            "secagg_applied": True,
        }
    )
    parsed = parse_egress_audit_lines([line, "not json"])
    assert parsed[0]["raw_samples_sent"] == 0
    assert parsed[0]["payload_kind"] == "model_weights"


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
