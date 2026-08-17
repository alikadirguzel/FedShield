"""Integration tests: local fit, SecAgg cancellation, mTLS, Non-IID, live round."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import flwr as fl
import numpy as np
import pytest

from src.client.client_node import FedShieldClient
from src.common.crypto import issue_dev_certificates
from src.common.grpc_mtls import start_numpy_client_mtls
from src.common.utils import get_parameters, set_parameters
from src.models.network import create_model
from src.server.custom_strategy import FedShieldStrategy


def _privacy_cfg(**overrides: object) -> dict:
    cfg = {
        "differential_privacy": {
            "target_epsilon": 8.0,
            "epsilon_max": 50.0,
            "target_delta": 1.0e-5,
            "noise_multiplier": 1.0,
            "max_grad_norm": 1.0,
            "calibrate_to_target": False,
            "poisson_sampling": False,
        },
        "training": {"local_epochs": 1, "batch_size": 32, "learning_rate": 0.05},
        "federated": {"num_rounds": 2},
        "aggregation": {"mode": "secagg", "max_update_norm": 50.0},
        "client": {"input_dim": 784, "num_classes": 10},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


@pytest.fixture()
def privacy_cfg() -> dict:
    return _privacy_cfg()


@pytest.fixture()
def partitioned_data(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for cid in ("client-alpha", "client-beta"):
        x = rng.normal(size=(120, 784)).astype(np.float32)
        y = rng.integers(0, 10, size=(120,)).astype(np.int64)
        np.save(tmp_path / f"{cid}_x.npy", x)
        np.save(tmp_path / f"{cid}_y.npy", y)
    return tmp_path


def test_parameter_roundtrip_serialization() -> None:
    model = create_model()
    params = get_parameters(model)
    clone = create_model()
    set_parameters(clone, params)
    for a, b in zip(get_parameters(model), get_parameters(clone)):
        np.testing.assert_allclose(a, b)


def test_one_round_local_fit_updates_weights(privacy_cfg: dict, partitioned_data: Path) -> None:
    client = FedShieldClient("client-alpha", privacy_cfg, str(partitioned_data))
    before = get_parameters(client.trainer.model)
    after, n, metrics = client.fit(before, {"server_round": 1, "num_rounds": 2})
    assert n > 0
    assert "epsilon" in metrics
    assert metrics["epsilon"] > 0
    assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_client_epsilon_composes_across_fit_rounds(privacy_cfg: dict, partitioned_data: Path) -> None:
    client = FedShieldClient("client-alpha", privacy_cfg, str(partitioned_data))
    params = get_parameters(client.trainer.model)
    _, _, first = client.fit(params, {"server_round": 1, "num_rounds": 2})
    _, _, second = client.fit(get_parameters(client.trainer.model), {"server_round": 2, "num_rounds": 2})
    assert second["epsilon"] > first["epsilon"]


def test_strategy_records_history() -> None:
    strategy = FedShieldStrategy(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        aggregation_mode="fedavg",
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(create_model())),
    )
    assert strategy.round_history == []


def test_mtls_dev_certs_required_for_secure_channel(tmp_path: Path) -> None:
    server_mat = issue_dev_certificates(tmp_path / "server", "fedshield-server", role="server")
    client_mat = issue_dev_certificates(tmp_path / "client", "client-alpha", role="client")
    from src.common.crypto import CertificateError, validate_certificate

    with pytest.raises(CertificateError):
        validate_certificate(client_mat.cert, server_mat.ca_cert)


def test_non_iid_class_skew_detectable(tmp_path: Path) -> None:
    from scripts.generate_non_iid_data import dirichlet_partition, synthesize_dataset

    features, labels = synthesize_dataset(3000, 784, 10, seed=7)
    parts = dirichlet_partition(labels, 3, 10, alpha=0.3, seed=7)
    hists = []
    for idx in parts:
        hist = np.bincount(labels[idx], minlength=10).astype(float)
        hist /= hist.sum()
        hists.append(hist)
    dist = np.abs(hists[0] - hists[1]).sum()
    assert dist > 0.15, f"Expected Non-IID skew, got L1={dist}"


def test_two_client_masked_fit_cancels(privacy_cfg: dict, partitioned_data: Path) -> None:
    alpha = FedShieldClient("client-alpha", privacy_cfg, str(partitioned_data))
    beta = FedShieldClient("client-beta", privacy_cfg, str(partitioned_data))
    keys = {
        "client-alpha": alpha.secagg_keypair,
        "client-beta": beta.secagg_keypair,
    }
    roster = {cid: keys[cid].public_b64 for cid in keys}
    start = get_parameters(create_model())
    config = {
        "server_round": 1,
        "num_rounds": 2,
        "aggregation_mode": "secagg",
        "secagg_roster": __import__("json").dumps(roster, separators=(",", ":"), sort_keys=True),
    }
    masked_a, n_a, metrics_a = alpha.fit(start, config)
    masked_b, n_b, metrics_b = beta.fit(start, config)
    assert metrics_a["secagg_applied"] is True
    assert metrics_b["secagg_applied"] is True

    true_a = get_parameters(alpha.trainer.model)
    true_b = get_parameters(beta.trainer.model)
    total = float(n_a + n_b)
    expected = [(ta * n_a + tb * n_b) / total for ta, tb in zip(true_a, true_b)]
    got = [(ma * n_a + mb * n_b) / total for ma, mb in zip(masked_a, masked_b)]
    for left, right in zip(expected, got):
        np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-4)


def _wait_for_port(host: str, port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"server did not listen on {host}:{port} within {timeout}s")


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def test_one_federated_round_over_mtls(tmp_path: Path, partitioned_data: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CERT_DIR", str(tmp_path))
    monkeypatch.setenv("FEDSHIELD_DEV_CERTS", "true")
    server_mat = issue_dev_certificates(tmp_path / "server", "server", role="server")
    alpha_mat = issue_dev_certificates(tmp_path / "client-alpha", "client-alpha", role="client")
    beta_mat = issue_dev_certificates(tmp_path / "client-beta", "client-beta", role="client")

    port = _free_port()
    strategy = FedShieldStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        aggregation_mode="secagg",
        num_rounds=1,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(create_model())),
        inplace=False,
        accept_failures=False,
        property_timeout=20.0,
    )
    errors: list[BaseException] = []

    def run_server() -> None:
        try:
            fl.server.start_server(
                server_address=f"localhost:{port}",
                config=fl.server.ServerConfig(num_rounds=1, round_timeout=120.0),
                strategy=strategy,
                grpc_max_message_length=32 * 1024 * 1024,
                certificates=(
                    server_mat.ca_cert.read_bytes(),
                    server_mat.cert.read_bytes(),
                    server_mat.key.read_bytes(),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    server_thread = threading.Thread(target=run_server, name="fl-server", daemon=True)
    server_thread.start()
    _wait_for_port("localhost", port)

    cfg = _privacy_cfg()

    def run_client(client_id: str, material) -> None:
        try:
            client = FedShieldClient(client_id, cfg, str(partitioned_data))
            start_numpy_client_mtls(
                server_address=f"localhost:{port}",
                client=client,
                material=material,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=run_client, args=("client-alpha", alpha_mat), daemon=True),
        threading.Thread(target=run_client, args=("client-beta", beta_mat), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)
    server_thread.join(timeout=30)

    assert not errors, errors
    assert strategy.round_history, "expected at least one aggregated round"
    last = strategy.round_history[-1]
    assert last.get("secagg_aborted", 1.0) == 0.0
    assert last.get("num_fit_clients", 0.0) == 2.0
