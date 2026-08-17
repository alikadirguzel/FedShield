"""Flower federated server entrypoint with strict mTLS and MLflow tracking."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import flwr as fl

from src.common.config import apply_logging_config, load_yaml
from src.common.crypto import provision_certificates
from src.common.grpc_mtls import DEFAULT_MAX_MESSAGE_LENGTH
from src.common.logger import setup_logger
from src.common.utils import get_parameters
from src.models.network import create_model
from src.server.custom_strategy import FedShieldStrategy, weighted_accuracy_metrics
from src.server.mlflow_tracker import MLflowTracker

logger = setup_logger("fedshield.server")


def fit_config_fn(server_round: int):
    """Send round index to clients (no raw data ever leaves clients)."""
    return {"server_round": server_round}


def main() -> None:
    parser = argparse.ArgumentParser(description="FedShield confidential FL server")
    parser.add_argument(
        "--server-config",
        default=os.getenv("SERVER_CONFIG", "config/server_config.yaml"),
    )
    parser.add_argument(
        "--privacy-config",
        default=os.getenv("PRIVACY_CONFIG", "config/privacy_config.yaml"),
    )
    parser.add_argument(
        "--cert-dir",
        default=os.getenv("CERT_DIR", "certs"),
    )
    args = parser.parse_args()
    os.environ["FEDSHIELD_SERVICE"] = "server"

    server_cfg = load_yaml(args.server_config)
    privacy_cfg = load_yaml(args.privacy_config)
    apply_logging_config(server_cfg)

    srv = server_cfg.get("server", {})
    mlflow_cfg = server_cfg.get("mlflow", {})
    grpc_cfg = server_cfg.get("grpc", {})
    agg_cfg = {**privacy_cfg.get("aggregation", {}), **server_cfg.get("aggregation", {})}

    if not grpc_cfg.get("require_mtls", True):
        raise RuntimeError("Architecture constraint violated: require_mtls must be true")

    address = srv.get("address", "0.0.0.0:8080")
    num_rounds = int(srv.get("num_rounds", 5))
    round_timeout = srv.get("round_timeout")
    min_fit = int(srv.get("min_fit_clients", 3))
    min_eval = int(srv.get("min_evaluate_clients", 3))
    min_available = int(srv.get("min_available_clients", 3))
    fraction_fit = float(srv.get("fraction_fit", 1.0))
    fraction_evaluate = float(srv.get("fraction_evaluate", 1.0))
    grpc_max_message = int(
        grpc_cfg.get("max_message_length", os.getenv("GRPC_MAX_MESSAGE_LENGTH", DEFAULT_MAX_MESSAGE_LENGTH))
    )

    client_cfg = privacy_cfg.get("client", {})
    input_dim = int(client_cfg.get("input_dim", 784))
    num_classes = int(client_cfg.get("num_classes", 10))

    material = provision_certificates(
        role="server",
        common_name="server",
        cert_dir=Path(args.cert_dir),
    )
    ca_bytes = material.ca_cert.read_bytes()
    cert_bytes = material.cert.read_bytes()
    key_bytes = material.key.read_bytes()

    tracker = MLflowTracker(
        tracking_uri=mlflow_cfg.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI"),
        experiment_name=mlflow_cfg.get("experiment_name", "fedshield-federated-dp"),
        run_name_prefix=mlflow_cfg.get("run_name_prefix", "round"),
        username=mlflow_cfg.get("username") or os.getenv("MLFLOW_TRACKING_USERNAME"),
        password=mlflow_cfg.get("password") or os.getenv("MLFLOW_TRACKING_PASSWORD"),
    )
    tracker.start_run(
        run_name="federated-session",
        tags={
            "framework": "flower",
            "dp_engine": "opacus",
            "mtls": "strict",
            "aggregation": str(agg_cfg.get("mode", "secagg")),
        },
    )

    initial_model = create_model(input_dim=input_dim, num_classes=num_classes)
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(initial_model))

    strategy = FedShieldStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit,
        min_evaluate_clients=min_eval,
        min_available_clients=min_available,
        on_fit_config_fn=fit_config_fn,
        evaluate_metrics_aggregation_fn=weighted_accuracy_metrics,
        initial_parameters=initial_parameters,
        mlflow_tracker=tracker,
        aggregation_mode=str(agg_cfg.get("mode", "secagg")),
        trim_beta=float(agg_cfg.get("trim_beta", 0.2)),
        num_malicious_clients=int(agg_cfg.get("num_malicious_clients", 0)),
        krum_to_keep=int(agg_cfg.get("krum_to_keep", 0)),
        num_rounds=num_rounds,
        inplace=False,
        accept_failures=str(agg_cfg.get("mode", "secagg")) != "secagg",
    )

    logger.info(
        "Starting Flower server with strict mTLS",
        extra={
            "address": address,
            "num_rounds": num_rounds,
            "round_timeout": round_timeout,
            "min_available_clients": min_available,
            "aggregation_mode": strategy.aggregation_mode,
            "grpc_max_message_length": grpc_max_message,
        },
    )

    timeout = float(round_timeout) if round_timeout is not None else None
    try:
        fl.server.start_server(
            server_address=address,
            config=fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=timeout),
            strategy=strategy,
            grpc_max_message_length=grpc_max_message,
            certificates=(ca_bytes, cert_bytes, key_bytes),
        )
    finally:
        tracker.end_run()
        logger.info("Federated server shut down")


if __name__ == "__main__":
    main()
