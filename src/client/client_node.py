"""Flower NumPyClient — local DP training with zero raw-data egress."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np

from src.client.data_loader import load_client_partition, make_dataloader
from src.client.dp_trainer import DPTrainConfig, DPTrainer
from src.common.config import apply_logging_config, load_yaml
from src.common.crypto import provision_certificates
from src.common.grpc_mtls import start_numpy_client_mtls
from src.common.logger import setup_logger
from src.common.secagg import (
    SecAggError,
    clip_update,
    decode_roster,
    generate_keypair,
    mask_weights,
    pairwise_masks,
)
from src.common.utils import get_parameters, set_parameters
from src.models.network import create_model

logger = setup_logger("fedshield.client")


class FedShieldClient(fl.client.NumPyClient):
    """Confidential federated client: DP-SGD, update clip, optional SecAgg masks."""

    def __init__(
        self,
        client_id: str,
        privacy_cfg: dict,
        data_dir: str,
        num_rounds: int | None = None,
    ) -> None:
        self.client_id = client_id
        train_cfg = privacy_cfg.get("training", {})
        dp_cfg = privacy_cfg.get("differential_privacy", {})
        client_cfg = privacy_cfg.get("client", {})
        fed_cfg = privacy_cfg.get("federated", {})
        agg_cfg = privacy_cfg.get("aggregation", {})

        self.input_dim = int(client_cfg.get("input_dim", 784))
        self.num_classes = int(client_cfg.get("num_classes", 10))
        self.batch_size = int(train_cfg.get("batch_size", 64))
        self.max_update_norm = float(agg_cfg.get("max_update_norm", 5.0))
        self.secagg_keypair = generate_keypair()

        rounds = int(num_rounds or fed_cfg.get("num_rounds", 5))
        self.dp_config = DPTrainConfig(
            epochs=int(train_cfg.get("local_epochs", 2)),
            learning_rate=float(train_cfg.get("learning_rate", 0.01)),
            batch_size=self.batch_size,
            noise_multiplier=float(dp_cfg.get("noise_multiplier", 1.1)),
            max_grad_norm=float(dp_cfg.get("max_grad_norm", 1.0)),
            target_delta=float(dp_cfg.get("target_delta", 1e-5)),
            epsilon_max=float(dp_cfg.get("epsilon_max", 8.0)),
            target_epsilon=float(dp_cfg.get("target_epsilon", 5.0)),
            num_rounds=rounds,
            calibrate_to_target=bool(dp_cfg.get("calibrate_to_target", True)),
            poisson_sampling=bool(dp_cfg.get("poisson_sampling", True)),
        )

        self.model = create_model(self.input_dim, self.num_classes)
        self.train_ds, self.val_ds = load_client_partition(client_id, data_dir)
        self.trainer = DPTrainer(self.model, self.dp_config, n_samples=len(self.train_ds))
        logger.info("Client initialized", extra={"client_id": client_id})

    def get_properties(self, config: Dict) -> Dict:
        return {
            "client_id": self.client_id,
            "secagg_pub": self.secagg_keypair.public_b64,
        }

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_parameters(self.trainer.get_underlying_model())

    def _maybe_mask(
        self,
        weights: List[np.ndarray],
        config: Dict,
        num_examples: int,
        metrics: Dict,
    ) -> List[np.ndarray]:
        mode = str(config.get("aggregation_mode", "secagg"))
        roster_raw = config.get("secagg_roster", "")
        if mode != "secagg" or not roster_raw:
            metrics["secagg_applied"] = False
            return weights
        try:
            roster = decode_roster(str(roster_raw))
            round_idx = int(config.get("server_round", 0))
            masks = pairwise_masks(
                client_id=self.client_id,
                keypair=self.secagg_keypair,
                roster=roster,
                round_idx=round_idx,
                shapes=[tuple(w.shape) for w in weights],
                num_examples=num_examples,
            )
            metrics["secagg_applied"] = True
            metrics["secagg_peers"] = int(max(len(roster) - 1, 0))
            return mask_weights(weights, masks)
        except SecAggError as exc:
            logger.error("SecAgg masking failed", extra={"error": str(exc), "client_id": self.client_id})
            metrics["secagg_applied"] = False
            metrics["secagg_error"] = str(exc)[:200]
            return weights

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        incoming = [np.array(p, copy=True) for p in parameters]
        if "num_rounds" in config:
            self.trainer.recalibrate_if_idle(int(config["num_rounds"]))
        set_parameters(self.trainer.model, incoming)

        train_loader = make_dataloader(self.train_ds, self.batch_size, shuffle=True)
        result = self.trainer.train(train_loader)
        updated = get_parameters(self.trainer.get_underlying_model())
        clipped = clip_update(incoming, updated, self.max_update_norm)
        set_parameters(self.trainer.model, clipped)

        metrics: Dict = {
            "client_id": self.client_id,
            "train_loss": float(result.loss),
            "train_accuracy": float(result.accuracy),
            "epsilon": float(result.epsilon),
            "delta": float(result.delta),
            "epochs_completed": int(result.epochs_completed),
            "stopped_early": bool(result.stopped_early),
            "reason": result.reason or "",
            "noise_multiplier": float(result.noise_multiplier),
            "steps_taken": int(result.steps_taken),
            "epsilon_cap_hit": bool(result.reason == "epsilon_cap_exceeded"),
        }
        outbound = self._maybe_mask(clipped, config, len(self.train_ds), metrics)
        return outbound, len(self.train_ds), metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        set_parameters(self.trainer.model, parameters)
        metrics = self.trainer.evaluate(make_dataloader(self.val_ds, self.batch_size, shuffle=False))
        return (
            float(metrics["loss"]),
            len(self.val_ds),
            {
                "accuracy": float(metrics["accuracy"]),
                "client_id": self.client_id,
                "epsilon": float(self.trainer.get_epsilon()),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FedShield confidential FL client")
    parser.add_argument("--client-id", default=os.getenv("CLIENT_ID", "client-alpha"))
    parser.add_argument(
        "--privacy-config",
        default=os.getenv("PRIVACY_CONFIG", "config/privacy_config.yaml"),
    )
    parser.add_argument(
        "--server-address",
        default=os.getenv("SERVER_ADDRESS", "server:8080"),
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "data/processed"),
    )
    parser.add_argument(
        "--cert-dir",
        default=os.getenv("CERT_DIR", "certs"),
    )
    args = parser.parse_args()

    os.environ["CLIENT_ID"] = args.client_id
    os.environ["FEDSHIELD_SERVICE"] = f"client-{args.client_id}"

    privacy_cfg = load_yaml(args.privacy_config)
    apply_logging_config(privacy_cfg)

    material = provision_certificates(
        role="client",
        common_name=args.client_id,
        cert_dir=Path(args.cert_dir),
    )

    client = FedShieldClient(
        client_id=args.client_id,
        privacy_cfg=privacy_cfg,
        data_dir=args.data_dir,
    )

    logger.info(
        "Starting Flower client with strict mTLS",
        extra={"server": args.server_address, "client_id": args.client_id},
    )
    start_numpy_client_mtls(
        server_address=args.server_address,
        client=client,
        material=material,
    )


if __name__ == "__main__":
    main()
