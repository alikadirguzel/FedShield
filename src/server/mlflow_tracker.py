"""MLflow experiment tracking for federated rounds."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.common.logger import setup_logger

logger = setup_logger("fedshield.mlflow")


class MLflowTracker:
    """Thin wrapper around MLflow for federated metric persistence.

    All round metrics are logged into one parent run with ``step=round_idx``
    so the MLflow UI charts (accuracy / loss / epsilon) are visible immediately.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "fedshield-federated-dp",
        run_name_prefix: str = "round",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.experiment_name = experiment_name
        self.run_name_prefix = run_name_prefix
        self.username = username or os.getenv("MLFLOW_TRACKING_USERNAME")
        self.password = password or os.getenv("MLFLOW_TRACKING_PASSWORD")
        self._mlflow = None
        self._active_run = None
        self._enabled = True

    def _ensure_client(self) -> bool:
        if self._mlflow is not None:
            return True
        try:
            import mlflow

            if self.username and self.password:
                os.environ["MLFLOW_TRACKING_USERNAME"] = self.username
                os.environ["MLFLOW_TRACKING_PASSWORD"] = self.password
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            logger.info(
                "MLflow tracker ready",
                extra={"tracking_uri": self.tracking_uri, "experiment": self.experiment_name},
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MLflow unavailable; metrics will be logged locally only",
                extra={"error": str(exc)},
            )
            self._enabled = False
            return False

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> None:
        if not self._ensure_client() or not self._enabled:
            return
        try:
            if self._mlflow.active_run() is not None:
                self._mlflow.end_run()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._active_run = self._mlflow.start_run(run_name=run_name, tags=tags or {})
            logger.info("MLflow parent run started", extra={"run_name": run_name})
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MLflow start_run failed; metrics will be logged locally only",
                extra={"error": str(exc)},
            )
            self._enabled = False
            self._active_run = None

    def end_run(self) -> None:
        if self._mlflow is None or not self._enabled:
            return
        try:
            self._mlflow.end_run()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow end_run failed", extra={"error": str(exc)})
        self._active_run = None

    def log_round(
        self,
        round_idx: int,
        metrics: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log aggregated federated metrics for a completed round onto the parent run."""
        logger.info(
            "Round metrics",
            extra={"round": round_idx, **{k: metrics[k] for k in list(metrics)[:20]}},
        )
        if not self._ensure_client() or not self._enabled:
            return

        if self._mlflow.active_run() is None:
            self.start_run(run_name="federated-session")

        try:
            if params:
                safe = {
                    f"r{round_idx}_{k}": str(v)[:250]
                    for k, v in params.items()
                }
                try:
                    self._mlflow.log_params(safe)
                except Exception:  # noqa: BLE001
                    pass

            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._mlflow.log_metric(str(key), float(value), step=int(round_idx))

            self._mlflow.set_tag("last_federated_round", str(round_idx))
            logger.info(
                "Logged metrics to MLflow parent run",
                extra={"round": round_idx, "run_id": self._mlflow.active_run().info.run_id},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log round to MLflow", extra={"error": str(exc), "round": round_idx})
