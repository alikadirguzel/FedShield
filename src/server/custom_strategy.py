"""Federated strategy: SecAgg-aware FedAvg, trimmed mean, or Krum."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Tuple, Union

import flwr as fl
from flwr.common import FitIns, FitRes, GetPropertiesIns, Metrics, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, aggregate_krum, aggregate_trimmed_avg

from src.common.logger import setup_logger
from src.server.mlflow_tracker import MLflowTracker

logger = setup_logger("fedshield.strategy")

AggregationMode = str


def _weighted_average(metrics: List[Tuple[int, Metrics]], key: str) -> Optional[float]:
    total_examples = 0
    weighted_sum = 0.0
    found = False
    for num_examples, metric in metrics:
        if key not in metric:
            continue
        value = metric[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            weighted_sum += float(value) * num_examples
            total_examples += num_examples
            found = True
    if not found or total_examples == 0:
        return None
    return weighted_sum / total_examples


def _max_metric(metrics: List[Tuple[int, Metrics]], key: str) -> Optional[float]:
    values = []
    for _, metric in metrics:
        if key in metric and isinstance(metric[key], (int, float)):
            values.append(float(metric[key]))
    return max(values) if values else None


def _as_bool(value: object) -> bool:
    return value in (True, "True", "true", 1, "1")


class FedShieldStrategy(fl.server.strategy.FedAvg):
    """FedAvg extended with SecAgg rostering, robust aggregators, and MLflow."""

    def __init__(
        self,
        *args,
        mlflow_tracker: Optional[MLflowTracker] = None,
        aggregation_mode: AggregationMode = "secagg",
        trim_beta: float = 0.2,
        num_malicious_clients: int = 0,
        krum_to_keep: int = 0,
        num_rounds: int = 5,
        property_timeout: float = 15.0,
        **kwargs,
    ) -> None:
        kwargs.setdefault("inplace", False)
        kwargs.setdefault("accept_failures", aggregation_mode != "secagg")
        super().__init__(*args, **kwargs)
        self.mlflow_tracker = mlflow_tracker
        self.aggregation_mode = aggregation_mode
        self.trim_beta = float(trim_beta)
        self.num_malicious_clients = int(num_malicious_clients)
        self.krum_to_keep = int(krum_to_keep)
        self.num_rounds = int(num_rounds)
        self.property_timeout = float(property_timeout)
        self.round_history: List[Dict[str, float]] = []
        self._round_roster: Dict[int, Set[str]] = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config.update(self.on_fit_config_fn(server_round))
        config["server_round"] = int(server_round)
        config["num_rounds"] = int(self.num_rounds)
        config["aggregation_mode"] = self.aggregation_mode

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if self.aggregation_mode == "secagg":
            roster = self._collect_secagg_roster(clients, server_round)
            config["secagg_roster"] = json.dumps(roster, separators=(",", ":"), sort_keys=True)
            self._round_roster[server_round] = set(roster.keys())
            logger.info(
                "SecAgg roster published",
                extra={"round": server_round, "n_clients": len(roster)},
            )

        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def _collect_secagg_roster(
        self, clients: List[ClientProxy], server_round: int
    ) -> Dict[str, str]:
        roster: Dict[str, str] = {}
        for client in clients:
            try:
                response = client.get_properties(
                    GetPropertiesIns(config={"q": "secagg"}),
                    timeout=self.property_timeout,
                    group_id=server_round,
                )
                properties = dict(response.properties or {})
                client_id = str(properties.get("client_id") or client.cid)
                public_key = str(properties.get("secagg_pub") or "")
                if not public_key:
                    logger.warning(
                        "Client omitted SecAgg public key",
                        extra={"proxy_cid": client.cid, "client_id": client_id},
                    )
                    continue
                roster[client_id] = public_key
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to fetch SecAgg properties",
                    extra={"proxy_cid": client.cid, "error": str(exc)},
                )
        return roster

    def _secagg_ready(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[bool, str]:
        if failures:
            return False, "fit_failures"
        expected = self._round_roster.get(server_round)
        if not expected:
            return False, "empty_roster"
        returned: Set[str] = set()
        for _, fit_res in results:
            metrics = dict(fit_res.metrics or {})
            client_id = str(metrics.get("client_id") or "")
            if not client_id or not _as_bool(metrics.get("secagg_applied")):
                return False, "mask_missing"
            returned.add(client_id)
        if returned != expected:
            return False, "dropout"
        return True, ""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            logger.warning(
                "Fit failures observed",
                extra={"round": server_round, "n_failures": len(failures)},
            )

        capped_clients = []
        fit_metrics: List[Tuple[int, Metrics]] = []
        for client, fit_res in results:
            metrics = dict(fit_res.metrics or {})
            fit_metrics.append((fit_res.num_examples, metrics))
            if _as_bool(metrics.get("epsilon_cap_hit")):
                capped_clients.append(metrics.get("client_id", client.cid))

        if capped_clients:
            logger.warning(
                "Clients hit epsilon cap",
                extra={"round": server_round, "clients": capped_clients},
            )

        aggregated_metrics: Dict[str, Scalar] = {}
        aggregated_params: Optional[Parameters] = None

        if self.aggregation_mode == "secagg":
            ready, reason = self._secagg_ready(server_round, results, failures)
            if not ready:
                logger.error(
                    "SecAgg round aborted",
                    extra={"round": server_round, "reason": reason},
                )
                aggregated_metrics["secagg_aborted"] = 1.0
                aggregated_metrics["secagg_abort_reason"] = reason
                self._log_round(server_round, aggregated_metrics)
                return None, aggregated_metrics
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_params = ndarrays_to_parameters(aggregate(weights_results))
            aggregated_metrics["secagg_aborted"] = 0.0
        else:
            if not results:
                return None, {}
            if failures and not self.accept_failures:
                return None, {}
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            if self.aggregation_mode == "trimmed_mean":
                aggregated_params = ndarrays_to_parameters(
                    aggregate_trimmed_avg(weights_results, self.trim_beta)
                )
            elif self.aggregation_mode == "krum":
                aggregated_params = ndarrays_to_parameters(
                    aggregate_krum(
                        weights_results,
                        self.num_malicious_clients,
                        self.krum_to_keep,
                    )
                )
            else:
                aggregated_params = ndarrays_to_parameters(aggregate(weights_results))

        avg_train_acc = _weighted_average(fit_metrics, "train_accuracy")
        avg_train_loss = _weighted_average(fit_metrics, "train_loss")
        max_epsilon = _max_metric(fit_metrics, "epsilon")
        avg_epsilon = _weighted_average(fit_metrics, "epsilon")

        if avg_train_acc is not None:
            aggregated_metrics["train_accuracy"] = avg_train_acc
        if avg_train_loss is not None:
            aggregated_metrics["train_loss"] = avg_train_loss
        if max_epsilon is not None:
            aggregated_metrics["epsilon_max_observed"] = max_epsilon
        if avg_epsilon is not None:
            aggregated_metrics["epsilon_avg"] = avg_epsilon
        aggregated_metrics["clients_epsilon_capped"] = float(len(capped_clients))
        aggregated_metrics["num_fit_clients"] = float(len(results))
        aggregated_metrics["aggregation_mode"] = self.aggregation_mode

        self._log_round(server_round, aggregated_metrics)
        logger.info(
            "Aggregated fit metrics",
            extra={"round": server_round, **{k: aggregated_metrics[k] for k in list(aggregated_metrics)[:15]}},
        )
        return aggregated_params, aggregated_metrics

    def _log_round(self, server_round: int, aggregated_metrics: Dict[str, Scalar]) -> None:
        record = {
            k: float(v)
            for k, v in aggregated_metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        record["round"] = float(server_round)
        self.round_history.append(record)
        if self.mlflow_tracker is not None:
            self.mlflow_tracker.log_round(server_round, record)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        metrics_aggregated = dict(metrics_aggregated or {})

        eval_metrics = [(res.num_examples, dict(res.metrics or {})) for _, res in results]
        acc = _weighted_average(eval_metrics, "accuracy")
        eps = _max_metric(eval_metrics, "epsilon")
        if acc is not None:
            metrics_aggregated["accuracy"] = acc
        if eps is not None:
            metrics_aggregated["epsilon_max_observed"] = eps

        if self.mlflow_tracker is not None:
            payload = {
                k: float(v)
                for k, v in metrics_aggregated.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if loss_aggregated is not None:
                payload["eval_loss"] = float(loss_aggregated)
            self.mlflow_tracker.log_round(server_round, payload, params={"phase": "evaluate"})

        return loss_aggregated, metrics_aggregated


def weighted_accuracy_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Flower `evaluate_metrics_aggregation_fn` compatible helper."""
    acc = _weighted_average(metrics, "accuracy")
    return {"accuracy": acc} if acc is not None else {}
