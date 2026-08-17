"""Unit tests for pairwise SecAgg masks and update clipping."""

from __future__ import annotations

import numpy as np

from src.common.secagg import (
    clip_update,
    decode_roster,
    encode_roster,
    generate_keypair,
    mask_weights,
    pairwise_masks,
)
from src.common.utils import get_parameters
from src.models.network import create_model
from src.server.custom_strategy import FedShieldStrategy
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays


def _fake_fit(client_id: str, weights, n: int, epsilon: float = 1.0, masked: bool = True) -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(weights),
        num_examples=n,
        metrics={
            "client_id": client_id,
            "epsilon": epsilon,
            "train_accuracy": 0.5,
            "train_loss": 1.0,
            "secagg_applied": masked,
            "epsilon_cap_hit": False,
        },
    )


class _Proxy:
    def __init__(self, cid: str) -> None:
        self.cid = cid


def test_pairwise_masks_cancel_in_weighted_average() -> None:
    ids = ("client-alpha", "client-beta", "client-gamma")
    keys = {cid: generate_keypair() for cid in ids}
    roster = {cid: keys[cid].public_b64 for cid in ids}
    shapes = [(8, 4), (4,)]
    true = {
        "client-alpha": [np.ones(shapes[0], dtype=np.float32), np.full(shapes[1], 2.0, dtype=np.float32)],
        "client-beta": [np.full(shapes[0], 3.0, dtype=np.float32), np.full(shapes[1], 4.0, dtype=np.float32)],
        "client-gamma": [np.full(shapes[0], 5.0, dtype=np.float32), np.full(shapes[1], 6.0, dtype=np.float32)],
    }
    ns = {"client-alpha": 10, "client-beta": 20, "client-gamma": 30}
    masked = {}
    for cid in ids:
        masks = pairwise_masks(
            client_id=cid,
            keypair=keys[cid],
            roster=roster,
            round_idx=3,
            shapes=shapes,
            num_examples=ns[cid],
        )
        masked[cid] = mask_weights(true[cid], masks)

    total = float(sum(ns.values()))
    expected = [
        sum(true[cid][layer] * ns[cid] for cid in ids) / total
        for layer in range(len(shapes))
    ]
    got = [
        sum(masked[cid][layer] * ns[cid] for cid in ids) / total
        for layer in range(len(shapes))
    ]
    for left, right in zip(expected, got):
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-5)


def test_roster_roundtrip() -> None:
    roster = {"client-alpha": generate_keypair().public_b64}
    assert decode_roster(encode_roster(roster)) == roster


def test_clip_update_bounds_l2_norm() -> None:
    glo = [np.zeros((4,), dtype=np.float32)]
    loc = [np.ones((4,), dtype=np.float32) * 10]
    clipped = clip_update(glo, loc, max_norm=2.0)
    delta = clipped[0] - glo[0]
    assert float(np.linalg.norm(delta)) <= 2.0 + 1e-4


def test_strategy_aborts_secagg_on_dropout() -> None:
    strategy = FedShieldStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        aggregation_mode="secagg",
        initial_parameters=ndarrays_to_parameters(get_parameters(create_model())),
    )
    strategy._round_roster[1] = {"client-alpha", "client-beta"}
    weights = get_parameters(create_model())
    results = [(_Proxy("a"), _fake_fit("client-alpha", weights, 10))]
    params, metrics = strategy.aggregate_fit(1, results, failures=[])
    assert params is None
    assert metrics["secagg_aborted"] == 1.0
    assert metrics["secagg_abort_reason"] == "dropout"


def test_strategy_aggregates_unmasked_fedavg() -> None:
    strategy = FedShieldStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        aggregation_mode="fedavg",
        initial_parameters=ndarrays_to_parameters(get_parameters(create_model())),
    )
    a = [np.zeros((2, 2), dtype=np.float32)]
    b = [np.ones((2, 2), dtype=np.float32)]
    results = [
        (_Proxy("a"), _fake_fit("client-alpha", a, 1, masked=False)),
        (_Proxy("b"), _fake_fit("client-beta", b, 1, masked=False)),
    ]
    params, metrics = strategy.aggregate_fit(1, results, failures=[])
    assert params is not None
    avg = parameters_to_ndarrays(params)
    np.testing.assert_allclose(avg[0], 0.5 * np.ones((2, 2)), atol=1e-6)
    assert metrics["num_fit_clients"] == 2.0
    assert strategy.round_history
