"""Pairwise-mask secure aggregation (honest-but-curious server).

Each selected client ECDH-agrees with every other participant and adds a
deterministic mask that cancels in the weighted sum. The server therefore
observes only the aggregate when the full roster returns; a dropout aborts
the round so leftover masks cannot poison the global model.

Masked payload is ``w_i + m_i / n_i`` so Flower-style weighted averaging
``sum(n_i * masked_i) / sum(n_i)`` equals ``sum(n_i * w_i) / sum(n_i)``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from src.common.logger import setup_logger

logger = setup_logger("fedshield.secagg")

_INFO = b"fedshield-secagg-v1"


class SecAggError(RuntimeError):
    """Raised when the roster is unusable or keys cannot be decoded."""


@dataclass
class SecAggKeypair:
    private: X25519PrivateKey
    public_raw: bytes

    @property
    def public_b64(self) -> str:
        return base64.b64encode(self.public_raw).decode("ascii")


def generate_keypair() -> SecAggKeypair:
    private = X25519PrivateKey.generate()
    public_raw = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return SecAggKeypair(private=private, public_raw=public_raw)


def decode_public_b64(value: str) -> bytes:
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:  # noqa: BLE001
        raise SecAggError(f"Invalid SecAgg public key: {exc}") from exc
    if len(raw) != 32:
        raise SecAggError(f"SecAgg public key must be 32 bytes, got {len(raw)}")
    return raw


def encode_roster(roster: Mapping[str, str]) -> str:
    """JSON object of client_id → base64 public key (Flower Scalar-safe)."""
    return json.dumps(dict(roster), separators=(",", ":"), sort_keys=True)


def decode_roster(payload: str) -> Dict[str, str]:
    if not payload:
        return {}
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SecAggError(f"Invalid SecAgg roster JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SecAggError("SecAgg roster must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def _shared_secret(private: X25519PrivateKey, peer_public_raw: bytes) -> bytes:
    peer = X25519PublicKey.from_public_bytes(peer_public_raw)
    raw = private.exchange(peer)
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=_INFO,
    ).derive(raw)


def _prf_layer(secret: bytes, round_idx: int, layer: int, shape: Sequence[int]) -> np.ndarray:
    info = _INFO + round_idx.to_bytes(4, "big") + layer.to_bytes(4, "big")
    digest = hmac.new(secret, info, hashlib.sha256).digest()
    seed = np.frombuffer(digest, dtype=np.uint32).tolist()
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    return rng.standard_normal(size=tuple(shape), dtype=np.float32)


def pairwise_masks(
    *,
    client_id: str,
    keypair: SecAggKeypair,
    roster: Mapping[str, str],
    round_idx: int,
    shapes: Iterable[Sequence[int]],
    num_examples: int,
) -> List[np.ndarray]:
    """Return ``m_i / n_i`` tensors that cancel across the full roster."""
    shape_list = [tuple(int(d) for d in shape) for shape in shapes]
    masks = [np.zeros(shape, dtype=np.float32) for shape in shape_list]
    peers = {cid: pub for cid, pub in roster.items() if cid != client_id}
    if client_id not in roster:
        raise SecAggError(f"{client_id} is not in the SecAgg roster")
    if not peers:
        raise SecAggError("SecAgg roster must contain at least one peer")

    for peer_id, peer_b64 in peers.items():
        secret = _shared_secret(keypair.private, decode_public_b64(peer_b64))
        sign = 1.0 if client_id < peer_id else -1.0
        for layer, shape in enumerate(shape_list):
            masks[layer] += sign * _prf_layer(secret, round_idx, layer, shape)

    denom = float(max(int(num_examples), 1))
    return [layer / denom for layer in masks]


def mask_weights(
    weights: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> List[np.ndarray]:
    if len(weights) != len(masks):
        raise SecAggError("Weight / mask count mismatch")
    masked: List[np.ndarray] = []
    for weight, mask in zip(weights, masks):
        if tuple(weight.shape) != tuple(mask.shape):
            raise SecAggError(f"Shape mismatch: {weight.shape} vs {mask.shape}")
        masked.append((weight.astype(np.float32, copy=False) + mask).astype(np.float32))
    return masked


def clip_update(
    global_weights: Sequence[np.ndarray],
    local_weights: Sequence[np.ndarray],
    max_norm: float,
) -> List[np.ndarray]:
    """L2-clip the client update before masking (bounds poisoning magnitude)."""
    if max_norm <= 0:
        return [np.array(w, copy=True) for w in local_weights]
    deltas = [loc.astype(np.float64) - glo.astype(np.float64) for loc, glo in zip(local_weights, global_weights)]
    sq = float(sum(float(np.sum(d * d)) for d in deltas))
    norm = sq ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        deltas = [d * scale for d in deltas]
    return [
        (glo.astype(np.float64) + delta).astype(np.float32)
        for glo, delta in zip(global_weights, deltas)
    ]
