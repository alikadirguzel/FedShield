"""Strict mTLS helpers for Flower gRPC (plaintext forbidden).

Flower 1.8's start_numpy_client only passes a root CA (server auth).
This module patches the channel factory in both `flwr.common.grpc` and
`flwr.client.grpc_client.connection` so the leaf certificate is presented.
The dual patch survives `from X import create_channel` binding at import time.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Tuple

import grpc

from src.common.crypto import TLSMaterial, flower_ssl_paths
from src.common.logger import setup_logger

if TYPE_CHECKING:
    import flwr as fl

logger = setup_logger("fedshield.grpc_mtls")

DEFAULT_MAX_MESSAGE_LENGTH = 32 * 1024 * 1024  # 32 MiB — enough for FedShieldNet


def _max_message_length() -> int:
    raw = os.getenv("GRPC_MAX_MESSAGE_LENGTH")
    if raw:
        return int(raw)
    return DEFAULT_MAX_MESSAGE_LENGTH


def load_mtls_bytes(material: TLSMaterial) -> Tuple[bytes, bytes, bytes]:
    """Return (ca_cert, certificate_chain, private_key) PEM bytes."""
    ca, cert, key = flower_ssl_paths(material)
    return Path(ca).read_bytes(), Path(cert).read_bytes(), Path(key).read_bytes()


def build_client_channel_credentials(material: TLSMaterial) -> grpc.ChannelCredentials:
    """Client credentials that present a leaf cert and verify the server via CA."""
    ca, cert, key = load_mtls_bytes(material)
    return grpc.ssl_channel_credentials(
        root_certificates=ca,
        private_key=key,
        certificate_chain=cert,
    )


def build_server_credentials(material: TLSMaterial) -> grpc.ServerCredentials:
    """Server credentials that require client certificate authentication."""
    ca, cert, key = load_mtls_bytes(material)
    return grpc.ssl_server_credentials(
        private_key_certificate_chain_pairs=[(key, cert)],
        root_certificates=ca,
        require_client_auth=True,
    )


def _mtls_create_channel(
    credentials: grpc.ChannelCredentials,
    max_message_length: int,
) -> Callable[..., grpc.Channel]:
    def _create_channel_mtls(
        server_address: str,
        insecure: bool,
        root_certificates: Optional[bytes] = None,
        max_message_length: int = max_message_length,
    ) -> grpc.Channel:
        if insecure:
            raise RuntimeError("Plaintext gRPC is forbidden by FedShield architecture constraints")
        logger.info("Opening mTLS gRPC channel to federated server", extra={"address": server_address})
        options = [
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
        ]
        return grpc.secure_channel(server_address, credentials, options=options)

    return _create_channel_mtls


@contextmanager
def mtls_channel_patch(
    material: TLSMaterial,
    max_message_length: Optional[int] = None,
) -> Iterator[None]:
    """Install (and later restore) mTLS channel factories used by Flower 1.8."""
    import flwr.client.grpc_client.connection as flwr_conn
    import flwr.common.grpc as flwr_grpc

    length = max_message_length if max_message_length is not None else _max_message_length()
    factory = _mtls_create_channel(build_client_channel_credentials(material), length)
    originals = {
        "common": getattr(flwr_grpc, "create_channel", None),
        "conn": getattr(flwr_conn, "create_channel", None),
    }
    flwr_grpc.create_channel = factory  # type: ignore[assignment]
    flwr_conn.create_channel = factory  # type: ignore[assignment]
    try:
        yield
    finally:
        if originals["common"] is not None:
            flwr_grpc.create_channel = originals["common"]  # type: ignore[assignment]
        if originals["conn"] is not None:
            flwr_conn.create_channel = originals["conn"]  # type: ignore[assignment]


def start_numpy_client_mtls(
    server_address: str,
    client: "fl.client.NumPyClient",
    material: TLSMaterial,
    max_message_length: Optional[int] = None,
) -> None:
    """Start a Flower NumPyClient over mutual TLS only (no insecure fallback)."""
    import flwr as fl

    length = max_message_length if max_message_length is not None else _max_message_length()
    with mtls_channel_patch(material, max_message_length=length):
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client,
            root_certificates=material.ca_cert.read_bytes(),
            insecure=False,
            grpc_max_message_length=length,
        )
