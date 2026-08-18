"""Strict mTLS helpers for Flower gRPC (plaintext forbidden).

Flower 1.8 gaps this module closes:

* ``start_numpy_client`` only forwards a root CA — we patch the channel
  factory so the leaf certificate is presented.
* ``start_server`` sets ``require_client_auth=False`` even when a CA is
  given — we wrap ``grpc.ssl_server_credentials`` so mutual TLS is required.
* Flower assigns a random UUID as ``ClientProxy.cid`` — we bind ``cid`` to
  the verified certificate Common Name so the strategy can reject spoofed
  ``client_id`` values.
"""

from __future__ import annotations

import os
import threading
import types
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Optional, Sequence, Tuple

import grpc

from src.common.crypto import TLSMaterial, flower_ssl_paths
from src.common.logger import setup_logger

if TYPE_CHECKING:
    import flwr as fl

logger = setup_logger("fedshield.grpc_mtls")

DEFAULT_MAX_MESSAGE_LENGTH = 32 * 1024 * 1024  # 32 MiB — enough for FedShieldNet

_join_tls = threading.local()
_channel_tls = threading.local()
_allowed_cns: Optional[frozenset[str]] = None
_channel_patch_lock = threading.Lock()
_channel_patch_depth = 0
_channel_patch_originals: dict[str, object] = {}


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


def peer_certificate_cn(context: grpc.ServicerContext) -> str:
    """Return the verified client certificate Common Name, or empty string."""
    auth = context.auth_context() or {}
    names = auth.get("x509_common_name") or auth.get(b"x509_common_name") or ()
    if names:
        raw = names[0]
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)
    pems = auth.get("x509_pem_cert") or auth.get(b"x509_pem_cert") or ()
    if not pems:
        return ""
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    pem = pems[0] if isinstance(pems[0], bytes) else str(pems[0]).encode("utf-8")
    cert = x509.load_pem_x509_certificate(pem)
    attrs = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    if not attrs:
        return ""
    value = attrs[0].value
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def force_client_auth_ssl_server_credentials(
    original: Callable[..., grpc.ServerCredentials],
) -> Callable[..., grpc.ServerCredentials]:
    """Wrap ``grpc.ssl_server_credentials`` so client certs are always required."""

    def wrapped(*args: object, **kwargs: object) -> grpc.ServerCredentials:
        kwargs["require_client_auth"] = True
        return original(*args, **kwargs)

    return wrapped


def _cn_aware_uuid4(original_uuid4: Callable[[], object]) -> Callable[[], object]:
    def factory() -> object:
        cn = getattr(_join_tls, "client_cn", None)
        if cn:
            return types.SimpleNamespace(hex=str(cn))
        return original_uuid4()

    return factory


def _wrap_flower_join(original_join: Callable) -> Callable:
    def join(self: object, request_iterator: object, context: grpc.ServicerContext) -> Iterator:
        cn = peer_certificate_cn(context)
        if not cn:
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "mTLS client certificate with a Common Name is required",
            )
        allowed = _allowed_cns
        if allowed is not None and cn not in allowed:
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                f"certificate CN '{cn}' is not in the allowlist",
            )
        _join_tls.client_cn = cn
        logger.info("Flower client joined with certificate identity", extra={"cert_cn": cn})
        try:
            yield from original_join(self, request_iterator, context)
        finally:
            _join_tls.client_cn = None

    return join


def _shared_mtls_create_channel(
    server_address: str,
    insecure: bool,
    root_certificates: Optional[bytes] = None,
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
) -> grpc.Channel:
    if insecure:
        raise RuntimeError("Plaintext gRPC is forbidden by FedShield architecture constraints")
    credentials = getattr(_channel_tls, "credentials", None)
    length = int(getattr(_channel_tls, "max_message_length", max_message_length) or max_message_length)
    if credentials is None:
        raise RuntimeError("mTLS channel credentials are not installed on this thread")
    logger.info("Opening mTLS gRPC channel to federated server", extra={"address": server_address})
    options = [
        ("grpc.max_send_message_length", length),
        ("grpc.max_receive_message_length", length),
    ]
    return grpc.secure_channel(server_address, credentials, options=options)


def _mtls_create_channel(
    credentials: grpc.ChannelCredentials,
    max_message_length: int,
) -> Callable[..., grpc.Channel]:
    """Single-thread factory used by unit tests that call the channel function directly."""

    def _create_channel_mtls(
        server_address: str,
        insecure: bool,
        root_certificates: Optional[bytes] = None,
        max_message_length: int = max_message_length,
    ) -> grpc.Channel:
        _channel_tls.credentials = credentials
        _channel_tls.max_message_length = max_message_length
        return _shared_mtls_create_channel(
            server_address,
            insecure,
            root_certificates=root_certificates,
            max_message_length=max_message_length,
        )

    return _create_channel_mtls


@contextmanager
def mtls_channel_patch(
    material: TLSMaterial,
    max_message_length: Optional[int] = None,
) -> Iterator[None]:
    """Install a process-wide channel factory that reads per-thread credentials.

    Two Flower clients in one process (integration tests) must not clobber each
    other's leaf certificates; production is one client per process/container.
    """
    import flwr.client.grpc_client.connection as flwr_conn
    import flwr.common.grpc as flwr_grpc

    global _channel_patch_depth

    length = max_message_length if max_message_length is not None else _max_message_length()
    _channel_tls.credentials = build_client_channel_credentials(material)
    _channel_tls.max_message_length = length

    with _channel_patch_lock:
        if _channel_patch_depth == 0:
            _channel_patch_originals["common"] = getattr(flwr_grpc, "create_channel", None)
            _channel_patch_originals["conn"] = getattr(flwr_conn, "create_channel", None)
            flwr_grpc.create_channel = _shared_mtls_create_channel  # type: ignore[assignment]
            flwr_conn.create_channel = _shared_mtls_create_channel  # type: ignore[assignment]
        _channel_patch_depth += 1
    try:
        yield
    finally:
        _channel_tls.credentials = None
        _channel_tls.max_message_length = None
        with _channel_patch_lock:
            _channel_patch_depth -= 1
            if _channel_patch_depth == 0:
                if _channel_patch_originals.get("common") is not None:
                    flwr_grpc.create_channel = _channel_patch_originals["common"]  # type: ignore[assignment]
                if _channel_patch_originals.get("conn") is not None:
                    flwr_conn.create_channel = _channel_patch_originals["conn"]  # type: ignore[assignment]
                _channel_patch_originals.clear()


@contextmanager
def mtls_server_runtime(
    allowed_client_ids: Optional[Iterable[str]] = None,
) -> Iterator[None]:
    """Force Flower's gRPC server to require client certs and bind cid to CN."""
    global _allowed_cns

    import flwr.server.superlink.fleet.grpc_bidi.flower_service_servicer as flwr_servicer

    ids = [str(item).strip() for item in (allowed_client_ids or []) if str(item).strip()]
    previous_allowed = _allowed_cns
    _allowed_cns = frozenset(ids) if ids else None

    original_ssl = grpc.ssl_server_credentials
    original_uuid4 = flwr_servicer.uuid.uuid4
    original_join = flwr_servicer.FlowerServiceServicer.Join

    grpc.ssl_server_credentials = force_client_auth_ssl_server_credentials(original_ssl)
    flwr_servicer.uuid.uuid4 = _cn_aware_uuid4(original_uuid4)  # type: ignore[assignment]
    flwr_servicer.FlowerServiceServicer.Join = _wrap_flower_join(original_join)  # type: ignore[assignment]

    logger.info(
        "Flower server mTLS runtime installed",
        extra={"require_client_auth": True, "allowlist": sorted(_allowed_cns) if _allowed_cns else "any-valid-cert"},
    )
    try:
        yield
    finally:
        grpc.ssl_server_credentials = original_ssl
        flwr_servicer.uuid.uuid4 = original_uuid4
        flwr_servicer.FlowerServiceServicer.Join = original_join
        _allowed_cns = previous_allowed


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


def start_server_mtls(
    *,
    server_address: str,
    config: "fl.server.ServerConfig",
    strategy: object,
    certificates: Tuple[bytes, bytes, bytes],
    grpc_max_message_length: Optional[int] = None,
    allowed_client_ids: Optional[Sequence[str]] = None,
) -> object:
    """Start a Flower server with mutual TLS and certificate-bound client ids."""
    import flwr as fl

    length = grpc_max_message_length if grpc_max_message_length is not None else _max_message_length()
    with mtls_server_runtime(allowed_client_ids=allowed_client_ids):
        return fl.server.start_server(
            server_address=server_address,
            config=config,
            strategy=strategy,
            grpc_max_message_length=length,
            certificates=certificates,
        )
