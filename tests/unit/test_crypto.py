"""Unit tests for certificate provisioning, validation, and channel policy."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.common.crypto import (
    CertificateError,
    issue_dev_certificates,
    validate_certificate,
    flower_ssl_paths,
    build_ssl_context,
)
from src.common.grpc_mtls import _mtls_create_channel, build_client_channel_credentials


def test_dev_certificate_issue_and_validate(tmp_path: Path) -> None:
    material = issue_dev_certificates(tmp_path, common_name="fedshield-server", role="server")
    cert = validate_certificate(material.cert, material.ca_cert)
    assert cert.subject
    ca, cert_path, key = flower_ssl_paths(material)
    assert Path(ca).exists() and Path(cert_path).exists() and Path(key).exists()


def test_shared_ca_when_cert_dir_env_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CERT_DIR", str(tmp_path))
    server = issue_dev_certificates(tmp_path / "server", common_name="server", role="server")
    client = issue_dev_certificates(tmp_path / "client-alpha", common_name="client-alpha", role="client")
    validate_certificate(client.cert, server.ca_cert)
    validate_certificate(server.cert, client.ca_cert)


def test_client_dev_certificate(tmp_path: Path) -> None:
    material = issue_dev_certificates(tmp_path, common_name="client-alpha", role="client")
    validate_certificate(material.cert, material.ca_cert)
    ctx = build_ssl_context(material, server_side=False)
    assert ctx.verify_mode != 0


def test_tampered_certificate_rejected(tmp_path: Path) -> None:
    material = issue_dev_certificates(tmp_path, common_name="server", role="server")
    other = issue_dev_certificates(tmp_path / "other", common_name="other-ca", role="server")
    with pytest.raises(CertificateError):
        validate_certificate(material.cert, other.ca_cert)


def test_plaintext_channel_rejected(tmp_path: Path) -> None:
    material = issue_dev_certificates(tmp_path, common_name="client-alpha", role="client")
    factory = _mtls_create_channel(build_client_channel_credentials(material), 1024)
    with pytest.raises(RuntimeError, match="Plaintext"):
        factory("127.0.0.1:1", insecure=True)


@pytest.mark.skipif(
    os.getenv("RUN_VAULT_TESTS", "0") != "1",
    reason="Set RUN_VAULT_TESTS=1 with a live Vault to exercise PKI issuance",
)
def test_vault_certificate_fetch(tmp_path: Path) -> None:
    from src.common.crypto import fetch_vault_certificate

    material = fetch_vault_certificate(
        role="server",
        common_name="fedshield-server",
        cert_dir=tmp_path,
        max_retries=3,
        retry_delay=1.0,
    )
    validate_certificate(material.cert, material.ca_cert)
