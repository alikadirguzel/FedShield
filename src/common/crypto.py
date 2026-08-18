"""Vault PKI certificate provisioning and gRPC mTLS credential builders.

Production path: short-lived x509 certs from HashiCorp Vault PKI.
Dev path (FEDSHIELD_DEV_CERTS=true): local self-signed certs for unit tests only.
Strict mTLS is always required for Flower gRPC — plaintext is never allowed.
"""

from __future__ import annotations

import os
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path
from typing import Optional, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID

from src.common.logger import setup_logger

logger = setup_logger("fedshield.crypto")


@dataclass(frozen=True)
class TLSMaterial:
    """Paths to CA, certificate, and private key for mTLS."""

    ca_cert: Path
    cert: Path
    key: Path


class CertificateError(RuntimeError):
    """Raised when certificate provisioning or validation fails."""


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes"}


def _resolve_ca_dir(cert_dir: Path) -> Path:
    """Issue all dev leaves from CERT_DIR's CA when that env var is set."""
    shared = os.getenv("CERT_DIR")
    if shared:
        return Path(shared)
    return cert_dir


def issue_dev_certificates(
    cert_dir: Path,
    common_name: str,
    role: str = "server",
    ttl_hours: int = 24,
) -> TLSMaterial:
    """Generate a local CA + leaf cert for development / unit tests.

    NEVER use in production. Controlled by FEDSHIELD_DEV_CERTS=true.
    Nested client directories reuse the shared CA under CERT_DIR when present.
    """
    cert_dir.mkdir(parents=True, exist_ok=True)
    ca_dir = _resolve_ca_dir(cert_dir)
    ca_dir.mkdir(parents=True, exist_ok=True)
    ca_key_path = ca_dir / "dev-ca.key"
    ca_cert_path = ca_dir / "ca.crt"
    leaf_cert_path = cert_dir / f"{role}.crt"
    leaf_key_path = cert_dir / f"{role}.key"

    if not ca_cert_path.exists() or not ca_key_path.exists():
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "FedShield-Dev-CA")])
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=1))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .sign(ca_key, hashes.SHA256())
        )
        ca_key_path.write_bytes(
            ca_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        ca_cert_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    else:
        ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
        ca_cert = x509.load_pem_x509_certificate(ca_cert_path.read_bytes())

    # Ensure leaf directory also has a copy of the shared CA for Flower
    if ca_dir != cert_dir:
        (cert_dir / "ca.crt").write_bytes(ca_cert_path.read_bytes())
        ca_cert_path = cert_dir / "ca.crt"

    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    san = x509.SubjectAlternativeName(
        [
            x509.DNSName("localhost"),
            x509.DNSName(common_name),
            x509.DNSName("server"),
            x509.DNSName("client"),
            x509.IPAddress(ip_address("127.0.0.1")),
        ]
    )
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(leaf_name)
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(hours=ttl_hours))
        .add_extension(san, critical=False)
        .add_extension(
            x509.ExtendedKeyUsage(
                [
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH
                    if role == "server"
                    else x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH
                    if role == "server"
                    else x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                ]
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    leaf_key_path.write_bytes(
        leaf_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    leaf_cert_path.write_bytes(leaf_cert.public_bytes(serialization.Encoding.PEM))
    # Also write generic client cert names expected by compose for shared volume tests
    if role == "client":
        (cert_dir / "client.crt").write_bytes(leaf_cert_path.read_bytes())
        (cert_dir / "client.key").write_bytes(leaf_key_path.read_bytes())
    if role == "server":
        (cert_dir / "server.crt").write_bytes(leaf_cert_path.read_bytes())
        (cert_dir / "server.key").write_bytes(leaf_key_path.read_bytes())

    logger.info("Issued development mTLS certificates", extra={"role": role, "cn": common_name})
    return TLSMaterial(ca_cert=ca_cert_path, cert=leaf_cert_path, key=leaf_key_path)


def fetch_vault_certificate(
    role: str,
    common_name: str,
    cert_dir: Path,
    vault_addr: Optional[str] = None,
    vault_token: Optional[str] = None,
    ttl: str = "24h",
    max_retries: int = 10,
    retry_delay: float = 2.0,
) -> TLSMaterial:
    """Request a short-lived certificate from Vault PKI and persist PEM files."""
    try:
        import hvac
    except ImportError as exc:
        raise CertificateError("hvac is required for Vault certificate provisioning") from exc

    addr = vault_addr or os.getenv("VAULT_ADDR", "http://vault:8200")
    token = vault_token or os.getenv("VAULT_TOKEN")
    if not token:
        raise CertificateError("VAULT_TOKEN is not set")

    cert_dir.mkdir(parents=True, exist_ok=True)
    pki_role = "fedshield-server" if role == "server" else "fedshield-client"

    client = hvac.Client(url=addr, token=token)
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            if not client.is_authenticated():
                raise CertificateError("Vault authentication failed")
            result = client.secrets.pki.generate_certificate(
                name=pki_role,
                common_name=common_name,
                mount_point="pki_int",
                extra_params={"ttl": ttl},
            )
            data = result["data"]
            ca_pem = data.get("issuing_ca") or data.get("ca_chain", [""])[0]
            cert_pem = data["certificate"]
            key_pem = data["private_key"]

            ca_path = cert_dir / "ca.crt"
            cert_path = cert_dir / f"{role}.crt"
            key_path = cert_dir / f"{role}.key"
            ca_path.write_text(ca_pem if isinstance(ca_pem, str) else "\n".join(ca_pem))
            cert_path.write_text(cert_pem)
            key_path.write_text(key_pem)
            key_path.chmod(0o600)

            validate_certificate(cert_path, ca_path)
            logger.info(
                "Provisioned Vault PKI certificate",
                extra={"role": role, "cn": common_name, "attempt": attempt},
            )
            return TLSMaterial(ca_cert=ca_path, cert=cert_path, key=key_path)
        except Exception as exc:  # noqa: BLE001 — retry loop for bootstrap race
            last_error = exc
            logger.warning(
                "Vault certificate fetch failed; retrying",
                extra={"attempt": attempt, "error": str(exc)},
            )
            time.sleep(retry_delay)

    raise CertificateError(f"Failed to fetch Vault certificate: {last_error}")


def load_existing_material(cert_dir: Path, role: str) -> Optional[TLSMaterial]:
    """Return TLSMaterial if ca/cert/key already exist and validate."""
    ca = cert_dir / "ca.crt"
    cert = cert_dir / f"{role}.crt"
    key = cert_dir / f"{role}.key"
    if ca.exists() and cert.exists() and key.exists():
        try:
            validate_certificate(cert, ca)
            return TLSMaterial(ca_cert=ca, cert=cert, key=key)
        except CertificateError as exc:
            logger.warning("Existing certificates invalid; will re-provision", extra={"error": str(exc)})
    return None


def provision_certificates(
    role: str,
    common_name: str,
    cert_dir: Optional[Path] = None,
) -> TLSMaterial:
    """Provision mTLS material via Vault, or local CA when FEDSHIELD_DEV_CERTS=true."""
    target = Path(cert_dir or os.getenv("CERT_DIR", "/certs"))
    existing = load_existing_material(target, role=role)
    if existing is not None:
        logger.info("Reusing provisioned certificates", extra={"role": role, "dir": str(target)})
        return existing
    if _env_bool("FEDSHIELD_DEV_CERTS"):
        logger.warning("Using development certificates — NOT FOR PRODUCTION")
        return issue_dev_certificates(target, common_name=common_name, role=role)
    return fetch_vault_certificate(role=role, common_name=common_name, cert_dir=target)


def certificate_common_name(cert_path: Path) -> str:
    """Return the leaf certificate Common Name (used as the mTLS identity)."""
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    attrs = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    if not attrs:
        raise CertificateError(f"Certificate {cert_path} has no Common Name")
    value = attrs[0].value
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def validate_certificate(cert_path: Path, ca_path: Path) -> x509.Certificate:
    """Validate that cert_path is signed by ca_path and currently valid."""
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    ca = x509.load_pem_x509_certificate(ca_path.read_bytes())
    now = datetime.now(timezone.utc)
    # cryptography >=42 uses timezone-aware properties
    not_before = cert.not_valid_before_utc if hasattr(cert, "not_valid_before_utc") else cert.not_valid_before.replace(tzinfo=timezone.utc)
    not_after = cert.not_valid_after_utc if hasattr(cert, "not_valid_after_utc") else cert.not_valid_after.replace(tzinfo=timezone.utc)
    if now < not_before or now > not_after:
        raise CertificateError(f"Certificate {cert_path} is outside its validity window")
    try:
        public_key = ca.public_key()
        hash_algo = cert.signature_hash_algorithm
        if hash_algo is None:
            raise CertificateError("Certificate missing signature hash algorithm")
        public_key.verify(
            cert.signature,
            cert.tbs_certificate_bytes,
            padding.PKCS1v15(),
            hash_algo,
        )
    except CertificateError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CertificateError(f"Certificate signature validation failed: {exc}") from exc
    return cert


def build_ssl_context(
    material: TLSMaterial,
    *,
    server_side: bool,
) -> ssl.SSLContext:
    """Build an SSLContext that requires mutual TLS (CERT_REQUIRED)."""
    purpose = ssl.Purpose.CLIENT_AUTH if server_side else ssl.Purpose.SERVER_AUTH
    ctx = ssl.create_default_context(purpose=purpose)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = False  # gRPC uses IP/DNS SANs; hostname check via cert CN/SAN
    ctx.load_verify_locations(cafile=str(material.ca_cert))
    ctx.load_cert_chain(certfile=str(material.cert), keyfile=str(material.key))
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def flower_ssl_paths(material: TLSMaterial) -> Tuple[str, str, str]:
    """Return (root_cert, certificate, private_key) path strings for Flower SSLConfig."""
    for path in (material.ca_cert, material.cert, material.key):
        if not path.exists():
            raise CertificateError(f"Missing TLS file: {path}")
    return str(material.ca_cert), str(material.cert), str(material.key)
