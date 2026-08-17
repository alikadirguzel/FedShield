#!/usr/bin/env python3
"""Bootstrap Vault PKI and issue FedShield mTLS certificates.

Uses Vault HTTP API via hvac adapter for compatibility across Vault 1.15 / hvac 2.x.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import hvac


def wait_for_vault(client: hvac.Client, timeout: int = 120) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if client.sys.is_initialized() and not client.sys.is_sealed():
                if client.is_authenticated():
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise RuntimeError("Vault did not become ready")


def ensure_mount(client: hvac.Client, path: str, max_lease_ttl: str) -> None:
    mounts = client.sys.list_mounted_secrets_engines()
    names = {k.rstrip("/") for k in mounts}
    if path not in names:
        client.sys.enable_secrets_engine(
            backend_type="pki",
            path=path,
            config={"max_lease_ttl": max_lease_ttl},
        )
    else:
        try:
            client.sys.tune_mount_configuration(path=path, max_lease_ttl=max_lease_ttl)
        except Exception:  # noqa: BLE001
            pass


def _as_data(resp) -> dict:
    """Normalize hvac adapter responses (raw Response or already-parsed dict)."""
    if isinstance(resp, dict):
        return resp.get("data") or resp
    body = resp.json() if getattr(resp, "content", None) else {}
    if isinstance(body, dict):
        return body.get("data") or body
    return {}


def write_json(client: hvac.Client, path: str, payload: dict) -> dict:
    """POST to Vault and return decoded JSON data (or empty dict)."""
    resp = client.adapter.post(f"/v1/{path}", json=payload)
    return _as_data(resp)


def read_json(client: hvac.Client, path: str) -> dict:
    resp = client.adapter.get(f"/v1/{path}")
    return _as_data(resp)


def write_cert_bundle(dest: Path, data: dict, role: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    ca = data.get("issuing_ca") or (data.get("ca_chain") or [""])[0]
    if isinstance(ca, list):
        ca = "\n".join(ca)
    (dest / "ca.crt").write_text(ca)
    (dest / f"{role}.crt").write_text(data["certificate"])
    key_path = dest / f"{role}.key"
    key_path.write_text(data["private_key"])
    # Readable by non-root app user (uid 10001) inside server/client containers
    for path in (dest / "ca.crt", dest / f"{role}.crt", key_path):
        try:
            path.chmod(0o644 if path.suffix == ".crt" else 0o640)
        except OSError:
            pass
    _chown_tree(dest, uid=10001, gid=10001)


def _chown_tree(path: Path, uid: int, gid: int) -> None:
    try:
        os.chown(path, uid, gid)
        if path.is_dir():
            for child in path.rglob("*"):
                os.chown(child, uid, gid)
    except OSError:
        # Windows / unsupported FS — chmod above is the fallback
        pass


def setup_root_ca(client: hvac.Client, addr: str) -> None:
    try:
        read_json(client, "pki/cert/ca")
        print("[vault-init] Root CA already present", flush=True)
    except Exception:  # noqa: BLE001
        print("[vault-init] Generating root CA", flush=True)
        write_json(
            client,
            "pki/root/generate/internal",
            {"common_name": "FedShield Root CA", "ttl": "87600h"},
        )

    write_json(
        client,
        "pki/config/urls",
        {
            "issuing_certificates": [f"{addr}/v1/pki/ca"],
            "crl_distribution_points": [f"{addr}/v1/pki/crl"],
        },
    )


def setup_intermediate(client: hvac.Client, addr: str) -> None:
    print("[vault-init] Configuring intermediate CA", flush=True)
    # If intermediate already has an issuer, skip regeneration
    try:
        issuers = read_json(client, "pki_int/issuers")
        keys = issuers.get("keys") or []
        if keys:
            print("[vault-init] Intermediate issuer already configured", flush=True)
            write_json(
                client,
                "pki_int/config/urls",
                {
                    "issuing_certificates": [f"{addr}/v1/pki_int/ca"],
                    "crl_distribution_points": [f"{addr}/v1/pki_int/crl"],
                },
            )
            return
    except Exception:  # noqa: BLE001
        pass

    csr_data = write_json(
        client,
        "pki_int/intermediate/generate/internal",
        {"common_name": "FedShield Intermediate CA", "ttl": "43800h"},
    )
    csr = csr_data["csr"]

    signed = write_json(
        client,
        "pki/root/sign-intermediate",
        {
            "csr": csr,
            "common_name": "FedShield Intermediate CA",
            "ttl": "43800h",
            "format": "pem_bundle",
        },
    )
    cert_pem = signed["certificate"]
    issuing = signed.get("issuing_ca") or ""
    bundle = cert_pem if "BEGIN CERTIFICATE" in cert_pem and issuing in cert_pem else (
        cert_pem + ("\n" + issuing if issuing else "")
    )

    write_json(client, "pki_int/intermediate/set-signed", {"certificate": bundle})

    # Vault 1.11+ issuer model: ensure a default issuer exists
    try:
        issuers = read_json(client, "pki_int/issuers")
        keys = issuers.get("keys") or []
        if keys:
            write_json(client, "pki_int/config/issuers", {"default": keys[0]})
            print(f"[vault-init] Default intermediate issuer set: {keys[0]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[vault-init] Issuer config note: {exc}", flush=True)

    write_json(
        client,
        "pki_int/config/urls",
        {
            "issuing_certificates": [f"{addr}/v1/pki_int/ca"],
            "crl_distribution_points": [f"{addr}/v1/pki_int/crl"],
        },
    )


def create_roles(client: hvac.Client) -> None:
    write_json(
        client,
        "pki_int/roles/fedshield-server",
        {
            "allowed_domains": ["fedshield-server", "server", "localhost"],
            "allow_subdomains": True,
            "allow_bare_domains": True,
            "allow_localhost": True,
            "client_flag": True,
            "server_flag": True,
            "ttl": "24h",
            "max_ttl": "72h",
        },
    )
    write_json(
        client,
        "pki_int/roles/fedshield-client",
        {
            "allowed_domains": [
                "client-alpha",
                "client-beta",
                "client-gamma",
                "client",
                "localhost",
            ],
            "allow_subdomains": True,
            "allow_bare_domains": True,
            "allow_localhost": True,
            "client_flag": True,
            "server_flag": False,
            "ttl": "24h",
            "max_ttl": "72h",
        },
    )


def issue_certificate(
    client: hvac.Client,
    role: str,
    common_name: str,
    alt_names: list[str] | None = None,
) -> dict:
    payload: dict = {"common_name": common_name, "ttl": "24h"}
    if alt_names:
        payload["alt_names"] = ",".join(alt_names)
    return write_json(client, f"pki_int/issue/{role}", payload)


def main() -> int:
    addr = os.getenv("VAULT_ADDR", "http://vault:8200")
    token = os.getenv("VAULT_TOKEN", "fedshield-dev-root-token")
    cert_dir = Path(os.getenv("CERT_DIR", "/certs"))
    policy_path = Path(os.getenv("VAULT_POLICY_PATH", "/config/vault_policy.hcl"))

    client = hvac.Client(url=addr, token=token)
    print(f"[vault-init] Waiting for Vault at {addr}", flush=True)
    wait_for_vault(client)

    print("[vault-init] Enabling PKI engines", flush=True)
    ensure_mount(client, "pki", "87600h")
    ensure_mount(client, "pki_int", "43800h")

    setup_root_ca(client, addr)
    setup_intermediate(client, addr)
    create_roles(client)

    if policy_path.exists():
        try:
            client.sys.create_or_update_policy(
                name="fedshield-pki",
                policy=policy_path.read_text(encoding="utf-8"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[vault-init] Policy note: {exc}", flush=True)

    cert_dir.mkdir(parents=True, exist_ok=True)

    print("[vault-init] Issuing server certificate", flush=True)
    # Clients connect to hostname "server" (compose service name) — must be on cert
    server_issue = issue_certificate(
        client,
        "fedshield-server",
        common_name="server",
        alt_names=["fedshield-server", "localhost"],
    )
    write_cert_bundle(cert_dir / "server", server_issue, role="server")

    for cn in ("client-alpha", "client-beta", "client-gamma"):
        print(f"[vault-init] Issuing certificate for {cn}", flush=True)
        issue = issue_certificate(client, "fedshield-client", cn)
        write_cert_bundle(cert_dir / cn, issue, role="client")

    ready = cert_dir / "server" / ".vault_ready"
    ready.write_text("ok", encoding="utf-8")
    for folder in ("server", "client-alpha", "client-beta", "client-gamma"):
        _chown_tree(cert_dir / folder, uid=10001, gid=10001)
    print("[vault-init] PKI bootstrap complete.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[vault-init] FAILED: {exc}", file=sys.stderr, flush=True)
        raise
