#!/bin/sh
# Bootstrap HashiCorp Vault PKI for FedShield mTLS certificates.
# Runs inside the vault-init container after Vault is unsealed/ready.
set -eu

VAULT_ADDR="${VAULT_ADDR:-http://vault:8200}"
VAULT_TOKEN="${VAULT_TOKEN:-fedshield-dev-root-token}"
CERT_DIR="${CERT_DIR:-/certs}"
export VAULT_ADDR VAULT_TOKEN

echo "[vault-init] Waiting for Vault at ${VAULT_ADDR}..."
i=0
until vault status >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -gt 60 ]; then
    echo "[vault-init] Vault did not become ready in time" >&2
    exit 1
  fi
  sleep 2
done

echo "[vault-init] Enabling PKI secrets engines..."
vault secrets enable -path=pki pki 2>/dev/null || true
vault secrets tune -max-lease-ttl=87600h pki 2>/dev/null || true

vault secrets enable -path=pki_int pki 2>/dev/null || true
vault secrets tune -max-lease-ttl=43800h pki_int 2>/dev/null || true

echo "[vault-init] Generating root CA..."
if ! vault read pki/cert/ca >/dev/null 2>&1; then
  vault write -field=certificate pki/root/generate/internal \
    common_name="FedShield Root CA" \
    ttl=87600h > /tmp/ca.crt
fi

vault write pki/config/urls \
  issuing_certificates="${VAULT_ADDR}/v1/pki/ca" \
  crl_distribution_points="${VAULT_ADDR}/v1/pki/crl" >/dev/null

echo "[vault-init] Generating intermediate CA..."
vault write -format=json pki_int/intermediate/generate/internal \
  common_name="FedShield Intermediate CA" \
  | awk -F'"' '/csr/{print $4}' | sed 's/\\n/\n/g' > /tmp/pki_int.csr || true

# Prefer JSON field extraction via vault when available
CSR=$(vault write -format=json pki_int/intermediate/generate/internal \
  common_name="FedShield Intermediate CA" 2>/dev/null | sed -n 's/.*"csr":"\([^"]*\)".*/\1/p' | sed 's/\\n/\n/g' || true)

if [ -z "${CSR}" ]; then
  # Intermediate may already exist — continue to roles
  echo "[vault-init] Intermediate CSR skipped (may already exist)"
else
  echo "$CSR" > /tmp/pki_int.csr
  vault write -format=json pki/root/sign-intermediate csr=@/tmp/pki_int.csr \
    format=pem_bundle ttl=43800h \
    | sed -n 's/.*"certificate":"\([^"]*\)".*/\1/p' | sed 's/\\n/\n/g' > /tmp/intermediate.cert.pem
  vault write pki_int/intermediate/set-signed certificate=@/tmp/intermediate.cert.pem >/dev/null
fi

vault write pki_int/config/urls \
  issuing_certificates="${VAULT_ADDR}/v1/pki_int/ca" \
  crl_distribution_points="${VAULT_ADDR}/v1/pki_int/crl" >/dev/null

echo "[vault-init] Creating PKI roles..."
vault write pki_int/roles/fedshield-server \
  allowed_domains="fedshield-server,server,localhost" \
  allow_subdomains=true \
  allow_bare_domains=true \
  allow_localhost=true \
  client_flag=true \
  server_flag=true \
  ttl=24h \
  max_ttl=72h >/dev/null

vault write pki_int/roles/fedshield-client \
  allowed_domains="client-alpha,client-beta,client-gamma,client,localhost" \
  allow_subdomains=true \
  allow_bare_domains=true \
  allow_localhost=true \
  client_flag=true \
  server_flag=false \
  ttl=24h \
  max_ttl=72h >/dev/null

if [ -f /config/vault_policy.hcl ]; then
  vault policy write fedshield-pki /config/vault_policy.hcl >/dev/null || true
fi

mkdir -p "${CERT_DIR}"

echo "[vault-init] Issuing server certificate..."
vault write -format=json pki_int/issue/fedshield-server \
  common_name="fedshield-server" ttl=24h > /tmp/server.json

# Minimal portable JSON field scrape
python3 - <<'PY' || true
import json, pathlib, re, sys
cert_dir = pathlib.Path("/certs")
cert_dir.mkdir(exist_ok=True)

def dump(path, payload, role):
    data = payload["data"]
    (cert_dir / "ca.crt").write_text(data.get("issuing_ca") or data["ca_chain"][0])
    (cert_dir / f"{role}.crt").write_text(data["certificate"])
    key_path = cert_dir / f"{role}.key"
    key_path.write_text(data["private_key"])
    key_path.chmod(0o600)

raw = pathlib.Path("/tmp/server.json").read_text()
dump(cert_dir, json.loads(raw), "server")
print("[vault-init] Wrote server certs")
PY

for CN in client-alpha client-beta client-gamma; do
  echo "[vault-init] Issuing certificate for ${CN}..."
  vault write -format=json pki_int/issue/fedshield-client \
    common_name="${CN}" ttl=24h > "/tmp/${CN}.json"
  mkdir -p "${CERT_DIR}/${CN}"
  python3 - <<PY
import json, pathlib
cert_dir = pathlib.Path("/certs/${CN}")
cert_dir.mkdir(parents=True, exist_ok=True)
root = pathlib.Path("/certs")
data = json.loads(pathlib.Path("/tmp/${CN}.json").read_text())["data"]
ca = data.get("issuing_ca") or data["ca_chain"][0]
(cert_dir / "ca.crt").write_text(ca)
(root / "ca.crt").write_text(ca)
(cert_dir / "client.crt").write_text(data["certificate"])
key = cert_dir / "client.key"
key.write_text(data["private_key"])
key.chmod(0o600)
print("[vault-init] Wrote ${CN} certs")
PY
done

echo "[vault-init] PKI bootstrap complete."
touch "${CERT_DIR}/.vault_ready"
