# HashiCorp Vault policy for FedShield PKI certificate issuance.
# Applied at bootstrap. Runtime server/clients do not receive a Vault token;
# vault-init writes per-identity cert volumes and exits.

path "pki_int/issue/fedshield-server" {
  capabilities = ["create", "update"]
}

path "pki_int/issue/fedshield-client" {
  capabilities = ["create", "update"]
}

path "pki_int/ca/pem" {
  capabilities = ["read"]
}

path "pki_int/cert/ca_chain" {
  capabilities = ["read"]
}

# Deny raw secret listing / arbitrary KV access
path "secret/*" {
  capabilities = ["deny"]
}
