# FedShield MLOps

Production-shaped **federated learning** pipeline: Flower + Opacus (DP-SGD) + HashiCorp Vault PKI + pairwise SecAgg + MLflow.

Raw samples never leave client containers. The network carries only **masked** model parameters (and scalar privacy metrics) over **mutual TLS**. When every selected client returns, pairwise masks cancel and the server obtains the weighted average without seeing an individual update.

## What this guarantees — and what it does not

| Guarantee | Mechanism |
|-----------|-----------|
| No raw data egress | Clients return NumPy weights + ε scalars only |
| Transport authentication | Flower gRPC mTLS (`require_client_auth=True`); plaintext raises |
| Individual updates hidden from the server | Pairwise ECDH masks; dropout **aborts** the round |
| (ε, δ)-DP on local training | Session-scoped RDP accountant; noise calibrated to `target_epsilon` |
| Hard privacy stop | Next DP step is refused once spent ε ≥ `epsilon_max` |
| Poisoning magnitude bound | Client L2 update clip before masking |
| Identity isolation | Per-client cert volumes; no Vault token on FL nodes |

This is **not** a TEE / confidential-computing enclave, and it is **not** malicious-client consensus. Byzantine-robust aggregators (`trimmed_mean`, `krum`) are available but **incompatible with SecAgg** because they need plaintext individual updates — pick one threat model in config.

Opacus `secure_mode` is off: `torchcsprng` is unmaintained. Noise generators are re-seeded from `os.urandom` each round.

## Architecture

```
Vault (PKI, control-plane only)
        │  vault-init writes per-identity volumes, then exits
        ▼
Flower Server :8080  ──mTLS──  Client-Alpha / Beta / Gamma
        │                         (local Non-IID data + Opacus)
        ▼
     MLflow :5000  (basic auth, control-plane)
```

Clients attach **only** to `fl-fabric`. They cannot reach Vault.

## Quick start

### Windows (PowerShell)

```powershell
Copy-Item .env.example .env   # set MLFLOW_TRACKING_PASSWORD
.\dev.ps1 install
.\dev.ps1 data
.\dev.ps1 test
.\dev.ps1 up
```

### Linux / macOS (Make)

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
make install
make data
make test
make up
```

- MLflow UI: http://localhost:5000 (basic auth from `.env`)
- Vault and the Flower gRPC port are **not** published to the host

```bash
make logs    # follow containers
make down    # tear down + volumes
```

### Local tests without Vault

```bash
set FEDSHIELD_DEV_CERTS=true   # PowerShell: $env:FEDSHIELD_DEV_CERTS="true"
pytest tests -q
```

Dev certificates are for unit/integration tests only — never for production.

## Project layout

```
config/           server, privacy, Vault policy
src/common/       crypto, mTLS, SecAgg, logging, utils
src/models/       LayerNorm MLP (BatchNorm forbidden for DP-SGD)
src/client/       NumPyClient, session DP trainer, Non-IID loader
src/server/       Flower server, SecAgg/robust strategy, MLflow
docker/           server / client / vault-init / mlflow images
tests/            unit + integration (including a live mTLS round)
scripts/          Non-IID data generator, Vault PKI bootstrap
```

## Configuration

- `config/server_config.yaml` — rounds, `round_timeout`, min clients, aggregation mode, gRPC size, MLflow
- `config/privacy_config.yaml` — `target_epsilon` / `epsilon_max` / δ, session `num_rounds`, update clip

`target_epsilon` sizes Gaussian noise for the **whole federated session** (`local_epochs × batches × num_rounds`). `epsilon_max` is a hard cap above that. Spent ε is tracked on one RDP accountant for the client process lifetime.

Aggregation modes:

- `secagg` (default) — pairwise masks; abort on dropout
- `trimmed_mean` / `krum` / `fedavg` — unmasked; use when you need robust aggregation more than hiding individual updates

## CI

GitHub Actions (`.github/workflows/ci-cd.yml`): ruff → unit/integration tests → Docker image builds → `docker compose config`.

## License

Proprietary / internal — adjust as needed for your organization.
