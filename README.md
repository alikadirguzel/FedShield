# FedShield MLOps

Production-shaped **federated learning** pipeline: Flower + Opacus (DP-SGD) + HashiCorp Vault PKI + pairwise SecAgg + MLflow.

Clone this repository on any machine. There are no personal tokens, home-directory paths, or machine-specific secrets in the tree. Local credentials live only in a gitignored `.env`.

Raw samples never leave client containers. The network carries only **masked** model parameters (and scalar privacy metrics) over **mutual TLS**. Flower client identities are the **verified certificate Common Name**.

## What this guarantees — and what it does not

| Guarantee | Mechanism |
|-----------|-----------|
| No raw data egress | Clients return NumPy weights + ε scalars only |
| Transport authentication | Flower gRPC mTLS (`require_client_auth=True`); plaintext raises |
| Individual updates hidden from the server | Pairwise ECDH masks; dropout **aborts** the round |
| (ε, δ)-DP on local training | Session-scoped RDP accountant; noise calibrated to `target_epsilon` |
| Hard privacy stop | Next DP step is refused once spent ε ≥ `epsilon_max` |
| Poisoning magnitude bound | Client L2 update clip before masking |
| Identity isolation | Per-client cert + data volumes; `cid` = cert CN; allowlisted ids |

This is **not** a TEE / confidential-computing enclave, and it is **not** malicious-client consensus. Byzantine-robust aggregators (`trimmed_mean`, `krum`) are available but **incompatible with SecAgg**.

Opacus `secure_mode` is off: `torchcsprng` is unmaintained. Noise is seeded from `os.urandom` each round.

## Architecture

```
Vault (file storage + init/unseal, pki-plane only)
        │  vault-init writes per-identity cert volumes, then exits
        ▼
Flower Server :8080  ──mTLS──  Client-Alpha / Beta / Gamma
        │                         (volume-private data + Opacus)
        ▼
     MLflow :5000  (basic auth, control-plane)
```

Clients attach **only** to `fl-fabric` (internal). They cannot reach Vault or MLflow. Each client mounts **only** its own data volume (`x.npy` / `y.npy`).

## Quick start

Requires Docker. Python 3.11 is optional (needed only for local pytest).

```bash
git clone https://github.com/<you>/FedShield.git
cd FedShield
python scripts/init_env.py          # writes gitignored .env with a random password
docker compose up -d --build
```

Windows (PowerShell): `python scripts/init_env.py` then `.\dev.ps1 up`.

- MLflow: http://127.0.0.1:5000 — username `fedshield`, password in `.env`
- Vault and Flower gRPC are **not** published to the host
- Data partitions are generated inside the `data-init` container (no host `data/` required)

```bash
docker compose logs -f
docker compose down -v              # stop + delete volumes
```

### Public / VPS bind

Default MLflow bind is loopback. To listen on all interfaces:

```bash
# in .env
MLFLOW_BIND=0.0.0.0
MLFLOW_PORT=5000
```

Then `docker compose up -d`. Put TLS (Caddy/nginx) in front of port 5000 before exposing it to the internet.

### Local tests without Vault

```bash
export FEDSHIELD_DEV_CERTS=true     # PowerShell: $env:FEDSHIELD_DEV_CERTS="true"
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
pytest tests -q
```

Dev certificates are for unit/integration tests only.

## Project layout

```
config/           server, privacy, Vault policy + server HCL
src/common/       crypto, mTLS, SecAgg, logging, utils
src/models/       LayerNorm MLP (BatchNorm forbidden for DP-SGD)
src/client/       NumPyClient, session DP trainer, Non-IID loader
src/server/       Flower server, SecAgg/robust strategy, MLflow
docker/           server / client / data-init / vault / mlflow images
tests/            unit + integration (including a live mTLS round)
scripts/          env bootstrap, Non-IID generator, Vault PKI bootstrap
```

## Configuration

- `config/server_config.yaml` — rounds, timeout, min clients, aggregation, gRPC, allowlist
- `config/privacy_config.yaml` — `target_epsilon` / `epsilon_max` / δ, session rounds, update clip

`target_epsilon` sizes Gaussian noise for the **whole federated session**. `epsilon_max` is a hard cap.

Aggregation modes: `secagg` (default), `trimmed_mean`, `krum`, `fedavg`.

## CI

GitHub Actions: ruff → tests → Docker image builds → `docker compose config`.

## License

MIT — see `LICENSE`.
