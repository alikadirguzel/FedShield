# Security

- Never commit `.env`, certificates, Vault unseal keys, or datasets.
- `python scripts/init_env.py` writes a gitignored `.env` with a random MLflow password.
- Default MLflow bind is `127.0.0.1`. Set `MLFLOW_BIND=0.0.0.0` only on hosts you intend to expose.
- Flower gRPC and Vault ports are not published; they stay on internal Compose networks.
- Report vulnerabilities privately; do not open issues that include secrets or absolute machine paths.
