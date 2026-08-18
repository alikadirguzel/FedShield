#!/bin/sh
# Deprecated: Compose runs scripts/bootstrap_vault.py via docker/Dockerfile.vault.
# Kept so older runbooks fail closed instead of issuing certs with a second, unmaintained path.
echo "Use scripts/bootstrap_vault.py (vault-init container). This shell bootstrap is retired." >&2
exit 1
