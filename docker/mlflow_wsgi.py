"""Gunicorn / Flask entry for MLflow with HTTP basic authentication."""

from __future__ import annotations

import os

from flask import Response, request
from mlflow.server import ARTIFACT_ROOT_ENV_VAR, BACKEND_STORE_URI_ENV_VAR, SERVE_ARTIFACTS_ENV_VAR
from mlflow.server import app as mlflow_app
from mlflow.server.handlers import initialize_backend_stores

os.environ.setdefault(BACKEND_STORE_URI_ENV_VAR, "sqlite:////mlflow/mlflow.db")
os.environ.setdefault(ARTIFACT_ROOT_ENV_VAR, "/mlflow/artifacts")
os.environ.setdefault(SERVE_ARTIFACTS_ENV_VAR, "true")

initialize_backend_stores(
    backend_store_uri=os.environ[BACKEND_STORE_URI_ENV_VAR],
    default_artifact_root=os.environ[ARTIFACT_ROOT_ENV_VAR],
)

_USER = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")


@mlflow_app.before_request
def require_basic_auth():
    if not _USER or not _PASSWORD:
        return Response("MLflow credentials are not configured", status=503)
    auth = request.authorization
    if auth is None or auth.username != _USER or auth.password != _PASSWORD:
        return Response(
            "Authentication required",
            status=401,
            headers={"WWW-Authenticate": 'Basic realm="FedShield-MLflow"'},
        )
    return None


app = mlflow_app
