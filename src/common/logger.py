"""Structured JSON logging for FedShield services."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line for aggregation-friendly logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": os.getenv("FEDSHIELD_SERVICE", "fedshield"),
            "client_id": os.getenv("CLIENT_ID"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "taskName",
            } and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def _json_logs_enabled(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    return os.getenv("FEDSHIELD_JSON_LOGS", "true").lower() in {"1", "true", "yes"}


def setup_logger(name: str, level: str | None = None, json_logs: bool | None = None) -> logging.Logger:
    """Configure and return a named logger (idempotent handlers, live level)."""
    logger = logging.getLogger(name)
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logger.setLevel(resolved)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    if _json_logs_enabled(json_logs):
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
