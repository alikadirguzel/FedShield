"""YAML loading and process-wide logging configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a mapping YAML file; empty/null documents become ``{}``."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def apply_logging_config(cfg: dict[str, Any]) -> None:
    """Apply ``logging.level`` / ``logging.json`` from a service YAML to the process."""
    logging_cfg = cfg.get("logging") or {}
    level = str(logging_cfg.get("level") or os.getenv("LOG_LEVEL", "INFO")).upper()
    json_logs = bool(logging_cfg.get("json", True))
    os.environ["LOG_LEVEL"] = level
    os.environ["FEDSHIELD_JSON_LOGS"] = "true" if json_logs else "false"
    manager = logging.Logger.manager.loggerDict
    for name in list(manager.keys()):
        if str(name).startswith("fedshield"):
            logging.getLogger(str(name)).setLevel(level)
