"""Shared hospital tenant ids (5-hospital demo)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _tenants_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "tenants.json"


def load_hospitals() -> list[dict[str, Any]]:
    path = _tenants_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    hospitals = list(payload.get("hospitals") or [])
    if len(hospitals) < 2:
        raise ValueError("config/tenants.json must list at least two hospitals")
    return hospitals


def hospital_ids() -> list[str]:
    return [str(item["id"]) for item in load_hospitals()]
