#!/usr/bin/env python3
"""Create a local .env from .env.example with a random MLflow password.

Never prints the password. Never overwrites an existing .env.
"""

from __future__ import annotations

import secrets
from pathlib import Path

PLACEHOLDER = "REPLACE_ME"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    example = root / ".env.example"
    target = root / ".env"
    if not example.exists():
        raise SystemExit("Missing .env.example")
    if target.exists():
        print(".env already exists — leaving it unchanged.")
        return 0
    text = example.read_text(encoding="utf-8")
    if PLACEHOLDER not in text:
        raise SystemExit(".env.example is missing the REPLACE_ME password placeholder")
    text = text.replace(PLACEHOLDER, secrets.token_urlsafe(24))
    target.write_text(text, encoding="utf-8")
    try:
        target.chmod(0o600)
    except OSError:
        pass
    print("Created .env with a generated MLFLOW_TRACKING_PASSWORD.")
    print("This file is gitignored — do not commit it.")
    print("For a public bind, set MLFLOW_BIND=0.0.0.0 in .env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
