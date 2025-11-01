"""Minimal live trading GUI utilities used by the test-suite."""

from __future__ import annotations

import os
from pathlib import Path


class AlpacaCredentialsError(RuntimeError):
    """Raised when the required Alpaca API credentials are missing."""


def _load_env_file() -> None:
    """Load key-value pairs from a local ``.env`` file into ``os.environ``.

    The helper mirrors the behaviour used by the original GUI code where the
    file is only read once per process.  Subsequent calls become no-ops unless
    the cached ``_loaded`` flag is manually cleared (which the tests do).
    """

    if getattr(_load_env_file, "_loaded", False):
        return

    env_path = Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip()

    _load_env_file._loaded = True  # type: ignore[attr-defined]


def _read_env(key: str) -> str:
    """Return the value of ``key`` from the environment or raise an error."""

    value = os.getenv(key)
    if value not in (None, ""):
        return value

    example = "ALPACA_API_KEY=your_key_here\nALPACA_API_SECRET=your_secret_here"
    raise AlpacaCredentialsError(
        f"Missing required environment variable: {key}\n"
        "Create a .env file with entries such as:\n"
        f"{example}"
    )


__all__ = ["AlpacaCredentialsError", "_load_env_file", "_read_env"]
