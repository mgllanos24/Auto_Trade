"""Minimal stub of the :mod:`requests` module for the test environment."""
from __future__ import annotations

import sys
import types


class RequestException(Exception):
    """Base exception matching the real requests API."""


class HTTPError(RequestException):
    """Specific error used by :mod:`alpaca_trade_api` when requests fail."""


class Response:
    def __init__(self, payload: dict | None = None, status_code: int = 200) -> None:
        self._payload = payload or {}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPError(f"HTTP error {self.status_code}")

    def json(self) -> dict:
        return self._payload


def get(*_args, **_kwargs) -> Response:  # pragma: no cover - trivial shim
    """Return an empty :class:`Response` instance."""

    return Response()


exceptions = types.ModuleType(f"{__name__}.exceptions")
exceptions.RequestException = RequestException
exceptions.HTTPError = HTTPError
sys.modules[exceptions.__name__] = exceptions


