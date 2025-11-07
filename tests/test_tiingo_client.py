from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import pattern_scanner

# Ensure the Tiingo helpers operate on the real pandas module.  The
# ``tests.test_pattern_scanner`` module replaces ``pattern_scanner.pd`` with a
# lightweight stub which is perfect for its isolated tests but prevents us from
# building DataFrames here.
pattern_scanner.pd = pd


class _DummyResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise pattern_scanner.requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


class _DummySession:
    def __init__(self, payload, expected_symbol: str):
        self._payload = payload
        self._expected_symbol = expected_symbol
        self.last_headers = None
        self.last_params = None
        self.last_url = None

    def get(self, url, *, params=None, headers=None, timeout=None):
        self.last_url = url
        self.last_params = params
        self.last_headers = headers
        assert self._expected_symbol.lower() in url
        assert timeout == 30
        return _DummyResponse(self._payload)


def test_get_tiingo_data_returns_dataframe():
    payload = [
        {
            "date": "2024-01-02T00:00:00.000Z",
            "open": 10.0,
            "high": 11.0,
            "low": 9.5,
            "close": 10.5,
            "adjClose": 10.4,
            "volume": 1200,
        },
        {
            "date": "2024-01-03T00:00:00.000Z",
            "open": 10.5,
            "high": 11.5,
            "low": 10.0,
            "close": 11.0,
            "adjClose": 10.9,
            "volume": 1400,
        },
    ]

    session = _DummySession(payload, expected_symbol="SPY")
    df = pattern_scanner.get_tiingo_data(
        "SPY",
        token="token-123",
        start_date=date(2024, 1, 1),
        end_date="2024-01-05",
        session=session,
    )

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "adj_close"]
    assert len(df) == 2
    assert float(df["close"].iloc[-1]) == 11.0
    assert session.last_headers["Authorization"] == "Token token-123"
    assert session.last_params["startDate"] == "2024-01-01"
    assert session.last_params["endDate"] == "2024-01-05"


def test_get_tiingo_data_requires_token():
    with pytest.raises(ValueError):
        pattern_scanner.get_tiingo_data("SPY", token="")

