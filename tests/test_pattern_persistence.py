import csv
import importlib
import json
import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_pattern_scanner(monkeypatch):
    modules_to_clear = [
        "pattern_scanner",
        "cup_handle_scanner",
        "double_bottom_scanner",
        "swing_trading_screener",
    ]
    for name in modules_to_clear:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv("AUTO_TRADE_DATA_DIR", raising=False)
    from tests import test_pattern_scanner as _scanner_tests

    _scanner_tests._install_stub_modules()
    monkeypatch.delitem(sys.modules, "pattern_scanner", raising=False)
    yield
    for name in modules_to_clear:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_log_watchlist_persists_and_loads_pattern(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_TRADE_DATA_DIR", str(tmp_path))

    pattern_scanner = importlib.import_module("pattern_scanner")
    pattern_scanner = importlib.reload(pattern_scanner)

    class DummySeries:
        def __init__(self, values):
            self._values = list(values)

        def tail(self, length):
            length = max(int(length), 0)
            return DummySeries(self._values[-length:])

        def sum(self):
            return sum(self._values)

        @property
        def iloc(self):
            return self

        def __getitem__(self, index):
            return self._values[index]

    class DummyFrame:
        def __init__(self, data):
            self._data = {key: list(values) for key, values in data.items()}
            self.empty = False

        def __getitem__(self, key):
            return DummySeries(self._data[key])

    df = DummyFrame(
        {
            "open": [10 + idx * 0.05 for idx in range(120)],
            "high": [10.5 + idx * 0.05 for idx in range(120)],
            "low": [9.5 + idx * 0.05 for idx in range(120)],
            "close": [10 + idx * 0.05 for idx in range(120)],
            "volume": [1_000_000] * 120,
        }
    )

    pattern = pattern_scanner.InverseHeadShouldersPattern(
        left_idx=90,
        head_idx=95,
        right_idx=100,
        left_low=9.8,
        head_low=9.0,
        right_low=9.9,
        neckline_left_idx=91,
        neckline_right_idx=101,
        neckline_left=11.0,
        neckline_right=11.2,
        breakout=True,
    )

    rr_levels = pattern_scanner.RiskRewardLevels(
        breakout=12.5,
        stop=9.0,
        target=15.0,
        rr_ratio=0.5,
    )

    pattern_scanner.log_watchlist(
        "TEST",
        "Inverse Head and Shoulders",
        entry=11.5,
        rr_levels=rr_levels,
        df=df,
        pattern_details=pattern,
    )

    details_path = pattern_scanner.PATTERN_DETAILS_DIR / "TEST.json"
    assert details_path.exists()
    persisted = json.loads(details_path.read_text())
    key = "inverse head and shoulders"
    assert key in {k.lower(): v for k, v in persisted.items()}

    loaded = pattern_scanner.load_pattern_dataclass(
        "TEST",
        "Inverse Head and Shoulders",
        pattern_scanner.InverseHeadShouldersPattern,
    )

    assert isinstance(loaded, pattern_scanner.InverseHeadShouldersPattern)
    assert loaded.left_idx == pattern.left_idx
    assert abs(loaded.neckline_right - pattern.neckline_right) < 1e-9


def test_log_watchlist_skips_corrupt_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_TRADE_DATA_DIR", str(tmp_path))

    pattern_scanner = importlib.import_module("pattern_scanner")
    pattern_scanner = importlib.reload(pattern_scanner)

    corrupt_rows = [
        ["", "", ""],  # empty symbol
        ["ONLY_SYMBOL"],  # truncated row
    ]

    watchlist_path = pattern_scanner.WATCHLIST_PATH
    watchlist_path.parent.mkdir(parents=True, exist_ok=True)
    with watchlist_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(pattern_scanner.WATCHLIST_HEADER)
        writer.writerows(corrupt_rows)

    class DummySeries:
        def __init__(self, values):
            self._values = list(values)

        def tail(self, length):
            length = max(int(length), 0)
            return DummySeries(self._values[-length:])

        def sum(self):
            return sum(self._values)

        @property
        def iloc(self):
            return self

        def __getitem__(self, index):
            return self._values[index]

    class DummyFrame:
        def __init__(self, data):
            self._data = {key: list(values) for key, values in data.items()}
            self.empty = False

        def __getitem__(self, key):
            return DummySeries(self._data[key])

    df = DummyFrame(
        {
            "open": [10.0],
            "high": [10.5],
            "low": [9.5],
            "close": [10.0],
            "volume": [1_000_000],
        }
    )

    rr_levels = pattern_scanner.RiskRewardLevels(
        breakout=11.0,
        stop=9.5,
        target=14.0,
        rr_ratio=0.5,
    )

    pattern_scanner.log_watchlist(
        "CLEAN",
        "Inverse Head and Shoulders",
        entry=10.0,
        rr_levels=rr_levels,
        df=df,
        pattern_details=None,
    )

    with watchlist_path.open("r", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == pattern_scanner.WATCHLIST_HEADER
    assert [row[0] for row in rows[1:]] == ["CLEAN"]


def test_initialize_watchlist_preserves_existing_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_TRADE_DATA_DIR", str(tmp_path))

    pattern_scanner = importlib.import_module("pattern_scanner")
    pattern_scanner = importlib.reload(pattern_scanner)

    watchlist_path = pattern_scanner.WATCHLIST_PATH
    watchlist_path.parent.mkdir(parents=True, exist_ok=True)
    with watchlist_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(pattern_scanner.WATCHLIST_HEADER)
        writer.writerow(["KEEP", "", "", "", "", "", "", "", ""])

    pattern_scanner.initialize_watchlist()

    with watchlist_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    assert rows[0] == list(pattern_scanner.WATCHLIST_HEADER)
    assert any(row and row[0] == "KEEP" for row in rows[1:])
