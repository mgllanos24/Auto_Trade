from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
import types


def _install_stub_modules() -> None:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

    if "alpaca_trade_api" not in sys.modules:
        alpaca_stub = types.ModuleType("alpaca_trade_api")
        alpaca_stub.REST = lambda *args, **kwargs: None
        sys.modules["alpaca_trade_api"] = alpaca_stub

    if "pandas" not in sys.modules:
        pandas_stub = types.ModuleType("pandas")
        pandas_stub.DataFrame = object
        pandas_stub.Series = object
        pandas_stub.date_range = lambda *args, **kwargs: []
        sys.modules["pandas"] = pandas_stub

    if "yfinance" not in sys.modules:
        yfinance_stub = types.ModuleType("yfinance")
        yfinance_stub.download = lambda *args, **kwargs: None
        sys.modules["yfinance"] = yfinance_stub

    if "numpy" not in sys.modules:
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.ndarray = object

        def _arange(start, stop=None, step=1, **kwargs):
            if stop is None:
                stop = start
                start = 0
            values = []
            current = start
            while current < stop:
                values.append(current)
                current += step
            return values

        def _linspace(start, stop, num, **kwargs):
            if num <= 1:
                return [start]
            step = (stop - start) / (num - 1)
            return [start + step * i for i in range(num)]

        numpy_stub.arange = _arange
        numpy_stub.linspace = _linspace
        numpy_stub.less = lambda a, b: a < b
        numpy_stub.greater = lambda a, b: a > b
        numpy_stub.abs = abs
        numpy_stub.sum = sum
        numpy_stub.isnan = lambda *args, **kwargs: False
        sys.modules["numpy"] = numpy_stub

    if "scipy" not in sys.modules:
        sys.modules["scipy"] = types.ModuleType("scipy")

    if "scipy.signal" not in sys.modules:
        signal_stub = types.ModuleType("scipy.signal")
        signal_stub.find_peaks = lambda *args, **kwargs: ([], {})
        signal_stub.argrelextrema = lambda *args, **kwargs: []
        sys.modules["scipy.signal"] = signal_stub

    if "sklearn" not in sys.modules:
        sys.modules["sklearn"] = types.ModuleType("sklearn")

    if "sklearn.linear_model" not in sys.modules:
        class DummyLinearRegression:
            def fit(self, X, y):
                self.coef_ = [0]
                return self

        linear_model_stub = types.ModuleType("sklearn.linear_model")
        linear_model_stub.LinearRegression = DummyLinearRegression
        sys.modules["sklearn.linear_model"] = linear_model_stub

    if "matplotlib" not in sys.modules:
        sys.modules["matplotlib"] = types.ModuleType("matplotlib")

    if "matplotlib.pyplot" not in sys.modules:
        pyplot_stub = types.ModuleType("matplotlib.pyplot")
        pyplot_stub.plot = lambda *args, **kwargs: None
        sys.modules["matplotlib.pyplot"] = pyplot_stub


_install_stub_modules()

from pattern_scanner import detect_double_bottom


class DummyDataFrame:
    def __init__(self, rows: int):
        self._rows = rows

    def __len__(self) -> int:
        return self._rows


def _make_dummy(rows: int = 100):
    df = DummyDataFrame(rows)
    index = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(rows)]
    return df, index


def test_detect_double_bottom_prefers_recent_hit(monkeypatch):
    df, index = _make_dummy()
    total_rows = len(df)

    old_hit = types.SimpleNamespace(
        start_idx=0,
        end_idx=59,
        left_idx=10,
        right_idx=20,
        left_timestamp=index[10],
        right_timestamp=index[20],
        support=49.5,
        neckline=55.0,
        left_low=49.0,
        right_low=49.5,
        bounce_pct=0.08,
        touch_count=2,
        contraction_pct=None,
        volume_contracted=None,
        breakout=True,
        breakout_idx=30,
        breakout_timestamp=index[30],
        breakout_price=56.0,
    )

    recent_hit = types.SimpleNamespace(
        start_idx=total_rows - 60,
        end_idx=total_rows - 1,
        left_idx=total_rows - 15,
        right_idx=total_rows - 8,
        left_timestamp=index[total_rows - 15],
        right_timestamp=index[total_rows - 8],
        support=58.5,
        neckline=60.0,
        left_low=58.0,
        right_low=58.2,
        bounce_pct=0.06,
        touch_count=2,
        contraction_pct=None,
        volume_contracted=None,
        breakout=True,
        breakout_idx=total_rows - 2,
        breakout_timestamp=index[total_rows - 2],
        breakout_price=60.5,
    )

    def fake_scan_double_bottoms(*args, **kwargs):
        return [old_hit, recent_hit]

    monkeypatch.setattr("pattern_scanner.scan_double_bottoms", fake_scan_double_bottoms)

    result = detect_double_bottom(df, window=60)

    assert result is recent_hit
