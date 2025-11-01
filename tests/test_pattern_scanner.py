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

        class _Array(list):
            @property
            def size(self):
                return len(self)

            def max(self):
                return max(self) if self else 0

            def min(self):
                return min(self) if self else 0

            def mean(self):
                return sum(self) / len(self) if self else 0

            def reshape(self, _rows, cols):
                if cols != 1:
                    raise NotImplementedError("Stub reshape only supports single column")
                return [[value] for value in self]

            def __getitem__(self, item):
                if isinstance(item, slice):
                    return _Array(list(super().__getitem__(item)))
                if isinstance(item, list):
                    return _Array([self[i] for i in item])
                return super().__getitem__(item)

        def _array(values, dtype=None):
            return _Array(values)

        def _arange(start, stop=None, step=1, **kwargs):
            if stop is None:
                stop = start
                start = 0
            result = _Array([])
            current = start
            while current < stop:
                result.append(current)
                current += step
            return result

        def _linspace(start, stop, num, **kwargs):
            if num <= 1:
                return _Array([start])
            step = (stop - start) / (num - 1)
            return _Array([start + step * i for i in range(num)])

        def _diff(values):
            return _Array([values[i + 1] - values[i] for i in range(len(values) - 1)])

        def _argmax(values):
            if not values:
                return 0
            max_idx = 0
            max_val = values[0]
            for idx, val in enumerate(values):
                if val > max_val:
                    max_idx = idx
                    max_val = val
            return max_idx

        numpy_stub.array = _array
        numpy_stub.asarray = _array
        numpy_stub.arange = _arange
        numpy_stub.linspace = _linspace
        numpy_stub.diff = _diff
        numpy_stub.argmax = _argmax
        numpy_stub.any = lambda iterable: any(iterable)
        numpy_stub.abs = abs
        numpy_stub.sum = sum
        numpy_stub.isnan = lambda *args, **kwargs: False
        numpy_stub.ndarray = _Array
        numpy_stub.less = lambda a, b: a < b
        numpy_stub.greater = lambda a, b: a > b
        sys.modules["numpy"] = numpy_stub

    numpy_mod = sys.modules.get("numpy")
    if numpy_mod is not None:
        if not hasattr(numpy_mod, "less"):
            numpy_mod.less = lambda a, b: a < b
        if not hasattr(numpy_mod, "greater"):
            numpy_mod.greater = lambda a, b: a > b

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
                xs = [row[0] for row in X]
                ys = list(y)
                if not xs:
                    self.coef_ = [0.0]
                    self.intercept_ = 0.0
                    return self
                n = len(xs)
                mean_x = sum(xs) / n
                mean_y = sum(ys) / n
                denom = sum((x - mean_x) ** 2 for x in xs)
                if denom == 0:
                    slope = 0.0
                else:
                    slope = sum((x - mean_x) * (y_val - mean_y) for x, y_val in zip(xs, ys)) / denom
                self.coef_ = [slope]
                self.intercept_ = mean_y - slope * mean_x
                return self

            def predict(self, X):
                slope = self.coef_[0]
                return [self.intercept_ + slope * row[0] for row in X]

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

import pattern_scanner
from pattern_scanner import (
    RiskRewardLevels,
    calculate_rr_price_action,
    detect_double_bottom,
    detect_ascending_triangle,
)


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


class _Series:
    def __init__(self, values):
        numpy_mod = sys.modules.get("numpy")
        if numpy_mod and hasattr(numpy_mod, "array"):
            self.values = numpy_mod.array(values)
        else:
            self.values = list(values)


class _Frame:
    def __init__(self, data):
        self._data = {key: list(value) for key, value in data.items()}
        self._length = len(next(iter(self._data.values()))) if self._data else 0

    def __len__(self):
        return self._length

    def __getitem__(self, item):
        return _Series(self._data[item])

    def tail(self, window):
        window = min(window, self._length)
        sliced = {key: value[-window:] for key, value in self._data.items()}
        return _Frame(sliced)


def _build_price_frame(highs, lows, closes):
    return _Frame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def _extend_price_frame(highs, lows, closes, extra_highs, extra_lows, extra_closes):
    return _build_price_frame(highs + extra_highs, lows + extra_lows, closes + extra_closes)


def test_detect_ascending_triangle_prefers_flat_resistance():
    highs = [10.0, 10.02, 10.01, 9.99, 10.03, 10.02, 10.04, 10.05, 10.6]
    lows = [9.5, 9.55, 9.6, 9.65, 9.7, 9.8, 9.9, 10.0, 10.45]
    closes = [9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.35, 10.4, 10.55]
    df = _build_price_frame(highs, lows, closes)

    pattern = detect_ascending_triangle(df, window=9, tolerance=0.02, min_touches=2)

    assert pattern is not None
    assert pattern.breakout is True


def test_detect_ascending_triangle_rejects_wide_plateau():
    highs = [10.0, 10.1, 10.25, 10.4, 10.2, 10.5, 10.7, 10.9, 11.0]
    lows = [9.5, 9.55, 9.6, 9.65, 9.7, 9.8, 9.9, 10.0, 10.1]
    closes = [9.8, 9.85, 9.95, 10.05, 10.15, 10.25, 10.35, 10.45, 10.55]
    df = _build_price_frame(highs, lows, closes)

    pattern = detect_ascending_triangle(df, window=9, tolerance=0.02, min_touches=2)

    assert pattern is None


def test_detect_bullish_rectangle_identifies_consolidation():
    flagpole_highs = [50 + 1.5 * i for i in range(15)]
    flagpole_lows = [value - 0.9 for value in flagpole_highs]
    flagpole_closes = [low + 0.8 for low in flagpole_lows]

    rectangle_highs = [
        72.05,
        71.6,
        71.7,
        71.65,
        72.08,
        71.58,
        71.7,
        71.6,
        72.1,
        71.62,
        71.68,
        71.6,
        72.06,
        71.6,
        71.7,
        71.58,
        72.09,
        71.6,
        71.7,
        71.62,
    ]
    rectangle_lows = [
        68.3,
        68.0,
        68.15,
        68.05,
        68.25,
        68.02,
        68.12,
        68.03,
        68.24,
        68.01,
        68.1,
        68.02,
        68.23,
        68.0,
        68.11,
        68.03,
        68.22,
        68.01,
        68.12,
        68.05,
    ]
    rectangle_closes = [
        70.1,
        69.8,
        69.95,
        69.9,
        70.05,
        69.82,
        69.94,
        69.88,
        70.04,
        69.82,
        69.92,
        69.86,
        70.03,
        69.82,
        69.93,
        69.85,
        70.05,
        69.82,
        69.94,
        69.88,
    ]

    df = _extend_price_frame(
        flagpole_highs,
        flagpole_lows,
        flagpole_closes,
        rectangle_highs,
        rectangle_lows,
        rectangle_closes,
    )

    window = len(rectangle_highs)
    pattern = pattern_scanner.detect_bullish_rectangle(df, window=window, tolerance=0.02, min_touches=4)

    assert pattern is not None
    assert pattern.high == rectangle_highs[8]
    assert pattern.low == rectangle_lows[1]
    assert len(pattern.high_touch_indices) >= 4
    assert len(pattern.low_touch_indices) >= 4


def test_detect_bullish_rectangle_rejects_trending_channel():
    flagpole_highs = [50 + 1.5 * i for i in range(15)]
    flagpole_lows = [value - 0.9 for value in flagpole_highs]
    flagpole_closes = [low + 0.8 for low in flagpole_lows]

    channel_highs = [72 + i * 0.4 for i in range(20)]
    channel_lows = [high - 1.2 for high in channel_highs]
    channel_closes = [low + 0.9 for low in channel_lows]

    df = _extend_price_frame(
        flagpole_highs,
        flagpole_lows,
        flagpole_closes,
        channel_highs,
        channel_lows,
        channel_closes,
    )

    window = len(channel_highs)
    pattern = pattern_scanner.detect_bullish_rectangle(df, window=window, tolerance=0.02, min_touches=4)

    assert pattern is None


def test_calculate_rr_price_action_uses_recent_swing_high(monkeypatch):
    highs = [12.0, 13.5, 14.2, 15.0, 14.8, 15.5, 15.2, 16.0, 17.5, 18.0]
    lows = [10.5, 11.0, 10.8, 11.5, 11.2, 12.0, 11.6, 11.0, 12.8, 13.5]
    df = _build_price_frame(highs, lows, lows)

    def _matches(expected, actual):
        if len(expected) != len(actual):
            return False
        return all(abs(e - a) < 1e-9 for e, a in zip(expected, actual))

    def fake_argrelextrema(values, _comparator, order=5):
        seq = list(values)
        if _matches(lows, seq):
            return (pattern_scanner.np.array([2, 7]),)
        if _matches(highs, seq):
            return (pattern_scanner.np.array([1, 5, 9]),)
        return (pattern_scanner.np.array([]),)

    monkeypatch.setattr(sys.modules["scipy.signal"], "argrelextrema", fake_argrelextrema)

    entry_price = 16.5
    rr_levels = calculate_rr_price_action(df, entry_price)

    assert isinstance(rr_levels, RiskRewardLevels)
    assert rr_levels.breakout == highs[9]
    assert rr_levels.stop == lows[7]
    expected_target = highs[9] + (highs[9] - lows[7])
    assert rr_levels.target == expected_target
    expected_rr = round((entry_price - lows[7]) / (expected_target - entry_price), 2)
    assert rr_levels.rr_ratio == expected_rr
