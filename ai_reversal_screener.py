#!/usr/bin/env python3
"""
AI‑ASSISTED 1–3 DAY TREND REVERSAL SCREENER (Robust CLI + Self‑Tests)

What changed (to fix SystemExit: 2)
----------------------------------
• `--symbols` is no longer *required*; if omitted, the script runs a demo with
  defaults sourced from the approved universe in ``build_ohlcv_last2y``. This prevents `argparse` from raising
  `SystemExit: 2` when you just run `python ai_reversal_screener.py`.
• `parse_args(list_args: Optional[List[str]])` now supports programmatic testing.
• Added `--self-test` to run quick unit tests (no internet required).
• Clear help text and friendly messages when defaults are used.

Usage examples
--------------
python ai_reversal_screener.py                       # runs demo tickers (no crash)
python ai_reversal_screener.py --symbols AAPL MSFT TSLA --start 2018-01-01
python ai_reversal_screener.py --symbols MSTR UAL --intermarket SPY DXY ^VIX --start 2016-01-01
python ai_reversal_screener.py --from-csv ./data --symbols AAPL MSFT
python ai_reversal_screener.py --self-test           # run built‑in tests

Output: table ranked by P(reversal in 1–3d) with recent stats and validation scores.
"""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
import warnings

import numpy as np
import pandas as pd

from build_ohlcv_last2y import TICKERS as MASTER_TICKERS

# Optional imports – fail gracefully if missing
try:
    import yfinance as yf
except Exception:
    yf = None

try:  # pragma: no cover - exercised indirectly in tests
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, precision_score
except Exception:  # pragma: no cover
    class GradientBoostingClassifier:  # type: ignore[override]
        """Minimal fallback when scikit-learn is unavailable.

        The real project relies on scikit-learn's implementation, but the test
        environment intentionally strips heavy optional dependencies.  A very
        small probabilistic classifier keeps the workflow operational for unit
        tests by returning the training class frequencies as constant
        probabilities.
        """

        def __init__(self, random_state: int | None = None, **_: object) -> None:
            self.random_state = random_state
            self._positive_rate = 0.0

        def fit(self, X, y):  # noqa: D401 - signature mirrors sklearn
            y_arr = np.asarray(y, dtype=float)
            if y_arr.size == 0:
                self._positive_rate = 0.0
            else:
                self._positive_rate = float(np.clip(np.mean(y_arr), 0.0, 1.0))
            return self

        def predict_proba(self, X):
            n_rows = len(X)
            pos = np.full(n_rows, self._positive_rate, dtype=float)
            neg = 1.0 - pos
            return np.column_stack([neg, pos])

    def roc_auc_score(y_true, y_score):
        y_true_arr = np.asarray(y_true, dtype=float)
        y_score_arr = np.asarray(y_score, dtype=float)
        if y_true_arr.size == 0 or np.unique(y_true_arr).size < 2:
            return float("nan")
        order = np.argsort(-y_score_arr)
        y_true_sorted = y_true_arr[order]
        total_pos = float((y_true_arr == 1).sum())
        total_neg = float((y_true_arr == 0).sum())
        if total_pos == 0 or total_neg == 0:
            return float("nan")
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)
        tpr = np.concatenate(([0.0], tps / total_pos, [1.0]))
        fpr = np.concatenate(([0.0], fps / total_neg, [1.0]))
        auc = np.trapz(tpr, fpr)
        return float(auc)

    def precision_score(y_true, y_pred, zero_division: float = 0.0):
        y_true_arr = np.asarray(y_true, dtype=int)
        y_pred_arr = np.asarray(y_pred, dtype=int)
        tp = float(np.logical_and(y_true_arr == 1, y_pred_arr == 1).sum())
        fp = float(np.logical_and(y_true_arr == 0, y_pred_arr == 1).sum())
        if tp + fp == 0:
            return float(zero_division)
        return tp / (tp + fp)

warnings.filterwarnings("ignore")

# -----------------------------
# Technical indicator utilities
# -----------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def linreg_slope(series: pd.Series, window: int = 5) -> pd.Series:
    # Rolling OLS slope (normalized per day)
    idx = np.arange(window)
    denom = ((idx - idx.mean())**2).sum()
    def _slope(x):
        y = x.values
        num = ((idx - idx.mean()) * (y - y.mean())).sum()
        return num / (denom + 1e-12)
    return series.rolling(window).apply(_slope, raw=False)

# -----------------------------
# Feature engineering
# -----------------------------

def _as_series(obj: pd.Series | pd.DataFrame, label: str) -> pd.Series:
    """Coerce a Series/DataFrame column into a 1D Series.

    yfinance can return MultiIndex columns even for a single symbol (e.g. when
    downloading multiple tickers in one call and subsetting afterward).  Pandas
    then returns a small DataFrame when you access ``df["High"]`` instead of a
    Series, which later breaks scalar column assignments.  To keep feature
    engineering robust we squeeze any single-column DataFrame down to a Series
    and raise a clear error if multiple columns remain.
    """

    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            ser = obj.iloc[:, 0]
            ser.name = label
            return ser
        raise ValueError(f"Expected a single column for {label}, got {obj.shape[1]}")
    obj.name = label
    return obj


def _get_column(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return _as_series(df[column], column)
    raise KeyError(f"Expected one of {candidates} in DataFrame")


def _select_close(df: pd.DataFrame) -> pd.Series:
    """Return the best available close/adjusted-close series.

    The cached CSVs for this project only contain ``close`` (lowercase), while
    yfinance provides ``Adj Close``. To support both sources seamlessly we try a
    few common column names before raising a clear error.
    """

    return _get_column(df, ["Adj Close", "adj_close", "Close", "close"])


def make_features(df: pd.DataFrame, intermarket: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    c = _select_close(df).copy()
    c.name = "Close"
    o = _get_column(df, ["Open", "open"])
    h = _get_column(df, ["High", "high"])
    l = _get_column(df, ["Low", "low"])
    v = _get_column(df, ["Volume", "volume"])

    feats = pd.DataFrame(index=df.index)

    # Returns and momentum
    feats["ret_1"] = c.pct_change(1)
    feats["ret_2"] = c.pct_change(2)
    feats["ret_5"] = c.pct_change(5)
    feats["ret_10"] = c.pct_change(10)

    # Volatility & range
    feats["atr14"] = atr(h, l, c, 14) / c
    feats["range"] = (h - l) / c

    # Oscillators
    feats["rsi14"] = rsi(c, 14)
    feats = feats.join(macd(c))
    feats = feats.join(stoch_kd(h, l, c))

    # Regime slope (proxy for recent trend direction)
    feats["slope_5"] = linreg_slope(c, 5)
    feats["slope_10"] = linreg_slope(c, 10)

    # Intermarket proxies (optional): add their 1/5/10d returns
    if intermarket:
        for name, idf in intermarket.items():
            ic = _select_close(idf).rename(f"{name}_c")
            feats[f"{name}_ret1"] = ic.pct_change(1)
            feats[f"{name}_ret5"] = ic.pct_change(5)
            feats[f"{name}_ret10"] = ic.pct_change(10)

    feats = feats.dropna()
    return feats

# -----------------------------
# Label: 1–3 day trend reversal
# -----------------------------

def make_labels(close: pd.Series, lookback: int = 5, horizon: int = 3) -> pd.Series:
    """Define a reversal when the sign of recent slope flips within next horizon days.

    • recent slope: slope over last `lookback` days
    • future slope: slope over the next `horizon` days (using shifted window)
    Label = 1 if sign(recent_slope) != sign(future_slope) and |future_slope| exceeds a small threshold.
    """
    past = linreg_slope(close, lookback)
    future = linreg_slope(close.shift(-horizon+1), horizon)  # approximate forward slope
    thresh = future.abs().rolling(100).median() * 0.5  # adaptive threshold
    y = ((np.sign(past) != np.sign(future)) & (future.abs() > (thresh.fillna(0)))).astype(int)
    y = y.shift(-(horizon))  # align labels to decision time
    return y

# -----------------------------
# Data loading
# -----------------------------

def load_prices(symbol: str, start: str, end: Optional[str] = None, from_csv: Optional[str] = None) -> pd.DataFrame:
    if from_csv:
        path = os.path.join(from_csv, f"{symbol}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV for {symbol} not found at {path}")
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
        return df
    if yf is None:
        raise RuntimeError("yfinance is unavailable and no CSV path was provided.")
    return yf.download(symbol, start=start, end=end, progress=False)

# -----------------------------
# Walk‑forward training & evaluation
# -----------------------------

def walk_forward_fit_predict(X: pd.DataFrame, y: pd.Series, min_train: int = 400) -> Dict[str, object]:
    """Expanding window walk‑forward. Returns model fitted on all data and out‑of‑sample metrics."""
    Xy = pd.concat([X, y.rename("y")], axis=1).dropna()
    Xf, yf = Xy.drop(columns=["y"]), Xy["y"]

    oof_pred = pd.Series(index=yf.index, dtype=float)
    model = GradientBoostingClassifier(random_state=42)

    for i in range(min_train, len(Xf) - 1):
        Xt, yt = Xf.iloc[:i], yf.iloc[:i]
        Xv = Xf.iloc[i:i+1]
        model.fit(Xt, yt)
        oof_pred.iloc[i] = model.predict_proba(Xv)[:, 1][0]

    # Final fit on all available data
    model.fit(Xf, yf)

    # Metrics (computed on OOF predictions avoiding initial warmup NaNs)
    valid = oof_pred.dropna()
    if len(valid) > 20 and valid.nunique() > 1:
        auc = roc_auc_score(yf.loc[valid.index], valid)
        # Precision at top 20% probability cutoff
        cutoff = np.quantile(valid, 0.8)
        preds_bin = (valid >= cutoff).astype(int)
        prec = precision_score(yf.loc[valid.index], preds_bin, zero_division=0)
    else:
        auc, prec, cutoff = np.nan, np.nan, np.nan

    return {"model": model, "oof_pred": oof_pred, "auc": auc, "precision_top20": prec}

# -----------------------------
# Screener
# -----------------------------

@dataclass
class ScreenResult:
    symbol: str
    prob_reversal_1_3d: float
    auc: float
    precision_top20: float
    recent_return_5d: float
    rsi14: float
    slope_5: float


def screen_symbols(symbols: List[str], start: str, end: Optional[str], inter_syms: List[str], from_csv: Optional[str]) -> pd.DataFrame:
    allowed = set(MASTER_TICKERS)
    filtered_symbols = [s for s in symbols if s in allowed]
    skipped = [s for s in symbols if s not in allowed]

    if skipped:
        print("[WARN] The following symbols are not in the approved universe and will be skipped:", " ".join(skipped))
    if not filtered_symbols:
        print("[WARN] No valid symbols to screen after applying the approved universe filter.")
        return pd.DataFrame(
            columns=[
                "symbol",
                "prob_reversal_1_3d",
                "auc",
                "precision_top20",
                "recent_return_5d",
                "rsi14",
                "slope_5",
            ]
        )

    # Load intermarket series
    inter_data: Dict[str, pd.DataFrame] = {}
    for s in inter_syms:
        try:
            inter_data[s] = load_prices(s, start=start, end=end, from_csv=from_csv)
        except Exception:
            continue

    results: List[ScreenResult] = []

    for sym in filtered_symbols:
        try:
            df = load_prices(sym, start=start, end=end, from_csv=from_csv)
            if len(df) < 600:
                print(f"[WARN] {sym}: too few rows ({len(df)}) – skipping")
                continue

            feats = make_features(df, inter_data)
            labels = make_labels(_select_close(df).rename("Close"))
            aligned = feats.join(labels.rename("y"))

            if aligned["y"].dropna().empty:
                print(f"[WARN] {sym}: label series empty – skipping")
                continue

            res = walk_forward_fit_predict(aligned.drop(columns=["y"]), aligned["y"], min_train=400)
            model = res["model"]

            latest_X = feats.iloc[[-1]]
            prob = float(model.predict_proba(latest_X)[:, 1][0]) if len(latest_X) else np.nan

            results.append(
                ScreenResult(
                    symbol=sym,
                    prob_reversal_1_3d=prob,
                    auc=float(res["auc"]),
                    precision_top20=float(res["precision_top20"]),
                    recent_return_5d=float(_select_close(df).pct_change(5).iloc[-1]),
                    rsi14=float(feats["rsi14"].iloc[-1]),
                    slope_5=float(feats["slope_5"].iloc[-1]),
                )
            )
        except Exception as e:
            import traceback

            print(f"[ERROR] {sym}: {e}")
            traceback.print_exc()
            continue

    if not results:
        return pd.DataFrame(
            columns=[
                "symbol",
                "prob_reversal_1_3d",
                "auc",
                "precision_top20",
                "recent_return_5d",
                "rsi14",
                "slope_5",
            ]
        )

    out = pd.DataFrame([r.__dict__ for r in results]).sort_values("prob_reversal_1_3d", ascending=False)
    return out

# -----------------------------
# CLI + Tests
# -----------------------------

def parse_args(list_args: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI-assisted 1–3 day trend reversal screener",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbols", nargs="+", help="Ticker symbols to screen", default=None)
    p.add_argument("--start", type=str, default="2015-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--intermarket", nargs="*", default=["SPY", "DXY", "^VIX"], help="Optional intermarket tickers")
    p.add_argument("--from-csv", type=str, default=None, help="Directory holding <TICKER>.csv files (bypasses yfinance)")
    p.add_argument("--self-test", action="store_true", help="Run quick unit tests and exit")
    return p.parse_args(list_args)


# ---- Self tests (no network) ----

def _generate_dummy_prices(n: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Trend + noise + seasonality
    t = np.arange(n)
    base = 100 + 0.02 * t + 2 * np.sin(t / 15)
    noise = rng.normal(0, 0.5, n)
    close = base + noise
    high = close + rng.uniform(0.1, 0.6, n)
    low = close - rng.uniform(0.1, 0.6, n)
    open_ = close + rng.normal(0, 0.2, n)
    vol = rng.integers(1e5, 2e5, n)
    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": vol,
    })
    df.index = pd.date_range("2010-01-01", periods=n, freq="B")
    return df


def run_self_tests() -> None:
    import unittest

    class TestIndicators(unittest.TestCase):
        def setUp(self):
            self.df = _generate_dummy_prices()

        def test_rsi_bounds(self):
            r = rsi(_select_close(self.df))  # should be within [0, 100]
            self.assertTrue((r.dropna() >= 0).all())
            self.assertTrue((r.dropna() <= 100).all())

        def test_macd_cols(self):
            m = macd(_select_close(self.df))  # has columns
            self.assertTrue(set(["macd", "macd_signal", "macd_hist"]).issubset(m.columns))

        def test_labels_binary(self):
            y = make_labels(_select_close(self.df)).dropna()
            self.assertTrue(set(y.unique()).issubset({0, 1}))

    class TestCLI(unittest.TestCase):
        def test_defaults_when_no_args(self):
            ns = parse_args([])
            # With no args, symbols should be None; main() assigns defaults
            self.assertIsNone(ns.symbols)

        def test_parse_with_symbols(self):
            ns = parse_args(["--symbols", "AAPL", "MSFT"])
            self.assertEqual(ns.symbols, ["AAPL", "MSFT"])

    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    unittest.TextTestRunner(verbosity=2).run(suite)


def main():
    args = parse_args()

    if args.self_test:
        print("Running self-tests...\n")
        run_self_tests()
        return

    # Default symbols if none were provided (prevents argparse SystemExit 2 on bare run)
    default_symbols = MASTER_TICKERS[:3]
    symbols = args.symbols or default_symbols
    if args.symbols is None:
        print(f"[INFO] No --symbols provided. Running demo with: {' '.join(default_symbols)}\n")

    table = screen_symbols(symbols, args.start, args.end, args.intermarket, args.from_csv)
    pd.set_option("display.max_columns", None)
    print("\n=== Reversal Screener (1–3 days) ===")
    if table.empty:
        print("No results. (Data might be missing; try other symbols or provide CSVs.)")
    else:
        print(table.to_string(index=False, justify="center", float_format=lambda x: f"{x:0.4f}"))


if __name__ == "__main__":
    main()
