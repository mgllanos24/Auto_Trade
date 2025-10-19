"""Swing trading stock screener utilities.

This module mirrors the layout used by the pattern specific scanners in the
project (see :mod:`double_bottom_scanner` and :mod:`cup_handle_scanner`).  It
exposes a small dataclass describing a qualifying swing candidate together with
helper functions that calculate technical indicators and a public screening
routine.

The implementation focuses on being deterministic and testable – the caller is
expected to supply OHLCV data as a :class:`pandas.DataFrame` and no network
operations are performed from within this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SwingCandidate",
    "SwingScreenerConfig",
    "evaluate_swing_setup",
    "screen_swing_candidates",
]


@dataclass(frozen=True)
class SwingCandidate:
    """Description of a symbol that meets the swing trading criteria."""

    symbol: str
    close: float
    trend_strength: float
    momentum_score: float
    atr_pct: float
    rsi: float
    pullback_pct: float


@dataclass(frozen=True)
class SwingScreenerConfig:
    """Configuration object that controls the swing screening heuristics."""

    min_candles: int = 90
    volume_window: int = 20
    min_average_volume: float = 500_000
    short_ma: int = 21
    long_ma: int = 50
    atr_period: int = 14
    max_atr_pct: float = 0.06
    rsi_period: int = 14
    rsi_bounds: tuple[float, float] = (45.0, 65.0)
    pullback_window: int = 20
    max_pullback_pct: float = 0.08


def _validate_ohlcv(df: pd.DataFrame) -> bool:
    required = {"open", "high", "low", "close", "volume"}
    return bool(required.issubset(df.columns))


def _ema(series: pd.Series, span: int) -> pd.Series:
    span = max(int(span), 1)
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    return series.rolling(window=window, min_periods=window).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - prev_close).abs()
    low_close = (df["low"] - prev_close).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    period = max(int(period), 1)
    tr = _true_range(df)
    return tr.rolling(window=period, min_periods=period).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    period = max(int(period), 1)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    if len(rsi) >= period:
        rsi = rsi.ewm(span=period, adjust=False).mean()
    return rsi.fillna(50.0)


def _pullback_pct(series: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    rolling_high = series.rolling(window=window, min_periods=window).max()
    pullback = (rolling_high - series) / rolling_high.replace(0, np.nan)
    return pullback.replace([np.inf, -np.inf], np.nan)


def evaluate_swing_setup(
    symbol: str,
    df: pd.DataFrame,
    config: SwingScreenerConfig | None = None,
) -> Optional[SwingCandidate]:
    """Return a :class:`SwingCandidate` if the symbol satisfies the filters."""

    if df is None or df.empty or not _validate_ohlcv(df):
        return None

    config = config or SwingScreenerConfig()

    if len(df) < config.min_candles:
        return None

    recent = df.tail(config.long_ma + 5).copy()
    if recent[["high", "low", "close", "volume"]].isnull().any().any():
        return None

    close = float(recent["close"].iloc[-1])

    sma_short = _sma(recent["close"], config.short_ma)
    sma_long = _sma(recent["close"], config.long_ma)

    short_ma_val = sma_short.iloc[-1]
    long_ma_val = sma_long.iloc[-1]
    if np.isnan(short_ma_val) or np.isnan(long_ma_val):
        return None

    if close <= long_ma_val or short_ma_val <= long_ma_val:
        return None

    avg_volume = recent["volume"].tail(config.volume_window).mean()
    if float(avg_volume) < config.min_average_volume:
        return None

    atr_series = _atr(recent, config.atr_period)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else np.nan
    if not np.isfinite(atr_val):
        return None
    atr_pct = atr_val / close if close else np.nan
    if not np.isfinite(atr_pct) or atr_pct > config.max_atr_pct:
        return None

    rsi_series = _rsi(recent["close"], config.rsi_period)
    rsi_val = float(rsi_series.iloc[-1])
    rsi_low, rsi_high = config.rsi_bounds
    if not (rsi_low <= rsi_val <= rsi_high):
        return None

    pullback_series = _pullback_pct(recent["close"], config.pullback_window)
    pullback_val = float(pullback_series.iloc[-1]) if not pullback_series.empty else np.nan
    if not np.isfinite(pullback_val) or pullback_val > config.max_pullback_pct:
        return None

    trend_strength = (short_ma_val - long_ma_val) / long_ma_val
    momentum_score = (close - short_ma_val) / short_ma_val

    return SwingCandidate(
        symbol=symbol,
        close=close,
        trend_strength=float(trend_strength),
        momentum_score=float(momentum_score),
        atr_pct=float(atr_pct),
        rsi=float(rsi_val),
        pullback_pct=float(pullback_val),
    )


def screen_swing_candidates(
    data: Mapping[str, pd.DataFrame],
    *,
    config: SwingScreenerConfig | None = None,
) -> List[SwingCandidate]:
    """Evaluate multiple symbols and return those matching the swing criteria."""

    if not data:
        return []

    config = config or SwingScreenerConfig()

    candidates: List[SwingCandidate] = []
    for symbol, df in data.items():
        candidate = evaluate_swing_setup(symbol, df, config)
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda c: (c.trend_strength, c.momentum_score), reverse=True)
    return candidates

