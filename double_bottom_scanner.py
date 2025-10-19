"""Utilities for detecting double bottom patterns.

This module mirrors the structure of the ascending triangle scanner by exposing a
`DoubleBottomHit` dataclass together with helper functions and a public scanning
routine.  The implementation focuses on identifying classic double-bottom
patterns within sliding windows of OHLCV data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from typing import Sequence


@dataclass
class DoubleBottomHit:
    """Container describing a detected double bottom pattern."""

    start_idx: int
    end_idx: int
    left_idx: int
    right_idx: int
    left_timestamp: pd.Timestamp
    right_timestamp: pd.Timestamp
    support: float
    neckline: float
    left_low: float
    right_low: float
    bounce_pct: float
    touch_count: int
    contraction_pct: Optional[float]
    volume_contracted: Optional[bool]
    breakout: bool
    breakout_idx: Optional[int]
    breakout_timestamp: Optional[pd.Timestamp]
    breakout_price: Optional[float]


def _linreg(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    """Return the slope and intercept of a simple linear regression."""

    x_arr = [float(val) for val in x]
    y_arr = [float(val) for val in y]

    if not x_arr or not y_arr or len(x_arr) != len(y_arr):
        raise ValueError("x and y must be non-empty sequences of equal length")

    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_arr, y_arr))
    denominator = sum((x - x_mean) ** 2 for x in x_arr)
    if denominator == 0:
        return 0.0, y_mean
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)


def _find_troughs(values: Sequence[float], distance: int) -> List[int]:
    """Return indices of local minima separated by at least ``distance``."""

    data = list(values)
    n = len(data)
    if n < 3:
        return []

    distance = max(int(distance), 1)
    troughs: List[int] = []
    for idx in range(1, n - 1):
        if data[idx] < data[idx - 1] and data[idx] < data[idx + 1]:
            if troughs and idx - troughs[-1] < distance:
                continue
            troughs.append(idx)
    return troughs


def _count_touches(values: Iterable[float], level: float, tolerance: float) -> int:
    """Count how many values lie within ``tolerance`` of ``level``."""

    if level == 0:
        return 0

    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0

    relative = np.abs(arr - level) / abs(level)
    return int(np.sum(relative <= tolerance))


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling mean with sane defaults."""

    window = max(int(window), 1)
    return series.rolling(window=window, min_periods=1).mean()


def scan_double_bottoms(
    df: pd.DataFrame,
    *,
    window: int = 60,
    step: int = 1,
    tolerance: float = 0.03,
    min_bounce: float = 0.05,
    require_breakout: bool = True,
    require_volume_contraction: bool = False,
) -> List[DoubleBottomHit]:
    """Scan a dataframe for double bottom patterns.

    Parameters mirror :func:`ascending_triangle_scanner.scan_ascending_triangles`
    while adapting the heuristics to the double bottom setup.
    """

    if df is None or df.empty:
        return []

    required_cols = ["high", "low", "close", "volume"]
    if not set(required_cols).issubset(df.columns):
        return []

    n_rows = len(df)
    if window <= 0:
        window = n_rows
    window = min(window, n_rows)
    step = max(step, 1)

    hits: List[DoubleBottomHit] = []
    for start in range(0, n_rows - window + 1, step):
        end = start + window
        segment = df.iloc[start:end]

        if segment[required_cols].isnull().any().any():
            continue

        lows = segment["low"].to_numpy()
        highs = segment["high"].to_numpy()

        troughs = _find_troughs(lows, distance=2)
        if len(troughs) < 2:
            continue

        for i in range(len(troughs) - 1):
            left = int(troughs[i])
            right = int(troughs[i + 1])
            separation = right - left
            min_sep = max(3, window // 10)
            if separation < min_sep:
                continue

            left_low = lows[left]
            right_low = lows[right]
            support = (left_low + right_low) / 2.0
            if support <= 0:
                continue

            symmetry = abs(left_low - right_low) / support
            if symmetry > tolerance:
                continue

            mid_slice = segment.iloc[left:right + 1]
            if mid_slice.empty:
                continue

            bounce_high = mid_slice["high"].max()
            support_bounce = (bounce_high - support) / support
            if support_bounce < min_bounce:
                continue

            x_vals = np.arange(left, right + 1)
            try:
                slope, intercept = _linreg(x_vals, highs[left:right + 1])
            except ValueError:
                continue

            neckline = slope * right + intercept
            if not np.isfinite(neckline):
                continue

            touches = _count_touches(highs[left:right + 1], neckline, tolerance)

            post = segment.iloc[right + 1 :]
            breakout = False
            breakout_idx: Optional[int] = None
            breakout_price: Optional[float] = None
            breakout_timestamp: Optional[pd.Timestamp] = None
            if not post.empty:
                post_x = np.arange(right + 1, right + 1 + len(post))
                neckline_post = slope * post_x + intercept
                breakout_mask = post["close"].to_numpy() > neckline_post * (1 + tolerance)
                if np.any(breakout_mask):
                    breakout = True
                    rel_idx = int(np.argmax(breakout_mask))
                    breakout_idx = start + right + 1 + rel_idx
                    breakout_price = float(post["close"].iloc[rel_idx])
                    breakout_timestamp = df.index[breakout_idx]

            if require_breakout and not breakout:
                continue

            lookback = max(separation, 1)
            pre_slice = segment["volume"].iloc[max(0, left - lookback) : left]
            mid_vol_slice = segment["volume"].iloc[left : right + 1]
            contraction_pct: Optional[float] = None
            volume_contracted: Optional[bool] = None
            if not pre_slice.empty and not mid_vol_slice.empty:
                pre_mean = _rolling_mean(pre_slice, lookback).iloc[-1]
                mid_mean = _rolling_mean(mid_vol_slice, lookback).iloc[-1]
                if pre_mean > 0:
                    contraction_pct = float((pre_mean - mid_mean) / pre_mean)
                    volume_contracted = bool(mid_mean < pre_mean)

            if require_volume_contraction and not volume_contracted:
                continue

            hits.append(
                DoubleBottomHit(
                    start_idx=start,
                    end_idx=end - 1,
                    left_idx=start + left,
                    right_idx=start + right,
                    left_timestamp=df.index[start + left],
                    right_timestamp=df.index[start + right],
                    support=float(support),
                    neckline=float(neckline),
                    left_low=float(left_low),
                    right_low=float(right_low),
                    bounce_pct=float(support_bounce),
                    touch_count=touches,
                    contraction_pct=contraction_pct,
                    volume_contracted=volume_contracted,
                    breakout=breakout,
                    breakout_idx=breakout_idx,
                    breakout_timestamp=breakout_timestamp,
                    breakout_price=breakout_price,
                )
            )

    return hits
