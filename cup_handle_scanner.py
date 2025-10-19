"""Cup and handle pattern detection utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from typing import Iterable


def _linear_regression_slope(x: Iterable[float], y: Iterable[float]) -> float:
    x_vals = [float(val) for val in x]
    y_vals = [float(val) for val in y]
    if not x_vals or len(x_vals) != len(y_vals):
        return 0.0
    x_mean = float(np.mean(x_vals))
    y_mean = float(np.mean(y_vals))
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass
class CupHandleHit:
    """Container describing a detected cup and handle pattern."""

    resistance: float
    cup_depth: float
    cup_depth_pct: float
    handle_length: int
    handle_pullback_pct: float
    handle_slope: float


_DEF_MIN_CUP_DEPTH_PCT = 0.12
_DEF_MAX_HANDLE_PULLBACK_PCT = 0.18
_DEF_HANDLE_SLOPE_BOUNDS = (-0.25, 0.05)


def detect_cup_and_handle(
    df: pd.DataFrame,
    *,
    cup_window: int = 60,
    handle_window: int = 15,
    tolerance: float = 0.1,
    min_cup_depth_pct: float | None = None,
    max_handle_pullback_pct: float | None = None,
    handle_slope_bounds: tuple[float, float] | None = None,
) -> Optional[CupHandleHit]:
    """Return a :class:`CupHandleHit` if a pattern is detected.

    The logic mirrors the historical boolean implementation but now captures
    richer metrics so downstream consumers can surface more insightful context
    without re-computing them.
    """

    if df is None or df.empty:
        return None

    required_cols = {"close"}
    if not required_cols.issubset(df.columns):
        return None

    min_cup_depth_pct = _DEF_MIN_CUP_DEPTH_PCT if min_cup_depth_pct is None else min_cup_depth_pct
    max_handle_pullback_pct = (
        _DEF_MAX_HANDLE_PULLBACK_PCT if max_handle_pullback_pct is None else max_handle_pullback_pct
    )
    handle_slope_bounds = _DEF_HANDLE_SLOPE_BOUNDS if handle_slope_bounds is None else handle_slope_bounds

    window = cup_window + handle_window
    if window <= 0 or len(df) < window:
        return None

    closes = df["close"].tail(window)
    if closes.isnull().any():
        return None

    cup = closes.iloc[:cup_window]
    handle = closes.iloc[cup_window:]
    if cup.empty or handle.empty:
        return None

    midpoint = cup_window // 2
    left = cup.iloc[: midpoint or 1]
    right = cup.iloc[midpoint:]
    if left.empty or right.empty:
        return None

    left_peak = float(left.max())
    right_peak = float(right.max())
    resistance = float(np.mean([left_peak, right_peak]))
    if resistance <= 0:
        return None

    peak_diff = abs(left_peak - right_peak) / resistance
    if peak_diff > tolerance:
        return None

    cup_min = float(cup.min())
    cup_depth = float(resistance - cup_min)
    if cup_depth <= 0:
        return None

    cup_depth_pct = cup_depth / resistance
    if cup_depth_pct < min_cup_depth_pct:
        return None

    x = range(handle_window)
    handle_slope = float(_linear_regression_slope(x, handle))

    slope_low, slope_high = handle_slope_bounds
    if not (slope_low <= handle_slope <= slope_high):
        return None

    handle_min = float(handle.min())
    handle_pullback = (resistance - handle_min) / resistance
    if handle_pullback < 0 or handle_pullback > max_handle_pullback_pct:
        return None

    return CupHandleHit(
        resistance=resistance,
        cup_depth=cup_depth,
        cup_depth_pct=cup_depth_pct,
        handle_length=int(handle_window),
        handle_pullback_pct=float(handle_pullback),
        handle_slope=handle_slope,
    )
