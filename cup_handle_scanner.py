"""Cup and handle pattern detection utilities.

This module exposes a :class:`CupHandleHit` dataclass describing the salient
properties of a detected pattern together with a :func:`scan_cup_handle`
routine that scans OHLCV data for qualifying setups.  The implementation keeps
parity with the heuristics historically used in :func:`pattern_scanner.detect_cup_and_handle`
while enriching the results with structured metadata that downstream callers can
consume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class CupHandleHit:
    """Container summarising a detected cup and handle formation."""

    start_idx: int
    end_idx: int
    cup_start_idx: int
    cup_end_idx: int
    handle_start_idx: int
    handle_end_idx: int
    left_peak: float
    right_peak: float
    cup_low: float
    cup_depth_pct: float
    handle_low: float
    handle_depth_pct: float
    handle_slope: float
    breakout: bool
    breakout_idx: Optional[int]
    breakout_price: Optional[float]


def _linreg_slope(values: Iterable[float]) -> float:
    """Return the slope of a simple linear regression fitted to ``values``."""

    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("values must contain at least one element")

    x = np.arange(arr.size, dtype=float).reshape(-1, 1)
    model = LinearRegression().fit(x, arr)
    return float(model.coef_[0])


def _validate_inputs(df: pd.DataFrame, total_window: int, handle_window: int) -> bool:
    if df is None or df.empty:
        return False
    if len(df) < total_window:
        return False
    if handle_window <= 0 or total_window <= handle_window:
        return False
    if "close" not in df.columns:
        return False
    return True


def _evaluate_window(
    window: pd.DataFrame,
    *,
    global_start: int,
    handle_window: int,
    tolerance: float,
    min_cup_depth: float,
    max_handle_slope: float,
    min_handle_slope: float,
    require_breakout: bool,
) -> Optional[CupHandleHit]:
    """Evaluate a single window for a cup and handle formation."""

    closes = window["close"].to_numpy(dtype=float)
    cup_length = len(window) - handle_window
    if cup_length <= 2:
        return None

    cup_segment = closes[:cup_length]
    handle_segment = closes[cup_length:]

    midpoint = cup_length // 2
    if midpoint == 0 or midpoint == cup_length:
        return None

    left = cup_segment[:midpoint]
    right = cup_segment[midpoint:]

    left_peak = float(np.max(left))
    right_peak = float(np.max(right))
    cup_low = float(np.min(cup_segment))

    peak_ref = max(left_peak, right_peak)
    if peak_ref <= 0:
        return None

    symmetry = abs(left_peak - right_peak) / peak_ref
    if symmetry > tolerance:
        return None

    left_depth = (left_peak - cup_low) / left_peak if left_peak else 0.0
    right_depth = (right_peak - cup_low) / right_peak if right_peak else 0.0
    cup_depth = min(left_depth, right_depth)
    if cup_depth < min_cup_depth:
        return None

    try:
        handle_slope = _linreg_slope(handle_segment)
    except ValueError:
        return None

    if not (min_handle_slope < handle_slope < max_handle_slope):
        return None

    handle_low = float(np.min(handle_segment))
    handle_depth_pct = (peak_ref - handle_low) / peak_ref if peak_ref else 0.0

    breakout_price = float(handle_segment[-1])
    breakout = breakout_price >= peak_ref
    if require_breakout and not breakout:
        return None

    start_idx = global_start
    end_idx = global_start + len(window) - 1
    cup_start_idx = start_idx
    cup_end_idx = start_idx + cup_length - 1
    handle_start_idx = cup_end_idx + 1
    handle_end_idx = end_idx
    breakout_idx = handle_end_idx if breakout else None

    return CupHandleHit(
        start_idx=start_idx,
        end_idx=end_idx,
        cup_start_idx=cup_start_idx,
        cup_end_idx=cup_end_idx,
        handle_start_idx=handle_start_idx,
        handle_end_idx=handle_end_idx,
        left_peak=left_peak,
        right_peak=right_peak,
        cup_low=cup_low,
        cup_depth_pct=float(cup_depth),
        handle_low=handle_low,
        handle_depth_pct=float(handle_depth_pct),
        handle_slope=float(handle_slope),
        breakout=breakout,
        breakout_idx=breakout_idx,
        breakout_price=breakout_price if breakout else None,
    )


def scan_cup_handle(
    df: pd.DataFrame,
    *,
    cup_window: int = 60,
    handle_window: int = 15,
    tolerance: float = 0.1,
    min_cup_depth: float = 0.1,
    min_handle_slope: float = -0.2,
    max_handle_slope: float = 0.05,
    require_breakout: bool = False,
) -> List[CupHandleHit]:
    """Scan ``df`` for cup and handle formations.

    Parameters mirror the legacy :func:`pattern_scanner.detect_cup_and_handle`
    defaults while exposing additional knobs for callers that need to restrict
    the required cup depth, handle slope, or breakout confirmation.
    """

    total_window = cup_window + handle_window
    if not _validate_inputs(df, total_window, handle_window):
        return []

    hits: List[CupHandleHit] = []
    for end in range(total_window, len(df) + 1):
        window = df.iloc[end - total_window : end]
        hit = _evaluate_window(
            window,
            global_start=end - total_window,
            handle_window=handle_window,
            tolerance=tolerance,
            min_cup_depth=min_cup_depth,
            max_handle_slope=max_handle_slope,
            min_handle_slope=min_handle_slope,
            require_breakout=require_breakout,
        )
        if hit:
            hits.append(hit)

    return hits
