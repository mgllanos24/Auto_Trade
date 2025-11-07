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
    left_peak_idx: Optional[int] = None
    right_peak_idx: Optional[int] = None
    cup_low_idx: Optional[int] = None
    handle_start_idx: Optional[int] = None
    handle_end_idx: Optional[int] = None
    handle_low_idx: Optional[int] = None


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
    handle_consolidation_pct: float = 0.1,
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

    if handle_consolidation_pct <= 0 or handle_consolidation_pct >= 1:
        handle_consolidation_pct = 0.1

    if handle_min < resistance * (1 - handle_consolidation_pct):
        return None

    handle_max = float(handle.max())
    if handle_max > resistance * (1 + tolerance):
        return None

    volumes = df["volume"] if "volume" in getattr(df, "columns", []) else None
    if volumes is not None:
        handle_volume = volumes.tail(window).iloc[cup_window:]
        if len(handle_volume) == handle_window and not handle_volume.isnull().any():
            cup_volume = volumes.tail(window).iloc[:cup_window]
            ref_volume = cup_volume.iloc[-len(handle_volume) :] if len(cup_volume) else cup_volume
            if ref_volume is not None and len(ref_volume):
                ref_mean = float(np.mean(ref_volume))
                handle_mean = float(np.mean(handle_volume))
                if ref_mean > 0 and handle_mean >= ref_mean:
                    return None
            volume_slope = float(_linear_regression_slope(range(len(handle_volume)), handle_volume))
            if volume_slope > 0:
                return None

    rsi_series = df["rsi"] if "rsi" in getattr(df, "columns", []) else None
    if rsi_series is not None:
        handle_rsi = rsi_series.tail(window).iloc[cup_window:]
        if len(handle_rsi) == handle_window and not handle_rsi.isnull().any():
            rsi_min = float(handle_rsi.min())
            rsi_max = float(handle_rsi.max())
            if rsi_min < 40 or rsi_max > 60:
                return None

    def _series_index(series: pd.Series, mode: str) -> Optional[object]:
        if series is None or getattr(series, "empty", True):
            return None
        try:
            if mode == "max":
                return series.idxmax()
            return series.idxmin()
        except AttributeError:
            pass

        try:
            values = series.to_numpy()
        except AttributeError:
            try:
                values = np.asarray(list(series))
            except Exception:
                return None

        if values.size == 0:
            return None

        if mode == "max":
            try:
                position = int(np.argmax(values))
            except AttributeError:
                position = int(max(range(len(values)), key=lambda idx: values[idx]))
        else:
            try:
                position = int(np.argmin(values))
            except AttributeError:
                position = int(min(range(len(values)), key=lambda idx: values[idx]))
        try:
            return series.index[position]
        except Exception:  # pragma: no cover - fallback for stub indices
            return position

    def _resolve_position(label: Optional[object]) -> Optional[int]:
        if label is None:
            return None
        index_obj = getattr(df, "index", None)
        if index_obj is None:
            return None
        try:
            if hasattr(index_obj, "get_loc"):
                loc = index_obj.get_loc(label)  # type: ignore[call-arg]
            else:
                loc = list(index_obj).index(label)
        except (KeyError, ValueError, AttributeError, TypeError):
            return None
        if isinstance(loc, slice):
            loc = loc.start
        elif isinstance(loc, np.ndarray):
            if loc.size == 0:
                return None
            loc = loc[0]
        if loc is None:
            return None
        try:
            return int(loc)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    left_peak_idx = _resolve_position(_series_index(left, "max")) if not left.empty else None
    right_peak_idx = _resolve_position(_series_index(right, "max")) if not right.empty else None
    cup_low_idx = _resolve_position(_series_index(cup, "min")) if not cup.empty else None
    handle_index = getattr(handle, "index", None)
    if handle_index is not None and len(handle_index):
        handle_start_idx = _resolve_position(handle_index[0])
        handle_end_idx = _resolve_position(handle_index[-1])
    else:  # pragma: no cover - defensive
        handle_start_idx = handle_end_idx = None
    handle_low_idx = _resolve_position(_series_index(handle, "min")) if not handle.empty else None

    return CupHandleHit(
        resistance=resistance,
        cup_depth=cup_depth,
        cup_depth_pct=cup_depth_pct,
        handle_length=int(handle_window),
        handle_pullback_pct=float(handle_pullback),
        handle_slope=handle_slope,
        left_peak_idx=left_peak_idx,
        right_peak_idx=right_peak_idx,
        cup_low_idx=cup_low_idx,
        handle_start_idx=handle_start_idx,
        handle_end_idx=handle_end_idx,
        handle_low_idx=handle_low_idx,
    )
