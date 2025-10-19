import alpaca_trade_api as tradeapi
import pandas as pd
import yfinance as yf
import os
import csv
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from scipy.signal import find_peaks, argrelextrema
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from double_bottom_scanner import DoubleBottomHit, scan_double_bottoms
from cup_handle_scanner import CupHandleHit, detect_cup_and_handle
from swing_trading_screener import (
    SwingCandidate,
    SwingScreenerConfig,
    evaluate_swing_setup,
)

API_KEY = 'PKWMYLAWJCU6ITACV6KP'
API_SECRET = 'k8T9M3XdpVcNQudgPudCfqtkRJ0IUCChFSsKYe07'
BASE_URL = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_PATH = SCRIPT_DIR / 'watchlist.csv'
WATCHLIST_HEADER = [
    'symbol',
    'breakout_high',
    'rr_ratio',
    'stop_loss',
    'target_price',
    'timestamp',
    'direction',
    'pattern',
    '3mo_volume'
]

SWING_CONFIG = SwingScreenerConfig()


@dataclass
class AscendingTrianglePattern:
    resistance: float
    support_slope: float
    support_intercept: float
    offset: int
    length: int
    resistance_indices: List[int]
    support_indices: List[int]
    breakout: bool


@dataclass
class InverseHeadShouldersPattern:
    left_idx: int
    head_idx: int
    right_idx: int
    left_low: float
    head_low: float
    right_low: float
    neckline_left_idx: int
    neckline_right_idx: int
    neckline_left: float
    neckline_right: float


@dataclass
class BullishPennantPattern:
    upper_slope: float
    upper_intercept: float
    lower_slope: float
    lower_intercept: float
    offset: int
    length: int


@dataclass
class BullishFlagPattern:
    slope: float
    intercept: float
    upper_offset: float
    lower_offset: float
    offset: int
    length: int


@dataclass
class BullishRectanglePattern:
    high: float
    low: float
    offset: int
    length: int
    high_touch_indices: List[int]
    low_touch_indices: List[int]


@dataclass
class RoundingBottomPattern:
    coeffs: Sequence[float]
    offset: int
    length: int


@dataclass
class BreakawayGapPattern:
    prev_close_idx: int
    curr_open_idx: int
    prev_close: float
    curr_open: float
    curr_close: float


@dataclass
class PrecomputedIndicators:
    entry: float
    ma50: Optional[float]
    ma200: Optional[float]
    price_slope: Optional[float]
    volume_slope: Optional[float]
    recent_high_20: Optional[float]

# Excluded ETFs
EXCLUDED_ETFS = ['VTIP', 'NFXS', 'ACWX', 'VXUS', 'NVD', 'NVDD', 'NVDL', 'TBIL', 'VRIG', 'CONL', 'PDBC', 'PFF',
    'EMB', 'EMXC', 'ESGE', 'ETHA', 'TLT', 'EUFN', 'FDNI', 'TQQQ', 'QQQ', 'QQQM', 'QYLD', 'TSDD',
    'TSLL', 'TSLQ', 'TSLR', 'RDVY', 'TSLS', 'IBIT', 'ICLN', 'IEF', 'IGF', 'IGIB', 'IGSB', 'ACWI',
    'ISTB', 'IUSB', 'SCZ', 'IXUS', 'JEPQ', 'USIG', 'BSCQ', 'SHV', 'SHY', 'VCIT', 'VCLT', 'VCSH',
    'VGIT', 'VGLT', 'SMH', 'VGSH', 'BNDX', 'BND', 'MBB', 'MCHI', 'AAPU', 'METU', 'VMBS', 'SOXX', 'SQQQ']

def flatten_yf_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def compute_precomputed_indicators(df: pd.DataFrame) -> PrecomputedIndicators:
    entry = float(df['close'].iloc[-1]) if not df.empty else float('nan')

    closes = df['close']
    ma50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else None
    ma200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else None

    price_tail = closes.tail(90)
    if len(price_tail) >= 2:
        x_price = np.arange(len(price_tail)).reshape(-1, 1)
        price_slope = LinearRegression().fit(x_price, price_tail).coef_[0]
    else:
        price_slope = None

    volume_tail = df['volume'].tail(60)
    if len(volume_tail) >= 2:
        x_vol = np.arange(len(volume_tail)).reshape(-1, 1)
        volume_slope = LinearRegression().fit(x_vol, volume_tail).coef_[0]
    else:
        volume_slope = None

    if len(df) >= 21:
        recent_high_20 = df['high'].rolling(20).max().shift(1).iloc[-1]
    else:
        recent_high_20 = None

    return PrecomputedIndicators(
        entry=entry,
        ma50=ma50,
        ma200=ma200,
        price_slope=price_slope,
        volume_slope=volume_slope,
        recent_high_20=recent_high_20,
    )

def hits_to_dataframe(hits: List[DoubleBottomHit]) -> pd.DataFrame:
    if not hits:
        return pd.DataFrame()
    return pd.DataFrame([hit.__dict__ for hit in hits])


def get_yf_data(symbol):
    try:
        df = yf.download(symbol, period='1y', interval='1d', progress=False, auto_adjust=False)
        df = flatten_yf_columns(df)
        if df.empty or len(df) < 90:
            print(f" Insufficient data for {symbol}: only {len(df)} rows")
            return pd.DataFrame()
        df.reset_index(inplace=True)
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        if df[['high', 'low', 'close', 'volume']].isnull().any().any():
            return pd.DataFrame()
        df.set_index('Date', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception:
        return pd.DataFrame()


def fetch_symbol_data(symbols: Sequence[str], max_workers: int = 8) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    if not symbols:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(get_yf_data, symbol): symbol for symbol in symbols
        }
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                print(f" Error fetching {symbol}: {exc}")
                results[symbol] = pd.DataFrame()

    return results

def anchored_vwap(df, anchor_index):
    cum_vol = df['volume'][anchor_index:].cumsum()
    cum_vol_price = (df['close'][anchor_index:] * df['volume'][anchor_index:]).cumsum()
    return (cum_vol_price / cum_vol).bfill()

def detect_swing_low(df, window=20):
    recent = df.tail(window)
    return recent['low'].idxmin()

def is_structure_uptrend(df):
    from scipy.signal import argrelextrema

    if len(df) < 60:
        return False

    highs = df['high'].values
    lows = df['low'].values

    swing_lows_idx = argrelextrema(lows, np.less, order=5)[0]
    swing_highs_idx = argrelextrema(highs, np.greater, order=5)[0]

    if len(swing_lows_idx) < 3 or len(swing_highs_idx) < 3:
        print(" Not enough swing points to evaluate HH+HL structure")
        return False

    last_lows = lows[swing_lows_idx][-3:]
    last_highs = highs[swing_highs_idx][-3:]

    print(f" Swing Lows: {last_lows}")
    print(f" Swing Highs: {last_highs}")

    margin = 0.02
    is_higher_lows = (last_lows[1] >= last_lows[0] * (1 - margin)) and (last_lows[2] > last_lows[1])
    is_higher_highs = (last_highs[1] >= last_highs[0] * (1 - margin)) and (last_highs[2] > last_highs[1])

    print(f" Higher Lows: {is_higher_lows},  Higher Highs: {is_higher_highs}")
    return is_higher_lows and is_higher_highs

def holds_monthly_breakout(df):
    one_month_high = df['high'].rolling(window=21).max().shift(1).iloc[-1]
    return df['close'].iloc[-1] > one_month_high

def is_uptrend(df, indicators: Optional[PrecomputedIndicators] = None):
    if len(df) < 200:
        print(f" Not enough data for trend check: {len(df)} rows")
        return False
    if indicators and indicators.ma50 is not None:
        ma50 = indicators.ma50
    else:
        ma50 = df['close'].rolling(50).mean().iloc[-1]
    if indicators and indicators.ma200 is not None:
        ma200 = indicators.ma200
    else:
        ma200 = df['close'].rolling(200).mean().iloc[-1]

    if indicators and indicators.price_slope is not None:
        slope = indicators.price_slope
    else:
        recent = df['close'].tail(90).values
        X = np.arange(len(recent)).reshape(-1, 1)
        slope = LinearRegression().fit(X, recent).coef_[0]

    swing_idx = detect_swing_low(df)
    vwap_series = anchored_vwap(df, swing_idx)
    vwap_ok = df['close'].iloc[-1] > vwap_series.iloc[-1]

    structure_ok = is_structure_uptrend(df)
    monthly_ok = holds_monthly_breakout(df)

    print(f"🔹 MA50: {ma50:.2f}, MA200: {ma200:.2f}, Slope: {slope:.4f}")
    print(f"🔹 Price > Anchored VWAP: {vwap_ok}, Structure HH+HL: {structure_ok}, Monthly Breakout: {monthly_ok}")

    pass_count = sum([ma50 > ma200, slope > 0, vwap_ok, structure_ok, monthly_ok])
    return pass_count >= 3

def detect_double_bottom(
    df,
    window=60,
    step=1,
    tolerance=0.03,
    min_bounce=0.05,
    require_breakout=True,
    require_volume_contraction=False,
):
    hits = scan_double_bottoms(
        df,
        window=window,
        step=step,
        tolerance=tolerance,
        min_bounce=min_bounce,
        require_breakout=require_breakout,
        require_volume_contraction=require_volume_contraction,
    )
    if not hits:
        return False

    total_rows = len(df)
    recent_threshold = max(5, window // 4)

    recent_hits = [
        hit
        for hit in hits
        if getattr(hit, "breakout_idx", None) is not None
        and total_rows - hit.breakout_idx <= recent_threshold
    ]

    if not recent_hits:
        recent_hits = [
            hit
            for hit in hits
            if getattr(hit, "right_idx", None) is not None
            and total_rows - hit.right_idx <= recent_threshold
        ]

    if not recent_hits:
        return False

    return max(recent_hits, key=lambda hit: hit.bounce_pct)

def _find_argmax(values: Sequence[float]) -> int:
    max_idx = 0
    max_val = values[0]
    for idx, val in enumerate(values):
        if val > max_val:
            max_val = val
            max_idx = idx
    return max_idx


def _shoulder_has_width(
    lows: Sequence[float],
    idx: int,
    window: int = 2,
    tolerance: float = 0.005,
) -> bool:
    """Ensure the shoulder trough spans multiple candles.

    A valid shoulder should have at least one neighbouring candle with a low
    close to the trough low. This helps avoid flagging one-candle shoulders.
    """

    start = max(0, idx - window)
    end = min(len(lows), idx + window + 1)
    if end - start <= 1:
        return False

    shoulder_low = lows[idx]
    proximity = shoulder_low * tolerance
    similar_candles = sum(
        1 for value in lows[start:end] if abs(value - shoulder_low) <= proximity
    )
    return similar_candles >= 2


def detect_inverse_head_shoulders(df) -> Optional[InverseHeadShouldersPattern]:
    recent = df.tail(60)
    lows = recent['low'].values
    highs = recent['high'].values
    closes = recent['close'].values

    troughs, _ = find_peaks(-lows, distance=5, prominence=0.5)
    if len(troughs) < 3:
        return None

    offset = len(df) - len(recent)

    recency_window = 10

    for i in range(len(troughs) - 2, -1, -1):
        l_idx = int(troughs[i])
        if i + 2 >= len(troughs):
            continue
        h_idx = int(troughs[i + 1])
        r_idx = int(troughs[i + 2])

        if offset + r_idx < len(df) - recency_window:
            break

        l = float(lows[l_idx])
        h = float(lows[h_idx])
        r = float(lows[r_idx])

        if not (h < l and h < r):
            continue
        if max(l, r) == 0:
            continue
        if abs(l - r) / max(l, r) > 0.1:
            continue
        if not (_shoulder_has_width(lows, l_idx) and _shoulder_has_width(lows, r_idx)):
            continue

        left_high_segment = highs[l_idx:h_idx + 1]
        right_high_segment = highs[h_idx:r_idx + 1]
        if len(left_high_segment) == 0 or len(right_high_segment) == 0:
            continue

        left_high_rel = _find_argmax(left_high_segment)
        right_high_rel = _find_argmax(right_high_segment)

        neckline_left_idx = offset + l_idx + left_high_rel
        neckline_right_idx = offset + h_idx + right_high_rel
        neckline_left = float(left_high_segment[left_high_rel])
        neckline_right = float(right_high_segment[right_high_rel])
        neckline_level = (neckline_left + neckline_right) / 2

        if closes[-1] > neckline_level:
            return InverseHeadShouldersPattern(
                left_idx=offset + l_idx,
                head_idx=offset + h_idx,
                right_idx=offset + r_idx,
                left_low=l,
                head_low=h,
                right_low=r,
                neckline_left_idx=neckline_left_idx,
                neckline_right_idx=neckline_right_idx,
                neckline_left=neckline_left,
                neckline_right=neckline_right,
            )

    return None

def detect_ascending_triangle(
    df,
    window=60,
    tolerance=0.02,
    min_touches=2,
) -> Optional[AscendingTrianglePattern]:
    window = min(window, 60, len(df))
    if window < min_touches + 2:
        return None

    recent = df.tail(window)
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    x = np.arange(len(lows)).reshape(-1, 1)

    if len(highs) < min_touches + 2:
        return None

    plateau_end = len(highs) - 2 if len(highs) > 2 else len(highs)
    if plateau_end <= 0:
        return None

    plateau_span = min(20, plateau_end)
    plateau_start = plateau_end - plateau_span
    plateau_highs = highs[plateau_start:plateau_end]

    if plateau_highs.size == 0:
        return None

    resistance = float(plateau_highs.max())
    if resistance == 0:
        return None

    plateau_range = float(plateau_highs.max() - plateau_highs.min())
    if plateau_range / resistance > tolerance:
        return None

    resistance_touches = [
        idx
        for idx in range(plateau_start, plateau_end)
        if abs(highs[idx] - resistance) / resistance < tolerance
    ]

    if len(resistance_touches) < min_touches:
        return None

    lr = LinearRegression().fit(x, lows)
    support_slope = float(lr.coef_[0])
    support_intercept = float(getattr(lr, "intercept_", lows.mean() - support_slope * (len(lows) / 2)))
    if support_slope <= 0:
        return None

    fitted_support = lr.predict(x)
    support_touch_indices = [
        idx
        for idx, low in enumerate(lows)
        if low != 0 and abs(low - fitted_support[idx]) / abs(low) < tolerance
    ]

    if len(support_touch_indices) < min_touches:
        return None

    support_touch_lows = [float(lows[idx]) for idx in support_touch_indices]
    if len(support_touch_lows) >= 2:
        if support_touch_lows[-1] <= support_touch_lows[0] * (1 + tolerance):
            return None

        for previous, current in zip(support_touch_lows, support_touch_lows[1:]):
            if current < previous * (1 - tolerance):
                return None

    breakout = (closes[-1] > resistance * (1 + tolerance)) or (
        highs[-1] > resistance * (1 + tolerance)
    )

    offset = len(df) - len(recent)
    return AscendingTrianglePattern(
        resistance=resistance,
        support_slope=support_slope,
        support_intercept=support_intercept,
        offset=offset,
        length=len(recent),
        resistance_indices=[offset + idx for idx in resistance_touches],
        support_indices=[offset + idx for idx in support_touch_indices],
        breakout=bool(breakout),
    )


def detect_bullish_pennant(
    df,
    window=60,
    flagpole_window=20,
    tolerance=0.03,
) -> Optional[BullishPennantPattern]:
    if len(df) < window + flagpole_window:
        return None

    flagpole = df.iloc[-(window + flagpole_window):-window]
    if flagpole.empty:
        return None
    pole_gain = flagpole['close'].iloc[-1] / flagpole['close'].iloc[0]
    if pole_gain < 1.2:
        return None

    recent = df.tail(window)
    highs = recent['high'].values
    lows = recent['low'].values
    x = np.arange(len(highs)).reshape(-1, 1)

    upper_model = LinearRegression().fit(x, highs)
    lower_model = LinearRegression().fit(x, lows)
    high_slope = float(upper_model.coef_[0])
    low_slope = float(lower_model.coef_[0])

    if high_slope >= 0 or low_slope <= 0:
        return None
    if abs(high_slope - low_slope) < 0.01:
        return None

    offset = len(df) - len(recent)
    return BullishPennantPattern(
        upper_slope=high_slope,
        upper_intercept=float(getattr(upper_model, "intercept_", highs.mean())),
        lower_slope=low_slope,
        lower_intercept=float(getattr(lower_model, "intercept_", lows.mean())),
        offset=offset,
        length=len(recent),
    )


def detect_bullish_flag(
    df,
    window=40,
    flagpole_window=20,
    slope_threshold=-0.1,
) -> Optional[BullishFlagPattern]:
    if len(df) < window + flagpole_window:
        return None

    flagpole = df.iloc[-(window + flagpole_window):-window]
    if flagpole.empty:
        return None
    pole_gain = flagpole['close'].iloc[-1] / flagpole['close'].iloc[0]
    if pole_gain < 1.2:
        return None

    recent = df.tail(window)
    x = np.arange(len(recent)).reshape(-1, 1)
    y = recent['close'].values
    model = LinearRegression().fit(x, y)
    slope = float(model.coef_[0])

    if not (slope_threshold < slope < 0):
        return None

    fitted = model.predict(x)
    upper_offset = float(np.max(recent['high'].values - fitted))
    lower_offset = float(np.min(recent['low'].values - fitted))

    offset = len(df) - len(recent)
    return BullishFlagPattern(
        slope=slope,
        intercept=float(getattr(model, "intercept_", y.mean())),
        upper_offset=upper_offset,
        lower_offset=lower_offset,
        offset=offset,
        length=len(recent),
    )


def detect_bullish_rectangle(
    df,
    window=60,
    tolerance=0.02,
    min_touches=2,
) -> Optional[BullishRectanglePattern]:
    if len(df) < window:
        return None

    recent = df.tail(window)
    high = float(recent['high'].max())
    low = float(recent['low'].min())
    if high == 0 or low == 0:
        return None

    high_touch_indices = [
        idx for idx, value in enumerate(recent['high'].values) if abs(value - high) / high < tolerance
    ]
    low_touch_indices = [
        idx for idx, value in enumerate(recent['low'].values) if abs(value - low) / low < tolerance
    ]

    if len(high_touch_indices) < min_touches or len(low_touch_indices) < min_touches:
        return None

    offset = len(df) - len(recent)
    return BullishRectanglePattern(
        high=high,
        low=low,
        offset=offset,
        length=len(recent),
        high_touch_indices=[offset + idx for idx in high_touch_indices],
        low_touch_indices=[offset + idx for idx in low_touch_indices],
    )


def detect_rounding_bottom(
    df,
    window=100,
    tolerance=0.02,
) -> Optional[RoundingBottomPattern]:
    if len(df) < window:
        return None

    recent = df.tail(window)
    prices = recent['close'].values
    if len(prices) < 3:
        return None
    x = np.arange(len(prices))
    poly = np.polyfit(x, prices, 2)
    a = float(poly[0])

    if a <= 0:
        return None

    mid = window // 2
    left = np.mean(prices[: max(1, mid // 2)])
    bottom = float(np.min(prices))
    right = np.mean(prices[-max(1, mid // 2):])

    if left == 0 or right == 0:
        return None
    if (left - bottom) / left < tolerance or (right - bottom) / right < tolerance:
        return None

    offset = len(df) - len(recent)
    return RoundingBottomPattern(coeffs=poly, offset=offset, length=len(recent))


def detect_breakaway_gap(
    df,
    gap_threshold=0.03,
    volume_multiplier=1.5,
) -> Optional[BreakawayGapPattern]:
    if len(df) < 35:
        return None

    prev_close = float(df['close'].iloc[-2])
    curr_open = float(df['open'].iloc[-1])
    curr_close = float(df['close'].iloc[-1])
    prev_vol = df['volume'].iloc[-31:-1].mean()
    curr_vol = df['volume'].iloc[-1]

    if prev_close == 0:
        return None

    gap = (curr_open - prev_close) / prev_close
    strong_gap = gap > gap_threshold
    volume_spike = curr_vol > prev_vol * volume_multiplier
    full_gap_fill = curr_close > curr_open

    if strong_gap and volume_spike and full_gap_fill:
        prev_close_idx = len(df) - 2
        curr_open_idx = len(df) - 1
        return BreakawayGapPattern(
            prev_close_idx=prev_close_idx,
            curr_open_idx=curr_open_idx,
            prev_close=prev_close,
            curr_open=curr_open,
            curr_close=curr_close,
        )

    return None


def volume_trend_up(df, window=60, slope: Optional[float] = None):
    if len(df) < window:
        return False
    if slope is None:
        vol = df['volume'].tail(window).values
        x = np.arange(len(vol)).reshape(-1, 1)
        model = LinearRegression().fit(x, vol)
        slope = model.coef_[0]
    print(f" Volume slope: {slope:.2f}")
    return slope > 0

def is_near_breakout(
    df,
    tolerance=0.03,
    recent_high: Optional[float] = None,
    last_close: Optional[float] = None,
):
    if recent_high is None:
        recent_high = df['high'].rolling(20).max().iloc[-2]
    if last_close is None:
        last_close = df['close'].iloc[-1]
    if not recent_high:
        return False
    return abs(last_close - recent_high) / recent_high <= tolerance

def calculate_rr_price_action(df, entry_price):
    from scipy.signal import argrelextrema

    lows = df['low'].values
    highs = df['high'].values
    swing_lows_idx = argrelextrema(lows, np.less, order=5)[0]
    swing_highs_idx = argrelextrema(highs, np.greater, order=5)[0]

    if len(swing_lows_idx) < 1 or len(swing_highs_idx) < 1:
        return None, None, None

    stop = lows[swing_lows_idx[-1]]
    recent_low = lows[swing_lows_idx[-1]]
    recent_high = highs[swing_highs_idx[-1]]
    breakout_height = recent_high - recent_low
    target = entry_price + breakout_height

    risk = entry_price - stop
    reward = target - entry_price

    if risk <= 0 or reward <= 0:
        return None, None, None

    rr = round(risk / reward, 2)
    return stop, target, rr

def log_watchlist(symbol, pattern, entry, rr, stop, target, df):
    path = WATCHLIST_PATH
    volume_3mo = int(df['volume'].tail(60).sum())

    new_entry = [
        symbol,
        round(entry, 2),
        rr,
        round(stop, 2),
        round(target, 2),
        datetime.now().strftime('%m-%d %H:%M'),
        'bullish',
        pattern,
        volume_3mo
    ]

    if path.exists():
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            existing = {row[0]: row for row in reader if row}
    else:
        existing = {}

    existing[symbol] = new_entry

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(WATCHLIST_HEADER)
        writer.writerows(sorted(existing.values(), key=lambda x: x[0]))


def initialize_watchlist():
    """Create or reset the watchlist file with only the header."""
    with open(WATCHLIST_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(WATCHLIST_HEADER)


def _demo_ascending_triangle_detection():
    """Demonstrate ascending triangle detection on synthetic data."""
    highs = [10.0, 10.2, 10.4, 10.6, 11.0, 10.95, 11.0, 11.5]
    lows = [9.5, 9.6, 9.7, 9.85, 9.9, 10.1, 10.2, 10.8]
    closes = [9.8, 10.0, 10.2, 10.5, 10.9, 10.8, 10.95, 11.4]
    volume = [1_000_000] * len(highs)

    df = pd.DataFrame({
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume
    })

    pattern_detected = detect_ascending_triangle(df, window=60, tolerance=0.02, min_touches=2)
    print("Synthetic ascending triangle detected:", pattern_detected)

def scan_all_symbols(symbols):
    initialize_watchlist()
    disqualified = []

    symbols_to_fetch = [s for s in symbols if s.upper() not in EXCLUDED_ETFS]
    data_by_symbol = fetch_symbol_data(symbols_to_fetch)

    for symbol in symbols:
        print(f"\n Scanning {symbol}...")
        if symbol.upper() in EXCLUDED_ETFS:
            print(" Skipped: Excluded ETF")
            continue

        try:
            df = data_by_symbol.get(symbol, pd.DataFrame())
            if df.empty:
                print(" Skipped: Bad or insufficient data")
                disqualified.append({'symbol': symbol, 'reason': 'bad data', 'entry': None, 'rr': None})
                continue

            indicators = compute_precomputed_indicators(df)
            entry = indicators.entry

            if not is_uptrend(df, indicators):
                print(" Skipped: Uptrend not confirmed")
                disqualified.append({'symbol': symbol, 'reason': 'not in uptrend', 'entry': entry, 'rr': None})
                continue

            if not volume_trend_up(df, slope=indicators.volume_slope):
                print(" Skipped: Volume trend is not increasing")
                disqualified.append({'symbol': symbol, 'reason': 'volume not picking up', 'entry': entry, 'rr': None})
                continue

            if not is_near_breakout(df, recent_high=indicators.recent_high_20, last_close=entry):
                print(" Skipped: Not near breakout")
                disqualified.append({'symbol': symbol, 'reason': 'not near breakout', 'entry': entry, 'rr': None})
                continue

            swing_candidate = evaluate_swing_setup(symbol, df, SWING_CONFIG)
            if isinstance(swing_candidate, SwingCandidate):
                print(
                    " Swing setup qualified → "
                    f"Trend Strength: {swing_candidate.trend_strength:.2%}, "
                    f"Momentum: {swing_candidate.momentum_score:.2%}, "
                    f"ATR%: {swing_candidate.atr_pct:.2%}, "
                    f"RSI: {swing_candidate.rsi:.1f}, "
                    f"Pullback: {swing_candidate.pullback_pct:.2%}"
                )
            else:
                print(" Swing setup did not meet the screener criteria")

            double_bottom_hit = detect_double_bottom(df, window=60)
            if double_bottom_hit:
                pattern = "Double Bottom"
                if isinstance(double_bottom_hit, DoubleBottomHit):
                    print(
                        "  Double Bottom details → "
                        f"Support: {double_bottom_hit.support:.2f}, "
                        f"Neckline: {double_bottom_hit.neckline:.2f}, "
                        f"Bounce: {double_bottom_hit.bounce_pct * 100:.1f}%, "
                        f"Touches: {double_bottom_hit.touch_count}, "
                        f"Breakout: {double_bottom_hit.breakout}, "
                        f"Volume Contracted: {double_bottom_hit.volume_contracted}"
                    )
            else:
                pattern = None
                cup_handle_hit = detect_cup_and_handle(df)
                if cup_handle_hit:
                    pattern = "Cup and Handle"
                    if isinstance(cup_handle_hit, CupHandleHit):
                        print(
                            "  Cup and Handle details → "
                            f"Resistance: {cup_handle_hit.resistance:.2f}, "
                            f"Cup Depth: {cup_handle_hit.cup_depth:.2f} ({cup_handle_hit.cup_depth_pct * 100:.1f}%), "
                            f"Handle Length: {cup_handle_hit.handle_length}, "
                            f"Handle Pullback: {cup_handle_hit.handle_pullback_pct * 100:.1f}%, "
                            f"Handle Slope: {cup_handle_hit.handle_slope:.4f}"
                        )
                else:
                    ihs = detect_inverse_head_shoulders(df)
                    if ihs:
                        pattern = "Inverse Head and Shoulders"
                    else:
                        if detect_ascending_triangle(df):
                            pattern = "Ascending Triangle"
                        elif detect_bullish_pennant(df):
                            pattern = "Bullish Pennant"
                        elif detect_bullish_flag(df):
                            pattern = "Bullish Flag"
                        elif detect_bullish_rectangle(df):
                            pattern = "Bullish Rectangle"
                        elif detect_rounding_bottom(df):
                            pattern = "Rounding Bottom"
                        elif detect_breakaway_gap(df):
                            pattern = "Breakaway Gap"

                if not pattern:
                    print(" Skipped: No pattern matched")
                    disqualified.append({'symbol': symbol, 'reason': 'no pattern matched', 'entry': entry, 'rr': None})
                    continue

            stop, target, rr = calculate_rr_price_action(df, entry)
            if rr is None or rr > 0.8:
                print(f" Skipped: RR too high or invalid → RR: {rr}")
                disqualified.append({'symbol': symbol, 'reason': 'rr too high or invalid', 'entry': entry, 'rr': rr})
                continue

            log_watchlist(symbol, pattern, entry, rr, stop, target, df)
            print(f" Match: {pattern} → Entry: {entry:.2f}, RR: {rr}, Stop: {stop:.2f}, Target: {target:.2f}")

        except Exception as e:
            print(f" Error scanning {symbol}: {e}")
            disqualified.append({'symbol': symbol, 'reason': f'error: {e}', 'entry': None, 'rr': None})

    if disqualified:
        pd.DataFrame(disqualified).to_csv('disqualified.csv', index=False)
        print(f"\n Saved disqualified symbols to disqualified.csv")

if __name__ == '__main__':
    _demo_ascending_triangle_detection()
    try:
        with open('filtered_symbols.txt', 'r') as f:
            symbols = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(" filtered_symbols.txt not found.")
        symbols = []
    if symbols:
        scan_all_symbols(symbols)
