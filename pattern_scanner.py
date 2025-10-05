import alpaca_trade_api as tradeapi
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import os
import csv
import time
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks, argrelextrema
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

API_KEY = 'PKWMYLAWJCU6ITACV6KP'
API_SECRET = 'k8T9M3XdpVcNQudgPudCfqtkRJ0IUCChFSsKYe07'
BASE_URL = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

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
        return df[['Date', 'open', 'high', 'low', 'close', 'volume']]
    except Exception:
        return pd.DataFrame()

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

def is_uptrend(df):
    if len(df) < 200:
        print(f" Not enough data for trend check: {len(df)} rows")
        return False
    ma50 = df['close'].rolling(50).mean().iloc[-1]
    ma200 = df['close'].rolling(200).mean().iloc[-1]
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

def detect_double_bottom(df, window=120, tolerance=0.03, min_bounce=0.03):
    """Detect a double bottom pattern in the recent price action."""
    recent = df.tail(window)
    lows = recent['low'].values
    highs = recent['high'].values

    # Identify local minima (potential bottoms)
    troughs, _ = find_peaks(-lows, distance=5)
    if len(troughs) < 2:
        return False

    # Use the last two troughs for the pattern
    left_idx, right_idx = troughs[-2], troughs[-1]
    if right_idx - left_idx < 5:
        return False

    left_low, right_low = lows[left_idx], lows[right_idx]

    # Bottoms should be within the tolerance band
    if abs(left_low - right_low) / max(left_low, right_low) > tolerance:
        return False

    # Ensure there is a meaningful bounce between the two lows
    mid_high = highs[left_idx:right_idx + 1].max()
    if (mid_high - min(left_low, right_low)) / min(left_low, right_low) < min_bounce:
        return False

    return True

def detect_inverse_head_shoulders(df):
    lows = df['low'].tail(60).values
    highs = df['high'].tail(60).values
    closes = df['close'].tail(60).values
    indexes = np.arange(len(lows))
    
    # Find potential troughs (local minima)
    troughs, _ = find_peaks(-lows, distance=5, prominence=0.5)
    if len(troughs) < 3:
        return False

    for i in range(len(troughs) - 2):
        l_idx = troughs[i]
        h_idx = troughs[i + 1]
        r_idx = troughs[i + 2]

        l = lows[l_idx]
        h = lows[h_idx]
        r = lows[r_idx]

        # Head must be lowest, shoulders similar height
        if not (h < l and h < r):
            continue
        if abs(l - r) / max(l, r) > 0.1:
            continue

        # Neckline from high between LS and head to high between head and RS
        left_high = max(highs[l_idx:h_idx])
        right_high = max(highs[h_idx:r_idx])
        neckline = (left_high + right_high) / 2

        # Breakout condition (optional)
        if closes[-1] > neckline:
            return True

    return False

def detect_ascending_triangle(df, window=60, tolerance=0.02, min_touches=2):
    highs = df['high'].tail(window).values
    lows = df['low'].tail(window).values
    closes = df['close'].tail(window).values
    x = np.arange(len(lows)).reshape(-1, 1)

    if len(highs) < min_touches + 2:
        return False

    # Step 1: Identify flat resistance level using highs prior to the most recent candles
    plateau_end = len(highs) - 2 if len(highs) > 2 else len(highs)
    if plateau_end <= 0:
        return False

    plateau_span = min(20, plateau_end)
    plateau_start = plateau_end - plateau_span
    plateau_highs = highs[plateau_start:plateau_end]

    if plateau_highs.size == 0:
        return False

    resistance = plateau_highs.max()
    if resistance == 0:
        return False

    resistance_touches = [idx for idx in range(plateau_start, plateau_end)
                          if abs(highs[idx] - resistance) / resistance < tolerance]

    if len(resistance_touches) < min_touches:
        return False

    # Step 2: Fit ascending trendline to lows
    lr = LinearRegression().fit(x, lows)
    support_slope = lr.coef_[0]
    if support_slope <= 0:
        return False

    # Step 3: Check how many points lie near the support trendline
    fitted_support = lr.predict(x)
    support_touches = np.sum(np.abs(lows - fitted_support) / lows < tolerance)
    
    if support_touches < min_touches:
        return False

    # Step 4 (optional): Check for breakout relative to the plateau resistance
    breakout = (closes[-1] > resistance * (1 + tolerance)) or \
               (highs[-1] > resistance * (1 + tolerance))

    return breakout or True  # Return True even if no breakout yet


def detect_bullish_pennant(df, window=60, flagpole_window=20, tolerance=0.03):
    if len(df) < window + flagpole_window:
        return False

    # Step 1: Check flagpole strength
    flagpole = df.iloc[-(window + flagpole_window):-window]
    pole_gain = flagpole['close'].iloc[-1] / flagpole['close'].iloc[0]
    if pole_gain < 1.2:
        return False

    # Step 2: Analyze pennant body
    recent = df.tail(window)
    highs = recent['high'].values
    lows = recent['low'].values
    x = np.arange(len(highs)).reshape(-1, 1)

    high_slope = LinearRegression().fit(x, highs).coef_[0]
    low_slope = LinearRegression().fit(x, lows).coef_[0]

    # Step 3: Confirm convergence
    if high_slope >= 0 or low_slope <= 0:
        return False
    if abs(high_slope - low_slope) < 0.01:
        return False

    return True

def detect_bullish_flag(df, window=40, flagpole_window=20, slope_threshold=-0.1):
    if len(df) < window + flagpole_window:
        return False

    flagpole = df.iloc[-(window + flagpole_window):-window]
    pole_gain = flagpole['close'].iloc[-1] / flagpole['close'].iloc[0]
    if pole_gain < 1.2:
        return False

    recent = df.tail(window)
    x = np.arange(len(recent)).reshape(-1, 1)
    y = recent['close'].values
    model = LinearRegression().fit(x, y)
    slope = model.coef_[0]

    return slope_threshold < slope < 0  # Channel should slope slightly downward
def detect_bullish_rectangle(df, window=60, tolerance=0.02, min_touches=2):
    if len(df) < window:
        return False

    recent = df.tail(window)
    high = recent['high'].max()
    low = recent['low'].min()

    highs_touch = sum(abs(row - high) / high < tolerance for row in recent['high'])
    lows_touch = sum(abs(row - low) / low < tolerance for row in recent['low'])

    if highs_touch < min_touches or lows_touch < min_touches:
        return False

    # Optional: Check breakout above resistance
    if recent['close'].iloc[-1] > high * (1 + tolerance):
        return True

    return True

def detect_cup_and_handle(df, cup_window=60, handle_window=15, tolerance=0.1):
    if len(df) < cup_window + handle_window:
        return False

    cup = df['close'].iloc[-(cup_window + handle_window):-handle_window].values
    handle = df['close'].tail(handle_window).values
    midpoint = len(cup) // 2
    left = cup[:midpoint]
    right = cup[midpoint:]

    # 1. Cup shape: left & right similar, U-shaped dip
    min_cup = np.min(cup)
    left_peak = np.max(left)
    right_peak = np.max(right)

    # Left and right sides should recover close to each other
    if abs(left_peak - right_peak) / max(left_peak, right_peak) > tolerance:
        return False

    # Cup should dip significantly
    if (left_peak - min_cup) / left_peak < 0.1 or (right_peak - min_cup) / right_peak < 0.1:
        return False

    # 2. Handle: slight downward drift after right cup
    x = np.arange(len(handle)).reshape(-1, 1)
    model = LinearRegression().fit(x, handle)
    handle_slope = model.coef_[0]

    # Slope should be slightly negative (falling handle)
    if not (-0.2 < handle_slope < 0):
        return False

    return True

def detect_rounding_bottom(df, window=100, tolerance=0.02):
    if len(df) < window:
        return False

    prices = df['close'].tail(window).values
    x = np.arange(window)
    poly = np.polyfit(x, prices, 2)
    a = poly[0]

    # Must be a U-shape (concave up)
    if a <= 0:
        return False

    # Optional: check dip and recovery
    mid = window // 2
    left = np.mean(prices[:mid//2])
    bottom = np.min(prices)
    right = np.mean(prices[-mid//2:])

    # Require a dip and partial recovery
    if (left - bottom) / left < 0.05 or (right - bottom) / right < 0.05:
        return False

    return True


def detect_breakaway_gap(df, gap_threshold=0.03, volume_multiplier=1.5):
    if len(df) < 35:
        return False

    prev_close = df['close'].iloc[-2]
    curr_open = df['open'].iloc[-1]
    curr_close = df['close'].iloc[-1]
    prev_vol = df['volume'].iloc[-31:-1].mean()
    curr_vol = df['volume'].iloc[-1]

    gap = (curr_open - prev_close) / prev_close
    strong_gap = gap > gap_threshold
    volume_spike = curr_vol > prev_vol * volume_multiplier
    full_gap_fill = curr_close > curr_open  # Strong close after open

    return strong_gap and volume_spike and full_gap_fill


def volume_trend_up(df, window=60):
    if len(df) < window:
        return False
    vol = df['volume'].tail(window).values
    x = np.arange(len(vol)).reshape(-1, 1)
    model = LinearRegression().fit(x, vol)
    slope = model.coef_[0]
    print(f" Volume slope: {slope:.2f}")
    return slope > 0

def is_near_breakout(df, tolerance=0.03):
    recent_high = df['high'].rolling(20).max().iloc[-2]
    last_close = df['close'].iloc[-1]
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
    path = 'watchlist.csv'
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

    header = [
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

    if os.path.exists(path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            existing = {row[0]: row for row in reader if row}
    else:
        existing = {}

    existing[symbol] = new_entry

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sorted(existing.values(), key=lambda x: x[0]))


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

    pattern_detected = detect_ascending_triangle(df, window=len(df), tolerance=0.02, min_touches=2)
    print("Synthetic ascending triangle detected:", pattern_detected)

def scan_all_symbols(symbols):
    disqualified = []
    for symbol in symbols:
        print(f"\n Scanning {symbol}...")
        if symbol.upper() in EXCLUDED_ETFS:
            print(" Skipped: Excluded ETF")
            continue
        try:
            df = get_yf_data(symbol)
            entry = df['close'].iloc[-1]

            if detect_double_bottom(df):
                pattern = "Double Bottom"
            elif detect_inverse_head_shoulders(df):
                pattern = "Inverse Head and Shoulders"
            elif detect_ascending_triangle(df):
                pattern = "Ascending Triangle"
            elif detect_bullish_pennant(df):
                pattern = "Bullish Pennant"
            elif detect_bullish_flag(df):
                pattern = "Bullish Flag"
            elif detect_bullish_rectangle(df):
                pattern = "Bullish Rectangle"
            elif detect_cup_and_handle(df):
                pattern = "Cup and Handle"
            elif detect_rounding_bottom(df):
                pattern = "Rounding Bottom"
            elif detect_breakaway_gap(df):
                pattern = "Breakaway Gap"
            else:
                print(" Skipped: No pattern matched")
                disqualified.append({'symbol': symbol, 'reason': 'no pattern matched', 'entry': entry, 'rr': None})
                continue

            if df.empty:
                print(" Skipped: Bad or insufficient data")
                disqualified.append({'symbol': symbol, 'reason': 'bad data', 'entry': None, 'rr': None})
                continue

            if not is_uptrend(df):
                print(" Skipped: Uptrend not confirmed")
                disqualified.append({'symbol': symbol, 'reason': 'not in uptrend', 'entry': None, 'rr': None})
                continue

            if not volume_trend_up(df):
                print(" Skipped: Volume trend is not increasing")
                disqualified.append({'symbol': symbol, 'reason': 'volume not picking up', 'entry': None, 'rr': None})
                continue

            if not is_near_breakout(df):
                print(" Skipped: Not near breakout")
                disqualified.append({'symbol': symbol, 'reason': 'not near breakout', 'entry': None, 'rr': None})
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
