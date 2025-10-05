import yfinance as yf
import pandas as pd
import numpy as np
from alpaca_trade_api.rest import REST
import datetime

# Alpaca API credentials
API_KEY = 'PKWMYLAWJCU6ITACV6KP'
API_SECRET = 'k8T9M3XdpVcNQudgPudCfqtkRJ0IUCChFSsKYe07'
BASE_URL = 'https://paper-api.alpaca.markets'

alpaca = REST(API_KEY, API_SECRET, BASE_URL)

# Screener settings
MIN_AVG_VOLUME = 500000
MIN_AVG_PRICE = 5.0
RR_MINIMUM = 2.0

# Get tradable tickers from Alpaca
def get_tradable_tickers(limit=1000):
    assets = alpaca.list_assets(status='active')
    tickers = [a.symbol for a in assets if a.tradable and a.exchange in ['NASDAQ', 'NYSE']]
    return tickers[:limit]

# Download OHLCV data
def fetch_data(ticker, period="2y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return pd.DataFrame()

# Volume profile calculation
def compute_volume_profile(df, bins=50):
    low = float(df['Low'].min())
    high = float(df['High'].max())
    price_range = np.linspace(low, high, bins)
    volume_profile = np.zeros(bins)

    for i in range(len(df)):
        row = df.iloc[i]
        price = (float(row['High']) + float(row['Low'])) / 2
        idx = np.searchsorted(price_range, price) - 1
        if 0 <= idx < bins:
            volume_profile[idx] += float(row['Volume'])

    return volume_profile, price_range

# Support/resistance detection
def detect_support_resistance(df, threshold=3):
    prices = df['Close'].values
    levels = []
    for price in prices:
        if not any(abs(price - l) / l < 0.02 for l in levels):
            count = sum(abs(price - prices) / price < 0.02)
            if count >= threshold:
                levels.append(price)
    if len(levels) >= 2:
        return min(levels), max(levels)
    return None, None

# Main screener
def screen_stocks(tickers):
    qualified = []

    for i, ticker in enumerate(tickers):
        print(f"({i+1}/{len(tickers)}) Screening {ticker}...")
        df = fetch_data(ticker)
        if df.empty or len(df) < 1000:
            continue

        # Exclude penny stocks
        avg_price = float(df['Close'].tail(90).mean())
        if avg_price < MIN_AVG_PRICE:
            continue

        # Volume check
        vol_series = df['Volume'].tail(90)
        avg_vol = float(vol_series.mean())
        if avg_vol < MIN_AVG_VOLUME:
            continue

        # Volume profile flatness
        volume_profile, _ = compute_volume_profile(df)
        vp_mean = np.mean(volume_profile)
        vp_std = np.std(volume_profile)
        if vp_mean == 0 or vp_std / vp_mean < 0.6:
            continue

        # Support/resistance logic
        support, resistance = detect_support_resistance(df)
        if not support or not resistance:
            continue

        entry = float(df['Close'].iloc[-1])
        stop_loss = float(support)
        target_price = float(resistance)
        risk = entry - stop_loss
        reward = target_price - entry

        if risk <= 0 or reward <= 0:
            continue

        rr_ratio = reward / risk
        if rr_ratio < RR_MINIMUM:
            continue

        qualified.append({
            "symbol": ticker,
            "breakout_high": round(entry, 2),
            "rr_ratio": round(rr_ratio, 2),
            "stop_loss": round(stop_loss, 2),
            "target_price": round(target_price, 2)
        })

    return pd.DataFrame(qualified)

# Run
if __name__ == "__main__":
    tickers = get_tradable_tickers(limit=1000)
    if not tickers:
        print("No tickers retrieved. Exiting.")
    else:
        results_df = screen_stocks(tickers)
        if results_df.empty:
            print("No stocks met the criteria.")
        else:
            print("\n✅ Qualified Stocks:\n")
            print(results_df.to_string(index=False))
            results_df.to_csv("qualified_stocks.csv", index=False)
