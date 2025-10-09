# Auto Trade Swing Trading Bot

This repository contains a swing trading helper that turns the raw output
from `swing_trading_screener.py` into actionable trading instructions.
The core logic lives in [`trading_bot.py`](trading_bot.py) where the
`SwingTradingBot` converts screener candidates into a structured trade
plan (entry, stop and target) and filters opportunities that do not meet
your minimum reward-to-risk threshold.

## Installation

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

2. Install the required Python dependencies:

   ```bash
   pip install -U pip
   pip install pandas numpy
   ```

   The trading bot itself only relies on `pandas`, but the test suite
   also uses `numpy`.

## Configuring Alpaca credentials

The optional desktop GUI in [`live_trading_gui.py`](live_trading_gui.py) reads
your Alpaca API credentials from environment variables.  For convenience it
also loads a local `.env` file from the project root if present.  Create the
file with the following contents (replace the placeholders with your own
credentials):

```text
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
```

When the GUI launches it will populate the environment from this file, so you
do not need to export the variables manually.

## Getting market data

The bot operates on historical OHLCV data supplied as a `pandas`
`DataFrame`. Each index entry should represent a trading session. A
minimal DataFrame must include the following columns:

- `open`
- `high`
- `low`
- `close`
- `volume`

You can obtain such data from a brokerage API, CSV exports, or any other
market data provider. Make sure all price values are floats and the
index is sorted in ascending chronological order.

## Using `SwingTradingBot`

1. Configure the screener parameters (optional). The defaults from
   `SwingScreenerConfig` are sensible for most swing strategies, but you
   can override them to match your preferences.

2. Instantiate the bot and pass in any custom configuration or risk
   parameters. The defaults enforce a minimum 1.2 reward-to-risk ratio
   and use 1× ATR for the stop with 2× ATR for the target.

3. Feed the bot a mapping of ticker symbols to their corresponding data
   frames. The bot will evaluate each symbol, filter out any setups that
   do not meet the reward-to-risk requirement, and emit both a
   `TradePlan` and concrete `TradeSignal` instructions.

```python
import pandas as pd

from trading_bot import SwingTradingBot
from swing_trading_screener import SwingScreenerConfig

# Load your own OHLCV data. Here we assume a CSV per symbol.
spy = pd.read_csv("data/SPY.csv", parse_dates=["date"], index_col="date")
tesla = pd.read_csv("data/TSLA.csv", parse_dates=["date"], index_col="date")

data = {"SPY": spy, "TSLA": tesla}

# Optional: tighten the screener filters
config = SwingScreenerConfig(min_average_volume=500_000, max_atr_pct=0.05)

bot = SwingTradingBot(
    screener_config=config,
    min_rr=1.3,             # require at least 1.3 reward-to-risk
    atr_stop_multiple=1.0,  # stop one ATR below the entry
    atr_target_multiple=2.5 # target 2.5 ATR above the entry
)

plans = bot.generate_trade_plans(data)
signals = bot.generate_trade_signals(data)

for plan in plans:
    print(
        f"{plan.symbol}: buy {plan.entry:.2f}, stop {plan.stop:.2f}, "
        f"target {plan.target:.2f} (RR={plan.risk_reward:.2f})"
    )

for signal in signals:
    print(signal)
```

`generate_trade_plans` returns structured `TradePlan` objects. You can
store them, rank them further, or feed them into your execution engine.
`generate_trade_signals` provides explicit buy/stop/target instructions
that you can forward to a broker integration.

## Running the tests

The repository includes a small pytest suite in `tests/test_trading_bot.py`.
Run it with:

```bash
pytest
```

The tests require both `pandas` and `numpy`. Install them via pip if you
have not already done so.

## Next steps

The bot does not connect to a brokerage or fetch live data. Integrate it
with your preferred data ingestion and order execution layers to build a
complete automated trading workflow.
