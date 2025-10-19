# Auto Trade Swing Trading Toolkit

This repository contains a collection of lightweight swing trading
utilities together with an optional desktop GUI. The primary analytical
component lives in [`swing_trading_screener.py`](swing_trading_screener.py)
which evaluates OHLCV data and highlights promising swing candidates.
[`live_trading_gui.py`](live_trading_gui.py) provides a ready-to-run
interface that periodically downloads data from Alpaca and presents the
resulting trade ideas and manual trade log in a friendly dashboard.

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

   The utilities themselves rely on `pandas`, while the broader test
   suite also uses `numpy`.

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

## Working with the screener

1. Configure the screener parameters (optional). The defaults from
   `SwingScreenerConfig` are sensible for most swing strategies, but you
   can override them to match your preferences.

2. Pass a `dict` mapping ticker symbols to `pandas.DataFrame` instances
   containing the OHLCV data. The `screen_swing_candidates` helper will
   return the strongest setups first.

```python
import pandas as pd

from swing_trading_screener import SwingScreenerConfig, screen_swing_candidates

# Load your own OHLCV data. Here we assume a CSV per symbol.
spy = pd.read_csv("data/SPY.csv", parse_dates=["date"], index_col="date")
tesla = pd.read_csv("data/TSLA.csv", parse_dates=["date"], index_col="date")

data = {"SPY": spy, "TSLA": tesla}

# Optional: tighten the screener filters
config = SwingScreenerConfig(min_average_volume=500_000, max_atr_pct=0.05)

candidates = screen_swing_candidates(data, config=config)

for candidate in candidates:
    print(
        f"{candidate.symbol}: close {candidate.close:.2f}, ATR% {candidate.atr_pct:.2%}, "
        f"RSI {candidate.rsi:.1f}"
    )
```

The GUI includes an embedded planner that mirrors the previous trading
bot behaviour, translating each candidate into entry, stop and target
levels before displaying them in the log.

## Running the tests

The repository includes a small pytest suite covering the screeners and
GUI helpers. Run it with:

```bash
pytest
```

The tests require both `pandas` and `numpy`. Install them via pip if you
have not already done so.

## Next steps

The bot does not connect to a brokerage or fetch live data. Integrate it
with your preferred data ingestion and order execution layers to build a
complete automated trading workflow.
