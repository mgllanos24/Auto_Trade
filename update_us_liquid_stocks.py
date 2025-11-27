import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

# =========================
# CONFIG
# =========================
YEARS_BACK = 2  # how many years of history
MIN_PRICE = 5.0  # non-penny threshold
MIN_DOLLAR_VOL = 5_000_000  # avg 20d dollar volume in dollars
OUT_FILE = "us_liquid_stocks_ohlcv_last2y.csv"

# =========================
# EMBEDDED TICKER LIST
# Mostly large / liquid US names (S&P 500 style universe).
# =========================
TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "GOOG",
    "META",
    "NVDA",
    "BRK-B",
    "LLY",
    "JPM",
    "XOM",
    "UNH",
    "AVGO",
    "JNJ",
    "V",
    "PG",
    "HD",
    "MA",
    "CVX",
    "MRK",
    "PEP",
    "ABBV",
    "COST",
    "ADBE",
    "KO",
    "WMT",
    "CRM",
    "ACN",
    "TMO",
    "CSCO",
    "MCD",
    "DHR",
    "LIN",
    "ABT",
    "NFLX",
    "INTU",
    "TXN",
    "AMD",
    "WFC",
    "MS",
    "HON",
    "IBM",
    "PM",
    "AMGN",
    "ORCL",
    "GE",
    "RTX",
    "CAT",
    "LOW",
    "SPGI",
    "GS",
    "LMT",
    "MDT",
    "UNP",
    "INTC",
    "AMT",
    "BKNG",
    "ISRG",
    "SYK",
    "PLD",
    "ADP",
    "AXP",
    "TJX",
    "ELV",
    "CI",
    "BLK",
    "EQIX",
    "GILD",
    "NOW",
    "T",
    "DE",
    "REGN",
    "MO",
    "BDX",
    "PFE",
    "MMC",
    "C",
    "CB",
    "ADI",
    "VRTX",
    "PANW",
    "CL",
    "ZTS",
    "DUK",
    "SO",
    "CSX",
    "NKE",
    "ICE",
    "EOG",
    "NSC",
    "PGR",
    "SHW",
    "BSX",
    "MU",
    "HCA",
    "ITW",
    "EW",
    "ETN",
    "APH",
    "FISV",
    "FDX",
    "COP",
    "MAR",
    "USB",
    "MPC",
    "GM",
    "F",
    "OXY",
    "KDP",
    "BK",
    "PBR",
    "HPQ",
    "DAL",
    "LULU",
    "ROST",
    "KR",
    "AEP",
    "SBUX",
    "PSX",
    "HUM",
    "CMCSA",
    "KLAC",
    "AON",
    "ROP",
    "AZO",
    "TRV",
    "CTAS",
    "MNST",
    "ADSK",
    "AIG",
    "PRU",
    "MET",
    "ALL",
    "GD",
    "APD",
    "CME",
    "AFL",
    "HES",
    "LRCX",
    "NXPI",
    "MCO",
    "MSI",
    "IDXX",
    "CDNS",
    "FTNT",
    "SNPS",
    "FICO",
    "ODFL",
    "PAYX",
    "SRE",
    "KMB",
    "KHC",
    "GIS",
    "MDLZ",
    "DLTR",
    "DG",
    "EBAY",
    "TGT",
    "WBA",
    "EL",
    "HLT",
    "HSY",
    "YUM",
    "CMG",
    "DHI",
    "LEN",
    "PHM",
    "NEM",
    "AEM",
    "FCX",
    "NUE",
    "STLD",
    "PSA",
    "O",
    "SPG",
    "VTR",
    "WELL",
    "EXR",
    "CCI",
    "DLR",
    "SBAC",
    "CSGP",
    "TRGP",
    "KMI",
    "WMB",
    "OKE",
    "ENB",
    "NEE",
    "D",
    "AEE",
    "ED",
    "EIX",
    "ES",
    "XEL",
    "PEG",
    "FE",
    "EXC",
    "PCG",
    "HIG",
    "DFS",
    "COF",
    "SYF",
    "KEY",
    "RF",
    "FITB",
    "PNC",
    "TFC",
    "STT",
    "SCHW",
    "NTRS",
    "BEN",
    "AMP",
    "RJF",
    "MTB",
    "HBAN",
    "CFG",
    "KEYS",
    "ANET",
    "ZS",
    "CRWD",
    "DDOG",
    "NET",
    "OKTA",
    "MDB",
    "TEAM",
    "SHOP",
    "SQ",
    "PYPL",
    "IONQ",
    "TSLA",
    "BABA",
    "ARM",
    "SNOW",
    "UBER",
    "PATH",
    "AI",
    "PLTR",
    "S",
    "RGTI",
    "QUBT",
    "QSI",
    "MELI",
    "ILMN",
    "NTLA",
    "MRVL",
    "VZ",
    "QCOM",
    "APA",
    "HPE",
    "ORLY",
    "TSCO",
    "CMI",
    "PCAR",
    "ROK",
    "EMR",
    "AME",
    "FAST",
    "GWW",
    "TT",
    "CARR",
    "IR",
    "ETR",
    "AWK",
    "WEC",
    "CMS",
    "DTE",
    "ROKU",
    "DBX",
    "BIDU",
    "TWLO",
    "ZI",
    "HUBS",
    "ESTC",
    "DUOL",
    "ASML",
    "ON",
    "STM",
    "UMC",
    "AMKR",
    "WOLF",
    "LSCC",
    "BTC-USD",
    "SOL-USD",
    "XRP-USD",
    "SHIB-USD",
    "DOGE-USD",
]

TICKERS = sorted(set(TICKERS))

COLS_TO_KEEP = ["date", "ticker", "open", "high", "low", "close", "volume"]


def _reshape_to_long(raw: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Convert yfinance MultiIndex columns into a long-form DataFrame."""
    rows: List[pd.DataFrame] = []
    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            print(f"Warning: no data for {ticker}, skipping.")
            continue

        df_t = raw[ticker].copy()
        df_t["ticker"] = ticker
        df_t = df_t.rename(columns=str.lower).reset_index().rename(columns={"Date": "date"})
        rows.append(df_t)

    if not rows:
        return pd.DataFrame(columns=COLS_TO_KEEP)

    df = pd.concat(rows, ignore_index=True)
    missing_cols = [c for c in COLS_TO_KEEP if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in data: {missing_cols}")

    return df[COLS_TO_KEEP].copy()


def _filter_liquid(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to liquid tickers using the latest 20 days of data per ticker."""
    if df.empty:
        return df

    df_sorted = df.sort_values(["ticker", "date"])
    df_sorted["dollar_volume"] = df_sorted["close"] * df_sorted["volume"]

    latest_20 = df_sorted.groupby("ticker").tail(20)
    counts = latest_20.groupby("ticker")["date"].count()
    enough_days = counts[counts >= 10].index
    latest_20 = latest_20[latest_20["ticker"].isin(enough_days)]

    stats = latest_20.groupby("ticker").agg(
        last_close=("close", "last"),
        avg20_dollar_vol=("dollar_volume", "mean"),
    )

    filtered_tickers = stats[
        (stats["last_close"] >= MIN_PRICE) & (stats["avg20_dollar_vol"] >= MIN_DOLLAR_VOL)
    ].index

    return df[df["ticker"].isin(filtered_tickers)].copy()


def _load_existing(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"No existing file found at {path}, starting fresh.")
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    missing_cols = [c for c in COLS_TO_KEEP if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Existing file missing expected columns: {missing_cols}")

    return df[COLS_TO_KEEP].copy()


def _determine_date_range(existing: Optional[pd.DataFrame]):
    end_dt = datetime.today()
    earliest_allowed = end_dt - timedelta(days=365 * YEARS_BACK)
    start_dt = earliest_allowed

    if existing is not None and not existing.empty:
        last_date = existing["date"].max()
        if pd.isna(last_date):
            raise ValueError("Existing data contains invalid dates.")
        start_dt = max(last_date + timedelta(days=1), earliest_allowed)

    return start_dt, end_dt, earliest_allowed


def _download_new_data(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    if start_dt >= end_dt:
        print("Dataset is already up to date; no download needed.")
        return pd.DataFrame(columns=COLS_TO_KEEP)

    print(f"Downloading data from {start_dt.date()} to {end_dt.date()} for {len(TICKERS)} tickers.")
    raw = yf.download(
        tickers=TICKERS,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError("No data returned from yfinance. Check internet connection / tickers.")

    return _reshape_to_long(raw, TICKERS)


def main():
    out_path = Path(OUT_FILE)
    existing = _load_existing(out_path)

    start_dt, end_dt, earliest_allowed = _determine_date_range(existing)
    new_data = _download_new_data(start_dt, end_dt)

    if existing is None:
        combined = new_data
    elif new_data.empty:
        combined = existing
    else:
        combined = pd.concat([existing, new_data], ignore_index=True)

    if combined.empty:
        raise ValueError("No data available after combining existing and new downloads.")

    combined = combined.drop_duplicates(subset=["date", "ticker"]).sort_values(["ticker", "date"])
    combined = combined[combined["date"] >= earliest_allowed]

    filtered = _filter_liquid(combined)
    filtered.to_csv(out_path, index=False)

    print(
        f"Saved filtered OHLCV dataset (last {YEARS_BACK} years) with {len(filtered)} rows for "
        f"{filtered['ticker'].nunique()} tickers to: {out_path}"
    )


if __name__ == "__main__":
    main()
