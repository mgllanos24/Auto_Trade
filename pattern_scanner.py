import argparse
import pandas as pd
import json
import yfinance as yf
import requests
import os
import csv
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from scipy.signal import find_peaks, argrelextrema
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, is_dataclass
from collections.abc import Sequence as ABCSequence
from typing import Any, Dict, List, Optional, Sequence, Union, get_args, get_origin

from cup_handle_scanner import CupHandleHit, detect_cup_and_handle
from swing_trading_screener import (
    SwingCandidate,
    SwingScreenerConfig,
    evaluate_swing_setup,
)
from ai_reversal_screener import screen_symbols as run_reversal_screen

SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_data_dir() -> Path:
    """Return the directory used to persist shared scanner state.

    The original implementation attempted to write ``watchlist.csv`` alongside
    the sources which works when the repository is writable.  When users install
    the package into a protected location (e.g. ``Program Files`` on Windows)
    the process no longer has permission to create or modify that file and the
    scanner crashes on start-up.  By default we now store the watchlist inside a
    user-scoped application directory (``~/.auto_trade``) which is writable for
    regular users.  The location can still be customised through environment
    variables for advanced setups.
    """

    env_path = os.environ.get("AUTO_TRADE_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".auto_trade"


def _resolve_master_csv_path(data_dir: Path) -> Path:
    env_path = os.environ.get("AUTO_TRADE_MASTER_CSV")
    if env_path:
        return Path(env_path).expanduser()

    packaged_copy = SCRIPT_DIR / "us_liquid_stocks_ohlcv_last2y.csv"
    if packaged_copy.exists():
        return packaged_copy

    return data_dir / "us_liquid_stocks_ohlcv_last2y.csv"


DATA_DIR = _resolve_data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = DATA_DIR / "yf_cache"
CACHE_TTL = timedelta(minutes=30)
PATTERN_DETAILS_DIR = DATA_DIR / "pattern_details"

MASTER_CSV_PATH = _resolve_master_csv_path(DATA_DIR)

try:
    MASTER_CSV_LOOKBACK_DAYS = int(
        os.environ.get("AUTO_TRADE_MASTER_LOOKBACK_DAYS", "0")
    )
except ValueError:
    MASTER_CSV_LOOKBACK_DAYS = 0

_MASTER_CSV_REQUIRED_COLUMNS = (
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
_MASTER_CSV_OPTIONAL_COLUMNS = ("adj_close",)
_MASTER_CSV_CACHE: Optional[pd.DataFrame] = None
_MASTER_CSV_FAILED = False
_MASTER_CSV_MTIME: Optional[float] = None

ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")
TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY")


_PD_TIMESTAMP = getattr(pd, "Timestamp", None)


def _resolve_watchlist_path() -> Path:
    env_path = os.environ.get("AUTO_TRADE_WATCHLIST")
    if env_path:
        return Path(env_path).expanduser()
    return DATA_DIR / "watchlist.csv"


WATCHLIST_PATH = _resolve_watchlist_path()
WATCHLIST_HEADER = [
    'symbol',
    'last_close',
    'breakout_high',
    'rr_ratio',
    'stop_loss',
    'target_price',
    'direction',
    'pattern',
    '3mo_volume'
]

SWING_CONFIG = SwingScreenerConfig()

REVERSAL_START_DATE = "2015-01-01"
REVERSAL_INTERMARKET = ["SPY", "DXY", "^VIX"]


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
    breakout: bool


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


@dataclass(frozen=True)
class RiskRewardLevels:
    """Container for breakout, stop and target levels."""

    breakout: float
    stop: float
    target: float
    rr_ratio: float


@dataclass
class PatternCandidate:
    """Normalized representation of a detected chart pattern."""

    name: str
    confidence: float
    details: Any = None

# ---------------------------------------------------------------------------
# Pattern detail persistence helpers


def _pattern_key(name: str) -> str:
    normalized = " ".join(str(name or "").strip().lower().split())
    return normalized


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, (float, int, bool)) or value is None:
        return value
    if isinstance(value, str):
        return value
    if isinstance(_PD_TIMESTAMP, type) and isinstance(value, _PD_TIMESTAMP):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(item) for item in value]
    if is_dataclass(value):
        return _normalize_for_json(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        try:
            return [_normalize_for_json(item) for item in value.tolist()]
        except Exception:  # pragma: no cover - defensive
            return value
    return value


def _ensure_pattern_details_dir() -> None:
    PATTERN_DETAILS_DIR.mkdir(parents=True, exist_ok=True)


def _load_pattern_file(symbol: str) -> Dict[str, Any]:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return {}
    path = PATTERN_DETAILS_DIR / f"{normalized_symbol}.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:  # pragma: no cover - defensive
        return {}
    return {}


def _write_pattern_file(symbol: str, data: Dict[str, Any]) -> None:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return
    _ensure_pattern_details_dir()
    path = PATTERN_DETAILS_DIR / f"{normalized_symbol}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def save_pattern_details(symbol: str, pattern_name: str, details: Any) -> None:
    if not pattern_name:
        return
    if details is None:
        return

    if is_dataclass(details):
        payload_details = asdict(details)
    elif isinstance(details, dict):
        payload_details = dict(details)
    else:
        # Fall back to the object's dictionary if available
        payload_details = getattr(details, "__dict__", None)
        if not isinstance(payload_details, dict):
            return

    normalized_details = _normalize_for_json(payload_details)
    store = _load_pattern_file(symbol)
    key = _pattern_key(pattern_name)
    store[key] = {
        "pattern": pattern_name,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "details": normalized_details,
    }
    _write_pattern_file(symbol, store)


def load_pattern_details(symbol: str, pattern_name: str) -> Optional[Dict[str, Any]]:
    if not pattern_name:
        return None
    store = _load_pattern_file(symbol)
    if not store:
        return None
    key = _pattern_key(pattern_name)
    entry = store.get(key)
    if entry:
        return entry
    for candidate in store.values():
        if not isinstance(candidate, dict):
            continue
        if _pattern_key(candidate.get("pattern", "")) == key:
            return candidate
    return None


def _convert_field_value(field_type: Any, value: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(field_type)
    if origin in (list, tuple, set):
        item_type = get_args(field_type)[0] if get_args(field_type) else Any
        converted = [_convert_field_value(item_type, item) for item in value]
        if origin is tuple:
            return tuple(converted)
        if origin is set:
            return set(converted)
        return converted
    if origin is ABCSequence:
        item_type = get_args(field_type)[0] if get_args(field_type) else Any
        return [_convert_field_value(item_type, item) for item in value]
    if origin is Union:
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        if len(args) == 1:
            return _convert_field_value(args[0], value)
        return value

    if field_type is float:
        return float(value)
    if field_type is int:
        return int(value)
    if field_type is bool:
        return bool(value)
    if field_type is str:
        return str(value)
    if field_type is datetime:
        return datetime.fromisoformat(value)
    if isinstance(_PD_TIMESTAMP, type) and field_type is _PD_TIMESTAMP:
        return pd.Timestamp(value)

    return value


def _dataclass_from_dict(cls: Any, data: Dict[str, Any]):
    if not isinstance(data, dict):
        return None
    try:
        field_values = {}
        for field in getattr(cls, "__dataclass_fields__", {}).values():
            field_name = field.name
            raw_value = data.get(field_name)
            field_values[field_name] = _convert_field_value(field.type, raw_value)
        return cls(**field_values)
    except Exception:  # pragma: no cover - defensive
        return None


def load_pattern_dataclass(symbol: str, pattern_name: str, cls: Any):
    entry = load_pattern_details(symbol, pattern_name)
    if not entry or not isinstance(entry, dict):
        return None
    details = entry.get("details")
    if not isinstance(details, dict):
        return None
    return _dataclass_from_dict(cls, details)

# Excluded ETFs
EXCLUDED_ETFS = ['VTIP', 'NFXS', 'ACWX', 'VXUS', 'NVD', 'NVDD', 'NVDL', 'TBIL', 'VRIG', 'CONL', 'PDBC', 'PFF',
    'EMB', 'EMXC', 'ESGE', 'ETHA', 'TLT', 'EUFN', 'FDNI', 'TQQQ', 'QQQ', 'QQQM', 'QYLD', 'TSDD',
    'TSLL', 'TSLQ', 'TSLR', 'RDVY', 'TSLS', 'IBIT', 'ICLN', 'IEF', 'IGF', 'IGIB', 'IGSB', 'ACWI',
    'ISTB', 'IUSB', 'SCZ', 'IXUS', 'JEPQ', 'USIG', 'BSCQ', 'SHV', 'SHY', 'VCIT', 'VCLT', 'VCSH',
    'VGIT', 'VGLT', 'SMH', 'VGSH', 'BNDX', 'BND', 'MBB', 'MCHI', 'AAPU', 'METU', 'VMBS', 'SOXX', 'SQQQ']

def _flatten_columns(columns) -> list[str]:
    """Return a 1-D list of column labels from a MultiIndex.

    Yahoo Finance occasionally returns price data with a two-level column
    index.  Depending on the ``group_by`` behaviour of ``yf.download`` the
    price field names (Open, High, Low, Close, Volume) may be found on either
    level of that index.  The previous implementation always selected the
    first
    level which breaks when the first level contains the ticker symbol for
    every column (resulting in duplicate column names such as ``['NE', 'NE', …]``).
    Later calls to :func:`pandas.to_numeric` expect a Series but instead
    receive a DataFrame whenever duplicate column names exist, triggering the
    user-facing ``"arg must be a list, tuple, 1-d array, or Series"`` error.

    This helper inspects every level and prefers the one that actually
    contains the OHLCV labels.  When no such level is present we gracefully
    fall back to joining the tuple entries, ensuring we always return a simple
    list of strings.
    """

    expected_labels = {"open", "high", "low", "close", "volume"}
    get_level = getattr(columns, "get_level_values", None)
    nlevels = getattr(columns, "nlevels", 1)

    if callable(get_level):
        for level in range(nlevels):
            level_values = list(get_level(level))
            normalized = {str(value).lower() for value in level_values}
            if expected_labels.issubset(normalized):
                # Use this level directly – it already contains the OHLCV
                # field names we need for downstream processing.
                return [str(value) for value in level_values]

    flattened: list[str] = []
    for column in columns:
        if isinstance(column, tuple):
            parts = [str(part) for part in column if part not in (None, "")]
            # Try to pick the first tuple component that looks like an OHLCV
            # label so that we keep familiar column names whenever possible.
            chosen = next(
                (
                    part
                    for part in parts
                    if isinstance(part, str)
                    and part.lower() in expected_labels
                ),
                None,
            )
            flattened.append(chosen if chosen is not None else "_".join(parts))
        else:
            flattened.append(str(column))

    return flattened


def flatten_yf_columns(df):
    columns = getattr(df, "columns", None)
    multi_index_cls = getattr(pd, "MultiIndex", None)

    if columns is None:
        return df

    is_multi_index = False
    if multi_index_cls is not None and isinstance(columns, multi_index_cls):
        is_multi_index = True
    elif getattr(columns, "nlevels", 1) > 1:
        is_multi_index = True

    if is_multi_index:
        df.columns = _flatten_columns(columns)
        # Drop duplicate columns (yfinance can occasionally return multiple
        # copies of the same field when the ticker level is preserved).
        duplicated = getattr(df.columns, "duplicated", None)
        if callable(duplicated):
            df = df.loc[:, ~duplicated()]

    return df


def _normalise_yf_response(symbol: str, raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    if raw is None or raw.empty:
        print(f" No data returned for {symbol}")
        return None

    df = flatten_yf_columns(raw)

    if df.empty or len(df) < 90:
        print(f" Insufficient data for {symbol}: only {len(df)} rows")
        return None

    df = df.reset_index()

    date_column = None
    for candidate in ('Date', 'date', 'Datetime', 'datetime', df.columns[0]):
        if candidate in df.columns:
            date_column = candidate
            break

    if date_column is None:
        print(f" Missing date column for {symbol}")
        return None

    df.rename(
        columns={
            date_column: 'Date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume',
        },
        inplace=True,
    )

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    df = df.dropna(subset=['Date'])

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    optional_numeric = []
    if 'adj_close' in df.columns:
        optional_numeric.append('adj_close')
    missing_columns = [column for column in numeric_cols if column not in df.columns]
    if missing_columns:
        print(f" Missing columns for {symbol}: {missing_columns}")
        return None

    for column in [*numeric_cols, *optional_numeric]:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    if 'adj_close' in df.columns:
        _apply_split_adjustments(df)

    df = df.dropna(subset=numeric_cols)

    if df.empty:
        print(f" Filtered out all rows for {symbol} due to NaNs")
        return None

    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    return df[numeric_cols]


def _download_with_yf_history(
    symbol: str, *, periods: Sequence[str] = ("1y", "5y", "max")
) -> Optional[pd.DataFrame]:
    for period in periods:
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval='1d', auto_adjust=False)
        except Exception as exc:
            print(f" Secondary yfinance history failed for {symbol} ({period}): {exc}")
            continue

        cleaned = _normalise_yf_response(symbol, history)
        if cleaned is not None:
            return cleaned

    return None


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

def _cache_path_for(symbol: str) -> Path:
    normalized = ''.join(ch if ch.isalnum() else '_' for ch in symbol.upper())
    return CACHE_DIR / f"{normalized}.csv"


def _load_cached_data(symbol: str, *, ignore_ttl: bool = False) -> Optional[pd.DataFrame]:
    path = _cache_path_for(symbol)
    if not path.exists():
        return None

    if not ignore_ttl:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if datetime.now(timezone.utc) - modified > CACHE_TTL:
            return None

    try:
        df = pd.read_csv(path, index_col='Date')
    except Exception as exc:
        print(f" Failed to load cache for {symbol}: {exc}")
        return None

    if df.empty:
        return None

    required = ['open', 'high', 'low', 'close', 'volume']
    if not set(required).issubset(df.columns):
        return None

    df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    df = df[required].dropna()
    return df if not df.empty else None


def _store_cached_data(symbol: str, df: pd.DataFrame) -> None:
    path = _cache_path_for(symbol)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index_label='Date')
    except Exception as exc:
        print(f" Failed to write cache for {symbol}: {exc}")


def _load_master_for_cache(symbol: str) -> Optional[pd.DataFrame]:
    """Load master CSV rows for *symbol* and persist them in the cache."""

    master_df = _load_symbol_from_master_csv(symbol)
    if master_df is None or master_df.empty:
        return None

    print(f" Using master CSV data for {symbol}")
    _store_cached_data(symbol, master_df)
    return master_df


def _apply_split_adjustments(df: pd.DataFrame) -> None:
    """Mutate *df* so that OHLC prices honour split adjustments."""

    adj_close = df.get('adj_close')
    if adj_close is None:
        return

    close = df.get('close')
    if close is None:
        return

    with np.errstate(divide='ignore', invalid='ignore'):
        adjustment = adj_close / close

    if hasattr(adjustment, 'replace'):
        adjustment = adjustment.replace([np.inf, -np.inf], np.nan)

    if hasattr(adjustment, 'notna'):
        valid_mask = adjustment.notna()
    else:  # pragma: no cover - defensive fallback
        valid_mask = [value is not None for value in adjustment]

    if hasattr(np, 'isfinite') and not isinstance(adjustment, (int, float)):
        finite_mask = np.isfinite(adjustment)
        if hasattr(valid_mask, '__and__'):
            valid_mask = valid_mask & finite_mask
        else:  # pragma: no cover - defensive fallback
            valid_mask = [v and f for v, f in zip(valid_mask, finite_mask)]

    if hasattr(close, 'ne'):
        nonzero = close.ne(0)
    else:  # pragma: no cover - defensive fallback
        nonzero = [value != 0 for value in close]

    if hasattr(valid_mask, '__and__'):
        mask = valid_mask & nonzero
        has_valid = bool(getattr(mask, 'any', lambda: mask)())
    else:  # pragma: no cover - defensive fallback
        mask = [v and nz for v, nz in zip(valid_mask, nonzero)]
        has_valid = any(mask)

    if not has_valid:
        return

    if hasattr(df, 'loc'):
        for column in ('open', 'high', 'low', 'close'):
            df.loc[mask, column] = df.loc[mask, column] * adjustment[mask]
    else:  # pragma: no cover - defensive fallback
        for idx, apply in enumerate(mask):
            if not apply:
                continue
            factor = adjustment[idx]
            for column in ('open', 'high', 'low', 'close'):
                df[column][idx] = df[column][idx] * factor


def _normalise_external_data(symbol: str, rows: Sequence[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not rows:
        return None

    try:
        df = pd.DataFrame(rows)
    except Exception as exc:
        # ``tests._pandas_stub`` mimics a subset of pandas and expects mapping
        # inputs instead of a sequence of dictionaries.  Attempt to coerce the
        # payload into the required structure before giving up.
        try:
            mapping = {key: [row.get(key) for row in rows] for key in rows[0]}
            df = pd.DataFrame(mapping)
        except Exception:
            print(f" Failed to build DataFrame for {symbol} fallback data: {exc}")
            return None

    if df.empty:
        return None

    if 'Date' not in df.columns:
        print(f" Missing Date column in fallback data for {symbol}")
        return None

    try:
        df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
    except TypeError:
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
    if hasattr(df, 'dropna'):
        df = df.dropna(subset=['Date'])
    else:
        date_values = list(df['Date'])
        valid_indices = [idx for idx, value in enumerate(date_values) if value is not None]
        if not valid_indices:
            return None
        filtered = {
            column: [list(df[column])[idx] for idx in valid_indices]
            for column in df.columns
        }
        df = pd.DataFrame(filtered)

    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [column for column in required if column not in df.columns]
    if missing:
        print(f" Missing columns in fallback data for {symbol}: {missing}")
        return None

    numeric_columns = required + (["adj_close"] if 'adj_close' in df.columns else [])
    for column in numeric_columns:
        try:
            converter = getattr(pd, 'to_numeric')
        except AttributeError:
            converter = None

        if converter is not None:
            try:
                df[column] = converter(df[column], errors='coerce')
                continue
            except TypeError:
                pass

        values = []
        for value in df[column]:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(None)
        df[column] = values

    if hasattr(df, 'dropna'):
        df = df.dropna(subset=required)
    else:
        masks = []
        for column in required:
            masks.append([value is not None for value in df[column]])
        combined = [all(values) for values in zip(*masks)] if masks else []
        valid_indices = [idx for idx, flag in enumerate(combined) if flag]
        if not valid_indices:
            return None
        filtered = {
            column: [list(df[column])[idx] for idx in valid_indices]
            for column in df.columns
        }
        df = pd.DataFrame(filtered)
    if df.empty:
        return None

    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df[numeric_columns]


def _serialise_date(value: Union[str, datetime, date, None]) -> Optional[str]:
    """Return an ISO formatted date string for API parameters.

    The Tiingo API accepts both date strings and :class:`datetime` objects.  The
    helper keeps the public ``get_tiingo_data`` interface flexible while keeping
    the request construction logic readable.
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return str(value)


def get_tiingo_data(
    symbol: str,
    token: Optional[str] = None,
    *,
    start_date: Union[str, datetime, date, None] = None,
    end_date: Union[str, datetime, date, None] = None,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Return OHLCV data for *symbol* from the Tiingo REST API.

    Parameters
    ----------
    symbol:
        The market ticker to request, e.g. ``"AAPL"``.
    token:
        Optional Tiingo API token.  When omitted the function falls back to the
        ``TIINGO_API_KEY`` environment variable.
    start_date / end_date:
        Optional date filters.  They accept either strings in ``YYYY-mm-dd``
        format or :class:`datetime.date` / :class:`datetime.datetime` instances.
        When left unset Tiingo will return the most recent candles (roughly 1
        year of data by default).
    session:
        Optional :class:`requests.Session` instance used to execute the HTTP
        request.  Supplying a session simplifies unit testing and allows callers
        to reuse HTTP connections.

    Returns
    -------
    pandas.DataFrame
        Historical OHLCV candles indexed by timestamp.

    Raises
    ------
    ValueError
        If the ticker symbol is missing or no API token is available.
    requests.RequestException
        If Tiingo responds with an HTTP error status.
    RuntimeError
        If Tiingo returns malformed JSON data.
    """

    if not symbol or not str(symbol).strip():
        raise ValueError("symbol must be a non-empty string")

    token = token or TIINGO_API_KEY
    if not token:
        raise ValueError("A Tiingo API token must be provided")

    url = f"https://api.tiingo.com/tiingo/daily/{symbol.lower()}/prices"

    params: Dict[str, Any] = {
        "resampleFreq": "daily",
    }

    start = _serialise_date(start_date)
    if start:
        params["startDate"] = start

    end = _serialise_date(end_date)
    if end:
        params["endDate"] = end

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {token}",
    }

    client = session or requests

    response = client.get(url, params=params, headers=headers, timeout=30)
    try:
        response.raise_for_status()
    except requests.RequestException:
        # Re-raise the original exception to preserve the error details.
        raise

    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Tiingo response did not contain valid JSON") from exc

    if not isinstance(payload, Sequence):
        raise RuntimeError("Tiingo response payload must be a list")

    rows = []
    for entry in payload:
        if not isinstance(entry, dict):  # pragma: no cover - defensive guard
            continue
        date_value = entry.get("date")
        if not date_value:
            continue
        rows.append(
            {
                "Date": date_value,
                "open": entry.get("open"),
                "high": entry.get("high"),
                "low": entry.get("low"),
                "close": entry.get("close"),
                "adj_close": entry.get("adjClose"),
                "volume": entry.get("volume"),
            }
        )

    if not rows:
        raise RuntimeError("Tiingo response did not contain price data")

    df = _normalise_external_data(symbol, rows)
    if df is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Tiingo data could not be normalised")
    return df


def _fetch_tiingo_data(symbol: str) -> Optional[pd.DataFrame]:
    token = TIINGO_API_KEY
    if not token:
        return None

    url = f"https://api.tiingo.com/tiingo/daily/{symbol.lower()}/prices"
    start_date = (datetime.utcnow() - timedelta(days=400)).strftime("%Y-%m-%d")
    params = {
        "startDate": start_date,
        "resampleFreq": "daily",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {token}",
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
    except Exception as exc:
        print(f" Tiingo request failed for {symbol}: {exc}")
        return None

    if response.status_code != 200:
        print(f" Tiingo returned status {response.status_code} for {symbol}: {response.text[:120]}")
        return None

    try:
        payload = response.json()
    except ValueError as exc:
        print(f" Failed to parse Tiingo response for {symbol}: {exc}")
        return None

    if not isinstance(payload, list):
        print(f" Unexpected Tiingo payload for {symbol}")
        return None

    rows = []
    for entry in payload:
        date = entry.get('date')
        if not date:
            continue
        rows.append(
            {
                'Date': date,
                'open': entry.get('open'),
                'high': entry.get('high'),
                'low': entry.get('low'),
                'close': entry.get('close'),
                'adj_close': entry.get('adjClose'),
                'volume': entry.get('volume'),
            }
        )

    df = _normalise_external_data(symbol, rows)
    if df is not None:
        print(f" Using Tiingo fallback data for {symbol}")
    return df


def _fetch_alpha_vantage_data(symbol: str) -> Optional[pd.DataFrame]:
    key = ALPHAVANTAGE_API_KEY
    if not key:
        return None

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": key,
        "outputsize": "full",
    }

    try:
        response = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    except Exception as exc:
        print(f" Alpha Vantage request failed for {symbol}: {exc}")
        return None

    if response.status_code != 200:
        print(f" Alpha Vantage returned status {response.status_code} for {symbol}: {response.text[:120]}")
        return None

    try:
        payload = response.json()
    except ValueError as exc:
        print(f" Failed to parse Alpha Vantage response for {symbol}: {exc}")
        return None

    time_series = payload.get("Time Series (Daily)")
    if not isinstance(time_series, dict):
        message = payload.get("Note") or payload.get("Error Message") or "Unexpected Alpha Vantage payload"
        print(f" Alpha Vantage payload issue for {symbol}: {message}")
        return None

    rows = []
    for date_str, values in time_series.items():
        rows.append(
            {
                'Date': date_str,
                'open': values.get('1. open'),
                'high': values.get('2. high'),
                'low': values.get('3. low'),
                'close': values.get('4. close'),
                'adj_close': values.get('5. adjusted close'),
                'volume': values.get('6. volume'),
            }
        )

    df = _normalise_external_data(symbol, rows)
    if df is not None:
        print(f" Using Alpha Vantage fallback data for {symbol}")
    return df


def _try_fallback_sources(symbol: str) -> Optional[pd.DataFrame]:
    for provider in (_fetch_tiingo_data, _fetch_alpha_vantage_data):
        try:
            df = provider(symbol)
        except Exception as exc:  # pragma: no cover - defensive
            print(f" Fallback provider {provider.__name__} raised for {symbol}: {exc}")
            continue
        if df is not None and not df.empty:
            return df
    return None


def _recover_with_fallback(symbol: str) -> Optional[pd.DataFrame]:
    fallback = _try_fallback_sources(symbol)
    if fallback is not None and not fallback.empty:
        _store_cached_data(symbol, fallback)
        return fallback

    cached = _load_cached_data(symbol, ignore_ttl=True)
    if cached is not None:
        print(f" Using cached data for {symbol}")
        return cached

    return None


def _load_master_csv_table() -> Optional[pd.DataFrame]:
    global _MASTER_CSV_CACHE, _MASTER_CSV_FAILED, _MASTER_CSV_MTIME

    if MASTER_CSV_PATH is None:
        return None

    if _MASTER_CSV_FAILED:
        return None

    if _MASTER_CSV_CACHE is not None:
        try:
            if MASTER_CSV_PATH.exists():
                mtime = MASTER_CSV_PATH.stat().st_mtime
                if _MASTER_CSV_MTIME == mtime:
                    return _MASTER_CSV_CACHE
        except Exception:
            pass

    if not hasattr(pd, "read_csv"):
        print(" Pandas does not provide read_csv(); master CSV disabled")
        _MASTER_CSV_FAILED = True
        return None

    path = MASTER_CSV_PATH

    try:
        table = pd.read_csv(path)
    except FileNotFoundError:
        print(f" Master CSV configured but not found at {path}")
        _MASTER_CSV_FAILED = True
        return None
    except Exception as exc:
        print(f" Failed to read master CSV {path}: {exc}")
        _MASTER_CSV_FAILED = True
        return None

    if table is None:
        _MASTER_CSV_FAILED = True
        print(f" Master CSV at {path} did not return a DataFrame")
        return None

    if not hasattr(table, "rename") or not hasattr(table, "dropna"):
        print(" Master CSV could not be processed (missing DataFrame helpers)")
        _MASTER_CSV_FAILED = True
        return None

    column_map = {str(col).strip().lower(): col for col in getattr(table, "columns", [])}

    # Gracefully handle common aliases used in upstream CSVs
    if "symbol" not in column_map:
        for alias in ("ticker",):
            original = column_map.get(alias)
            if original:
                column_map["symbol"] = original
                break
    missing = [col for col in _MASTER_CSV_REQUIRED_COLUMNS if col not in column_map]
    if missing:
        print(f" Master CSV is missing required columns: {missing}")
        _MASTER_CSV_FAILED = True
        return None

    rename_map = {column_map[name]: name for name in _MASTER_CSV_REQUIRED_COLUMNS}
    for optional in _MASTER_CSV_OPTIONAL_COLUMNS:
        original = column_map.get(optional)
        if original:
            rename_map[original] = optional

    table = table.rename(columns=rename_map)

    if not hasattr(pd, "to_datetime"):
        print(" Pandas does not provide to_datetime(); master CSV disabled")
        _MASTER_CSV_FAILED = True
        return None

    table = table.dropna(subset=["symbol", "date"])
    table["symbol"] = table["symbol"].astype(str)
    if hasattr(table["symbol"], "str"):
        table["symbol"] = table["symbol"].str.upper()
    else:  # pragma: no cover - defensive fallback
        table["symbol"] = [str(value).upper() for value in table["symbol"]]

    table["date"] = pd.to_datetime(table["date"], errors="coerce", utc=True)
    table = table.dropna(subset=["date"])

    if not hasattr(pd, "to_numeric"):
        print(" Pandas does not provide to_numeric(); master CSV disabled")
        _MASTER_CSV_FAILED = True
        return None

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for column in numeric_cols:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    optional_cols = [col for col in _MASTER_CSV_OPTIONAL_COLUMNS if col in table.columns]
    for column in optional_cols:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    table = table.dropna(subset=numeric_cols)
    table = table.sort_values(["symbol", "date"])

    _MASTER_CSV_CACHE = table
    try:
        _MASTER_CSV_MTIME = MASTER_CSV_PATH.stat().st_mtime
    except Exception:
        _MASTER_CSV_MTIME = None
    return _MASTER_CSV_CACHE


def list_symbols_from_master_csv() -> List[str]:
    """Return a sorted list of unique symbols present in the master CSV.

    The helper is intended for CLI callers that want to scan every ticker
    already present in ``MASTER_CSV_PATH`` (for example, after running
    ``update_us_liquid_stocks.py``).  It reuses the cached master table when
    available to avoid unnecessary disk I/O.
    """

    table = _load_master_csv_table()
    if table is None or table.empty:
        return []

    try:
        symbols = table.get("symbol")
        if symbols is None:
            return []

        unique = [str(value).strip().upper() for value in symbols.dropna().unique()]
        return sorted(symbol for symbol in unique if symbol)
    except Exception:
        return []


def _load_symbol_from_master_csv(symbol: str) -> Optional[pd.DataFrame]:
    table = _load_master_csv_table()
    if table is None or table.empty:
        return None

    target = symbol.upper()
    try:
        symbol_rows = table[table["symbol"] == target]
    except Exception:  # pragma: no cover - defensive
        return None

    if symbol_rows is None or symbol_rows.empty:
        return None

    symbol_rows = symbol_rows.copy()

    if MASTER_CSV_LOOKBACK_DAYS > 0 and "date" in symbol_rows.columns:
        latest = symbol_rows["date"].max()
        if pd.isna(latest):
            return None
        cutoff = latest - timedelta(days=max(MASTER_CSV_LOOKBACK_DAYS, 0))
        symbol_rows = symbol_rows[symbol_rows["date"] >= cutoff]
        if symbol_rows.empty:
            return None

    if "adj_close" in symbol_rows.columns:
        _apply_split_adjustments(symbol_rows)

    try:
        symbol_rows = symbol_rows.set_index("date")
    except KeyError:
        return None

    if hasattr(symbol_rows, "sort_index"):
        symbol_rows = symbol_rows.sort_index()

    cleaned = symbol_rows[["open", "high", "low", "close", "volume"]].dropna()

    if cleaned.empty or len(cleaned) < 90:
        print(
            f" Master CSV for {target} has insufficient rows: {len(cleaned)}"
        )
        return None

    _store_cached_data(target, cleaned)
    return cleaned


def load_symbol_from_master_csv(symbol: str) -> Optional[pd.DataFrame]:
    """Public wrapper used by external consumers such as the GUI."""

    return _load_symbol_from_master_csv(symbol)


def _prepare_master_csv_rows(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    working = df.copy()
    if "Date" in working.columns:
        date_series = working["Date"]
    elif "date" in working.columns:
        date_series = working["date"]
    else:
        date_series = working.index

    date_values = pd.to_datetime(date_series, errors="coerce", utc=True)
    if hasattr(working, "set_index"):
        working = working.copy()

    required_columns = ["open", "high", "low", "close", "volume"]
    if not set(required_columns).issubset(working.columns):
        return pd.DataFrame()

    payload = {
        "symbol": str(symbol).upper(),
        "date": date_values,
        "open": working["open"],
        "high": working["high"],
        "low": working["low"],
        "close": working["close"],
        "volume": working["volume"],
    }

    if "adj_close" in working.columns:
        payload["adj_close"] = working["adj_close"]

    rows = pd.DataFrame(payload)
    rows = rows.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    return rows


def update_master_csv(data_by_symbol: Dict[str, pd.DataFrame]) -> Optional[Path]:
    """Append the latest OHLCV data into the shared master CSV file."""

    if not data_by_symbol:
        return MASTER_CSV_PATH

    target_path = MASTER_CSV_PATH
    frames: List[pd.DataFrame] = []
    for symbol, df in data_by_symbol.items():
        prepared = _prepare_master_csv_rows(symbol, df)
        if not prepared.empty:
            frames.append(prepared)

    if not frames:
        return target_path

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f" Failed to create master CSV directory {target_path.parent}: {exc}")
        return None

    try:
        new_rows = pd.concat(frames, ignore_index=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f" Failed to combine master CSV rows: {exc}")
        return None

    try:
        if target_path.exists():
            existing = pd.read_csv(target_path)
        else:
            existing = pd.DataFrame(columns=[*_MASTER_CSV_REQUIRED_COLUMNS, *_MASTER_CSV_OPTIONAL_COLUMNS])
    except Exception as exc:  # pragma: no cover - defensive
        print(f" Failed to read existing master CSV {target_path}: {exc}")
        existing = pd.DataFrame(columns=[*_MASTER_CSV_REQUIRED_COLUMNS, *_MASTER_CSV_OPTIONAL_COLUMNS])

    optional_cols = [col for col in _MASTER_CSV_OPTIONAL_COLUMNS if col in new_rows.columns or col in existing.columns]
    all_columns = [*_MASTER_CSV_REQUIRED_COLUMNS, *optional_cols]
    missing_in_existing = [col for col in all_columns if col not in existing.columns]
    for column in missing_in_existing:
        existing[column] = np.nan
    missing_in_new = [col for col in all_columns if col not in new_rows.columns]
    for column in missing_in_new:
        new_rows[column] = np.nan

    try:
        new_rows["date"] = pd.to_datetime(new_rows["date"], errors="coerce", utc=True)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce", utc=True)
    except Exception:  # pragma: no cover - defensive
        pass

    latest_existing: Dict[str, pd.Timestamp] = {}
    try:
        if not existing.empty:
            grouped = existing.dropna(subset=["symbol", "date"]).groupby("symbol")
            latest_existing = grouped["date"].max().to_dict()
    except Exception:  # pragma: no cover - defensive
        latest_existing = {}

    def _is_newer(row):
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            return False
        latest = latest_existing.get(symbol)
        if latest is None:
            return True
        date_value = row.get("date")
        if pd.isna(date_value):
            return False
        return date_value > latest

    try:
        if hasattr(new_rows, "iterrows"):
            new_rows = new_rows[[_is_newer(row) for _, row in new_rows.iterrows()]]
    except Exception:  # pragma: no cover - defensive
        pass

    populated_optional = [
        col
        for col in optional_cols
        if (col in new_rows and new_rows[col].notna().any())
        or (col in existing and existing[col].notna().any())
    ]
    all_columns = [*_MASTER_CSV_REQUIRED_COLUMNS, *populated_optional]
    new_rows = new_rows[all_columns]
    existing = existing.reindex(columns=all_columns)

    if existing.empty:
        combined = new_rows.copy()
    else:
        combined = pd.concat([existing[all_columns], new_rows], ignore_index=True)
    combined = combined.dropna(subset=["symbol", "date"])
    try:
        combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
        combined = combined.sort_values(["symbol", "date"])
    except Exception:  # pragma: no cover - defensive
        pass

    try:
        combined.to_csv(target_path, index=False)
    except Exception as exc:  # pragma: no cover - defensive
        print(f" Failed to write master CSV {target_path}: {exc}")
        return None

    global _MASTER_CSV_CACHE, _MASTER_CSV_FAILED, _MASTER_CSV_MTIME
    _MASTER_CSV_CACHE = None
    _MASTER_CSV_FAILED = False
    try:
        _MASTER_CSV_MTIME = target_path.stat().st_mtime
    except Exception:
        _MASTER_CSV_MTIME = None
    return target_path


def _latest_trading_day(today: Optional[date] = None) -> date:
    """Return the most recent expected trading day (Mon–Fri)."""

    if today is None:
        today = datetime.now(timezone.utc).date()

    weekday = today.weekday()
    if weekday >= 5:
        offset = weekday - 4
        return today - timedelta(days=offset)
    return today


def get_yf_data(symbol):
    cached = _load_cached_data(symbol)
    if cached is not None:
        return cached

    master = _load_master_for_cache(symbol)
    if master is not None:
        _store_cached_data(symbol, master)
        return master

    print(
        f" No cached data for {symbol}. Please run build_ohlcv_last2y.py to refresh the master CSV."
    )
    return pd.DataFrame()


def fetch_symbol_data(symbols: Sequence[str], max_workers: int = 8) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    if not symbols:
        return results

    latest_expected = _latest_trading_day()
    last_dates: Dict[str, Optional[date]] = {}

    for symbol in symbols:
        csv_df = _load_symbol_from_master_csv(symbol)
        if csv_df is None or csv_df.empty:
            results[symbol] = pd.DataFrame()
            last_dates[symbol] = None
            continue

        last_date = None
        try:
            last_date = csv_df.index.max()
        except Exception:
            pass

        if last_date is not None and getattr(last_date, "date", None):
            last_dates[symbol] = last_date.date()
        else:
            last_dates[symbol] = None

        results[symbol] = csv_df

    latest_available = max((d for d in last_dates.values() if d is not None), default=None)
    if latest_available is not None and latest_expected > latest_available:
        confirmations = sum(1 for d in last_dates.values() if d == latest_available)
        if confirmations >= 3 and (latest_expected - latest_available).days <= 3:
            print(
                f" Detected {confirmations} symbols capped at {latest_available}; "
                "assuming that was the most recent trading day."
            )
            latest_expected = latest_available

    for symbol, last_date_value in last_dates.items():
        csv_df = results[symbol]
        if csv_df is None or csv_df.empty:
            continue

        if last_date_value is not None and last_date_value >= latest_expected:
            continue

        if last_date_value is None:
            print(
                f" Missing recent data for {symbol}; run build_ohlcv_last2y.py to refresh the master CSV."
            )
        else:
            print(
                f" Data for {symbol} is stale (latest {last_date_value}); run build_ohlcv_last2y.py for updates."
            )

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

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _score_cup_handle_hit(hit: CupHandleHit) -> float:
    score = 0.55
    depth_pct = max(float(getattr(hit, "cup_depth_pct", 0.0)), 0.0)
    score += _clamp(depth_pct, 0.0, 0.25)
    pullback = max(float(getattr(hit, "handle_pullback_pct", 0.0)), 0.0)
    score += _clamp(0.18 - pullback, 0.0, 0.12)
    slope = abs(float(getattr(hit, "handle_slope", 0.0)))
    score += _clamp(0.05 - slope, 0.0, 0.08)
    return _clamp(score, 0.0, 0.95)


def _score_inverse_head_shoulders(pattern: InverseHeadShouldersPattern) -> float:
    score = 0.58
    avg_low = max((pattern.left_low + pattern.right_low) / 2 or 0.0, 1e-6)
    symmetry = 1 - abs(pattern.left_low - pattern.right_low) / avg_low
    score += _clamp(symmetry, 0.0, 0.12)
    avg_neckline = max((pattern.neckline_left + pattern.neckline_right) / 2 or 0.0, 1e-6)
    depth_pct = (avg_neckline - pattern.head_low) / avg_neckline
    score += _clamp(depth_pct, 0.0, 0.18)
    neckline_slope = (pattern.neckline_right - pattern.neckline_left) / max(pattern.neckline_left, 1e-6)
    score += _clamp(neckline_slope, 0.0, 0.08)
    if getattr(pattern, "breakout", False):
        score += 0.05
    return _clamp(score, 0.0, 0.9)


def _score_ascending_triangle(pattern: AscendingTrianglePattern) -> float:
    score = 0.55
    slope_bonus = _clamp(pattern.support_slope, 0.0, 0.1)
    score += slope_bonus
    score += _clamp(len(pattern.resistance_indices) * 0.02, 0.0, 0.12)
    if pattern.breakout:
        score += 0.08
    return _clamp(score, 0.0, 0.85)


def _score_bullish_pennant(pattern: BullishPennantPattern) -> float:
    score = 0.5
    convergence = abs(pattern.upper_slope - pattern.lower_slope)
    score += _clamp(convergence, 0.0, 0.15)
    score += _clamp(pattern.length * 0.005, 0.0, 0.1)
    return _clamp(score, 0.0, 0.82)


def _score_bullish_flag(pattern: BullishFlagPattern) -> float:
    score = 0.5
    score += _clamp(-pattern.slope, 0.0, 0.15)
    channel_width = max(pattern.upper_offset - pattern.lower_offset, 0.0)
    if channel_width > 0:
        score += _clamp(0.1 / (channel_width + 1e-6), 0.0, 0.08)
    return _clamp(score, 0.0, 0.8)


def _score_bullish_rectangle(pattern: BullishRectanglePattern) -> float:
    score = 0.48
    range_pct = (pattern.high - pattern.low) / max(pattern.high, 1e-6)
    score += _clamp(0.2 - range_pct, 0.0, 0.12)
    touches = len(pattern.high_touch_indices) + len(pattern.low_touch_indices)
    score += _clamp(touches * 0.015, 0.0, 0.12)
    return _clamp(score, 0.0, 0.8)


def _score_rounding_bottom(pattern: RoundingBottomPattern) -> float:
    score = 0.52
    curvature = float(pattern.coeffs[0]) if pattern.coeffs else 0.0
    score += _clamp(curvature * 5000, 0.0, 0.12)
    score += _clamp(pattern.length * 0.003, 0.0, 0.08)
    return _clamp(score, 0.0, 0.8)


def _score_breakaway_gap(pattern: BreakawayGapPattern) -> float:
    score = 0.6
    gap_pct = (pattern.curr_open - pattern.prev_close) / max(pattern.prev_close, 1e-6)
    score += _clamp(gap_pct, 0.0, 0.25)
    body_pct = (pattern.curr_close - pattern.curr_open) / max(pattern.curr_open, 1e-6)
    score += _clamp(body_pct, 0.0, 0.1)
    return _clamp(score, 0.0, 0.9)


def _collect_pattern_candidates(df: pd.DataFrame) -> List[PatternCandidate]:
    candidates: List[PatternCandidate] = []

    cup_handle_hit = detect_cup_and_handle(df)
    if cup_handle_hit:
        confidence = _score_cup_handle_hit(cup_handle_hit)
        candidates.append(PatternCandidate("Cup and Handle", confidence, cup_handle_hit))

    ihs = detect_inverse_head_shoulders(df)
    if ihs:
        confidence = _score_inverse_head_shoulders(ihs)
        candidates.append(PatternCandidate("Inverse Head and Shoulders", confidence, ihs))

    ascending_triangle = detect_ascending_triangle(df)
    if ascending_triangle:
        confidence = _score_ascending_triangle(ascending_triangle)
        candidates.append(PatternCandidate("Ascending Triangle", confidence, ascending_triangle))

    pennant = detect_bullish_pennant(df)
    if pennant:
        confidence = _score_bullish_pennant(pennant)
        candidates.append(PatternCandidate("Bullish Pennant", confidence, pennant))

    flag = detect_bullish_flag(df)
    if flag:
        confidence = _score_bullish_flag(flag)
        candidates.append(PatternCandidate("Bullish Flag", confidence, flag))

    rectangle = detect_bullish_rectangle(df)
    if rectangle:
        confidence = _score_bullish_rectangle(rectangle)
        candidates.append(PatternCandidate("Bullish Rectangle", confidence, rectangle))

    rounding_bottom = detect_rounding_bottom(df)
    if rounding_bottom:
        confidence = _score_rounding_bottom(rounding_bottom)
        candidates.append(PatternCandidate("Rounding Bottom", confidence, rounding_bottom))

    gap = detect_breakaway_gap(df)
    if gap:
        confidence = _score_breakaway_gap(gap)
        candidates.append(PatternCandidate("Breakaway Gap", confidence, gap))

    candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
    return candidates


def _print_pattern_details(candidate: PatternCandidate) -> None:
    details = candidate.details
    if isinstance(details, CupHandleHit):
        print(
            "  Cup and Handle details → "
            f"Resistance: {details.resistance:.2f}, "
            f"Cup Depth: {details.cup_depth:.2f} ({details.cup_depth_pct * 100:.1f}%), "
            f"Handle Length: {details.handle_length}, "
            f"Handle Pullback: {details.handle_pullback_pct * 100:.1f}%, "
            f"Handle Slope: {details.handle_slope:.4f}"
        )
    elif isinstance(details, InverseHeadShouldersPattern):
        print(
            "  Inverse Head and Shoulders details → "
            f"L/R Lows: {details.left_low:.2f}/{details.right_low:.2f}, "
            f"Head Low: {details.head_low:.2f}, "
            f"Neckline: {details.neckline_left:.2f}→{details.neckline_right:.2f}"
        )
    elif isinstance(details, AscendingTrianglePattern):
        print(
            "  Ascending Triangle details → "
            f"Resistance: {details.resistance:.2f}, "
            f"Support slope: {details.support_slope:.4f}, "
            f"Touches: {len(details.resistance_indices)}/{len(details.support_indices)}, "
            f"Breakout: {details.breakout}"
        )
    elif isinstance(details, BullishPennantPattern):
        print(
            "  Bullish Pennant details → "
            f"Upper slope: {details.upper_slope:.4f}, "
            f"Lower slope: {details.lower_slope:.4f}, "
            f"Length: {details.length}"
        )
    elif isinstance(details, BullishFlagPattern):
        print(
            "  Bullish Flag details → "
            f"Slope: {details.slope:.4f}, "
            f"Upper offset: {details.upper_offset:.2f}, "
            f"Lower offset: {details.lower_offset:.2f}, "
            f"Length: {details.length}"
        )
    elif isinstance(details, BullishRectanglePattern):
        print(
            "  Bullish Rectangle details → "
            f"Range: {details.low:.2f}-{details.high:.2f}, "
            f"Touches: {len(details.low_touch_indices)}/{len(details.high_touch_indices)}"
        )
    elif isinstance(details, RoundingBottomPattern):
        print(
            "  Rounding Bottom details → "
            f"Curvature: {details.coeffs[0]:.6f}, "
            f"Length: {details.length}"
        )
    elif isinstance(details, BreakawayGapPattern):
        gap_pct = (details.curr_open - details.prev_close) / max(details.prev_close, 1e-6)
        body_pct = (details.curr_close - details.curr_open) / max(details.curr_open, 1e-6)
        print(
            "  Breakaway Gap details → "
            f"Prev Close: {details.prev_close:.2f}, "
            f"Open: {details.curr_open:.2f}, Close: {details.curr_close:.2f}, "
            f"Gap: {gap_pct * 100:.1f}%, Body: {body_pct * 100:.1f}%"
        )


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


def _has_prior_downtrend(
    closes: Sequence[float],
    pivot_idx: int,
    min_drop: float = 0.03,
    lookback: int = 15,
) -> bool:
    """Return ``True`` when prices were trending lower into the pattern."""

    if pivot_idx <= 1:
        return False

    start = max(0, pivot_idx - lookback)
    segment = closes[start : pivot_idx + 1]
    if len(segment) < 4:
        return False

    start_price = float(segment[0])
    end_price = float(segment[-1])
    if start_price <= 0:
        return False

    drop = (start_price - end_price) / start_price
    if drop < min_drop:
        return False

    return end_price < start_price


def _has_meaningful_bounce(low: float, high: float, min_bounce: float = 0.01) -> bool:
    if low <= 0:
        return False
    return (high - low) / low >= min_bounce


def _neckline_is_consistent(left: float, right: float, tolerance: float = 0.05) -> bool:
    level = (left + right) / 2 if (left or right) else 0.0
    if level == 0:
        return False
    return abs(left - right) / level <= tolerance


def _fallback_troughs(lows: Sequence[float], distance: int = 3) -> List[int]:
    """Simple local minima finder used when SciPy peaks are unavailable."""

    troughs: List[int] = []
    for idx in range(1, len(lows) - 1):
        curr = lows[idx]
        if curr >= lows[idx - 1] or curr > lows[idx + 1]:
            continue

        if troughs and idx - troughs[-1] < distance:
            prev_idx = troughs[-1]
            if curr < lows[prev_idx]:
                troughs[-1] = idx
            continue

        troughs.append(idx)

    return troughs


def detect_inverse_head_shoulders(df) -> Optional[InverseHeadShouldersPattern]:
    recent = df.tail(60)
    lows = recent['low'].values
    highs = recent['high'].values
    closes = recent['close'].values

    troughs, _ = find_peaks(-lows, distance=3, prominence=0.5)
    if len(troughs) < 3:
        fallback = _fallback_troughs(lows, distance=3)
        if len(fallback) >= 3:
            troughs = fallback
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
        if abs(l - r) / max(l, r) > 0.05:
            continue
        if not (_shoulder_has_width(lows, l_idx) and _shoulder_has_width(lows, r_idx)):
            continue
        if (min(l, r) - h) / min(l, r) < 0.02:
            continue
        if h_idx - l_idx < 3 or r_idx - h_idx < 3:
            continue
        if not _has_prior_downtrend(closes, l_idx):
            continue

        left_high_segment = highs[l_idx : h_idx + 1]
        right_high_segment = highs[h_idx : r_idx + 1]
        if len(left_high_segment) == 0 or len(right_high_segment) == 0:
            continue

        left_high_rel = _find_argmax(left_high_segment)
        right_high_rel = _find_argmax(right_high_segment)

        if left_high_rel == 0 or left_high_rel == len(left_high_segment) - 1:
            continue
        if right_high_rel == 0 or right_high_rel == len(right_high_segment) - 1:
            continue

        left_high_value = float(left_high_segment[left_high_rel])
        right_high_value = float(right_high_segment[right_high_rel])
        if not _has_meaningful_bounce(l, left_high_value):
            continue
        if not _has_meaningful_bounce(r, right_high_value):
            continue

        neckline_left_idx = offset + l_idx + left_high_rel
        neckline_right_idx = offset + h_idx + right_high_rel
        neckline_left = left_high_value
        neckline_right = right_high_value

        if not _neckline_is_consistent(neckline_left, neckline_right):
            continue

        neckline_level = (neckline_left + neckline_right) / 2
        close_price = float(closes[-1]) if closes.size else neckline_level
        high_price = float(highs[-1]) if highs.size else neckline_level
        breakout = close_price > neckline_level or high_price > neckline_level
        within_neckline_zone = close_price >= neckline_level * 0.99

        if breakout or within_neckline_zone:
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
                breakout=breakout,
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
    tolerance=0.05,
    min_touches=4,
) -> Optional[BullishRectanglePattern]:
    if len(df) < window:
        return None

    recent = df.tail(window)
    highs = [float(value) for value in recent['high'].values]
    lows = [float(value) for value in recent['low'].values]
    closes = [float(value) for value in recent['close'].values]

    if not highs or not lows:
        return None

    high_level = max(highs)
    low_level = min(lows)
    if high_level <= 0 or low_level <= 0:
        return None

    range_pct = (high_level - low_level) / high_level
    if range_pct > 0.06:
        return None

    def _collect_touches(values, extreme, is_high: bool) -> List[int]:
        if extreme == 0:
            return []
        candidates = [
            idx
            for idx, value in enumerate(values)
            if abs(value - extreme) / abs(extreme) <= tolerance
        ]
        if not candidates:
            return []
        deduped: List[int] = []
        for idx in sorted(candidates):
            if deduped and idx - deduped[-1] <= 2:
                if is_high:
                    better = idx if values[idx] >= values[deduped[-1]] else deduped[-1]
                else:
                    better = idx if values[idx] <= values[deduped[-1]] else deduped[-1]
                deduped[-1] = better
            else:
                deduped.append(int(idx))
        return deduped

    high_touch_indices = _collect_touches(highs, high_level, True)
    low_touch_indices = _collect_touches(lows, low_level, False)

    if len(high_touch_indices) < min_touches or len(low_touch_indices) < min_touches:
        return None

    half = max(len(highs) // 2, 1)
    if not (
        any(idx < half for idx in high_touch_indices)
        and any(idx >= half for idx in high_touch_indices)
        and any(idx < half for idx in low_touch_indices)
        and any(idx >= half for idx in low_touch_indices)
    ):
        return None

    combined_sequence = sorted(
        [(idx, "H") for idx in high_touch_indices] + [(idx, "L") for idx in low_touch_indices],
        key=lambda item: item[0],
    )
    transitions = sum(1 for (_, prev), (_, curr) in zip(combined_sequence, combined_sequence[1:]) if prev != curr)
    if transitions < 3:
        return None

    allowed_high = high_level * (1 + tolerance * 1.5)
    allowed_low = low_level * (1 - tolerance * 1.5)
    breaches = sum(1 for high, low in zip(highs, lows) if high > allowed_high or low < allowed_low)
    if breaches > max(2, int(0.1 * len(highs))):
        return None

    if len(closes) >= 2:
        x = [[idx] for idx in range(len(closes))]
        lr = LinearRegression().fit(x, closes)
        slope = float(lr.coef_[0])
        avg_price = sum(closes) / len(closes)
        if avg_price != 0 and abs(slope) / abs(avg_price) > 0.0025:
            return None

    total_len = len(df)
    pole_window = min(20, max(total_len - len(recent), 0))
    if pole_window >= 5:
        combined = df.tail(len(recent) + pole_window)
        combined_closes = [float(value) for value in combined['close'].values]
        flagpole_closes = combined_closes[:pole_window]
        if len(flagpole_closes) >= 2:
            start_price = flagpole_closes[0]
            end_price = flagpole_closes[-1]
            if start_price == 0:
                return None
            gain = end_price / start_price
            if gain < 1.08:
                return None

    offset = total_len - len(recent)
    return BullishRectanglePattern(
        high=high_level,
        low=low_level,
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

def calculate_rr_price_action(df, entry_price) -> Optional[RiskRewardLevels]:
    from scipy.signal import argrelextrema

    lows = df['low'].values
    highs = df['high'].values
    swing_lows_idx = argrelextrema(lows, np.less, order=5)[0]
    swing_highs_idx = argrelextrema(highs, np.greater, order=5)[0]

    if len(swing_lows_idx) < 1 or len(swing_highs_idx) < 1:
        return None

    last_low_idx = int(swing_lows_idx[-1])
    recent_low = float(lows[last_low_idx])

    subsequent_highs = [idx for idx in swing_highs_idx if idx > last_low_idx]
    if subsequent_highs:
        breakout_idx = int(subsequent_highs[-1])
    else:
        trailing_highs = highs[last_low_idx:]
        if trailing_highs.size == 0:
            return None
        breakout_idx = last_low_idx + int(np.argmax(trailing_highs))

    recent_high = float(highs[breakout_idx])
    breakout_height = recent_high - recent_low
    if breakout_height <= 0:
        return None

    breakout = recent_high
    target = breakout + breakout_height

    risk = entry_price - recent_low
    reward = target - entry_price

    if risk <= 0 or reward <= 0:
        return None

    rr = round(risk / reward, 2)
    return RiskRewardLevels(
        breakout=float(breakout),
        stop=float(recent_low),
        target=float(target),
        rr_ratio=rr,
    )

def _normalise_watchlist_row(row: list[str], header: Optional[Sequence[str]] = None) -> list[str]:
    """Pad or trim a CSV row so it matches ``WATCHLIST_HEADER``."""

    values = list(row)
    header_list = list(header or [])

    if header_list:
        while "timestamp" in header_list:
            idx = header_list.index("timestamp")
            del header_list[idx]
            if idx < len(values):
                del values[idx]
    elif len(values) == len(WATCHLIST_HEADER) + 1:
        # Legacy rows may still include a timestamp column even without a
        # header.  The timestamp used to sit between ``target_price`` and
        # ``direction``.
        del values[6]

    if len(values) < len(WATCHLIST_HEADER):
        # Insert an empty placeholder for the ``last_close`` column right after
        # the symbol.  This keeps backwards compatibility with watchlists
        # written before that column existed.
        values.insert(1, "")

    if len(values) > len(WATCHLIST_HEADER):
        values = values[: len(WATCHLIST_HEADER)]

    return values


def log_watchlist(
    symbol,
    pattern,
    entry,
    rr_levels: RiskRewardLevels,
    df,
    pattern_details: Optional[Any] = None,
):
    path = WATCHLIST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    volume_3mo = int(df['volume'].tail(60).sum())

    last_close_value = None
    if df is not None and not df.empty:
        try:
            last_close_value = round(float(df['close'].iloc[-1]), 2)
        except (KeyError, ValueError, TypeError):  # pragma: no cover - defensive
            last_close_value = None

    if rr_levels and rr_levels.breakout is not None:
        breakout_value = round(rr_levels.breakout, 2)
    else:
        breakout_value = round(entry, 2) if entry is not None else None
    stop_value = (
        round(rr_levels.stop, 2) if rr_levels and rr_levels.stop is not None else None
    )
    target_value = (
        round(rr_levels.target, 2) if rr_levels and rr_levels.target is not None else None
    )

    new_entry = [
        symbol,
        last_close_value,
        breakout_value,
        rr_levels.rr_ratio if rr_levels else None,
        stop_value,
        target_value,
        'bullish',
        pattern,
        volume_3mo
    ]

    if path.exists():
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            existing = {}
            for row in reader:
                if not row:
                    continue
                normalised = _normalise_watchlist_row(row, header)
                existing[normalised[0]] = normalised
    else:
        existing = {}

    existing[symbol] = new_entry

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(WATCHLIST_HEADER)
        writer.writerows(sorted(existing.values(), key=lambda x: x[0]))

    if pattern_details is not None:
        try:
            save_pattern_details(symbol, pattern, pattern_details)
        except Exception:
            pass


def initialize_watchlist():
    """Create or reset the watchlist file with only the header."""
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCHLIST_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(WATCHLIST_HEADER)


def _scan_trend_reversal(symbols: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Run the short-term trend reversal screener for ``symbols``.

    The screener reuses the standalone ``ai_reversal_screener`` logic so that
    we evaluate the exact same universe of tickers that the pattern scanner is
    analysing.  Results are returned as a mapping keyed by the upper-case
    symbol for easy lookups during the main scan loop.
    """

    unique_symbols = sorted({sym.upper() for sym in symbols})
    if not unique_symbols:
        return {}

    try:
        reversal_table = run_reversal_screen(
            unique_symbols,
            REVERSAL_START_DATE,
            None,
            REVERSAL_INTERMARKET,
            None,
        )
    except Exception as exc:  # pragma: no cover - defensive and network errors
        print(f" Trend-reversal scan failed: {exc}")
        return {}

    summary: Dict[str, Dict[str, Any]] = {}
    for _, row in reversal_table.iterrows():
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue
        summary[symbol] = row.to_dict()
    return summary


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


def _load_symbols_from_file(path: Path) -> List[str]:
    try:
        with path.open("r") as handle:
            return [line.strip() for line in handle.readlines() if line.strip()]
    except FileNotFoundError:
        print(f" {path} not found.")
        return []


def _run_cli():
    parser = argparse.ArgumentParser(description="Scan symbols for bullish patterns")
    parser.add_argument(
        "-f",
        "--symbols-file",
        type=Path,
        default=Path("filtered_symbols.txt"),
        help="Path to a text file with one symbol per line (default: filtered_symbols.txt)",
    )
    parser.add_argument(
        "--from-master",
        action="store_true",
        help="Scan every symbol available in the configured master CSV",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Skip the synthetic ascending triangle demonstration",
    )

    args = parser.parse_args()

    if not args.skip_demo:
        _demo_ascending_triangle_detection()

    symbols: List[str] = []
    if args.from_master:
        symbols = list_symbols_from_master_csv()
        if symbols:
            print(f" Loaded {len(symbols)} symbols from master CSV at {MASTER_CSV_PATH}")
        else:
            print(" Master CSV did not yield any symbols; falling back to symbols file")

    if not symbols:
        symbols = _load_symbols_from_file(args.symbols_file)

    if not symbols:
        print(" No symbols to scan. Provide a symbols file or use --from-master.")
        return

    scan_all_symbols(symbols)

def scan_all_symbols(symbols):
    initialize_watchlist()
    disqualified = []

    symbols_to_fetch = [s for s in symbols if s.upper() not in EXCLUDED_ETFS]
    data_by_symbol = fetch_symbol_data(symbols_to_fetch)
    update_master_csv(data_by_symbol)
    reversal_results = _scan_trend_reversal(symbols_to_fetch)

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

            candidates = _collect_pattern_candidates(df)
            if not candidates:
                print(" Skipped: No pattern matched")
                disqualified.append({'symbol': symbol, 'reason': 'no pattern matched', 'entry': entry, 'rr': None})
                continue

            best_candidate = candidates[0]
            pattern = best_candidate.name
            print(
                " Selected pattern → "
                f"{pattern} (confidence {best_candidate.confidence:.2f})"
            )
            _print_pattern_details(best_candidate)

            if len(candidates) > 1:
                print("  Alternate pattern signals:")
                for alternate in candidates[1:3]:
                    print(
                        "   • "
                        f"{alternate.name} (confidence {alternate.confidence:.2f})"
                    )

            rr_levels = calculate_rr_price_action(df, entry)
            if rr_levels is None or rr_levels.rr_ratio > 0.8:
                rr_value = None if rr_levels is None else rr_levels.rr_ratio
                print(f" Skipped: RR too high or invalid → RR: {rr_value}")
                disqualified.append({'symbol': symbol, 'reason': 'rr too high or invalid', 'entry': entry, 'rr': rr_value})
                continue

            log_watchlist(
                symbol,
                pattern,
                entry,
                rr_levels,
                df,
                pattern_details=best_candidate.details,
            )
            print(
                " Match: "
                f"{pattern} → Breakout: {rr_levels.breakout:.2f}, "
                f"Entry: {entry:.2f}, RR: {rr_levels.rr_ratio}, "
                f"Stop: {rr_levels.stop:.2f}, Target: {rr_levels.target:.2f}"
            )

            reversal_data = reversal_results.get(symbol.upper())
            if reversal_data:
                prob = reversal_data.get("prob_reversal_1_3d")
                auc = reversal_data.get("auc")
                precision = reversal_data.get("precision_top20")

                try:
                    prob_value = float(prob)
                    prob_display = "N/A" if np.isnan(prob_value) else f"{prob_value:.2%}"
                except (TypeError, ValueError):
                    prob_display = "N/A"

                try:
                    auc_value = float(auc)
                    if np.isnan(auc_value):
                        auc_value = None
                except (TypeError, ValueError):
                    auc_value = None

                try:
                    precision_value = float(precision)
                    if np.isnan(precision_value):
                        precision_value = None
                except (TypeError, ValueError):
                    precision_value = None

                message_parts = [f" Trend-reversal scan → P(reversal 1-3d): {prob_display}"]
                if auc_value is not None:
                    message_parts.append(f"AUC: {auc_value:.3f}")
                if precision_value is not None:
                    message_parts.append(f"Precision@20%: {precision_value:.3f}")
                print(", ".join(message_parts))
            else:
                print(" Trend-reversal scan → No data (insufficient history or download failure)")

        except Exception as e:
            print(f" Error scanning {symbol}: {e}")
            disqualified.append({'symbol': symbol, 'reason': f'error: {e}', 'entry': None, 'rr': None})

    if disqualified:
        pd.DataFrame(disqualified).to_csv('disqualified.csv', index=False)
        print(f"\n Saved disqualified symbols to disqualified.csv")

if __name__ == '__main__':
    _run_cli()
