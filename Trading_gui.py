
def navigate_chart(direction):
    selected = tree.selection()
    items = tree.get_children()
    if not selected or not items:
        return
    idx = items.index(selected[0])
    if direction == "up" and idx > 0:
        tree.selection_set(items[idx - 1])
    elif direction == "down" and idx < len(items) - 1:
        tree.selection_set(items[idx + 1])

import os
import sys
import shutil
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import math
import pandas as pd
import yfinance as yf
import numpy as np
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import threading
import signal
import json
import csv
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

from cup_handle_scanner import CupHandleHit, detect_cup_and_handle
from double_bottom_scanner import DoubleBottomHit, scan_double_bottoms
from pattern_scanner import (
    AscendingTrianglePattern,
    BreakawayGapPattern,
    BullishFlagPattern,
    BullishPennantPattern,
    BullishRectanglePattern,
    InverseHeadShouldersPattern,
    RoundingBottomPattern,
    detect_ascending_triangle,
    detect_breakaway_gap,
    detect_bullish_flag,
    detect_bullish_pennant,
    detect_bullish_rectangle,
    detect_inverse_head_shoulders,
    detect_rounding_bottom,
    load_pattern_dataclass,
    WATCHLIST_PATH,
    WATCHLIST_HEADER,
)

API_KEY = 'PKWMYLAWJCU6ITACV6KP'
API_SECRET = 'k8T9M3XdpVcNQudgPudCfqtkRJ0IUCChFSsKYe07'
paper_api = tradeapi.REST(API_KEY, API_SECRET, 'https://paper-api.alpaca.markets', api_version='v2')
live_api = tradeapi.REST(API_KEY, API_SECRET, 'https://api.alpaca.markets', api_version='v2')

def is_crypto(symbol):
    return symbol.endswith("USD") and len(symbol) > 3

def get_api(symbol):
    return live_api if is_crypto(symbol) else paper_api

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
MY_WATCHLIST_PATH = DATA_DIR / "my_watchlist.json"

root = tk.Tk()
root.title("Stock Scanner GUI")
root.geometry("1400x900")

symbol_data = {}
_long_name_cache = {}
_institution_activity_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_institution_snapshot_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_quote_summary_cache: dict[tuple[str, tuple[str, ...]], tuple[float, dict[str, Any]]] = {}
_alpha_vantage_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
INSTITUTION_CACHE_TTL = 60 * 30  # 30 minutes
QUOTE_SUMMARY_CACHE_TTL = 60 * 15  # 15 minutes
ALPHAVANTAGE_CACHE_TTL = 60 * 60  # 1 hour
WATCHLIST_COLUMNS = [
    "symbol",
    "last_close",
    "breakout_high",
    "target_price",
    "stop_loss",
    "rr_ratio",
    "timestamp",
    "pattern",
    "direction",
]
MONITOR_FILE = str(SCRIPT_DIR / "active_monitors.json")

ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY") or os.environ.get("AV_API_KEY")

SECTOR_ETF_MAP = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}


def _resolve_min_size(kwargs: dict[str, Any], fallback: int = 40) -> int:
    window = kwargs.get("window")
    flagpole_window = kwargs.get("flagpole_window")
    cup_window = kwargs.get("cup_window")
    handle_window = kwargs.get("handle_window")

    if window is not None:
        if flagpole_window:
            return int(window) + int(flagpole_window)
        return int(window)

    if cup_window is not None and handle_window is not None:
        return int(cup_window) + int(handle_window)

    return fallback


def _format_share_amount(value: Optional[float]) -> str:
    if value is None:
        return "0"
    absolute = abs(value)
    suffixes = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
    for threshold, suffix in suffixes:
        if absolute >= threshold:
            return f"{value / threshold:.1f}{suffix}"
    return f"{value:.0f}"


def _format_price(value: Optional[float]) -> str:
    if value is None or (isinstance(value, (float, int)) and pd.isna(value)):
        return "N/A"
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


def _format_dollar_amount(value: Optional[float]) -> str:
    if value is None or (isinstance(value, (float, int)) and pd.isna(value)):
        return "N/A"

    try:
        numeric = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)

    sign = "-" if numeric < 0 else ""
    absolute = abs(numeric)
    suffixes = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
    for threshold, suffix in suffixes:
        if absolute >= threshold:
            return f"{sign}${absolute / threshold:.1f}{suffix}"

    if absolute >= 1:
        return f"{sign}${absolute:,.2f}"
    return f"{sign}${absolute:.2f}"


_WATCHLIST_PRICE_COLUMNS = {"last_close", "breakout_high", "target_price", "stop_loss"}


_WATCHLIST_REFRESH_CACHE: dict[str, tuple[float, Optional[float]]] = {}
_WATCHLIST_REFRESH_TTL = 60 * 5

_LAST_ADDED_MY_WATCHLIST_SYMBOL: Optional[str] = None

_MY_WATCHLIST_LISTBOX: Optional[tk.Listbox] = None
_MY_WATCHLIST_SYMBOLS: list[str] = []


def _mark_last_added_my_watchlist_symbol(symbol: str) -> None:
    """Record the most recently added My Watchlist symbol."""

    global _LAST_ADDED_MY_WATCHLIST_SYMBOL
    normalized = (symbol or "").strip().upper()
    _LAST_ADDED_MY_WATCHLIST_SYMBOL = normalized or None


def _load_persisted_pattern(symbol: str, pattern_name: str, cls: type[Any]):
    if not symbol or not pattern_name:
        return None
    try:
        return load_pattern_dataclass(symbol, pattern_name, cls)
    except Exception:
        return None


def _consume_last_added_my_watchlist_symbol() -> Optional[str]:
    """Return and clear the pending My Watchlist symbol highlight."""

    global _LAST_ADDED_MY_WATCHLIST_SYMBOL
    symbol = _LAST_ADDED_MY_WATCHLIST_SYMBOL
    _LAST_ADDED_MY_WATCHLIST_SYMBOL = None
    return symbol


def _focus_watchlist_symbol(event=None) -> None:
    """Select the highlighted watchlist symbol in the main results table."""

    if not _MY_WATCHLIST_SYMBOLS:
        return

    listbox = _MY_WATCHLIST_LISTBOX
    if listbox is None:
        return

    selection = listbox.curselection()
    if not selection:
        return

    index = selection[0]
    if index >= len(_MY_WATCHLIST_SYMBOLS):
        return

    target_symbol = _MY_WATCHLIST_SYMBOLS[index]
    tree_widget = globals().get("tree")
    if tree_widget is None:
        return

    for item in tree_widget.get_children():
        item_values = tree_widget.item(item).get("values")
        if item_values and str(item_values[0]).strip().upper() == target_symbol:
            tree_widget.selection_set(item)
            tree_widget.focus(item)
            tree_widget.see(item)
            break


def _refresh_my_watchlist_list(
    *, target_symbol: Optional[str] = None, current_symbol: Optional[str] = None
) -> None:
    """Refresh the "My Watchlist" listbox contents."""

    listbox = _MY_WATCHLIST_LISTBOX
    if listbox is None:
        return

    normalized_target = (target_symbol or "").strip().upper() or None
    normalized_current = (current_symbol or "").strip().upper() or None

    listbox.configure(state=tk.NORMAL)
    listbox.delete(0, tk.END)
    listbox.selection_clear(0, tk.END)
    _MY_WATCHLIST_SYMBOLS.clear()

    try:
        entries = _read_my_watchlist_entries()
    except Exception as exc:  # pragma: no cover - defensive GUI feedback
        listbox.insert("end", "Unable to load My Watchlist.")
        listbox.configure(state=tk.DISABLED)
        messagebox.showwarning("My Watchlist", f"Failed to load My Watchlist:\n{exc}")
        return

    selection_index: Optional[int] = None
    for entry in entries:
        symbol_value = str(entry.get("symbol", "")).strip().upper()
        if not symbol_value:
            continue

        label = str(entry.get("label", "")).strip()
        display_text = f"{symbol_value} — {label}" if label else symbol_value

        _MY_WATCHLIST_SYMBOLS.append(symbol_value)
        current_index = len(_MY_WATCHLIST_SYMBOLS) - 1
        listbox.insert("end", display_text)

        if normalized_target and symbol_value == normalized_target:
            selection_index = current_index
        elif not normalized_target and normalized_current and symbol_value == normalized_current:
            selection_index = current_index

    if not _MY_WATCHLIST_SYMBOLS:
        listbox.insert("end", "My Watchlist is empty.")
        listbox.configure(state=tk.DISABLED)
        return

    listbox.configure(state=tk.NORMAL)
    listbox.selection_clear(0, tk.END)
    if selection_index is not None:
        listbox.selection_set(selection_index)
        listbox.activate(selection_index)
        listbox.see(selection_index)


def _init_my_watchlist_section(parent: tk.Misc) -> None:
    """Initialise the persistent "My Watchlist" section within the UI."""

    global _MY_WATCHLIST_LISTBOX

    container = tk.LabelFrame(parent, text="My Watchlist", padx=6, pady=6)
    container.pack(fill="both", expand=True, pady=(10, 0))

    list_container = tk.Frame(container, bg="#ffffff", bd=1, relief="solid")
    list_container.pack(fill="both", expand=True)

    listbox = tk.Listbox(
        list_container,
        height=8,
        exportselection=False,
        activestyle="none",
        bg="#ffffff",
        bd=0,
        highlightthickness=0,
    )
    listbox.pack(side="left", fill="both", expand=True)

    scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.configure(yscrollcommand=scrollbar.set)

    listbox.bind("<Double-Button-1>", _focus_watchlist_symbol)
    listbox.bind("<Return>", _focus_watchlist_symbol)

    tk.Label(
        container,
        text="Double-click a symbol in the list above to focus it in the main watchlist.",
        font=("Arial", 8),
        anchor="w",
        justify="left",
        wraplength=260,
    ).pack(fill="x", padx=2, pady=(6, 2))

    _MY_WATCHLIST_LISTBOX = listbox
    _refresh_my_watchlist_list()


def _extract_fast_info_price(fast_info: Any) -> Optional[float]:
    if fast_info is None:
        return None

    candidates = (
        "last_price",
        "lastPrice",
        "last",
        "regular_market_price",
        "regularMarketPrice",
        "previous_close",
        "previousClose",
    )

    for key in candidates:
        value = getattr(fast_info, key, None)
        if value is None and isinstance(fast_info, dict):
            value = fast_info.get(key)
        numeric = _coerce_numeric(value)
        if numeric is not None and numeric > 0:
            return float(numeric)

    return None


def _lookup_latest_close(symbol: str) -> Optional[float]:
    now = time.time()
    cached = _WATCHLIST_REFRESH_CACHE.get(symbol)
    if cached and now - cached[0] < _WATCHLIST_REFRESH_TTL:
        return cached[1]

    latest_close: Optional[float] = None
    try:
        ticker = yf.Ticker(symbol)
        latest_close = _extract_fast_info_price(getattr(ticker, "fast_info", None))

        if latest_close is None:
            history = ticker.history(period="5d", interval="1d", auto_adjust=True)
            history = flatten_yf_columns(history)
            if isinstance(history, pd.DataFrame) and not history.empty:
                for column in ("Close", "close", "Adj Close", "adj_close"):
                    if column in history.columns:
                        series = history[column].dropna()
                        if not series.empty:
                            candidate = _coerce_numeric(series.iloc[-1])
                            if candidate is not None and candidate > 0:
                                latest_close = float(candidate)
                        break
    except Exception:
        latest_close = None

    _WATCHLIST_REFRESH_CACHE[symbol] = (now, latest_close)
    return latest_close


def _maybe_refresh_watchlist_row(
    row: dict[str, Any],
    price_cache: dict[str, Optional[float]],
) -> bool:
    symbol = row.get("symbol")
    if not symbol:
        return False

    stored_close = _coerce_numeric(row.get("last_close"))

    if symbol in price_cache:
        latest_close = price_cache[symbol]
    else:
        latest_close = _lookup_latest_close(symbol)
        price_cache[symbol] = latest_close

    if latest_close is None or not math.isfinite(latest_close) or latest_close <= 0:
        return False

    updated = False
    if stored_close is None or not math.isfinite(stored_close) or stored_close <= 0:
        scale = None
    else:
        scale = latest_close / stored_close if stored_close else None
        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            scale = None

    if stored_close is None or abs(latest_close - stored_close) >= 0.01:
        row["last_close"] = f"{latest_close:.2f}"
        updated = True

    if scale is not None:
        for key in ("breakout_high", "target_price", "stop_loss"):
            numeric = _coerce_numeric(row.get(key))
            if numeric is None:
                continue
            adjusted = round(numeric * scale, 2)
            if abs(adjusted - numeric) >= 0.01:
                row[key] = f"{adjusted:.2f}"
                updated = True

    return updated


def _format_watchlist_value(column: str, value: Any, row: Optional[dict[str, Any]] = None) -> str:
    if column in _WATCHLIST_PRICE_COLUMNS:
        formatted_value = _format_price(value)
        return formatted_value

    if column == "rr_ratio":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value) if value not in (None, "") else "N/A"
        if not math.isfinite(numeric):
            return "N/A"
        return f"{numeric:.2f}"

    if value in (None, ""):
        return ""

    if isinstance(value, (float, int)) and pd.isna(value):
        return ""

    return str(value)


def _normalise_watchlist_fieldnames(fieldnames: Sequence[str] | None) -> list[str]:
    """Return a normalised list of watchlist fieldnames."""

    normalised = list(fieldnames or [])

    if "symbol" in normalised and "last_close" not in normalised:
        insert_at = normalised.index("symbol") + 1
        normalised.insert(insert_at, "last_close")
    elif "last_close" not in normalised:
        normalised.append("last_close")

    seen = set(normalised)
    for column in WATCHLIST_HEADER:
        if column not in seen:
            normalised.append(column)
            seen.add(column)

    return normalised


def _read_watchlist_rows() -> tuple[list[str], list[dict[str, Any]]]:
    """Read the persisted watchlist file and return its rows."""

    if not WATCHLIST_PATH.exists():
        return list(WATCHLIST_HEADER), []

    with WATCHLIST_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = _normalise_watchlist_fieldnames(reader.fieldnames or WATCHLIST_HEADER)
        rows: list[dict[str, Any]] = []
        for raw_row in reader:
            if not raw_row:
                continue
            rows.append({key: raw_row.get(key, "") for key in fieldnames})
    return fieldnames, rows


def _read_my_watchlist_entries() -> list[dict[str, str]]:
    """Return the persisted "My Watchlist" entries."""

    if not MY_WATCHLIST_PATH.exists():
        return []

    with MY_WATCHLIST_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    entries: list[dict[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                entries.append({"symbol": item.strip().upper(), "label": ""})
            elif isinstance(item, dict):
                symbol = str(item.get("symbol", "")).strip().upper()
                label = str(item.get("label", "")).strip()
                if symbol:
                    entries.append({"symbol": symbol, "label": label})
    return entries


def _write_my_watchlist_entries(entries: Sequence[dict[str, str]]) -> None:
    serialised: list[dict[str, str]] = []
    for entry in entries:
        symbol = str(entry.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        label = str(entry.get("label", "")).strip()
        serialised.append({"symbol": symbol, "label": label})

    MY_WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MY_WATCHLIST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(serialised, handle, ensure_ascii=False, indent=2)


def add_symbol_to_my_watchlist(symbol: str, *, label: str = "") -> bool:
    """Add the provided symbol to the personalised "My Watchlist" list."""

    normalized = (symbol or "").strip().upper()
    if not normalized:
        messagebox.showerror("My Watchlist", "Please provide a valid ticker symbol.")
        return False

    try:
        entries = _read_my_watchlist_entries()
    except Exception as exc:  # pragma: no cover - defensive GUI feedback
        messagebox.showerror("My Watchlist", f"Failed to read My Watchlist:\n{exc}")
        return False

    existing: dict[str, dict[str, str]] = {entry["symbol"]: dict(entry) for entry in entries if entry.get("symbol")}
    entry = existing.get(normalized)
    if entry is None:
        entry = {"symbol": normalized, "label": label.strip()}
        entries.append(entry)
    else:
        if not entry.get("label") and label:
            entry["label"] = label.strip()

    try:
        _write_my_watchlist_entries(entries)
    except Exception as exc:  # pragma: no cover - defensive GUI feedback
        messagebox.showerror("My Watchlist", f"Failed to update My Watchlist:\n{exc}")
        return False

    return True


def add_symbol_to_watchlist(symbol: str, *, source_label: str = "Screener Candidate") -> bool:
    """Persist a manually provided symbol to the shared watchlist."""

    normalized = (symbol or "").strip().upper()
    if not normalized:
        messagebox.showerror("Watchlist", "Please provide a valid ticker symbol.")
        return False

    try:
        fieldnames, rows = _read_watchlist_rows()
    except Exception as exc:  # pragma: no cover - defensive GUI feedback
        messagebox.showerror("Watchlist", f"Failed to read the watchlist:\n{exc}")
        return False

    entries: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_symbol = (row.get("symbol") or "").strip().upper()
        if row_symbol:
            entries[row_symbol] = {key: row.get(key, "") for key in fieldnames}

    if normalized in entries:
        entry = entries[normalized]
    else:
        entry = {key: "" for key in fieldnames}

    entry["symbol"] = normalized
    if source_label and not entry.get("pattern"):
        entry["pattern"] = source_label

    if "direction" in fieldnames and not entry.get("direction"):
        entry["direction"] = "bullish"

    entry["timestamp"] = datetime.now().strftime("%m-%d %H:%M")

    price_cache: dict[str, Optional[float]] = {}
    _maybe_refresh_watchlist_row(entry, price_cache)

    entries[normalized] = entry

    serialised_rows = [
        {key: value if value is not None else "" for key, value in entries[symbol_key].items() if key in fieldnames}
        for symbol_key in sorted(entries)
    ]

    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with WATCHLIST_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in serialised_rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    except Exception as exc:  # pragma: no cover - defensive GUI feedback
        messagebox.showerror("Watchlist", f"Failed to update the watchlist:\n{exc}")
        return False

    return True


def _coerce_numeric(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return numeric
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


NA_TEXT_COLOR = "#000000"


def _score_to_color(score: int) -> str:
    if score > 0:
        return "#0f9d58"
    if score < 0:
        return "#d93025"
    return "#000000"


def _score_to_icon(score: int) -> str:
    if score > 0:
        return "↑"
    if score < 0:
        return "↓"
    return "↔"


def _score_mfi(value: Any) -> int:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return 0
    if numeric <= 20:
        return 1
    if numeric >= 80:
        return -1
    return 0


def _score_rsi(value: Any) -> int:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return 0
    if numeric <= 30:
        return 1
    if numeric >= 70:
        return -1
    return 0


def _score_ad_slope(value: Any) -> int:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return 0
    if numeric > 0:
        return 1
    if numeric < 0:
        return -1
    return 0


def _score_obv_trend(value: Any) -> int:
    if not value:
        return 0
    text = str(value).lower()
    if any(keyword in text for keyword in ("up", "rise", "bull")):
        return 1
    if any(keyword in text for keyword in ("down", "fall", "bear")):
        return -1
    return 0


def _score_beta(value: Any) -> int:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return 0
    if numeric <= 0.9:
        return 1
    if numeric >= 1.3:
        return -1
    return 0


def _score_signal(description: str) -> int:
    if not description:
        return 0
    lowered = description.lower()
    if "bullish" in lowered:
        return 1
    if "bearish" in lowered:
        return -1
    return 0


def _score_positive_growth(value: Any) -> int:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return 0
    if numeric > 0:
        return 1
    if numeric < 0:
        return -1
    return 0


def _score_debt_to_equity(value: Any) -> int:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return 0
    if numeric <= 0.5:
        return 1
    if numeric >= 1.5:
        return -1
    return 0


def _score_roe(value: Any) -> int:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return 0
    if numeric >= 15:
        return 1
    if numeric < 5:
        return -1
    return 0


def _score_peg(value: Any) -> int:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return 0
    if numeric <= 1:
        return 1
    if numeric >= 2:
        return -1
    return 0


def _score_pe(value: Any) -> int:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return 0
    if 10 <= numeric <= 25:
        return 1
    if numeric < 5 or numeric > 40:
        return -1
    return 0


def _score_dividend_yield(value: Any) -> int:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return 0
    if numeric >= 0.03:
        return 1
    if numeric <= 0:
        return -1
    return 0


def _score_institutional_pct(value: Any) -> int:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return 0
    if numeric >= 0.6:
        return 1
    if numeric <= 0.3:
        return -1
    return 0


def _score_institutional_trend(value: Any) -> int:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return 0
    if numeric > 0:
        return 1
    if numeric < 0:
        return -1
    return 0


def _compute_institutional_strength(snapshot_summary: dict[str, Any]) -> Optional[int]:
    if not snapshot_summary:
        return None

    scores: list[int] = []
    scores.append(_score_institutional_pct(snapshot_summary.get("pct_held_by_institutions")))
    scores.append(_score_institutional_trend(snapshot_summary.get("inst_shares_qoq_pct")))
    scores.append(_score_positive_growth(snapshot_summary.get("insider_buys_3m")))
    scores.append(-_score_positive_growth(snapshot_summary.get("insider_sells_3m")))

    valid_scores = [score for score in scores if score is not None]
    if not valid_scores:
        return None

    # Map -1/0/1 to 0/50/100 to create an intuitive scorecard.
    total = sum(valid_scores)
    max_total = len(valid_scores)
    normalized = (total / max_total + 1) / 2  # Range 0-1
    return int(round(normalized * 100))


def _format_percent(value: Optional[float], *, signed: bool = False) -> str:
    if value is None or (isinstance(value, (float, int)) and pd.isna(value)):
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)

    if signed:
        return f"{numeric:+.2f}%"
    return f"{numeric:.2f}%"


def _format_int(value: Optional[float]) -> str:
    if value is None or (isinstance(value, (float, int)) and pd.isna(value)):
        return "N/A"
    try:
        return f"{int(value)}"
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


def _format_shares(value: Optional[float]) -> str:
    if value is None or (isinstance(value, (float, int)) and pd.isna(value)):
        return "N/A"
    return _format_share_amount(value)


def _format_decimal(value: Optional[float], *, decimals: int = 2) -> str:
    if value is None or (isinstance(value, (float, int)) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


def _nice_tick_size(value: float, *, default: float = 1.0) -> float:
    if not value or not math.isfinite(value):
        return default

    value = abs(value)
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return nice_fraction * (10 ** exponent)


def _derive_signal(
    mfi_value: Optional[float], rsi_value: Optional[float]
) -> tuple[str, str]:
    metrics: list[float] = []
    for raw in (mfi_value, rsi_value):
        if raw is None:
            continue
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        metrics.append(numeric)

    if not metrics:
        return "⚪", "Neutral"

    if any(value >= 65 for value in metrics):
        return "🟢", "Bullish"
    if any(value <= 35 for value in metrics):
        return "🔴", "Bearish"

    average = sum(metrics) / len(metrics)
    if average >= 55:
        return "⚪", "Neutral-Bullish"
    if average <= 45:
        return "⚪", "Neutral-Bearish"
    return "⚪", "Neutral"


def _coerce_float(value: Any) -> Optional[float]:
    """Attempt to coerce Yahoo Finance numeric payloads into floats."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            if math.isnan(float(value)):
                return None
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
        return float(value)

    if isinstance(value, dict):
        for key in ("raw", "value", "amount", "fmt"):
            if key in value:
                coerced = _coerce_float(value[key])
                if coerced is not None:
                    return coerced
        return None

    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None

        multiplier = 1.0
        upper = cleaned.upper()
        for suffix, factor in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
            if upper.endswith(suffix):
                multiplier = factor
                cleaned = cleaned[: -len(suffix)]
                break

        cleaned = cleaned.replace("$", "").replace("%", "")
        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None

    return None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, int)):
        try:
            return pd.isna(value)
        except Exception:  # pragma: no cover - defensive
            return False
    if isinstance(value, str):
        return not value.strip()
    return False


def _assign_if_missing(summary: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    current = summary.get(key)
    if _is_missing(current):
        summary[key] = value


def _fetch_quote_summary(
    symbol: str, modules: tuple[str, ...]
) -> tuple[dict[str, Any], Optional[str]]:
    """Fetch and cache Yahoo Finance quote summary data for the given modules."""

    cache_key = (symbol, modules)
    now = time.time()
    cached = _quote_summary_cache.get(cache_key)
    if cached and now - cached[0] < QUOTE_SUMMARY_CACHE_TTL:
        return cached[1], None

    module_param = ",".join(dict.fromkeys(modules))
    url = (
        f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules={module_param}"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request) as response:
            payload = json.load(response)
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        return {}, f"Unable to load quote summary ({reason})."
    except Exception as exc:  # pragma: no cover - defensive
        return {}, f"Unable to load quote summary ({exc})."

    try:
        results = payload.get("quoteSummary", {}).get("result", [])
        if not results:
            error = payload.get("quoteSummary", {}).get("error")
            if isinstance(error, dict):
                message = error.get("description") or error.get("message")
                if message:
                    return {}, f"Unable to load quote summary ({message})."
            return {}, "Unable to load quote summary."
        summary = results[0]
    except AttributeError:
        return {}, "Unable to load quote summary."

    _quote_summary_cache[cache_key] = (now, summary)
    return summary, None


def _fetch_alpha_vantage_payload(
    symbol: str, function: str
) -> tuple[dict[str, Any], Optional[str]]:
    if not ALPHAVANTAGE_API_KEY:
        return {}, "Alpha Vantage API key not configured."

    cache_key = (symbol, function)
    now = time.time()
    cached = _alpha_vantage_cache.get(cache_key)
    if cached and now - cached[0] < ALPHAVANTAGE_CACHE_TTL:
        return cached[1], None

    params = urlencode(
        {
            "function": function,
            "symbol": symbol,
            "apikey": ALPHAVANTAGE_API_KEY,
        }
    )
    url = f"https://www.alphavantage.co/query?{params}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request) as response:
            payload = json.load(response)
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        return {}, f"Unable to load Alpha Vantage data ({reason})."
    except Exception as exc:  # pragma: no cover - defensive
        return {}, f"Unable to load Alpha Vantage data ({exc})."

    if not isinstance(payload, dict):
        return {}, "Unexpected Alpha Vantage response."

    if payload.get("Error Message"):
        return {}, payload.get("Error Message")

    for key in ("Note", "Information"):
        if key in payload:
            return {}, payload[key]

    _alpha_vantage_cache[cache_key] = (now, payload)
    return payload, None


def _merge_alpha_vantage_overview(
    summary: dict[str, Any], payload: dict[str, Any], close_price: Optional[float]
) -> None:
    if not payload:
        return

    def update_numeric(field: str, key: str, *, percent: bool = False) -> None:
        raw = payload.get(key)
        value = _coerce_float(raw)
        if value is None:
            return
        if percent and abs(value) <= 1:
            value *= 100.0
        _assign_if_missing(summary, field, float(value))

    update_numeric("pe_ttm", "PERatio")
    update_numeric("forward_pe", "ForwardPE")
    update_numeric("peg_ratio", "PEGRatio")
    update_numeric("dividend_yield_pct", "DividendYield", percent=True)
    update_numeric("roe_pct", "ReturnOnEquityTTM", percent=True)
    update_numeric("de_ratio", "DebtToEquity")
    update_numeric("eps_growth_yoy_pct", "QuarterlyEarningsGrowthYOY", percent=True)
    update_numeric("revenue_growth_yoy_pct", "QuarterlyRevenueGrowthYOY", percent=True)

    free_cash_flow = _coerce_float(payload.get("FreeCashFlowTTM"))
    if free_cash_flow is not None:
        _assign_if_missing(summary, "free_cash_flow", float(free_cash_flow))

    target_price = _coerce_float(payload.get("AnalystTargetPrice"))
    if target_price is not None:
        _assign_if_missing(summary, "avg_target_price", float(target_price))
        if not _is_missing(summary.get("avg_target_upside_pct")):
            pass
        elif close_price and close_price != 0:
            upside = (float(target_price) - close_price) / close_price * 100.0
            summary["avg_target_upside_pct"] = float(upside)

    beta_value = _coerce_float(payload.get("Beta"))
    if beta_value is not None:
        _assign_if_missing(summary, "beta", float(beta_value))

    next_earnings = payload.get("NextEarningsDate")
    parsed_earnings = _parse_report_datetime(next_earnings)
    if parsed_earnings is not None:
        _assign_if_missing(summary, "next_earnings", parsed_earnings)

    sector_name = payload.get("Sector")
    if isinstance(sector_name, str) and sector_name.strip():
        cleaned = sector_name.strip()
        _assign_if_missing(summary, "sector", cleaned)
        if _is_missing(summary.get("sector_etf")):
            summary["sector_etf"] = SECTOR_ETF_MAP.get(cleaned)


def _merge_alpha_vantage_rating(summary: dict[str, Any], payload: dict[str, Any]) -> None:
    if not payload:
        return

    recommendation = payload.get("ratingRecommendation")
    if isinstance(recommendation, str) and recommendation.strip():
        _assign_if_missing(summary, "analyst_rating", recommendation.strip().title())

    target = _coerce_float(payload.get("ratingDetailsTargetPriceScore"))
    if target is not None:
        summary.setdefault("_alpha_rating_score", float(target))

def _normalise_report_date(raw: Any) -> str:
    if raw is None:
        return "N/A"
    if isinstance(raw, str):
        cleaned = raw.strip()
        return cleaned or "N/A"
    if isinstance(raw, (datetime, pd.Timestamp)):
        return raw.strftime("%Y-%m-%d")
    if pd.isna(raw):
        return "N/A"
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(raw).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            return str(raw)
    return str(raw)


def _parse_report_datetime(raw: Any) -> Optional[datetime]:
    if raw in (None, "", "N/A"):
        return None

    if isinstance(raw, (int, float)):
        for divisor in (1, 1000):
            try:
                parsed = datetime.fromtimestamp(raw / divisor)
            except (OverflowError, OSError, ValueError):
                continue
            else:
                if parsed.year >= 1970:
                    return parsed
        return None

    try:
        parsed = pd.to_datetime(raw, errors="coerce")
    except Exception:  # pragma: no cover - defensive
        return None

    if pd.isna(parsed):
        return None

    if hasattr(parsed, "to_pydatetime"):
        return parsed.to_pydatetime()

    return None


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    raw_money_flow = typical_price * df["Volume"]

    positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0.0)
    negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0.0)

    pos_mf = pd.Series(positive_flow, index=df.index).rolling(period, min_periods=period).sum()
    neg_mf = pd.Series(negative_flow, index=df.index).rolling(period, min_periods=period).sum()

    money_flow_ratio = pos_mf / neg_mf.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi.fillna(method="bfill").fillna(50.0)


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"])
    clv = clv.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return (clv * df["Volume"]).cumsum()


def relative_strength_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the RSI for the provided OHLC dataframe."""

    delta = df["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    avg_gain = avg_gain.fillna(method="bfill")
    avg_loss = avg_loss.fillna(method="bfill")

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """Compute the On-Balance Volume (OBV) for the provided dataframe."""

    obv = [0.0]
    closes = df["Close"].to_numpy(dtype=float)
    volumes = df["Volume"].to_numpy(dtype=float)
    for idx in range(1, len(df)):
        if closes[idx] > closes[idx - 1]:
            obv.append(obv[-1] + volumes[idx])
        elif closes[idx] < closes[idx - 1]:
            obv.append(obv[-1] - volumes[idx])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)


def average_true_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Average True Range (ATR)."""

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.rolling(period, min_periods=period).mean()
    return atr


def slope(series: pd.Series, lookback: int = 60) -> float:
    clean = series.dropna()
    if len(clean) < lookback:
        return float("nan")

    y = clean.iloc[-lookback:].to_numpy(dtype=float)
    x = np.arange(len(y))
    denominator = x - x.mean()
    denominator = np.sum(denominator ** 2)
    if denominator == 0:
        return float("nan")

    numerator = np.sum((x - x.mean()) * (y - y.mean()))
    return float(numerator / denominator)


def _normalise_share_change_percent(entry: dict[str, Any]) -> Optional[float]:
    for key in (
        "pctChange",
        "sharesChangePercent",
        "shareChangePercent",
        "sharePercentChange",
        "percentageChange",
    ):
        value = entry.get(key)
        numeric = _coerce_float(value)
        if numeric is None:
            continue
        if abs(numeric) <= 1:
            return numeric * 100
        return numeric
    return None


def fetch_institution_activity(symbol: str) -> tuple[list[dict[str, Any]], Optional[str]]:
    now = time.time()
    cached = _institution_activity_cache.get(symbol)
    if cached and now - cached[0] < INSTITUTION_CACHE_TTL:
        return cached[1], None

    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=institutionOwnership"
    try:
        with urlopen(url, timeout=10) as response:
            payload = json.load(response)
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        return [], f"Unable to load institutional activity ({reason})."
    except Exception as exc:  # pragma: no cover - defensive
        return [], f"Unable to load institutional activity ({exc})."

    try:
        results = payload.get("quoteSummary", {}).get("result", [])
        if not results:
            _institution_activity_cache[symbol] = (now, [])
            return [], None
        ownership = results[0].get("institutionOwnership", {})
        raw_entries = ownership.get("ownershipList", [])
    except AttributeError:
        _institution_activity_cache[symbol] = (now, [])
        return [], None

    parsed_entries: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        net_activity = _coerce_float(entry.get("netActivity"))
        purchased = _coerce_float(entry.get("purchased"))
        sold = _coerce_float(entry.get("soldOut") or entry.get("sold"))
        organization = entry.get("organization") or "Unknown"
        report_datetime = _parse_report_datetime(entry.get("reportDate"))
        shares_held = _coerce_float(entry.get("position") or entry.get("sharesHeld"))
        share_change_pct = _normalise_share_change_percent(entry)
        market_value = _coerce_float(entry.get("value"))
        if net_activity is None:
            net_activity = _coerce_float(entry.get("sharesChanged"))
        if purchased is None:
            purchased = 0.0
        if sold is None:
            sold = 0.0
        if net_activity is None:
            net_activity = 0.0
        parsed_entries.append(
            {
                "organization": organization,
                "net_activity": net_activity,
                "purchased": purchased,
                "sold": sold,
                "shares_held": shares_held,
                "share_change_pct": share_change_pct,
                "market_value": market_value,
                "report_date": _normalise_report_date(entry.get("reportDate")),
                "report_datetime": report_datetime,
            }
        )

    parsed_entries.sort(key=lambda item: abs(item.get("net_activity", 0)), reverse=True)
    cutoff = datetime.utcnow() - timedelta(days=182)
    recent_entries = [
        entry for entry in parsed_entries if entry.get("report_datetime") and entry["report_datetime"] >= cutoff
    ]

    _institution_activity_cache[symbol] = (now, recent_entries)
    return recent_entries, None


def fetch_institution_snapshot(
    symbol: str, price_history: Optional[pd.DataFrame] = None
) -> tuple[dict[str, Any], Optional[str]]:
    now = time.time()
    cached = _institution_snapshot_cache.get(symbol)
    if cached and now - cached[0] < INSTITUTION_CACHE_TTL:
        return cached[1], None

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    hist = None
    if price_history is not None:
        normalized_columns = {str(col).strip().title() for col in price_history.columns}
        if required_cols.issubset(normalized_columns):
            hist = price_history.copy()

    if hist is None or hist.empty:
        try:
            fetched = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
        except Exception as exc:  # pragma: no cover - network
            return {}, f"Unable to load price history ({exc})."
        hist = flatten_yf_columns(fetched)

    if hist is None or hist.empty:
        return {}, "Unable to load price history."

    hist = hist.rename(columns=str.title)
    if not required_cols.issubset(hist.columns):
        return {}, "Incomplete price history for institutional summary."

    hist = hist.sort_index()
    ohlcv = hist[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if ohlcv.empty:
        return {}, "Insufficient data for institutional summary."

    mfi_14 = money_flow_index(ohlcv, period=14).iloc[-1]
    ad_line = accumulation_distribution(ohlcv)
    ad_slope_60 = slope(ad_line, lookback=60)
    price_change_21d = float("nan")
    if len(ohlcv) >= 21:
        price_change_21d = (ohlcv["Close"].iloc[-1] / ohlcv["Close"].iloc[-21] - 1.0) * 100.0

    close_price = float(ohlcv["Close"].iloc[-1])

    rsi_14 = relative_strength_index(ohlcv, period=14).iloc[-1]
    obv_series = on_balance_volume(ohlcv)
    obv_lookback = min(len(obv_series), 30)
    obv_slope_30 = slope(obv_series, lookback=obv_lookback) if obv_lookback >= 2 else float("nan")
    if math.isnan(obv_slope_30):
        obv_trend = "→"
    elif obv_slope_30 > 0:
        obv_trend = "↑"
    elif obv_slope_30 < 0:
        obv_trend = "↓"
    else:
        obv_trend = "→"

    volume_avg_30 = float("nan")
    if len(ohlcv) >= 5:
        volume_avg_30 = float(ohlcv["Volume"].tail(30).mean())

    atr_value = float("nan")
    try:
        atr_series = average_true_range(ohlcv, period=14)
        if atr_series.notna().any():
            atr_value = float(atr_series.dropna().iloc[-1])
    except Exception:  # pragma: no cover - defensive
        atr_value = float("nan")

    inst_summary = {
        "close": round(close_price, 2),
        "mfi_14": round(float(mfi_14), 2) if not math.isnan(float(mfi_14)) else float("nan"),
        "rsi_14": round(float(rsi_14), 2) if not math.isnan(float(rsi_14)) else float("nan"),
        "ad_slope_60": round(float(ad_slope_60), 2) if not math.isnan(ad_slope_60) else float("nan"),
        "price_chg_21d_pct": round(float(price_change_21d), 2)
        if not math.isnan(price_change_21d)
        else float("nan"),
        "volume_avg_30": round(float(volume_avg_30), 2) if not math.isnan(volume_avg_30) else float("nan"),
        "obv_trend": obv_trend,
        "obv_slope_30": round(float(obv_slope_30), 4) if not math.isnan(obv_slope_30) else float("nan"),
        "inst_count": float("nan"),
        "inst_shares_sum": float("nan"),
        "inst_latest_report": None,
        "pct_held_by_institutions": float("nan"),
        "beta": float("nan"),
        "top_holders": [],
        "insider_buys_3m": None,
        "insider_sells_3m": None,
        "pe_ttm": float("nan"),
        "forward_pe": float("nan"),
        "peg_ratio": float("nan"),
        "dividend_yield_pct": float("nan"),
        "roe_pct": float("nan"),
        "de_ratio": float("nan"),
        "eps_growth_yoy_pct": float("nan"),
        "revenue_growth_yoy_pct": float("nan"),
        "free_cash_flow": float("nan"),
        "analyst_rating": None,
        "analyst_buy_count": None,
        "analyst_hold_count": None,
        "analyst_sell_count": None,
        "avg_target_price": float("nan"),
        "avg_target_upside_pct": float("nan"),
        "short_interest_pct_float": float("nan"),
        "next_earnings": None,
        "sector": None,
        "sector_etf": None,
        "correlation": float("nan"),
        "atr_14": round(float(atr_value), 2) if not math.isnan(atr_value) else float("nan"),
    }

    modules = (
        "financialData",
        "summaryDetail",
        "defaultKeyStatistics",
        "calendarEvents",
        "recommendationTrend",
        "assetProfile",
        "price",
    )
    quote_summary, _ = _fetch_quote_summary(symbol, modules)
    if quote_summary:
        financial_data = quote_summary.get("financialData") or {}
        summary_detail = quote_summary.get("summaryDetail") or {}
        statistics = quote_summary.get("defaultKeyStatistics") or {}
        calendar_events = quote_summary.get("calendarEvents") or {}
        recommendation_trend = quote_summary.get("recommendationTrend") or {}
        asset_profile = quote_summary.get("assetProfile") or {}
        price_section = quote_summary.get("price") or {}

        trailing_pe = _coerce_float(
            statistics.get("trailingPE") or summary_detail.get("trailingPE")
        )
        if trailing_pe is not None:
            inst_summary["pe_ttm"] = float(trailing_pe)

        forward_pe = _coerce_float(
            statistics.get("forwardPE") or summary_detail.get("forwardPE")
        )
        if forward_pe is not None:
            inst_summary["forward_pe"] = float(forward_pe)

        peg_ratio = _coerce_float(statistics.get("pegRatio") or financial_data.get("pegRatio"))
        if peg_ratio is not None:
            inst_summary["peg_ratio"] = float(peg_ratio)

        dividend_yield = _coerce_float(summary_detail.get("dividendYield"))
        if dividend_yield is not None:
            if abs(dividend_yield) <= 1:
                dividend_yield *= 100.0
            inst_summary["dividend_yield_pct"] = float(dividend_yield)

        roe_value = _coerce_float(financial_data.get("returnOnEquity"))
        if roe_value is not None:
            if abs(roe_value) <= 1:
                roe_value *= 100.0
            inst_summary["roe_pct"] = float(roe_value)

        debt_to_equity = _coerce_float(
            financial_data.get("debtToEquity") or statistics.get("debtToEquity")
        )
        if debt_to_equity is not None:
            inst_summary["de_ratio"] = float(debt_to_equity)

        eps_growth = _coerce_float(
            financial_data.get("earningsQuarterlyGrowth")
            or financial_data.get("earningsGrowth")
        )
        if eps_growth is not None:
            if abs(eps_growth) <= 1:
                eps_growth *= 100.0
            inst_summary["eps_growth_yoy_pct"] = float(eps_growth)

        revenue_growth = _coerce_float(financial_data.get("revenueGrowth"))
        if revenue_growth is not None:
            if abs(revenue_growth) <= 1:
                revenue_growth *= 100.0
            inst_summary["revenue_growth_yoy_pct"] = float(revenue_growth)

        free_cash_flow = _coerce_float(
            financial_data.get("freeCashflow") or financial_data.get("freeCashFlow")
        )
        if free_cash_flow is not None:
            inst_summary["free_cash_flow"] = float(free_cash_flow)

        target_price = _coerce_float(financial_data.get("targetMeanPrice"))
        if target_price is None:
            target_price = _coerce_float(financial_data.get("targetMedianPrice"))
        if target_price is not None:
            inst_summary["avg_target_price"] = float(target_price)
            if close_price:
                try:
                    if close_price != 0:
                        upside = (target_price - close_price) / close_price * 100.0
                    else:
                        upside = float("nan")
                except Exception:  # pragma: no cover - defensive
                    upside = float("nan")
                if not math.isnan(upside):
                    inst_summary["avg_target_upside_pct"] = float(upside)

        short_interest = _coerce_float(
            statistics.get("shortPercentOfFloat")
            or summary_detail.get("shortPercentOfFloat")
        )
        if short_interest is not None:
            if abs(short_interest) <= 1:
                short_interest *= 100.0
            inst_summary["short_interest_pct_float"] = float(short_interest)

        beta_candidate = _coerce_float(
            summary_detail.get("beta") or statistics.get("beta")
        )
        if beta_candidate is not None:
            inst_summary["beta"] = float(beta_candidate)

        recommendation_key = financial_data.get("recommendationKey")
        if isinstance(recommendation_key, str) and recommendation_key.strip():
            inst_summary["analyst_rating"] = recommendation_key.replace("_", " ").title()

        if isinstance(recommendation_trend, dict):
            trend_list = recommendation_trend.get("trend")
        else:
            trend_list = recommendation_trend
        if isinstance(trend_list, list) and trend_list:
            latest_trend = trend_list[0]

            def _extract_trend_count(*keys: str) -> Optional[int]:
                total = 0.0
                found = False
                for key in keys:
                    if not isinstance(latest_trend, dict):
                        break
                    value = _coerce_float(latest_trend.get(key))
                    if value is None:
                        continue
                    total += value
                    found = True
                if not found:
                    return None
                return int(round(total))

            buy_count = _extract_trend_count("strongBuy", "buy")
            hold_count = _extract_trend_count("hold")
            sell_count = _extract_trend_count("strongSell", "sell")

            if buy_count is not None:
                inst_summary["analyst_buy_count"] = buy_count
            if hold_count is not None:
                inst_summary["analyst_hold_count"] = hold_count
            if sell_count is not None:
                inst_summary["analyst_sell_count"] = sell_count

        earnings_info = calendar_events.get("earnings") if isinstance(calendar_events, dict) else {}
        earnings_dates = []
        if isinstance(earnings_info, dict):
            earnings_dates = earnings_info.get("earningsDate") or []
        if not isinstance(earnings_dates, list):
            earnings_dates = [earnings_dates]
        for entry in earnings_dates:
            candidate = None
            if isinstance(entry, dict):
                candidate = _parse_report_datetime(entry.get("raw"))
                if candidate is None:
                    candidate = _parse_report_datetime(entry.get("fmt"))
            else:
                candidate = _parse_report_datetime(entry)
            if candidate is not None:
                inst_summary["next_earnings"] = candidate
                break

        sector_name = None
        if isinstance(asset_profile, dict):
            sector_name = asset_profile.get("sector")
        if not sector_name and isinstance(price_section, dict):
            sector_name = price_section.get("sector")
        if isinstance(sector_name, str) and sector_name.strip():
            cleaned_sector = sector_name.strip()
            inst_summary["sector"] = cleaned_sector
            inst_summary["sector_etf"] = SECTOR_ETF_MAP.get(cleaned_sector)

    try:
        ticker = yf.Ticker(symbol)
        holders = ticker.institutional_holders
        if isinstance(holders, pd.DataFrame) and not holders.empty:
            inst_summary["inst_count"] = float(holders.shape[0])
            if "Shares" in holders.columns:
                inst_summary["inst_shares_sum"] = float(holders["Shares"].fillna(0).sum())

            for column_name in ("Date Reported", "Date", "date"):
                if column_name in holders.columns:
                    latest = pd.to_datetime(holders[column_name], errors="coerce").max()
                    if pd.notna(latest):
                        inst_summary["inst_latest_report"] = latest.to_pydatetime()
                    break

            top_holders: list[dict[str, Any]] = []
            normalized_columns = {str(col).strip().lower(): col for col in holders.columns}
            holder_col = normalized_columns.get("holder") or normalized_columns.get("name")
            pct_columns = [
                col
                for key, col in normalized_columns.items()
                if any(token in key for token in ("pct", "%", "percent"))
            ]
            for _, row in holders.head(3).iterrows():
                holder_name = None
                if holder_col:
                    holder_name = row.get(holder_col)
                if holder_name is None:
                    for candidate in ("Holder", "holder", "Name", "name"):
                        if candidate in row.index:
                            holder_name = row.get(candidate)
                            if holder_name is not None:
                                break
                pct_value: Optional[float] = None
                for pct_col in pct_columns:
                    raw_pct = row.get(pct_col)
                    numeric_pct = _coerce_float(raw_pct)
                    if numeric_pct is None:
                        continue
                    if abs(numeric_pct) <= 1:
                        numeric_pct *= 100.0
                    pct_value = float(numeric_pct)
                    break
                if holder_name is None and pct_value is None:
                    continue
                top_holders.append(
                    {
                        "name": holder_name or "Unknown",
                        "pct": pct_value,
                    }
                )
            inst_summary["top_holders"] = top_holders

        major = ticker.get_major_holders()
        if isinstance(major, pd.DataFrame) and not major.empty:
            major = major.copy()
            if major.shape[1] >= 2:
                major = major.iloc[:, :2]
                major.columns = ["metric", "value"]
            else:
                first_col = major.columns[0]
                major = major.rename(columns={first_col: "metric"})
                major["value"] = pd.NA

            major["metric"] = major["metric"].astype(str)
            mask = major["metric"].str.contains("Institutions", case=False, na=False)
            if mask.any():
                value_series = major.loc[mask, "value"].dropna()
                if not value_series.empty:
                    pct_value = _coerce_float(value_series.iloc[0])
                    if pct_value is not None:
                        if abs(pct_value) <= 1:
                            pct_value *= 100.0
                        inst_summary["pct_held_by_institutions"] = float(pct_value)

        info: dict[str, Any] = {}
        try:
            raw_info = ticker.get_info()
        except Exception:
            raw_info = {}
        if isinstance(raw_info, dict):
            info = raw_info

        trailing_pe = _coerce_float(info.get("trailingPE")) if info else None
        if trailing_pe is not None:
            inst_summary["pe_ttm"] = float(trailing_pe)

        forward_pe = _coerce_float(info.get("forwardPE")) if info else None
        if forward_pe is not None:
            inst_summary["forward_pe"] = float(forward_pe)

        peg_ratio = _coerce_float(info.get("pegRatio")) if info else None
        if peg_ratio is not None:
            inst_summary["peg_ratio"] = float(peg_ratio)

        dividend_yield = _coerce_float(info.get("dividendYield")) if info else None
        if dividend_yield is not None:
            if abs(dividend_yield) <= 1:
                dividend_yield *= 100.0
            inst_summary["dividend_yield_pct"] = float(dividend_yield)

        roe_value = _coerce_float(info.get("returnOnEquity")) if info else None
        if roe_value is not None:
            if abs(roe_value) <= 1:
                roe_value *= 100.0
            inst_summary["roe_pct"] = float(roe_value)

        debt_to_equity = _coerce_float(info.get("debtToEquity")) if info else None
        if debt_to_equity is not None:
            inst_summary["de_ratio"] = float(debt_to_equity)

        eps_growth = _coerce_float(info.get("earningsQuarterlyGrowth")) if info else None
        if eps_growth is None and info:
            eps_growth = _coerce_float(info.get("earningsGrowth"))
        if eps_growth is not None:
            if abs(eps_growth) <= 1:
                eps_growth *= 100.0
            inst_summary["eps_growth_yoy_pct"] = float(eps_growth)

        revenue_growth = _coerce_float(info.get("revenueGrowth")) if info else None
        if revenue_growth is not None:
            if abs(revenue_growth) <= 1:
                revenue_growth *= 100.0
            inst_summary["revenue_growth_yoy_pct"] = float(revenue_growth)

        free_cash_flow = _coerce_float(info.get("freeCashflow")) if info else None
        if free_cash_flow is not None:
            inst_summary["free_cash_flow"] = float(free_cash_flow)

        avg_target_price = _coerce_float(info.get("targetMeanPrice")) if info else None
        if avg_target_price is None and info:
            avg_target_price = _coerce_float(info.get("targetMedianPrice"))
        if avg_target_price is not None:
            inst_summary["avg_target_price"] = float(avg_target_price)
            if close_price:
                try:
                    if close_price != 0:
                        upside = (avg_target_price - close_price) / close_price * 100.0
                    else:
                        upside = float("nan")
                except Exception:  # pragma: no cover - defensive
                    upside = float("nan")
                if not math.isnan(upside):
                    inst_summary["avg_target_upside_pct"] = float(upside)

        short_interest = _coerce_float(info.get("shortPercentOfFloat")) if info else None
        if short_interest is not None:
            if abs(short_interest) <= 1:
                short_interest *= 100.0
            inst_summary["short_interest_pct_float"] = float(short_interest)

        earnings_timestamp = None
        if info:
            for key in ("earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd"):
                candidate = info.get(key)
                earnings_timestamp = _parse_report_datetime(candidate)
                if earnings_timestamp is not None:
                    break
        if earnings_timestamp is not None:
            inst_summary["next_earnings"] = earnings_timestamp

        sector_name = info.get("sector") if info else None
        if isinstance(sector_name, str) and sector_name.strip():
            cleaned_sector = sector_name.strip()
            inst_summary["sector"] = cleaned_sector
            inst_summary["sector_etf"] = SECTOR_ETF_MAP.get(cleaned_sector)

        recommendation = info.get("recommendationKey") if info else None
        if isinstance(recommendation, str) and recommendation:
            inst_summary["analyst_rating"] = recommendation.replace("_", " ").title()

        try:
            fast_info = ticker.fast_info
        except Exception:
            fast_info = None
        beta_candidate = None
        if fast_info is not None:
            beta_candidate = getattr(fast_info, "beta", None)
            if beta_candidate is None and isinstance(fast_info, dict):
                beta_candidate = fast_info.get("beta")
        if beta_candidate is None:
            beta_candidate = info.get("beta") if isinstance(info, dict) else None
        beta_numeric = _coerce_float(beta_candidate)
        if beta_numeric is not None:
            inst_summary["beta"] = float(beta_numeric)

        try:
            insider_df = ticker.insider_transactions
        except Exception:
            insider_df = None
        if isinstance(insider_df, pd.DataFrame) and not insider_df.empty:
            if "Date" in insider_df.columns and "Transaction" in insider_df.columns:
                parsed_dates = pd.to_datetime(insider_df["Date"], errors="coerce")
                recent_cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=90)
                recent_mask = parsed_dates >= recent_cutoff
                if recent_mask.any():
                    recent = insider_df.loc[recent_mask]
                    transactions = recent["Transaction"].astype(str).str.lower()
                    buys = int(transactions.str.contains("buy", na=False).sum())
                    sells = int(transactions.str.contains("sell", na=False).sum())
                    inst_summary["insider_buys_3m"] = buys
                    inst_summary["insider_sells_3m"] = sells
    except Exception as exc:  # pragma: no cover - network
        return inst_summary, f"Unable to load institutional snapshot ({exc})."

    if ALPHAVANTAGE_API_KEY:
        overview_payload, _ = _fetch_alpha_vantage_payload(symbol, "OVERVIEW")
        if overview_payload:
            _merge_alpha_vantage_overview(inst_summary, overview_payload, close_price)

        rating_payload, _ = _fetch_alpha_vantage_payload(symbol, "RATING")
        if rating_payload:
            _merge_alpha_vantage_rating(inst_summary, rating_payload)

    _institution_snapshot_cache[symbol] = (now, inst_summary)
    return inst_summary, None


def _detect_with_trimming(
    detector: Callable[..., Any],
    df: pd.DataFrame,
    *,
    max_trim: int = 10,
    min_size: Optional[int] = None,
    **kwargs: Any,
):
    if min_size is None:
        min_size = _resolve_min_size(kwargs, fallback=40)

    for trim in range(1, max_trim + 1):
        if len(df) - trim < min_size:
            break
        trimmed = df.iloc[:-trim]
        result = detector(trimmed, **kwargs)
        if result:
            return result

    return None


def _scan_with_trimming(
    scanner: Callable[..., Any],
    df: pd.DataFrame,
    *,
    max_trim: int = 10,
    min_size: Optional[int] = None,
    **kwargs: Any,
):
    if min_size is None:
        min_size = _resolve_min_size(kwargs, fallback=40)

    for trim in range(1, max_trim + 1):
        if len(df) - trim < min_size:
            break
        trimmed = df.iloc[:-trim]
        results = scanner(trimmed, **kwargs)
        if results:
            return results

    return []

def save_active_monitor(iid):
    data = order_tree.item(iid)['values']
    monitor_entry = {
        "symbol": data[0],
        "qty": float(data[1]),
        "entry": float(data[2]),
        "stop": float(data[3]),
        "status": data[4]
    }
    if os.path.exists(MONITOR_FILE):
        with open(MONITOR_FILE, 'r') as f:
            monitors = json.load(f)
    else:
        monitors = []
    for mon in monitors:
        if mon["symbol"] == monitor_entry["symbol"] and mon["entry"] == monitor_entry["entry"]:
            return
    monitors.append(monitor_entry)
    with open(MONITOR_FILE, 'w') as f:
        json.dump(monitors, f)

def rerun_stop_loss_monitors():
    if not os.path.exists(MONITOR_FILE):
        messagebox.showinfo("Reload Monitors", "No saved monitors to reload.")
        return

    with open(MONITOR_FILE, 'r') as f:
        monitors = json.load(f)

    existing_orders = set()
    for iid in order_tree.get_children():
        val = order_tree.item(iid)["values"]
        if len(val) >= 3:
            try:
                existing_orders.add((val[0], float(val[2])))
            except Exception:
                continue

    for entry in monitors:
        key = (entry['symbol'], float(entry['entry']))
        already_in_gui = key in existing_orders

        alpaca = get_api(entry['symbol'])
        try:
            open_orders = alpaca.list_orders(status="open")
            already_ordered = any(
                o.symbol == entry['symbol'] and
                o.side == 'buy' and
                o.status in ['accepted', 'new']
                for o in open_orders
            )
        except Exception as e:
            print(f"[Monitor] Error checking Alpaca orders: {e}")
            already_ordered = False

        if already_in_gui:
            print(f"[Reload] Skipping GUI duplicate: {key}")
            continue

        print(f"[Reload] Monitoring {entry['symbol']} @ {entry['entry']} (already_ordered={already_ordered})")
        iid = order_tree.insert("", "end", values=(entry['symbol'], entry['qty'], entry['entry'], entry['stop'], "Monitoring...", ""))

        if already_ordered:
            print(f"[Reload] Skipping order placement for {entry['symbol']} (already open). Monitoring only.")
            continue

        monitor_order(iid, entry['symbol'], entry['entry'], entry['stop'])

def monitor_order(iid, symbol, entry, stop):
    def monitor():
        try:
            qty = float(order_tree.set(iid, "Shares"))
            alpaca = get_api(symbol)
            save_active_monitor(iid)
            try:
                order = alpaca.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')
            except Exception as e:
                if "rejected by user request" in str(e).lower():
                    order_tree.set(iid, "Status", "Error: trading is disabled in account settings")
                else:
                    try:
                        order = alpaca.submit_order(symbol=symbol, qty=qty, side='buy', type='limit', time_in_force='gtc', extended_hours=True, limit_price=entry)
                        order_tree.set(iid, "Status", "Buy Order Placed")
                        order_tree.set(iid, "order_id", order.id)
                    except Exception as e2:
                        order_tree.set(iid, "Status", f"Error: {e2}")
                        return
                return
            order_tree.set(iid, "Status", "Buy Order Placed")
            order_tree.set(iid, "order_id", order.id)
            while True:
                bars = list(alpaca.get_bars(symbol, TimeFrame.Minute, limit=1))
                print(f"[Monitor] Checking {symbol}: Latest Price = {bars[-1].c if bars else 'N/A'} | Stop Loss = {stop}")
                if not bars:
                    time.sleep(10)
                    continue
                current_price = bars[-1].c
                if current_price <= stop:
                    try:
                        order = alpaca.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
                    except Exception:
                        order = alpaca.submit_order(symbol=symbol, qty=qty, side='sell', type='limit', time_in_force='gtc', extended_hours=True, limit_price=stop)
                    order_tree.set(iid, "Status", "Stop Loss Triggered — Sell Order Placed")
                    break
                time.sleep(10)
        except Exception as e:
            order_tree.set(iid, "Status", f"Error: {e}")
    threading.Thread(target=monitor, daemon=True).start()

def place_order():
    symbol = symbol_var.get().strip().upper()
    qty = qty_var.get().strip()
    entry = entry_var.get().strip()
    sl = sl_var.get().strip()
    if not symbol or not qty or not entry or not sl:
        messagebox.showerror("Error", "All fields required.")
        return
    try:
        entry = float(entry)
        sl = float(sl)
        qty = round(float(qty), 6)
    except ValueError:
        messagebox.showerror("Error", "Invalid number input.")
        return
    iid = order_tree.insert("", "end", values=(symbol, qty, entry, sl, "Waiting...", ""))
    monitor_order(iid, symbol, entry, sl)

def delete_selected_order():
    selected = order_tree.selection()
    if not selected:
        messagebox.showinfo("Delete Order", "No order selected.")
        return
    for item in selected:
        order_id = order_tree.set(item, "order_id")
        symbol = order_tree.set(item, "Symbol")
        if order_id:
            try:
                get_api(symbol).cancel_order(order_id)
            except Exception as e:
                messagebox.showwarning("Cancel Order", f"Could not cancel Alpaca order:\n{e}")
        order_tree.delete(item)

scan_status_var = None
scan_button = None
is_scanning = False


def run_scan():
    global is_scanning

    if is_scanning or scan_status_var is None or scan_button is None:
        return

    script_path = Path(__file__).with_name("pattern_scanner.py")
    if not script_path.exists():
        messagebox.showerror("Pattern Scanner", f"Scanner script not found: {script_path}")
        return

    is_scanning = True
    scan_status_var.set("Running…")
    scan_button.config(state="disabled")

    def worker():
        try:
            subprocess.run([sys.executable, str(script_path)], check=True)
        except subprocess.CalledProcessError as exc:
            status = f"Failed (code {exc.returncode})"
        except Exception as exc:  # pragma: no cover - GUI feedback path
            status = f"Error: {exc}"
        else:
            status = "Completed"

        def finalize():
            global is_scanning
            scan_status_var.set(status)
            scan_button.config(state="normal")
            if status == "Completed":
                load_watchlist()
            is_scanning = False

        root.after(0, finalize)

    threading.Thread(target=worker, daemon=True).start()

def _flatten_columns(columns) -> list[str]:
    expected_labels = {"open", "high", "low", "close", "volume"}
    get_level = getattr(columns, "get_level_values", None)
    nlevels = getattr(columns, "nlevels", 1)

    if callable(get_level):
        for level in range(nlevels):
            level_values = list(get_level(level))
            normalized = {str(value).lower() for value in level_values}
            if expected_labels.issubset(normalized):
                return [str(value) for value in level_values]

    flattened: list[str] = []
    for column in columns:
        if isinstance(column, tuple):
            parts = [str(part) for part in column if part not in (None, "")]
            chosen = next(
                (
                    part
                    for part in parts
                    if isinstance(part, str) and part.lower() in expected_labels
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
        duplicated = getattr(df.columns, "duplicated", None)
        if callable(duplicated):
            df = df.loc[:, ~duplicated()]

    return df

def download_all_data():
    symbol_data.clear()
    if not WATCHLIST_PATH.exists():
        return
    df = pd.read_csv(WATCHLIST_PATH)
    for sym in df["symbol"]:
        try:
            data = yf.download(sym, period="12mo", progress=False)
            data = flatten_yf_columns(data)
            symbol_data[sym] = data
            data.to_csv(DATA_DIR / f"{sym}.csv")
        except Exception as e:
            print(f"Download failed for {sym}: {e}")

def show_candlestick():
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    sel = tree.selection()
    if not sel:
        return
    item = tree.item(sel[0])
    row_values = item.get("values", [])
    columns = tree["columns"] or ()

    value_by_column = {col: row_values[idx] for idx, col in enumerate(columns)}

    sym = str(value_by_column.get("symbol", row_values[0] if row_values else "")).strip()

    pattern_value = value_by_column.get("pattern", "")
    pattern_name = str(pattern_value).strip() if pattern_value not in (None, "") else ""
    data_file = DATA_DIR / f"{sym}.csv"
    if sym not in symbol_data:
        if data_file.exists():
            symbol_data[sym] = pd.read_csv(data_file, index_col=0, parse_dates=True)
        else:
            df = yf.download(sym, period="12mo", auto_adjust=False, progress=False)
            df = flatten_yf_columns(df)
            if df.empty:
                messagebox.showinfo("Chart", f"No data available for {sym}.")
                return
            symbol_data[sym] = df
            df.to_csv(data_file)

    df = symbol_data[sym]

    target_for_refresh = _consume_last_added_my_watchlist_symbol()
    _refresh_my_watchlist_list(target_symbol=target_for_refresh, current_symbol=sym)

    for w in chart_frame.winfo_children():
        w.destroy()

    snapshot_summary, snapshot_error = fetch_institution_snapshot(sym, df)

    info_container = tk.Frame(chart_frame, bg="#f5f5f5", width=320)
    info_container.pack(side="left", fill="y", padx=(0, 10), pady=5)
    info_container.pack_propagate(False)

    info_canvas = tk.Canvas(info_container, bg="#f5f5f5", highlightthickness=0)
    info_canvas.pack(side="left", fill="both", expand=True)

    info_scrollbar = ttk.Scrollbar(info_container, orient="vertical", command=info_canvas.yview)
    info_scrollbar.pack(side="right", fill="y")

    info_canvas.configure(yscrollcommand=info_scrollbar.set)

    info_panel = tk.Frame(info_canvas, bg="#f5f5f5")
    info_window = info_canvas.create_window((0, 0), window=info_panel, anchor="nw")

    def _configure_scroll_region(event=None):
        info_canvas.configure(scrollregion=info_canvas.bbox("all"))

    def _sync_panel_width(event):
        info_canvas.itemconfigure(info_window, width=event.width)
        _configure_scroll_region()

    info_panel.bind("<Configure>", _configure_scroll_region)
    info_canvas.bind("<Configure>", _sync_panel_width)

    def _on_mousewheel(event):
        if event.delta:
            if sys.platform == "darwin":
                step = -1 if event.delta > 0 else 1
            else:
                step = int(-event.delta / 120) or (-1 if event.delta > 0 else 1)
            info_canvas.yview_scroll(step, "units")
        elif event.num == 4:
            info_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            info_canvas.yview_scroll(1, "units")

    def _bind_mousewheel(_):
        info_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        info_canvas.bind_all("<Button-4>", _on_mousewheel)
        info_canvas.bind_all("<Button-5>", _on_mousewheel)

    def _unbind_mousewheel(_):
        info_canvas.unbind_all("<MouseWheel>")
        info_canvas.unbind_all("<Button-4>")
        info_canvas.unbind_all("<Button-5>")

    info_canvas.bind("<Enter>", _bind_mousewheel)
    info_canvas.bind("<Leave>", _unbind_mousewheel)
    info_panel.bind("<Enter>", _bind_mousewheel)
    info_panel.bind("<Leave>", _unbind_mousewheel)
    info_canvas.yview_moveto(0)


    tk.Label(
        info_panel,
        text="Symbol Overview",
        font=("Arial", 12, "bold"),
        anchor="w",
        bg="#f5f5f5",
    ).pack(fill="x")

    summary_frame = tk.Frame(info_panel, bg="#f5f5f5")
    summary_frame.pack(fill="x", padx=4, pady=(2, 4))

    tk.Label(
        summary_frame,
        text=f"Symbol: {sym}",
        font=("Arial", 11),
        anchor="w",
        bg="#f5f5f5",
    ).pack(side="left")

    last_close_value = snapshot_summary.get("close") if snapshot_summary else None
    tk.Label(
        summary_frame,
        text=f"   |   Last Close: {_format_price(last_close_value)}",
        font=("Arial", 11),
        anchor="w",
        bg="#f5f5f5",
    ).pack(side="left")

    price_change = snapshot_summary.get("price_chg_21d_pct") if snapshot_summary else None
    change_value = (
        price_change
        if isinstance(price_change, (int, float)) and not pd.isna(price_change)
        else None
    )
    if change_value is None or change_value == 0:
        change_color = "#333333"
    elif change_value > 0:
        change_color = "#006400"  # Dark green for better visibility
    else:
        change_color = "#8B0000"  # Dark red for better visibility

    tk.Label(
        summary_frame,
        text=f"   |   21-Day Change: {_format_percent(price_change, signed=True)}",
        font=("Arial", 11),
        anchor="w",
        bg="#f5f5f5",
        fg=change_color,
    ).pack(side="left")

    content_pad = {"fill": "x", "anchor": "w", "padx": 4, "pady": 2}

    if snapshot_summary:
        separator_line = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        def add_separator(pady: tuple[int, int] = (4, 4)) -> None:
            tk.Label(
                info_panel,
                text=separator_line,
                font=("Arial", 10),
                anchor="w",
                bg="#f5f5f5",
            ).pack(fill="x", padx=4, pady=pady)

        def add_section(title: str, body: str, *, title_pady: tuple[int, int] = (0, 2)) -> None:
            tk.Label(
                info_panel,
                text=title,
                font=("Arial", 11, "bold"),
                anchor="w",
                bg="#f5f5f5",
            ).pack(fill="x", padx=4, pady=title_pady)
            tk.Label(
                info_panel,
                text=body,
                justify="left",
                wraplength=300,
                bg="#f5f5f5",
                anchor="w",
            ).pack(**content_pad)

        def add_metric_section(
            title: str,
            metrics: list[tuple[str, str, int]],
            *,
            title_pady: tuple[int, int] = (0, 2),
        ) -> None:
            tk.Label(
                info_panel,
                text=title,
                font=("Arial", 11, "bold"),
                anchor="w",
                bg="#f5f5f5",
            ).pack(fill="x", padx=4, pady=title_pady)

            section_frame = tk.Frame(info_panel, bg="#f5f5f5")
            section_frame.pack(fill="x", padx=4, pady=(0, 4))

            for label_text, value_text, score in metrics:
                row = tk.Frame(section_frame, bg="#f5f5f5")
                row.pack(fill="x", pady=1)

                tk.Label(
                    row,
                    text=f"{label_text}:",
                    font=("Arial", 10, "bold"),
                    anchor="w",
                    bg="#f5f5f5",
                ).pack(side="left")

                fg_color = _score_to_color(score)
                icon = _score_to_icon(score)
                if isinstance(value_text, str) and value_text.strip().upper() in {"N/A", "NA"}:
                    fg_color = NA_TEXT_COLOR
                tk.Label(
                    row,
                    text=f"  {icon} {value_text}",
                    font=("Arial", 10),
                    anchor="w",
                    bg="#f5f5f5",
                    fg=fg_color,
                ).pack(side="left")

        add_separator(pady=(4, 6))

        mfi_text = _format_decimal(snapshot_summary.get("mfi_14"))
        rsi_text = _format_decimal(snapshot_summary.get("rsi_14"))
        ad_text = _format_decimal(snapshot_summary.get("ad_slope_60"))
        obv_trend = snapshot_summary.get("obv_trend") or "→"
        volume_value = snapshot_summary.get("volume_avg_30")
        if volume_value is None or (
            isinstance(volume_value, (int, float)) and math.isnan(volume_value)
        ):
            volume_text = "N/A"
        else:
            volume_text = _format_share_amount(volume_value)
        beta_text = _format_decimal(snapshot_summary.get("beta"))
        signal_icon, signal_description = _derive_signal(
            snapshot_summary.get("mfi_14"), snapshot_summary.get("rsi_14")
        )
        pattern_display = pattern_name or "N/A"

        technical_metrics = [
            ("MFI (14)", mfi_text, _score_mfi(snapshot_summary.get("mfi_14"))),
            ("RSI (14)", rsi_text, _score_rsi(snapshot_summary.get("rsi_14"))),
            ("A/D Slope (60)", ad_text, _score_ad_slope(snapshot_summary.get("ad_slope_60"))),
            ("OBV Trend", obv_trend, _score_obv_trend(snapshot_summary.get("obv_trend"))),
            ("Volume (30D Avg)", volume_text, 0),
            ("Beta", beta_text, _score_beta(snapshot_summary.get("beta"))),
            ("Pattern", pattern_display, 0),
            (
                "Signal",
                f"{signal_icon} {signal_description}",
                _score_signal(signal_description),
            ),
        ]
        add_metric_section("📊 Technical Summary", technical_metrics, title_pady=(0, 2))

        add_separator()

        filers_text = _format_int(snapshot_summary.get("inst_count"))
        shares_text = _format_shares(snapshot_summary.get("inst_shares_sum"))
        pct_value = snapshot_summary.get("pct_held_by_institutions")
        pct_text = _format_percent(pct_value)
        qoq_value = snapshot_summary.get("inst_shares_qoq_pct")
        qoq_text = _format_percent(qoq_value, signed=True)

        top_holders = snapshot_summary.get("top_holders") or []
        if top_holders:
            holder_parts: list[str] = []
            for holder in top_holders:
                holder_name = holder.get("name", "Unknown")
                pct_holder = holder.get("pct")
                if isinstance(pct_holder, (int, float)) and not math.isnan(pct_holder):
                    holder_parts.append(f"{holder_name} ({pct_holder:.1f}%)")
                else:
                    holder_parts.append(str(holder_name))
            top_holder_text = ", ".join(holder_parts)
        else:
            top_holder_text = "N/A"

        insider_buys = snapshot_summary.get("insider_buys_3m")
        insider_sells = snapshot_summary.get("insider_sells_3m")

        if isinstance(insider_buys, (int, float)) and not pd.isna(insider_buys):
            insider_buys_text = f"{int(insider_buys):+d}" if insider_buys else "0"
        elif insider_buys is None:
            insider_buys_text = "N/A"
        else:
            insider_buys_text = str(insider_buys)

        if isinstance(insider_sells, (int, float)) and not pd.isna(insider_sells):
            insider_sells_text = f"{int(insider_sells)}"
        elif insider_sells is None:
            insider_sells_text = "N/A"
        else:
            insider_sells_text = str(insider_sells)

        inst_metrics = [
            ("Filers", filers_text, 0),
            ("Shares Held", shares_text, 0),
            (
                "Institutional %",
                pct_text,
                _score_institutional_pct(pct_value),
            ),
            (
                "QoQ Change",
                qoq_text,
                _score_institutional_trend(qoq_value),
            ),
            ("Top Holders", top_holder_text, 0),
            (
                "Insider Buys (3M)",
                insider_buys_text,
                _score_positive_growth(insider_buys),
            ),
            (
                "Insider Sells (3M)",
                insider_sells_text,
                -_score_positive_growth(insider_sells),
            ),
        ]

        add_metric_section("🏦 Institutional Snapshot", inst_metrics, title_pady=(0, 2))

        strength_score = _compute_institutional_strength(snapshot_summary)
        if strength_score is not None:
            score_band = 1 if strength_score >= 67 else -1 if strength_score <= 33 else 0
            tk.Label(
                info_panel,
                text=f"Institutional Strength Score: {strength_score}",
                font=("Arial", 10, "bold"),
                anchor="w",
                bg="#f5f5f5",
                fg=_score_to_color(score_band),
            ).pack(fill="x", padx=4, pady=(0, 4))

        add_separator()

        pe_text = _format_decimal(snapshot_summary.get("pe_ttm"))
        fwd_pe_text = _format_decimal(snapshot_summary.get("forward_pe"))
        peg_text = _format_decimal(snapshot_summary.get("peg_ratio"))
        dividend_value = snapshot_summary.get("dividend_yield_pct")
        dividend_text = _format_percent(dividend_value)
        roe_value = snapshot_summary.get("roe_pct")
        roe_text = _format_percent(roe_value)
        de_text = _format_decimal(snapshot_summary.get("de_ratio"))
        eps_growth_value = snapshot_summary.get("eps_growth_yoy_pct")
        eps_growth_text = _format_percent(eps_growth_value, signed=True)
        revenue_growth_value = snapshot_summary.get("revenue_growth_yoy_pct")
        revenue_growth_text = _format_percent(revenue_growth_value, signed=True)
        fcf_value = snapshot_summary.get("free_cash_flow")
        fcf_text = _format_dollar_amount(fcf_value)

        dividend_score_value = None
        numeric_dividend = _coerce_numeric(dividend_value)
        if numeric_dividend is not None:
            dividend_score_value = numeric_dividend / 100.0

        roe_score_value = None
        numeric_roe = _coerce_numeric(roe_value)
        if numeric_roe is not None:
            roe_score_value = numeric_roe

        eps_score_value = None
        numeric_eps = _coerce_numeric(eps_growth_value)
        if numeric_eps is not None:
            eps_score_value = numeric_eps / 100.0

        revenue_score_value = None
        numeric_revenue = _coerce_numeric(revenue_growth_value)
        if numeric_revenue is not None:
            revenue_score_value = numeric_revenue / 100.0

        fundamentals_metrics = [
            ("P/E (TTM)", pe_text, _score_pe(snapshot_summary.get("pe_ttm"))),
            ("Forward P/E", fwd_pe_text, _score_pe(snapshot_summary.get("forward_pe"))),
            ("PEG Ratio", peg_text, _score_peg(snapshot_summary.get("peg_ratio"))),
            (
                "Dividend Yield",
                dividend_text,
                _score_dividend_yield(dividend_score_value),
            ),
            ("ROE", roe_text, _score_roe(roe_score_value)),
            (
                "Debt/Equity",
                de_text,
                -_score_debt_to_equity(snapshot_summary.get("de_ratio")),
            ),
            (
                "EPS Growth (YoY)",
                eps_growth_text,
                _score_positive_growth(eps_score_value),
            ),
            (
                "Revenue Growth (YoY)",
                revenue_growth_text,
                _score_positive_growth(revenue_score_value),
            ),
            (
                "Free Cash Flow (TTM)",
                fcf_text,
                _score_positive_growth(fcf_value),
            ),
        ]
        add_metric_section("💰 Fundamentals", fundamentals_metrics, title_pady=(0, 2))

        add_separator()

        rating_text = snapshot_summary.get("analyst_rating") or "N/A"
        buy_count = snapshot_summary.get("analyst_buy_count")
        hold_count = snapshot_summary.get("analyst_hold_count")
        sell_count = snapshot_summary.get("analyst_sell_count")
        counts_text = ""
        if all(
            isinstance(value, (int, float)) and not pd.isna(value)
            for value in (buy_count, hold_count, sell_count)
        ):
            counts_text = f" ({int(buy_count)} Buy / {int(hold_count)} Hold / {int(sell_count)} Sell)"

        target_price_text = _format_price(snapshot_summary.get("avg_target_price"))
        upside_text = _format_percent(
            snapshot_summary.get("avg_target_upside_pct"), signed=True
        )
        short_interest_text = _format_percent(snapshot_summary.get("short_interest_pct_float"))

        next_earnings = snapshot_summary.get("next_earnings")
        if isinstance(next_earnings, datetime):
            next_earnings_text = next_earnings.strftime("%m/%d/%Y")
        elif isinstance(next_earnings, str) and next_earnings.strip():
            next_earnings_text = next_earnings
        else:
            next_earnings_text = "N/A"

        market_lines = [
            f"Analyst Rating: {rating_text}{counts_text}",
            f"Avg Target: {target_price_text} ({upside_text} Upside)",
            f"Short Interest: {short_interest_text}",
            f"Next Earnings: {next_earnings_text}",
        ]
        add_section("📈 Market Sentiment", "\n".join(market_lines), title_pady=(0, 2))

        add_separator()

        sector = snapshot_summary.get("sector")
        sector_etf = snapshot_summary.get("sector_etf")
        if sector_etf and sector:
            sector_text = f"{sector_etf} ({sector})"
        elif sector_etf:
            sector_text = sector_etf
        elif sector:
            sector_text = sector
        else:
            sector_text = "N/A"

        correlation_text = _format_decimal(snapshot_summary.get("correlation"))
        atr_text = _format_decimal(snapshot_summary.get("atr_14"))

        risk_lines = [
            f"Sector ETF: {sector_text}",
            f"Correlation: {correlation_text}",
            f"Volatility (ATR 14): {atr_text}",
        ]
        add_section("⚙️ Risk & Correlation", "\n".join(risk_lines), title_pady=(0, 2))

        add_separator(pady=(6, 4))

    if snapshot_error:
        tk.Label(
            info_panel,
            text=snapshot_error,
            justify="left",
            wraplength=240,
            fg="red",
            bg="#f5f5f5",
            anchor="w",
        ).pack(fill="x", padx=4, pady=(4, 2))

    canvas_container = tk.Frame(chart_frame)
    canvas_container.pack(side="left", fill="both", expand=True)

    price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    plot_df = df[price_columns].dropna().copy()
    if plot_df.empty:
        messagebox.showinfo("Chart", f"No complete OHLC data available for {sym}.")
        return

    plot_df.sort_index(inplace=True)
    analysis_df = plot_df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=lambda c: c.lower())
    plot_df['Date'] = mdates.date2num(plot_df.index.to_pydatetime())
    ohlc = plot_df[['Date', 'Open', 'High', 'Low', 'Close']].values

    data_price_min = float(plot_df['Low'].min())
    data_price_max = float(plot_df['High'].max())
    price_range = data_price_max - data_price_min

    if price_range == 0:
        reference = data_price_max if data_price_max else 1.0
        price_range = max(reference * 0.01, 0.5)

    tick_hint = price_range / 6 if price_range > 0 else 1.0
    tick_size = _nice_tick_size(tick_hint, default=1.0)

    price_min = math.floor(data_price_min / tick_size) * tick_size
    price_max = math.ceil(data_price_max / tick_size) * tick_size

    if price_max <= data_price_max:
        price_max += tick_size

    if data_price_min - price_min < 0.25 * tick_size:
        price_min -= tick_size

    if price_min < 0 < data_price_min:
        price_min = 0.0

    display_range = max(price_max - price_min, price_range)

    target_bins = 60
    bin_size = max(display_range / target_bins, 0.01)

    bins = np.arange(price_min, price_max + bin_size, bin_size)
    if bins[-1] < price_max - 1e-9:
        bins = np.append(bins, price_max)

    price_levels = 0.5 * (bins[1:] + bins[:-1])
    # Build a price-volume profile that aligns each candle's volume with the
    # actual price range traded during that candle.  Distributing volume across
    # the high/low range ties the histogram bars to the price axis instead of
    # assigning all of the volume to the closing price.
    volume_by_price = pd.Series(0.0, index=price_levels)

    bin_low_edges = bins[:-1]
    bin_high_edges = bins[1:]

    for _, row in plot_df.iterrows():
        low = row['Low']
        high = row['High']
        volume = row['Volume']

        if pd.isna(low) or pd.isna(high) or pd.isna(volume):
            continue

        if high == low:
            bin_idx = np.searchsorted(bins, high, side='right') - 1
            bin_idx = min(max(bin_idx, 0), len(price_levels) - 1)
            volume_by_price.iloc[bin_idx] += volume
            continue

        start_idx = np.searchsorted(bins, low, side='right') - 1
        end_idx = np.searchsorted(bins, high, side='right') - 1

        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(price_levels) - 1)

        if end_idx < start_idx:
            continue

        relevant_lows = bin_low_edges[start_idx:end_idx + 1]
        relevant_highs = bin_high_edges[start_idx:end_idx + 1]

        overlaps = np.minimum(high, relevant_highs) - np.maximum(low, relevant_lows)
        overlaps = np.clip(overlaps, 0, None)
        total_overlap = overlaps.sum()

        if total_overlap <= 0:
            continue

        volume_distribution = volume * (overlaps / total_overlap)
        volume_by_price.iloc[start_idx:end_idx + 1] += volume_distribution
    norm_denominator = volume_by_price.max()
    norm_vol = volume_by_price / norm_denominator if norm_denominator else volume_by_price

    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[26, 6], height_ratios=[3, 1, 1], hspace=0.05, wspace=0.04)

    ax_price = fig.add_subplot(gs[0, 0])
    ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_rsi = fig.add_subplot(gs[2, 0], sharex=ax_price)
    ax_vp = fig.add_subplot(gs[0, 1], sharey=ax_price)
    fig.subplots_adjust(left=0.06, right=0.98)
    for row in (1, 2):
        fig.add_subplot(gs[row, 1]).axis('off')

    for t, o, h, l, c in ohlc:
        color = 'green' if c >= o else 'red'
        ax_price.plot([t, t], [l, h], color='black')
        ax_price.add_patch(plt.Rectangle((t - 0.2, min(o, c)), 0.4, abs(c - o), color=color))

    ax_price.set_ylim(price_min, price_max)

    overlay_added = False

    if pattern_name:
        name_lower = pattern_name.strip().lower()
        if name_lower == "double bottom":
            hits = scan_double_bottoms(analysis_df, window=60)
            if not hits:
                hits = scan_double_bottoms(analysis_df, window=60, require_breakout=False)
            if not hits:
                hits = _scan_with_trimming(scan_double_bottoms, analysis_df, window=60)
            if not hits:
                hits = _scan_with_trimming(
                    scan_double_bottoms,
                    analysis_df,
                    window=60,
                    require_breakout=False,
                )

            hit: Optional[DoubleBottomHit] = None
            if hits:
                hit = max(hits, key=lambda h: h.right_idx)
            if hit is None:
                hit = _load_persisted_pattern(sym, pattern_name, DoubleBottomHit)

            if isinstance(hit, DoubleBottomHit):
                try:
                    left_idx = hit.left_idx
                    right_idx = hit.right_idx
                    support = hit.support

                    if 0 <= left_idx < len(plot_df) and 0 <= right_idx < len(plot_df):
                        ax_price.axhline(support, color='#1f77b4', linestyle='--', linewidth=1.5, label='Support')
                        overlay_added = True

                        ax_price.scatter(
                            [plot_df['Date'].iloc[left_idx], plot_df['Date'].iloc[right_idx]],
                            [hit.left_low, hit.right_low],
                            color='#d62728',
                            zorder=5,
                            label='Swing Low',
                        )
                        overlay_added = True

                        indices = np.arange(left_idx, right_idx + 1)
                        highs_segment = analysis_df['high'].iloc[left_idx:right_idx + 1].to_numpy()
                        if highs_segment.size >= 2 and np.all(np.isfinite(highs_segment)):
                            slope, intercept = np.polyfit(indices, highs_segment, 1)
                            line_idx = np.array([left_idx, right_idx])
                            line_prices = slope * line_idx + intercept
                            line_dates = plot_df['Date'].iloc[[left_idx, right_idx]].to_numpy()
                            ax_price.plot(line_dates, line_prices, color='#9467bd', linewidth=1.5, label='Neckline')
                            overlay_added = True

                            if hit.breakout_idx is not None and 0 <= hit.breakout_idx < len(plot_df):
                                breakout_date = plot_df['Date'].iloc[hit.breakout_idx]
                                breakout_price = hit.breakout_price if hit.breakout_price is not None else float(
                                    slope * hit.breakout_idx + intercept
                                )
                                ax_price.scatter(
                                    [breakout_date],
                                    [breakout_price],
                                    color='#ff7f0e',
                                    marker='^',
                                    s=60,
                                    zorder=6,
                                    label='Breakout',
                                )
                                overlay_added = True
                except Exception:
                    pass
        elif name_lower == "cup and handle":
            hit = detect_cup_and_handle(analysis_df, cup_window=60, handle_window=15)
            if not hit:
                hit = _detect_with_trimming(
                    detect_cup_and_handle,
                    analysis_df,
                    cup_window=60,
                    handle_window=15,
                )
            if not hit:
                hit = _load_persisted_pattern(sym, pattern_name, CupHandleHit)
            if isinstance(hit, CupHandleHit):
                resistance = hit.resistance
                ax_price.axhline(
                    resistance,
                    color='#9467bd',
                    linestyle='--',
                    linewidth=1.5,
                    label='Resistance',
                )
                overlay_added = True

                cup_points: list[tuple[int, float, float]] = []
                for idx in (hit.left_peak_idx, hit.cup_low_idx, hit.right_peak_idx):
                    if idx is None or not (0 <= idx < len(plot_df)):
                        continue
                    date_val = float(plot_df['Date'].iloc[idx])
                    price_val = float(analysis_df['close'].iloc[idx])
                    cup_points.append((idx, date_val, price_val))

                if len(cup_points) >= 3:
                    cup_points.sort(key=lambda point: point[0])
                    ax_price.plot(
                        [point[1] for point in cup_points],
                        [point[2] for point in cup_points],
                        color='#1f77b4',
                        linewidth=1.4,
                        label='Cup',
                    )
                    ax_price.scatter(
                        [point[1] for point in cup_points],
                        [point[2] for point in cup_points],
                        color='#1f77b4',
                        s=35,
                        zorder=6,
                    )
                    overlay_added = True

                handle_points: list[tuple[int, float, float]] = []
                for idx in (hit.handle_start_idx, hit.handle_low_idx, hit.handle_end_idx):
                    if idx is None or not (0 <= idx < len(plot_df)):
                        continue
                    date_val = float(plot_df['Date'].iloc[idx])
                    price_val = float(analysis_df['close'].iloc[idx])
                    handle_points.append((idx, date_val, price_val))

                if len(handle_points) >= 2:
                    handle_points.sort(key=lambda point: point[0])
                    ax_price.plot(
                        [point[1] for point in handle_points],
                        [point[2] for point in handle_points],
                        color='#ff7f0e',
                        linewidth=1.4,
                        label='Handle',
                    )
                    ax_price.scatter(
                        [point[1] for point in handle_points],
                        [point[2] for point in handle_points],
                        color='#ff7f0e',
                        s=30,
                        zorder=6,
                    )
                    overlay_added = True
        elif name_lower == "inverse head and shoulders":
            ihs = detect_inverse_head_shoulders(analysis_df)
            if not ihs:
                ihs = _detect_with_trimming(
                    detect_inverse_head_shoulders,
                    analysis_df,
                    min_size=60,
                )
            if not ihs:
                ihs = _load_persisted_pattern(sym, pattern_name, InverseHeadShouldersPattern)
            if isinstance(ihs, InverseHeadShouldersPattern):
                try:
                    structure_points: list[tuple[int, float, float]] = []
                    for idx, fallback_low in (
                        (ihs.left_idx, ihs.left_low),
                        (ihs.head_idx, ihs.head_low),
                        (ihs.right_idx, ihs.right_low),
                    ):
                        if 0 <= idx < len(plot_df):
                            date_value = float(plot_df['Date'].iloc[idx])
                            candle_low = float(analysis_df['low'].iloc[idx])
                            if not np.isfinite(candle_low):
                                candle_low = float(fallback_low)
                            structure_points.append((idx, date_value, candle_low))

                    if len(structure_points) == 3:
                        structure_points.sort(key=lambda point: point[0])
                        dates = [point[1] for point in structure_points]
                        lows = [point[2] for point in structure_points]
                        ax_price.scatter(dates, lows, color='#d62728', s=50, label='Shoulders/Head', zorder=6)
                        ax_price.plot(
                            dates,
                            lows,
                            color='#1f77b4',
                            linewidth=1.4,
                            label='Structure',
                        )
                        overlay_added = True

                    if 0 <= ihs.neckline_left_idx < len(plot_df) and 0 <= ihs.neckline_right_idx < len(plot_df):
                        neck_dates = [
                            float(plot_df['Date'].iloc[ihs.neckline_left_idx]),
                            float(plot_df['Date'].iloc[ihs.neckline_right_idx]),
                        ]
                        neck_prices = []
                        for idx, fallback_price in (
                            (ihs.neckline_left_idx, ihs.neckline_left),
                            (ihs.neckline_right_idx, ihs.neckline_right),
                        ):
                            price_val = float(analysis_df['high'].iloc[idx])
                            if not np.isfinite(price_val):
                                price_val = float(fallback_price)
                            neck_prices.append(price_val)
                        ax_price.plot(neck_dates, neck_prices, color='#9467bd', linestyle='--', linewidth=1.5, label='Neckline')
                        overlay_added = True

                    if getattr(ihs, 'breakout', False):
                        breakout_idx = len(plot_df) - 1
                        breakout_date = plot_df['Date'].iloc[breakout_idx]
                        breakout_price = analysis_df['close'].iloc[breakout_idx]
                        ax_price.scatter(
                            [breakout_date],
                            [breakout_price],
                            color='#ff7f0e',
                            marker='^',
                            s=60,
                            zorder=7,
                            label='Breakout',
                        )
                        overlay_added = True
                except Exception:
                    pass
        elif name_lower == "ascending triangle":
            triangle = detect_ascending_triangle(analysis_df, window=60, tolerance=0.02, min_touches=2)
            if not triangle:
                triangle = _detect_with_trimming(
                    detect_ascending_triangle,
                    analysis_df,
                    window=60,
                    tolerance=0.02,
                    min_touches=2,
                )
            if not triangle:
                triangle = _load_persisted_pattern(sym, pattern_name, AscendingTrianglePattern)
            if isinstance(triangle, AscendingTrianglePattern):
                start_idx = triangle.offset
                end_idx = triangle.offset + triangle.length - 1
                if 0 <= start_idx < end_idx < len(plot_df):
                    dates = plot_df['Date'].iloc[[start_idx, end_idx]].to_numpy()
                    support_start = triangle.support_intercept
                    support_end = triangle.support_slope * (triangle.length - 1) + triangle.support_intercept
                    ax_price.plot(dates, [support_start, support_end], color='#ff7f0e', linewidth=1.5, label='Rising Support')
                    overlay_added = True

                    ax_price.hlines(
                        triangle.resistance,
                        dates[0],
                        dates[-1],
                        colors='#2ca02c',
                        linestyles='--',
                        linewidth=1.5,
                        label='Flat Resistance',
                    )
                    overlay_added = True

                    res_label = 'Resistance Touch'
                    for idx in triangle.resistance_indices:
                        if 0 <= idx < len(plot_df):
                            ax_price.scatter(
                                plot_df['Date'].iloc[idx],
                                analysis_df['high'].iloc[idx],
                                color='#1f77b4',
                                s=35,
                                zorder=6,
                                label=res_label,
                            )
                            res_label = None
                            overlay_added = True

                    sup_label = 'Support Touch'
                    for idx in triangle.support_indices:
                        if 0 <= idx < len(plot_df):
                            ax_price.scatter(
                                plot_df['Date'].iloc[idx],
                                analysis_df['low'].iloc[idx],
                                color='#d62728',
                                s=35,
                                zorder=6,
                                label=sup_label,
                            )
                            sup_label = None
                            overlay_added = True
        elif name_lower == "bullish pennant":
            pennant = detect_bullish_pennant(analysis_df, window=60, flagpole_window=20)
            if not pennant:
                pennant = _detect_with_trimming(
                    detect_bullish_pennant,
                    analysis_df,
                    window=60,
                    flagpole_window=20,
                )
            if not pennant:
                pennant = _load_persisted_pattern(sym, pattern_name, BullishPennantPattern)
            if isinstance(pennant, BullishPennantPattern):
                start_idx = pennant.offset
                end_idx = pennant.offset + pennant.length - 1
                if 0 <= start_idx < end_idx < len(plot_df):
                    dates = plot_df['Date'].iloc[[start_idx, end_idx]].to_numpy()
                    upper_start = pennant.upper_intercept
                    upper_end = pennant.upper_slope * (pennant.length - 1) + pennant.upper_intercept
                    lower_start = pennant.lower_intercept
                    lower_end = pennant.lower_slope * (pennant.length - 1) + pennant.lower_intercept

                    ax_price.plot(dates, [upper_start, upper_end], color='#9467bd', linewidth=1.5, label='Pennant Upper')
                    ax_price.plot(dates, [lower_start, lower_end], color='#ff7f0e', linewidth=1.5, label='Pennant Lower')
                    overlay_added = True
        elif name_lower == "bullish flag":
            flag = detect_bullish_flag(analysis_df, window=40, flagpole_window=20)
            if not flag:
                flag = _detect_with_trimming(
                    detect_bullish_flag,
                    analysis_df,
                    window=40,
                    flagpole_window=20,
                )
            if not flag:
                flag = _load_persisted_pattern(sym, pattern_name, BullishFlagPattern)
            if isinstance(flag, BullishFlagPattern):
                start_idx = flag.offset
                end_idx = flag.offset + flag.length - 1
                if 0 <= start_idx < end_idx < len(plot_df):
                    dates = plot_df['Date'].iloc[[start_idx, end_idx]].to_numpy()
                    base_start = flag.intercept
                    base_end = flag.slope * (flag.length - 1) + flag.intercept
                    upper_line = [base_start + flag.upper_offset, base_end + flag.upper_offset]
                    lower_line = [base_start + flag.lower_offset, base_end + flag.lower_offset]

                    ax_price.plot(dates, upper_line, color='#9467bd', linewidth=1.5, label='Flag Upper')
                    ax_price.plot(dates, lower_line, color='#ff7f0e', linewidth=1.5, label='Flag Lower')
                    overlay_added = True
        elif name_lower == "bullish rectangle":
            rectangle = detect_bullish_rectangle(analysis_df, window=60)
            if not rectangle:
                rectangle = _detect_with_trimming(
                    detect_bullish_rectangle,
                    analysis_df,
                    window=60,
                )
            if not rectangle:
                rectangle = _load_persisted_pattern(sym, pattern_name, BullishRectanglePattern)
            if isinstance(rectangle, BullishRectanglePattern):
                start_idx = rectangle.offset
                end_idx = rectangle.offset + rectangle.length - 1
                if 0 <= start_idx < end_idx < len(plot_df):
                    dates = plot_df['Date'].iloc[[start_idx, end_idx]].to_numpy()
                    ax_price.hlines(rectangle.high, dates[0], dates[-1], colors='#1f77b4', linestyles='--', linewidth=1.5, label='Rectangle High')
                    ax_price.hlines(rectangle.low, dates[0], dates[-1], colors='#d62728', linestyles='--', linewidth=1.5, label='Rectangle Low')
                    overlay_added = True
        elif name_lower == "rounding bottom":
            rounding = detect_rounding_bottom(analysis_df, window=100)
            if not rounding:
                rounding = _detect_with_trimming(
                    detect_rounding_bottom,
                    analysis_df,
                    window=100,
                )
            if not rounding:
                rounding = _load_persisted_pattern(sym, pattern_name, RoundingBottomPattern)
            if isinstance(rounding, RoundingBottomPattern):
                start_idx = rounding.offset
                end_idx = rounding.offset + rounding.length - 1
                if 0 <= start_idx < end_idx < len(plot_df):
                    poly = np.poly1d(rounding.coeffs)
                    local_x = np.linspace(0, rounding.length - 1, 100)
                    curve_y = poly(local_x)
                    global_indices = np.linspace(start_idx, end_idx, 100)
                    index_array = np.arange(len(plot_df))
                    date_values = np.interp(global_indices, index_array, plot_df['Date'].values)
                    ax_price.plot(date_values, curve_y, color='#2ca02c', linewidth=1.5, label='Rounded Base')
                    overlay_added = True
        elif name_lower == "breakaway gap":
            gap = detect_breakaway_gap(analysis_df)
            if not gap:
                gap = _detect_with_trimming(
                    detect_breakaway_gap,
                    analysis_df,
                    min_size=35,
                )
            if not gap:
                gap = _load_persisted_pattern(sym, pattern_name, BreakawayGapPattern)
            if isinstance(gap, BreakawayGapPattern):
                prev_idx = gap.prev_close_idx
                curr_idx = gap.curr_open_idx
                if 0 <= prev_idx < len(plot_df) and 0 <= curr_idx < len(plot_df):
                    prev_date = plot_df['Date'].iloc[prev_idx]
                    curr_date = plot_df['Date'].iloc[curr_idx]
                    ax_price.vlines(curr_date, gap.prev_close, gap.curr_open, colors='#ff7f0e', linewidth=2, label='Gap')
                    ax_price.scatter([curr_date], [gap.curr_close], color='#2ca02c', marker='^', s=60, zorder=6, label='Gap Close')
                    overlay_added = True

    if overlay_added:
        handles, labels = ax_price.get_legend_handles_labels()
        legend_map = {label: handle for handle, label in zip(handles, labels) if label}
        if legend_map:
            ax_price.legend(list(legend_map.values()), list(legend_map.keys()), loc='upper left')


    bar_positions = bins[:-1]
    bar_heights = np.diff(bins)
    ax_vp.barh(
        bar_positions,
        norm_vol.values,
        height=bar_heights,
        align='edge',
        color='#8a8a8a',
        alpha=0.8,
    )
    ax_vp.set_ylim(price_min, price_max)
    vp_max = float(norm_vol.max()) if norm_vol.size else 0.0
    if not np.isfinite(vp_max) or vp_max <= 0:
        vp_max = 1.0
    ax_vp.set_xlim(0, vp_max * 1.05)
    ax_vp.invert_xaxis()
    ax_vp.set_xticks([])
    ax_vp.set_xlabel('Volume')
    ax_vp.tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False)
    ax_vp.spines['left'].set_visible(False)

    volume_colors = ['green' if c >= o else 'red' for o, c in zip(plot_df['Open'], plot_df['Close'])]
    ax_volume.bar(plot_df['Date'], plot_df['Volume'], width=0.6, color=volume_colors, align='center')
    ax_volume.set_ylabel('Volume')
    ax_volume.yaxis.set_label_position('right')

    delta = plot_df['Close'].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=14, min_periods=14).mean()
    avg_loss = losses.rolling(window=14, min_periods=14).mean()
    avg_loss_replaced = avg_loss.replace(0, pd.NA)
    rs = avg_gain / avg_loss_replaced
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100
    rsi[(avg_loss == 0) & (avg_gain == 0)] = 50

    ax_rsi.plot(plot_df['Date'], rsi, color='purple', linewidth=1)
    ax_rsi.axhline(70, color='red', linestyle='--', linewidth=1)
    ax_rsi.axhline(30, color='green', linestyle='--', linewidth=1)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel('RSI (14)')
    ax_rsi.yaxis.set_label_position('right')
    ax_rsi.set_yticks([0, 30, 50, 70, 100])

    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_price.xaxis.set_major_locator(mticker.MaxNLocator(10))
    fig.autofmt_xdate(rotation=45)
    fig.align_xlabels([ax_rsi])

    ax_price.tick_params(axis='x', labelbottom=False)
    ax_volume.tick_params(axis='x', labelbottom=False)
    ax_volume.tick_params(axis='y', labelright=True, right=True, labelleft=False, left=False)
    ax_rsi.tick_params(axis='y', labelright=True, right=True, labelleft=False, left=False)
    ax_rsi.set_xlabel('Date')
    ax_price.set_xlim(plot_df['Date'].min() - 0.5, plot_df['Date'].max() + 0.5)

    long_name = _long_name_cache.get(sym)
    if long_name is None:
        try:
            long_name = yf.Ticker(sym).info.get("longName")
        except Exception:
            long_name = None
        _long_name_cache[sym] = long_name

    title_name = long_name or sym
    chart_title = f"{title_name} ({sym}) Candlestick with Volume Profile"
    if pattern_name:
        chart_title += f" — {pattern_name.strip()}"

    ax_price.set_ylabel('Price')
    ax_price.set_title(chart_title)
    ax_price.yaxis.set_label_position('right')
    ax_price.tick_params(axis='y', labelright=True, right=True, labelleft=False, left=False)

    left_price_axis = ax_price.secondary_yaxis('left', functions=(lambda y: y, lambda y: y))
    left_price_axis.set_ylabel('Price')
    left_price_axis.tick_params(axis='y', labelleft=True, left=True)

    canvas = FigureCanvasTkAgg(fig, master=canvas_container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
def load_watchlist():
    tree.delete(*tree.get_children())
    try:
        fieldnames, rows = _read_watchlist_rows()
    except Exception as exc:
        messagebox.showerror("Watchlist", f"Failed to load watchlist:\n{exc}")
        return

    price_cache: dict[str, Optional[float]] = {}
    persisted_rows: list[dict[str, Any]] = []
    any_updates = False

    for row in rows:
        working_row: dict[str, Any] = {key: row.get(key, "") for key in fieldnames}
        updated = _maybe_refresh_watchlist_row(working_row, price_cache)
        if updated:
            any_updates = True

        persisted_rows.append({key: working_row.get(key, "") for key in fieldnames})

        values = []
        for column in WATCHLIST_COLUMNS:
            raw_value = working_row.get(column, "") if working_row else ""
            values.append(_format_watchlist_value(column, raw_value, working_row))
        tree.insert("", "end", values=values)

    if any_updates:
        try:
            with WATCHLIST_PATH.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(persisted_rows)
        except Exception as exc:
            messagebox.showwarning(
                "Watchlist",
                f"Updated prices could not be saved to the watchlist:\n{exc}",
            )

def refresh_watchlist():
    script_dir = SCRIPT_DIR
    transfer_script = script_dir / "transfer_watchlist.py"
    local_watchlist = WATCHLIST_PATH

    try:
        if transfer_script.exists():
            subprocess.run([sys.executable, str(transfer_script)], check=True, capture_output=True, text=True)
        else:
            shared_dirs = []
            env_shared = os.getenv("WATCHLIST_SHARED_DIR")
            if env_shared:
                shared_dirs.append(Path(env_shared))
            shared_dirs.append(script_dir / "shared")

            destination = local_watchlist
            shared_found = False
            for shared_dir in shared_dirs:
                if not shared_dir:
                    continue
                source = Path(shared_dir) / "watchlist.csv"
                if source.exists():
                    shutil.copy2(source, destination)
                    shared_found = True
                    break

            if not shared_found:
                if local_watchlist.exists():
                    print(
                        "[Watchlist] Shared watchlist not found. Using existing local copy.")
                else:
                    messagebox.showerror(
                        "Reload Watchlist",
                        "Could not locate watchlist.csv in shared directories or locally.",
                    )
                    return
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.strip() if e.stderr else str(e)
        messagebox.showerror("Reload Watchlist", f"Failed to refresh watchlist:\n{error_message}")
        return
    except Exception as e:
        messagebox.showerror("Reload Watchlist", f"Failed to refresh watchlist:\n{e}")
        return

    load_watchlist()

def sort_treeview(tree, col, descending=False):
    def sort_key(item):
        value = tree.set(item, col)
        if value in ("", None):
            return (2, "")
        try:
            return (0, float(value))
        except (ValueError, TypeError):
            return (1, str(value).lower())

    items = list(tree.get_children(""))
    items.sort(key=sort_key, reverse=descending)

    for index, iid in enumerate(items):
        tree.move(iid, "", index)

    tree.heading(col, command=lambda: sort_treeview(tree, col, not descending))

def setup_layout():
    global tree, chart_frame, symbol_var, qty_var, entry_var, sl_var, total_value_var, order_tree
    global scan_status_var, scan_button

    last_edited = {"field": None}

    def update_shares(*args):
        if last_edited["field"] != "total_value":
            return
        try:
            entry = float(entry_var.get())
            total_value = float(total_value_var.get())
            if entry > 0:
                qty = round(total_value / entry, 6)
                qty_var.set(qty)
        except ValueError:
            pass

    def update_total_value(*args):
        if last_edited["field"] != "qty":
            return
        try:
            entry = float(entry_var.get())
            qty = float(qty_var.get())
            if entry > 0 and qty > 0:
                total_value = round(entry * qty, 2)
                total_value_var.set(total_value)
        except ValueError:
            pass

    def update_from_entry(*args):
        if last_edited["field"] == "entry":
            try:
                entry = float(entry_var.get())
                total_value = float(total_value_var.get())
                if entry > 0:
                    qty = round(total_value / entry, 6)
                    qty_var.set(qty)
                else:
                    qty_var.set("")
            except ValueError:
                pass
            update_total_value()

    def mark_last_edited(field_name):
        last_edited["field"] = field_name

    top = tk.Frame(root); top.pack(fill="x", padx=10, pady=5)
    scan_status_var = tk.StringVar(value="Idle")
    scan_button = tk.Button(top, text="Pattern Scanner", command=run_scan)
    scan_button.pack(side="left")
    tk.Label(top, textvariable=scan_status_var).pack(side="left", padx=10)
    tk.Button(top, text="Reload Watchlist", command=refresh_watchlist).pack(side="right", padx=5)

    tree_frame = tk.Frame(root); tree_frame.pack(fill="x", padx=10)
    tree = ttk.Treeview(tree_frame, columns=WATCHLIST_COLUMNS, show="headings", height=8)
    for col in WATCHLIST_COLUMNS:
        tree.heading(col, text=col.replace("_", " ").title(), command=lambda c=col: sort_treeview(tree, c))
        tree.column(col, width=110, anchor="center")
    tree.pack(side="left", fill="x", expand=True)
    ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview).pack(side="right", fill="y")

    def _handle_tree_double_click(event):
        item_id = tree.identify_row(event.y)
        if not item_id:
            return
        values = tree.item(item_id).get("values")
        if not values:
            return
        symbol = str(values[0]).strip().upper()
        if not symbol:
            return
        if add_symbol_to_my_watchlist(symbol):
            _mark_last_added_my_watchlist_symbol(symbol)
            show_candlestick()

    tree.bind("<Double-1>", _handle_tree_double_click)

    middle = tk.Frame(root); middle.pack(fill="both", expand=True, padx=10)
    chart_frame = tk.Frame(middle); chart_frame.pack(side="left", fill="both", expand=True)
    order_frame = tk.LabelFrame(middle, text="Trading Controls", padx=10, pady=10)
    order_frame.pack(side="right", fill="y", padx=10)

    symbol_var = tk.StringVar()
    qty_var = tk.StringVar()
    entry_var = tk.StringVar()
    sl_var = tk.StringVar()
    total_value_var = tk.StringVar()

    qty_var.trace_add("write", lambda *args: mark_last_edited("qty") or update_total_value())
    entry_var.trace_add("write", lambda *args: mark_last_edited("entry") or update_from_entry())
    total_value_var.trace_add("write", lambda *args: mark_last_edited("total_value") or update_shares())

    tk.Label(order_frame, text="Symbol").pack()
    tk.Entry(order_frame, textvariable=symbol_var).pack()

    tk.Label(order_frame, text="Total Traded Value ($)").pack()
    tk.Entry(order_frame, textvariable=total_value_var).pack()

    tk.Label(order_frame, text="Entry Price").pack()
    tk.Entry(order_frame, textvariable=entry_var).pack()

    tk.Label(order_frame, text="Shares").pack()
    tk.Entry(order_frame, textvariable=qty_var).pack()

    tk.Label(order_frame, text="Stop Loss Price").pack()
    tk.Entry(order_frame, textvariable=sl_var).pack()

    tk.Button(order_frame, text="Place Order", command=place_order, bg="blue", fg="white").pack(pady=10)

    _init_my_watchlist_section(order_frame)

    tk.Label(root, text="Order Window", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)
    order_frame = tk.Frame(root); order_frame.pack(fill="x", padx=10)
    order_tree = ttk.Treeview(order_frame, columns=("Symbol", "Shares", "Entry Price", "Stop Loss Price", "Status", "order_id"), show="headings")
    for col in ("Symbol", "Shares", "Entry Price", "Stop Loss Price", "Status", "order_id"):
        width = 0 if col == "order_id" else 150
        order_tree.heading(col, text=col, command=lambda c=col: sort_treeview(order_tree, c))
        order_tree.column(col, width=width, anchor="center", stretch=(col != "order_id"))
    order_tree.pack(side="left", fill="x", expand=True)
    ttk.Scrollbar(order_frame, orient="vertical", command=order_tree.yview).pack(side="right", fill="y")

    controls = tk.Frame(root); controls.pack(fill="x", padx=10, pady=5)
    tk.Button(controls, text="Delete from Watchlist", command=lambda: tree.delete(*tree.selection())).pack(side="left", padx=5)
    tk.Button(controls, text="Delete Selected Order", command=delete_selected_order).pack(side="left", padx=5)
    tk.Button(controls, text="Reload Active Monitors", command=rerun_stop_loss_monitors).pack(side="left", padx=5)

def force_exit():
    os.kill(os.getpid(), signal.SIGTERM)

root.protocol("WM_DELETE_WINDOW", force_exit)
setup_layout()
load_watchlist()
tree.bind("<<TreeviewSelect>>", lambda e: root.after(100, show_candlestick))
root.bind("<Up>", lambda e: navigate_chart("up"))
root.bind("<Down>", lambda e: navigate_chart("down"))
root.mainloop()