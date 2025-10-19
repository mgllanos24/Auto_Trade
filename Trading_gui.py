
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
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import threading
import signal
import json
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from urllib.error import URLError
from urllib.request import urlopen
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
WATCHLIST_PATH = SCRIPT_DIR / "watchlist.csv"
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

root = tk.Tk()
root.title("Stock Scanner GUI")
root.geometry("1400x900")

symbol_data = {}
_long_name_cache = {}
_institution_activity_cache: dict[str, dict[str, Any]] = {}
INSTITUTION_CACHE_TTL = 60 * 30  # 30 minutes
WATCHLIST_COLUMNS = ["symbol", "breakout_high", "rr_ratio", "target_price", "stop_loss", "timestamp", "pattern", "direction"]
MONITOR_FILE = str(SCRIPT_DIR / "active_monitors.json")


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


def _normalise_report_date(raw: Any) -> str:
    if not raw:
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


def fetch_institution_activity(symbol: str) -> tuple[list[dict[str, Any]], Optional[str], str]:
    now = time.time()
    cached_payload = _institution_activity_cache.get(symbol) or {}

    if isinstance(cached_payload, tuple) and len(cached_payload) == 2:
        cached_payload = {
            "timestamp": cached_payload[0],
            "data_timestamp": cached_payload[0],
            "entries": cached_payload[1],
            "message": None,
            "level": "info",
        }

    cached_timestamp = cached_payload.get("timestamp")
    cached_entries = cached_payload.get("entries", [])
    cached_message = cached_payload.get("message")
    cached_level = cached_payload.get("level", "info")
    data_timestamp = cached_payload.get("data_timestamp", cached_timestamp)

    if cached_timestamp is not None and now - cached_timestamp < INSTITUTION_CACHE_TTL:
        return cached_entries, cached_message, cached_level

    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=institutionOwnership"

    def _cache_and_return(
        entries: list[dict[str, Any]],
        message: Optional[str],
        level: str,
        *,
        timestamp: float,
        data_ts: Optional[float] = None,
    ) -> tuple[list[dict[str, Any]], Optional[str], str]:
        payload = {
            "timestamp": timestamp,
            "data_timestamp": data_ts if data_ts is not None else timestamp,
            "entries": entries,
            "message": message,
            "level": level,
        }
        _institution_activity_cache[symbol] = payload
        return entries, message, level

    def _format_cached_timestamp(ts: Optional[float]) -> str:
        if not ts:
            return "previously"
        try:
            return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M UTC")
        except (OverflowError, OSError, ValueError):
            return "previously"

    try:
        with urlopen(url, timeout=10) as response:
            payload = json.load(response)
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        if cached_entries:
            message = (
                f"Showing cached institutional activity from {_format_cached_timestamp(data_timestamp)} "
                f"because the latest request failed: {reason}."
            )
            return _cache_and_return(cached_entries, message, "warning", timestamp=now, data_ts=data_timestamp)
        return [], f"Unable to load institutional activity ({reason}).", "error"
    except Exception as exc:  # pragma: no cover - defensive
        if cached_entries:
            message = (
                f"Showing cached institutional activity from {_format_cached_timestamp(data_timestamp)} "
                f"because the latest request failed: {exc}."
            )
            return _cache_and_return(cached_entries, message, "warning", timestamp=now, data_ts=data_timestamp)
        return [], f"Unable to load institutional activity ({exc}).", "error"

    try:
        results = payload.get("quoteSummary", {}).get("result", [])
        if not results:
            return _cache_and_return([], None, "info", timestamp=now)
        ownership = results[0].get("institutionOwnership", {})
        raw_entries = ownership.get("ownershipList", [])
    except AttributeError:
        return _cache_and_return([], None, "info", timestamp=now)

    parsed_entries: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        net_activity = entry.get("netActivity")
        purchased = entry.get("purchased")
        sold = entry.get("soldOut")
        organization = entry.get("organization") or "Unknown"
        report_datetime = _parse_report_datetime(entry.get("reportDate"))
        parsed_entries.append(
            {
                "organization": organization,
                "net_activity": net_activity if net_activity is not None else 0,
                "purchased": purchased if purchased is not None else 0,
                "sold": sold if sold is not None else 0,
                "report_date": _normalise_report_date(entry.get("reportDate")),
                "report_datetime": report_datetime,
            }
        )

    parsed_entries.sort(key=lambda item: abs(item.get("net_activity", 0)), reverse=True)
    cutoff = datetime.utcnow() - timedelta(days=182)
    recent_entries = [
        entry for entry in parsed_entries if entry.get("report_datetime") and entry["report_datetime"] >= cutoff
    ]

    return _cache_and_return(recent_entries, None, "info", timestamp=now)


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
            is_scanning = False

        root.after(0, finalize)

    threading.Thread(target=worker, daemon=True).start()

def flatten_yf_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
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
    import numpy as np

    sel = tree.selection()
    if not sel:
        return
    row_values = tree.item(sel[0])["values"]
    sym = row_values[0]
    pattern_name = ""
    if len(row_values) > 6 and row_values[6]:
        pattern_name = str(row_values[6])
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
    for w in chart_frame.winfo_children():
        w.destroy()

    activity_entries, activity_message, activity_level = fetch_institution_activity(sym)

    info_panel = tk.Frame(chart_frame, bg="#f5f5f5", width=260)
    info_panel.pack(side="left", fill="y", padx=(0, 10), pady=5)
    info_panel.pack_propagate(False)

    tk.Label(
        info_panel,
        text="Institutional Activity",
        font=("Arial", 12, "bold"),
        anchor="w",
        bg="#f5f5f5",
    ).pack(fill="x")

    content_pad = {"fill": "x", "anchor": "w", "padx": 4, "pady": 2}

    if activity_message:
        color_map = {"error": "red", "warning": "#c27d00", "info": "#333333"}
        tk.Label(
            info_panel,
            text=activity_message,
            justify="left",
            wraplength=240,
            fg=color_map.get(activity_level, "#333333"),
            bg="#f5f5f5",
            anchor="w",
        ).pack(fill="x", padx=4, pady=6)

    if activity_entries:
        total_net = sum(entry.get("net_activity", 0) or 0 for entry in activity_entries)
        net_color = "green" if total_net > 0 else "red" if total_net < 0 else "#555555"
        tk.Label(
            info_panel,
            text=f"Net activity (6M): {_format_share_amount(total_net)} shares",
            fg=net_color,
            bg="#f5f5f5",
            anchor="w",
        ).pack(fill="x", padx=4, pady=(6, 2))

        top_accumulators = sorted(
            (entry for entry in activity_entries if (entry.get("net_activity", 0) or 0) > 0),
            key=lambda entry: entry.get("net_activity", 0) or 0,
            reverse=True,
        )[:3]
        top_sellers = sorted(
            (entry for entry in activity_entries if (entry.get("net_activity", 0) or 0) < 0),
            key=lambda entry: entry.get("net_activity", 0) or 0,
        )[:3]

        if not top_accumulators and not top_sellers:
            tk.Label(
                info_panel,
                text="No institutional accumulation or selling in the past 6 months.",
                justify="left",
                wraplength=240,
                bg="#f5f5f5",
                anchor="w",
            ).pack(fill="x", padx=4, pady=6)

        def _render_section(title: str, entries: list[dict[str, Any]], positive: bool) -> None:
            if not entries:
                return

            tk.Label(
                info_panel,
                text=title,
                font=("Arial", 11, "bold"),
                bg="#f5f5f5",
                anchor="w",
            ).pack(fill="x", padx=4, pady=(8, 2))

            for entry in entries:
                frame = tk.Frame(info_panel, bg="#f5f5f5")
                frame.pack(fill="x", padx=2, pady=(4, 0))

                tk.Label(
                    frame,
                    text=entry.get("organization", "Unknown"),
                    font=("Arial", 10, "bold"),
                    bg="#f5f5f5",
                    anchor="w",
                    justify="left",
                    wraplength=240,
                ).pack(fill="x")

                net_value = entry.get("net_activity", 0) or 0
                action_color = "green" if positive else "red"
                net_display = _format_share_amount(abs(net_value))
                action_text = "Accumulated" if positive else "Sold"

                tk.Label(
                    frame,
                    text=f"{action_text}: {net_display} shares",
                    fg=action_color,
                    bg="#f5f5f5",
                    anchor="w",
                ).pack(**content_pad)

                tk.Label(
                    frame,
                    text=f"Report: {entry.get('report_date', 'N/A')}",
                    fg="#666666",
                    bg="#f5f5f5",
                    anchor="w",
                ).pack(**content_pad)

        _render_section("Top accumulators (6M)", top_accumulators, True)
        _render_section("Top sellers (6M)", top_sellers, False)
    else:
        tk.Label(
            info_panel,
            text="No institutional accumulation or selling reported in the past 6 months.",
            justify="left",
            wraplength=240,
            bg="#f5f5f5",
            anchor="w",
        ).pack(fill="x", padx=4, pady=6)

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

    price_min = plot_df['Low'].min()
    price_max = plot_df['High'].max()
    price_range = price_max - price_min

    if price_range == 0:
        price_range = max(price_max * 0.01, 0.5)

    target_bins = 60
    bin_size = max(price_range / target_bins, 0.01)

    bins = np.arange(price_min, price_max + bin_size, bin_size)
    if bins[-1] < price_max:
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

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=[20, 5], height_ratios=[3, 1, 1], hspace=0.05, wspace=0.05)

    ax_price = fig.add_subplot(gs[0, 0])
    ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_rsi = fig.add_subplot(gs[2, 0], sharex=ax_price)
    ax_vp = fig.add_subplot(gs[0, 1], sharey=ax_price)
    for row in (1, 2):
        fig.add_subplot(gs[row, 1]).axis('off')

    for t, o, h, l, c in ohlc:
        color = 'green' if c >= o else 'red'
        ax_price.plot([t, t], [l, h], color='black')
        ax_price.add_patch(plt.Rectangle((t - 0.2, min(o, c)), 0.4, abs(c - o), color=color))

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
            if hits:
                hit = max(hits, key=lambda h: h.right_idx)
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
            if isinstance(hit, CupHandleHit):
                resistance = hit.resistance
                ax_price.axhline(resistance, color='#9467bd', linestyle='--', linewidth=1.5, label='Resistance')
                overlay_added = True
        elif name_lower == "inverse head and shoulders":
            ihs = detect_inverse_head_shoulders(analysis_df)
            if not ihs:
                ihs = _detect_with_trimming(
                    detect_inverse_head_shoulders,
                    analysis_df,
                    min_size=60,
                )
            if isinstance(ihs, InverseHeadShouldersPattern):
                try:
                    indices = [ihs.left_idx, ihs.head_idx, ihs.right_idx]
                    lows = [ihs.left_low, ihs.head_low, ihs.right_low]
                    valid = [idx for idx in indices if 0 <= idx < len(plot_df)]
                    if len(valid) == 3:
                        dates = [plot_df['Date'].iloc[idx] for idx in indices]
                        ax_price.scatter(dates, lows, color='#d62728', s=50, label='Shoulders/Head', zorder=6)
                        overlay_added = True

                    if 0 <= ihs.neckline_left_idx < len(plot_df) and 0 <= ihs.neckline_right_idx < len(plot_df):
                        neck_dates = [
                            plot_df['Date'].iloc[ihs.neckline_left_idx],
                            plot_df['Date'].iloc[ihs.neckline_right_idx],
                        ]
                        neck_prices = [ihs.neckline_left, ihs.neckline_right]
                        ax_price.plot(neck_dates, neck_prices, color='#9467bd', linestyle='--', linewidth=1.5, label='Neckline')
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
    ax_vp.barh(bar_positions, norm_vol.values, height=bar_heights, align='edge', color='gray')
    ax_vp.set_ylim(price_min, price_max)
    ax_vp.set_xticks([])
    ax_vp.set_xlabel('Volume')
    ax_vp.tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False)

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

    canvas = FigureCanvasTkAgg(fig, master=canvas_container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
def load_watchlist():
    tree.delete(*tree.get_children())
    if not WATCHLIST_PATH.exists():
        return
    df = pd.read_csv(WATCHLIST_PATH)
    for _, row in df.iterrows():
        tree.insert("", "end", values=[row[col] for col in WATCHLIST_COLUMNS])

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