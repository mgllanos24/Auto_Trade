
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
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

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
WATCHLIST_COLUMNS = ["symbol", "breakout_high", "rr_ratio", "target_price", "stop_loss", "timestamp", "pattern", "direction"]
MONITOR_FILE = str(SCRIPT_DIR / "active_monitors.json")

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
    sym = tree.item(sel[0])["values"][0]
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

    price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    plot_df = df[price_columns].dropna().copy()
    if plot_df.empty:
        messagebox.showinfo("Chart", f"No complete OHLC data available for {sym}.")
        return

    plot_df.sort_index(inplace=True)
    plot_df['Date'] = mdates.date2num(plot_df.index.to_pydatetime())
    ohlc = plot_df[['Date', 'Open', 'High', 'Low', 'Close']].values

    price_min = plot_df['Low'].min()
    price_max = plot_df['High'].max()
    price_range = price_max - price_min

    if price_range == 0:
        price_range = max(price_max * 0.01, 0.5)

    target_bins = 60

    price_points = plot_df[['Open', 'High', 'Low', 'Close']].to_numpy().ravel()
    price_points = price_points[~np.isnan(price_points)]
    if price_points.size > 1:
        unique_prices = np.unique(price_points)
        price_steps = np.diff(unique_prices)
        price_steps = price_steps[price_steps > 0]
        native_step = price_steps.min() if price_steps.size else 0
    else:
        native_step = 0

    if native_step > 0:
        raw_bin_size = price_range / target_bins if price_range else native_step
        multiples = max(1, np.ceil(raw_bin_size / native_step))
        bin_size = multiples * native_step
    else:
        bin_size = max(price_range / target_bins, 0.01)

    bin_start = np.floor(price_min / bin_size) * bin_size
    bin_end = np.ceil(price_max / bin_size) * bin_size
    bins = np.arange(bin_start, bin_end + bin_size, bin_size)
    if bin_size > 0:
        precision = max(0, int(np.ceil(-np.log10(bin_size))) + 2)
        precision = min(10, precision)
    else:
        precision = 6
    bins = np.round(bins, decimals=precision)
    if bins[-1] < bin_end:
        bins = np.append(bins, bin_end)
    if bins.size < 2:
        bins = np.array([bin_start, bin_start + bin_size])

    price_levels = 0.5 * (bins[1:] + bins[:-1])
    price_index = pd.Index(price_levels, name='Price')
    # Build a price-volume profile that aligns each candle's volume with the
    # actual price range traded during that candle.  Distributing volume across
    # the high/low range ties the histogram bars to the price axis instead of
    # assigning all of the volume to the closing price.
    volume_by_price = pd.Series(0.0, index=price_index)

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
    ax_vp = fig.add_subplot(gs[:, 1], sharey=ax_price)

    for t, o, h, l, c in ohlc:
        color = 'green' if c >= o else 'red'
        ax_price.plot([t, t], [l, h], color='black')
        ax_price.add_patch(plt.Rectangle((t - 0.2, min(o, c)), 0.4, abs(c - o), color=color))

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

    ax_price.set_ylabel('Price')
    ax_price.set_title(f"{title_name} ({sym}) Candlestick with Volume Profile")
    ax_price.yaxis.set_label_position('right')
    ax_price.tick_params(axis='y', labelright=True, right=True, labelleft=False, left=False)

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
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