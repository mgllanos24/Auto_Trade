
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
import subprocess
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

root = tk.Tk()
root.title("Stock Scanner GUI")
root.geometry("1400x900")

symbol_data = {}
WATCHLIST_COLUMNS = ["symbol", "breakout_high", "rr_ratio", "target_price", "stop_loss", "timestamp", "pattern", "direction"]
MONITOR_FILE = "active_monitors.json"

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

def run_scan():
    subprocess.Popen([sys.executable, "scan_eod.py"])

def flatten_yf_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def download_all_data():
    symbol_data.clear()
    if not os.path.exists("watchlist.csv"):
        return
    df = pd.read_csv("watchlist.csv")
    for sym in df["symbol"]:
        try:
            data = yf.download(sym, period="12mo", progress=False)
            data = flatten_yf_columns(data)
            symbol_data[sym] = data
            data.to_csv(f"data/{sym}.csv")
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
    if sym not in symbol_data:
        if os.path.exists(f"data/{sym}.csv"):
            symbol_data[sym] = pd.read_csv(f"data/{sym}.csv", index_col=0, parse_dates=True)
        else:
            df = yf.download(sym, period="12mo", auto_adjust=False, progress=False)
            df = flatten_yf_columns(df)
            if df.empty:
                messagebox.showinfo("Chart", f"No data available for {sym}.")
                return
            symbol_data[sym] = df
            df.to_csv(f"data/{sym}.csv")

    df = symbol_data[sym]
    for w in chart_frame.winfo_children():
        w.destroy()

    df['Date'] = mdates.date2num(df.index.to_pydatetime())
    ohlc = df[['Date', 'Open', 'High', 'Low', 'Close']].dropna().values

    bin_size = 1.0
    price_min = df['Low'].min()
    price_max = df['High'].max()
    bins = np.arange(price_min, price_max + bin_size, bin_size)
    price_levels = 0.5 * (bins[1:] + bins[:-1])
    df['price_bin'] = np.digitize(df['Close'], bins)
    volume_by_price = df.groupby('price_bin')['Volume'].sum()
    volume_by_price.index = price_levels[volume_by_price.index - 1]
    norm_vol = volume_by_price / volume_by_price.max()

    fig, (ax_vp, ax) = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 4), width_ratios=[1.2, 4], sharey=True
    )

    for t, o, h, l, c in ohlc:
        color = 'green' if c >= o else 'red'
        ax.plot([t, t], [l, h], color='black')
        ax.add_patch(plt.Rectangle((t - 0.2, min(o, c)), 0.4, abs(c - o), color=color))

    ax_vp.barh(volume_by_price.index, norm_vol, height=bin_size * 0.9, color='gray')
    ax_vp.set_xticks([])
    ax_vp.set_xlabel('Volume')
    ax_vp.invert_xaxis()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    fig.autofmt_xdate()
    ax.set_ylabel('Price')
    ax.set_title(f"{sym} Candlestick with Volume Profile")
    ax.yaxis.set_label_position('right')
    ax.tick_params(axis='y', labelright=True, right=True, labelleft=False, left=False)
    ax_vp.tick_params(axis='y', labelleft=True, left=True, labelright=False, right=False)

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
def load_watchlist():
    tree.delete(*tree.get_children())
    if not os.path.exists("watchlist.csv"):
        return
    df = pd.read_csv("watchlist.csv")
    for _, row in df.iterrows():
        tree.insert("", "end", values=[row[col] for col in WATCHLIST_COLUMNS])

def setup_layout():
    global tree, chart_frame, symbol_var, qty_var, entry_var, sl_var, total_value_var, order_tree

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
    tk.Button(top, text="Run scan_eod.py", command=run_scan).pack(side="left")

    tree_frame = tk.Frame(root); tree_frame.pack(fill="x", padx=10)
    tree = ttk.Treeview(tree_frame, columns=WATCHLIST_COLUMNS, show="headings", height=8)
    for col in WATCHLIST_COLUMNS:
        tree.heading(col, text=col.replace("_", " ").title())
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
        order_tree.heading(col, text=col)
        order_tree.column(col, width=width, anchor="center", stretch=(col != "order_id"))
    order_tree.pack(side="left", fill="x", expand=True)
    ttk.Scrollbar(order_frame, orient="vertical", command=order_tree.yview).pack(side="right", fill="y")

    controls = tk.Frame(root); controls.pack(fill="x", padx=10, pady=5)
    tk.Button(controls, text="Download All Data", command=download_all_data).pack(side="left", padx=5)
    tk.Button(controls, text="Show Candlestick Chart", command=show_candlestick).pack(side="left", padx=5)
    tk.Button(controls, text="Delete from Watchlist", command=lambda: tree.delete(*tree.selection())).pack(side="left", padx=5)
    tk.Button(controls, text="Reload Watchlist", command=load_watchlist).pack(side="left", padx=5)
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