"""Simple desktop GUI for monitoring swing trading setups with live data.

The application is intentionally lightweight so it can be run on most
machines that already have ``tkinter`` available.  It communicates with the
Alpaca REST API using credentials supplied through environment variables and
feeds the fetched market data into :class:`SwingTradePlanner` to generate
trade plans similar to the retired ``trading_bot`` module.

The GUI exposes a start/stop button that controls a background worker thread.
That worker periodically downloads the latest bar data for the configured
symbols, evaluates the setup using the trading bot, and prints the resulting
trade plans in a scrolling log widget.

This module keeps all networking concerns isolated from the trading logic so
it is straightforward to extend the automation in the future (for example to
place orders automatically instead of just logging recommendations).
"""

from __future__ import annotations

import os
from pathlib import Path
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd
import requests
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox

from swing_trading_screener import (
    SwingCandidate,
    SwingScreenerConfig,
    evaluate_swing_setup,
)


@dataclass
class TradeTransaction:
    """Simple container describing a manual trade entry."""

    timestamp: pd.Timestamp
    symbol: str
    side: str
    quantity: float
    price: float
    notes: str = ""


@dataclass(frozen=True)
class TradePlan:
    """Simplified trade plan derived directly from screener candidates."""

    symbol: str
    entry: float
    stop: float
    target: float
    risk_reward: float
    candidate: SwingCandidate


class SwingTradePlanner:
    """Lightweight helper that mirrors the old trading bot's calculations."""

    def __init__(
        self,
        *,
        screener_config: Optional[SwingScreenerConfig] = None,
        min_rr: float = 1.2,
        atr_stop_multiple: float = 1.0,
        atr_target_multiple: float = 2.0,
    ) -> None:
        if min_rr <= 0:
            raise ValueError("min_rr must be positive")
        if atr_stop_multiple <= 0 or atr_target_multiple <= 0:
            raise ValueError("ATR multiples must be positive")
        if atr_target_multiple <= atr_stop_multiple:
            raise ValueError("atr_target_multiple must be greater than atr_stop_multiple")

        self.config = screener_config or SwingScreenerConfig()
        self.min_rr = float(min_rr)
        self.atr_stop_multiple = float(atr_stop_multiple)
        self.atr_target_multiple = float(atr_target_multiple)

    def generate_trade_plan(self, symbol: str, df: pd.DataFrame) -> Optional[TradePlan]:
        """Return a trade plan when the screener emits a qualifying candidate."""

        candidate = evaluate_swing_setup(symbol, df, self.config)
        if candidate is None:
            return None

        atr_value = candidate.atr_pct * candidate.close
        if not pd.notna(atr_value) or atr_value <= 0:
            return None

        entry = float(candidate.close)
        stop = entry - self.atr_stop_multiple * atr_value
        target = entry + self.atr_target_multiple * atr_value

        if stop <= 0 or target <= entry:
            return None

        risk = entry - stop
        reward = target - entry
        if risk <= 0:
            return None

        rr_ratio = reward / risk
        if rr_ratio < self.min_rr:
            return None

        return TradePlan(
            symbol=symbol,
            entry=entry,
            stop=stop,
            target=target,
            risk_reward=rr_ratio,
            candidate=candidate,
        )


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

DEFAULT_DATA_URL = "https://data.alpaca.markets"
DEFAULT_PAPER_TRADING_URL = "https://paper-api.alpaca.markets"


class AlpacaCredentialsError(RuntimeError):
    """Raised when Alpaca credentials are missing."""


def _load_env_file(path: str = ".env") -> None:
    """Populate :mod:`os.environ` from a simple ``.env`` file.

    The desktop application is often run outside of a managed shell where
    environment variables are inconvenient to define.  This helper mirrors the
    behaviour of popular ``dotenv`` packages without introducing an additional
    dependency.  Each non-empty, non-comment line must follow ``KEY=VALUE``
    syntax.  Existing environment variables always take precedence over values
    defined in the file.
    """

    env_path = Path(path)
    if getattr(_load_env_file, "_loaded", False):  # pragma: no cover - guard rail
        return

    setattr(_load_env_file, "_loaded", True)

    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _read_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise AlpacaCredentialsError(
            (
                "Missing required environment variable: {name}.\n\n"
                "Set it before launching the app or add it to a .env file in the "
                "project directory with entries such as:\n"
                "ALPACA_API_KEY=your_key_here\n"
                "ALPACA_API_SECRET=your_secret_here"
            ).format(name=name)
        )
    return value


@dataclass
class AlpacaClient:
    """Very small HTTP client for retrieving bar data from Alpaca."""

    api_key: str
    api_secret: str
    data_url: str = DEFAULT_DATA_URL
    base_url: str = DEFAULT_PAPER_TRADING_URL

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Hour",
        limit: int = 200,
    ) -> Optional[pd.DataFrame]:
        """Return a :class:`pandas.DataFrame` of recent bars for ``symbol``."""

        params = {"timeframe": timeframe, "limit": limit}
        url = f"{self.data_url}/v2/stocks/{symbol}/bars"
        try:
            response = requests.get(url, params=params, headers=self._headers(), timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network errors
            raise RuntimeError(f"Failed to fetch data for {symbol}: {exc}") from exc

        payload = response.json()
        bars = payload.get("bars", [])
        if not bars:
            return None

        df = pd.DataFrame.from_records(bars)
        # Normalise column names to match expectations of trading bot utilities
        rename_map = {
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
        df.rename(columns=rename_map, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        return df


def build_alpaca_client_from_env() -> AlpacaClient:
    """Load API credentials from environment variables."""

    _load_env_file()
    api_key = _read_env("ALPACA_API_KEY")
    api_secret = _read_env("ALPACA_API_SECRET")
    data_url = os.getenv("ALPACA_DATA_URL", DEFAULT_DATA_URL)
    base_url = os.getenv("ALPACA_BASE_URL", DEFAULT_PAPER_TRADING_URL)
    return AlpacaClient(api_key=api_key, api_secret=api_secret, data_url=data_url, base_url=base_url)


# ---------------------------------------------------------------------------
# GUI implementation
# ---------------------------------------------------------------------------


class TradingBotWorker(threading.Thread):
    """Background worker responsible for fetching data and running the bot."""

    def __init__(
        self,
        client: AlpacaClient,
        symbols: Sequence[str],
        timeframe: str,
        poll_interval: float,
        output_queue: "queue.Queue[str]",
    ) -> None:
        super().__init__(daemon=True)
        self.client = client
        self.symbols = [symbol.upper() for symbol in symbols if symbol]
        self.timeframe = timeframe
        self.poll_interval = poll_interval
        self.output_queue = output_queue
        self._stop_event = threading.Event()
        self._planner = SwingTradePlanner()

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - integration behaviour
        if not self.symbols:
            self.output_queue.put("No symbols configured; worker exiting.")
            return

        while not self._stop_event.is_set():
            for symbol in self.symbols:
                if self._stop_event.is_set():
                    break
                try:
                    df = self.client.get_bars(symbol, timeframe=self.timeframe)
                except Exception as exc:
                    self.output_queue.put(f"[{symbol}] Error fetching data: {exc}")
                    continue

                if df is None or df.empty:
                    self.output_queue.put(f"[{symbol}] No market data returned.")
                    continue

                plan = self._planner.generate_trade_plan(symbol, df)
                if plan is None:
                    self.output_queue.put(f"[{symbol}] No trade setup detected.")
                else:
                    self.output_queue.put(_format_trade_plan(plan))

            # Sleep outside the symbol loop so we can interrupt quickly when stopping
            for _ in range(int(self.poll_interval * 10)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)


def _format_trade_plan(plan: TradePlan) -> str:
    """Create a human friendly summary of a :class:`TradePlan`."""

    return (
        f"[{plan.symbol}] Entry={plan.entry:.2f} Stop={plan.stop:.2f} "
        f"Target={plan.target:.2f} RR={plan.risk_reward:.2f}"
    )


class TradingBotApp:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Live Swing Trading Bot")
        self.root.geometry("720x480")

        self.status_var = tk.StringVar(value="Idle")
        self.symbols_var = tk.StringVar(value="AAPL,MSFT,SPY")
        self.timeframe_var = tk.StringVar(value="1Hour")
        self.interval_var = tk.DoubleVar(value=15.0)

        self._output_queue: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[TradingBotWorker] = None
        self._transaction_window: Optional[tk.Toplevel] = None
        self._transactions: list[TradeTransaction] = []
        self._transaction_entries: dict[str, tk.Variable] = {}
        self._transactions_tree: Optional[ttk.Treeview] = None

        self._build_widgets()
        self._poll_output_queue()

    # ------------------------------------------------------------------
    # GUI construction helpers
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        control_frame = ttk.LabelFrame(self.root, text="Configuration")
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(control_frame, text="Symbols (comma separated)").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.symbols_var, width=40).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Timeframe").grid(row=1, column=0, sticky=tk.W)
        ttk.Combobox(
            control_frame,
            textvariable=self.timeframe_var,
            values=["1Min", "5Min", "15Min", "1Hour", "1Day"],
            state="readonly",
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(control_frame, text="Refresh interval (seconds)").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(control_frame, from_=5, to=3600, textvariable=self.interval_var, increment=5).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=5
        )

        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))

        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_bot, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Transactions", command=self._open_transaction_window).pack(
            side=tk.LEFT, padx=5
        )

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10)
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)

        log_frame = ttk.LabelFrame(self.root, text="Activity Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.log_widget = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, wrap=tk.WORD)
        self.log_widget.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def start_bot(self) -> None:
        if self._worker is not None:
            messagebox.showinfo("Trading Bot", "Bot is already running.")
            return

        try:
            client = build_alpaca_client_from_env()
        except AlpacaCredentialsError as exc:
            messagebox.showerror("Trading Bot", str(exc))
            return

        symbols = [symbol.strip() for symbol in self.symbols_var.get().split(",")]
        timeframe = self.timeframe_var.get()
        interval = float(self.interval_var.get())

        self._worker = TradingBotWorker(
            client=client,
            symbols=symbols,
            timeframe=timeframe,
            poll_interval=interval,
            output_queue=self._output_queue,
        )
        self._worker.start()
        self.status_var.set("Running")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self._append_log("Trading bot started.")

    def stop_bot(self) -> None:
        worker = self._worker
        if worker is None:
            return

        worker.stop()
        worker.join(timeout=5)
        self._worker = None
        self.status_var.set("Stopped")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self._append_log("Trading bot stopped.")

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------
    def _append_log(self, message: str) -> None:
        timestamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self.log_widget.configure(state=tk.NORMAL)
        self.log_widget.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_widget.configure(state=tk.DISABLED)
        self.log_widget.see(tk.END)

    def _poll_output_queue(self) -> None:
        try:
            while True:
                message = self._output_queue.get_nowait()
                self._append_log(message)
        except queue.Empty:
            pass
        finally:
            self.root.after(500, self._poll_output_queue)

    # ------------------------------------------------------------------
    # Trade transaction window
    # ------------------------------------------------------------------
    def _open_transaction_window(self) -> None:
        if self._transaction_window is not None and tk.Toplevel.winfo_exists(self._transaction_window):
            self._transaction_window.deiconify()
            self._transaction_window.lift()
            return

        self._transaction_window = tk.Toplevel(self.root)
        self._transaction_window.title("Trade Transactions")
        self._transaction_window.geometry("640x360")
        self._transaction_window.protocol("WM_DELETE_WINDOW", self._close_transaction_window)
        self._build_transaction_window(self._transaction_window)

    def _close_transaction_window(self) -> None:
        if self._transaction_window is None:
            return
        self._transaction_window.destroy()
        self._transaction_window = None
        self._transaction_entries.clear()
        self._transactions_tree = None

    def _build_transaction_window(self, window: tk.Toplevel) -> None:
        form_frame = ttk.LabelFrame(window, text="New Transaction")
        form_frame.pack(fill=tk.X, padx=10, pady=10)

        fields: list[tuple[str, tk.Variable, Optional[Sequence[str]]]] = [
            ("Symbol", tk.StringVar(), None),
            ("Side", tk.StringVar(value="Buy"), ("Buy", "Sell")),
            ("Quantity", tk.StringVar(), None),
            ("Price", tk.StringVar(), None),
            ("Notes", tk.StringVar(), None),
        ]

        for row, (label, variable, choices) in enumerate(fields):
            ttk.Label(form_frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
            if choices is None:
                entry = ttk.Entry(form_frame, textvariable=variable)
            else:
                entry = ttk.Combobox(form_frame, textvariable=variable, values=choices, state="readonly")
            entry.grid(row=row, column=1, sticky=tk.W + tk.E, padx=5, pady=5)
            self._transaction_entries[label.lower()] = variable

        add_button = ttk.Button(form_frame, text="Add Transaction", command=self._add_transaction)
        add_button.grid(row=len(fields), column=0, columnspan=2, pady=(10, 0))

        tree_frame = ttk.LabelFrame(window, text="Transaction History")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        columns = ("timestamp", "symbol", "side", "quantity", "price", "notes")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        headings = {
            "timestamp": "Timestamp",
            "symbol": "Symbol",
            "side": "Side",
            "quantity": "Quantity",
            "price": "Price",
            "notes": "Notes",
        }
        for column, heading in headings.items():
            tree.heading(column, text=heading)
            if column == "notes":
                tree.column(column, width=180)
            else:
                tree.column(column, width=90, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._transactions_tree = tree
        self._refresh_transactions_view()

    def _add_transaction(self) -> None:
        entries = self._transaction_entries
        symbol = entries.get("symbol")
        side = entries.get("side")
        quantity = entries.get("quantity")
        price = entries.get("price")
        notes = entries.get("notes")

        if symbol is None or side is None or quantity is None or price is None or notes is None:
            return

        symbol_value = symbol.get().strip().upper()
        side_value = side.get().strip().capitalize()
        quantity_value = quantity.get().strip()
        price_value = price.get().strip()
        notes_value = notes.get().strip()

        if not symbol_value:
            messagebox.showerror("Trade Transactions", "Symbol is required.")
            return

        try:
            quantity_float = float(quantity_value)
            price_float = float(price_value)
        except ValueError:
            messagebox.showerror(
                "Trade Transactions",
                "Quantity and Price must be numeric values.",
            )
            return

        transaction = TradeTransaction(
            timestamp=pd.Timestamp.utcnow(),
            symbol=symbol_value,
            side=side_value,
            quantity=quantity_float,
            price=price_float,
            notes=notes_value,
        )
        self._transactions.append(transaction)
        self._refresh_transactions_view()

        for variable in (symbol, side, quantity, price, notes):
            if variable is side:
                variable.set("Buy")
            else:
                variable.set("")

        self._append_log(
            "Recorded transaction: "
            f"{transaction.symbol} {transaction.side} {transaction.quantity} @ {transaction.price}"
        )

    def _refresh_transactions_view(self) -> None:
        tree = self._transactions_tree
        if tree is None:
            return

        for item in tree.get_children():
            tree.delete(item)

        for tx in self._transactions:
            tree.insert(
                "",
                tk.END,
                values=(
                    tx.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    tx.symbol,
                    tx.side,
                    f"{tx.quantity:g}",
                    f"{tx.price:.2f}",
                    tx.notes,
                ),
            )


def main() -> None:  # pragma: no cover - thin wrapper
    root = tk.Tk()
    app = TradingBotApp(root)

    def _on_close() -> None:
        app.stop_bot()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

