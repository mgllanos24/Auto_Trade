"""Simple desktop GUI for running the swing trading bot with live data.

The application is intentionally lightweight so it can be run on most
machines that already have ``tkinter`` available.  It communicates with the
Alpaca REST API using credentials supplied through environment variables and
feeds the fetched market data into :class:`~trading_bot.SwingTradingBot` to
generate trade plans.

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
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd
import requests
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox

from trading_bot import SwingTradingBot, TradePlan


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

DEFAULT_DATA_URL = "https://data.alpaca.markets"
DEFAULT_PAPER_TRADING_URL = "https://paper-api.alpaca.markets"


class AlpacaCredentialsError(RuntimeError):
    """Raised when Alpaca credentials are missing."""


def _read_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise AlpacaCredentialsError(
            "Missing required environment variable: %s" % name
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
        self._bot = SwingTradingBot()

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

                plan = self._bot.generate_trade_plan(symbol, df)
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

