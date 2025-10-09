import numpy as np
import pandas as pd

from trading_bot import SwingTradingBot
from swing_trading_screener import SwingScreenerConfig


def _build_dataframe(prices, volume: float = 1_000_000) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=len(prices), freq="D")
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": prices * 0.995,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": volume,
        },
        index=dates,
    )


def _trend_with_noise(start: float, slope: float, count: int = 120) -> np.ndarray:
    steps = np.arange(count)
    oscillation = np.sin(np.linspace(0, 6 * np.pi, count)) * 0.5
    return start + slope * steps + oscillation


def test_generate_trade_plan_filters_by_risk_reward():
    prices = _trend_with_noise(40.0, 0.1)
    df = _build_dataframe(prices)

    config = SwingScreenerConfig(
        min_average_volume=100_000,
        max_atr_pct=0.05,
        rsi_bounds=(40.0, 80.0),
    )

    bot = SwingTradingBot(
        screener_config=config,
        min_rr=1.2,
        atr_stop_multiple=1.0,
        atr_target_multiple=1.5,
    )

    plan = bot.generate_trade_plan("TREND", df)
    assert plan is not None
    assert plan.risk_reward >= 1.2

    conservative_bot = SwingTradingBot(
        screener_config=config,
        min_rr=1.2,
        atr_stop_multiple=1.0,
        atr_target_multiple=1.0,
    )

    assert conservative_bot.generate_trade_plan("TREND", df) is None


def test_generate_trade_signals_returns_buy_and_sell_orders():
    prices = _trend_with_noise(30.0, 0.08)
    df = _build_dataframe(prices)

    bot = SwingTradingBot(
        min_rr=1.2,
        atr_stop_multiple=1.0,
        atr_target_multiple=2.0,
    )

    signals = bot.generate_trade_signals({"TREND": df})
    assert len(signals) == 3
    actions = {signal.action for signal in signals}
    assert actions == {"buy", "sell"}
    reasons = {signal.reason for signal in signals if signal.action == "sell"}
    assert reasons == {"risk_management_stop", "profit_target"}
