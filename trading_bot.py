"""Swing trading bot built on top of the screener utilities.

The original project focuses on identifying potential setups via the
``swing_trading_screener`` module.  This file adds a thin orchestration
layer that turns those raw screening results into actionable trading
instructions with deterministic position sizing rules.

The bot operates purely on historical data supplied by the caller.  No
network or brokerage interaction is performed which makes the module
fully unit testable and safe to use in environments without market
connectivity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence

import math
import pandas as pd

from swing_trading_screener import (
    SwingCandidate,
    SwingScreenerConfig,
    evaluate_swing_setup,
)

__all__ = [
    "TradePlan",
    "TradeSignal",
    "SwingTradingBot",
]


@dataclass(frozen=True)
class TradePlan:
    """Description of a planned trade derived from a screener candidate."""

    symbol: str
    entry: float
    stop: float
    target: float
    risk_reward: float
    candidate: SwingCandidate


@dataclass(frozen=True)
class TradeSignal:
    """Concrete instruction for the trading bot to execute."""

    symbol: str
    action: str
    price: float
    reason: str
    timestamp: pd.Timestamp


class SwingTradingBot:
    """Convert screener results into actionable buy and sell instructions.

    Parameters
    ----------
    screener_config:
        Optional custom configuration for the swing trading screener.
    min_rr:
        Minimal acceptable reward to risk ratio.  Trades falling below
        the threshold are discarded.
    atr_stop_multiple:
        Multiplier applied to the ATR value to determine the stop price.
    atr_target_multiple:
        Multiplier applied to the ATR value to determine the profit
        target.  Must be strictly greater than ``atr_stop_multiple`` in
        order to produce a positive reward.
    """

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
        self.config = screener_config or SwingScreenerConfig()
        self.min_rr = float(min_rr)
        self.atr_stop_multiple = float(atr_stop_multiple)
        self.atr_target_multiple = float(atr_target_multiple)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_trade_plan(
        self, symbol: str, df: pd.DataFrame
    ) -> Optional[TradePlan]:
        """Return a :class:`TradePlan` when the screener issues a buy signal.

        The screener determines whether a symbol qualifies for a swing
        entry.  When a candidate is returned we translate it into entry,
        stop and target prices using fixed ATR multiples.  Candidates
        that do not meet the required reward to risk ratio are filtered
        out.
        """

        candidate = evaluate_swing_setup(symbol, df, self.config)
        if candidate is None:
            return None

        atr_value = candidate.atr_pct * candidate.close
        if not math.isfinite(atr_value) or atr_value <= 0:
            return None

        entry = float(candidate.close)
        stop = entry - self.atr_stop_multiple * atr_value
        target = entry + self.atr_target_multiple * atr_value

        if stop <= 0 or target <= entry:
            return None

        risk = entry - stop
        reward = target - entry
        if risk <= 0 or reward <= 0:
            return None

        rr = reward / risk
        if rr < self.min_rr:
            return None

        return TradePlan(
            symbol=symbol,
            entry=entry,
            stop=stop,
            target=target,
            risk_reward=float(rr),
            candidate=candidate,
        )

    def generate_trade_plans(
        self, data: Mapping[str, pd.DataFrame]
    ) -> List[TradePlan]:
        """Evaluate multiple symbols and return qualifying trade plans."""

        plans: List[TradePlan] = []
        for symbol, df in data.items():
            plan = self.generate_trade_plan(symbol, df)
            if plan is not None:
                plans.append(plan)

        plans.sort(
            key=lambda plan: (
                plan.candidate.trend_strength,
                plan.candidate.momentum_score,
            ),
            reverse=True,
        )
        return plans

    def generate_trade_signals(
        self, data: Mapping[str, pd.DataFrame]
    ) -> List[TradeSignal]:
        """Return buy and sell instructions for the provided dataset."""

        signals: List[TradeSignal] = []
        for plan in self.generate_trade_plans(data):
            df = data.get(plan.symbol)
            timestamp = _latest_timestamp(df)
            signals.extend(_plan_to_signals(plan, timestamp))
        return signals


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _plan_to_signals(plan: TradePlan, timestamp: pd.Timestamp) -> List[TradeSignal]:
    """Translate a :class:`TradePlan` into explicit trading instructions."""

    buy = TradeSignal(
        symbol=plan.symbol,
        action="buy",
        price=plan.entry,
        reason="screener_long_entry",
        timestamp=timestamp,
    )
    stop = TradeSignal(
        symbol=plan.symbol,
        action="sell",
        price=plan.stop,
        reason="risk_management_stop",
        timestamp=timestamp,
    )
    target = TradeSignal(
        symbol=plan.symbol,
        action="sell",
        price=plan.target,
        reason="profit_target",
        timestamp=timestamp,
    )
    return [buy, stop, target]


def _latest_timestamp(df: Optional[pd.DataFrame]) -> pd.Timestamp:
    if df is None or df.empty:
        return pd.NaT
    index = df.index
    if isinstance(index, pd.RangeIndex):
        return pd.Timestamp(index[-1])
    value = index[-1]
    try:
        return pd.Timestamp(value)
    except Exception:
        return pd.Timestamp.now()

