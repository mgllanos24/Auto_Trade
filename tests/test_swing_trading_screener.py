import numpy as np
import pandas as pd
from typing import Sequence

from swing_trading_screener import (
    SwingCandidate,
    SwingEvaluation,
    SwingScreenerConfig,
    evaluate_swing_setup,
    screen_swing_candidates,
)


def _build_dataframe(prices: Sequence[float], volume: float = 1_000_000) -> pd.DataFrame:
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
    oscillation = np.sin(np.linspace(0, 8 * np.pi, count)) * 0.8
    return start + slope * steps + oscillation


def test_evaluate_swing_setup_returns_candidate_for_trending_symbol():
    prices = _trend_with_noise(50.0, 0.08)
    df = _build_dataframe(prices)

    config = SwingScreenerConfig(
        min_average_volume=100_000,
        max_atr_pct=0.05,
        rsi_bounds=(40.0, 80.0),
    )

    candidate = evaluate_swing_setup("TREND", df, config)
    assert isinstance(candidate, SwingCandidate)
    assert candidate.symbol == "TREND"
    assert candidate.trend_strength > 0
    assert 40.0 <= candidate.rsi <= 80.0
    assert candidate.atr_pct <= config.max_atr_pct


def test_evaluate_swing_setup_filters_out_low_volume_symbols():
    prices = _trend_with_noise(40.0, 0.05)
    df = _build_dataframe(prices, volume=50_000)

    config = SwingScreenerConfig(min_average_volume=200_000)
    candidate = evaluate_swing_setup("LOWVOL", df, config)
    assert candidate is None


def test_screen_swing_candidates_orders_by_trend_and_momentum():
    base = _trend_with_noise(40.0, 0.06)
    strong = _trend_with_noise(40.0, 0.09)

    data = {
        "BASE": _build_dataframe(base),
        "STRONG": _build_dataframe(strong),
    }

    config = SwingScreenerConfig(
        min_average_volume=100_000,
        max_atr_pct=0.05,
        rsi_bounds=(40.0, 80.0),
    )

    results = screen_swing_candidates(data, config=config)
    assert [c.symbol for c in results] == ["STRONG", "BASE"]
    assert results[0].trend_strength >= results[1].trend_strength


def test_evaluate_swing_setup_exposes_rejection_reason():
    df = _build_dataframe(np.linspace(10, 20, 10))

    result = evaluate_swing_setup("TOO_SHORT", df, return_reason=True)

    assert isinstance(result, SwingEvaluation)
    assert result.candidate is None
    assert result.reason == "not enough candles"


def test_evaluate_swing_setup_returns_candidate_with_reason_flag():
    prices = _trend_with_noise(50.0, 0.08)
    df = _build_dataframe(prices)

    config = SwingScreenerConfig(
        min_average_volume=100_000,
        max_atr_pct=0.05,
        rsi_bounds=(40.0, 80.0),
    )

    result = evaluate_swing_setup("TREND", df, config, return_reason=True)

    assert isinstance(result, SwingEvaluation)
    assert isinstance(result.candidate, SwingCandidate)
    assert result.reason is None

