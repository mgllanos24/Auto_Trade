import pandas as pd

from cup_handle_scanner import CupHandleHit, scan_cup_handle
from pattern_scanner import detect_cup_and_handle


def _build_ohlcv(closes):
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    data = {
        "open": [c * 0.99 for c in closes],
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.98 for c in closes],
        "close": closes,
        "volume": [1_000_000] * len(closes),
    }
    return pd.DataFrame(data, index=index)


def test_scan_cup_handle_returns_hit_without_breakout():
    closes = [
        100,
        99,
        97,
        95,
        94,
        96,
        97,
        98,
        99,
        100,
        99,
        98.5,
        98,
        98.2,
        98.4,
    ]
    df = _build_ohlcv(closes)

    hits = scan_cup_handle(df, cup_window=10, handle_window=5, tolerance=0.15)

    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, CupHandleHit)
    assert not hit.breakout
    assert hit.cup_depth_pct > 0.1
    assert hit.handle_slope < 0


def test_scan_cup_handle_honours_breakout_requirement():
    closes = [
        100,
        99,
        97,
        95,
        94,
        96,
        97,
        98,
        99,
        100,
        99,
        98.5,
        98,
        98.2,
        101,
    ]
    df = _build_ohlcv(closes)

    hits = scan_cup_handle(
        df,
        cup_window=10,
        handle_window=5,
        tolerance=0.15,
        require_breakout=True,
    )

    assert len(hits) == 1
    assert hits[0].breakout
    assert hits[0].breakout_price >= hits[0].right_peak


def test_detect_cup_and_handle_wraps_scanner():
    closes = [
        100,
        99,
        97,
        95,
        94,
        96,
        97,
        98,
        99,
        100,
        99,
        98.5,
        98,
        98.2,
        99,
    ]
    df = _build_ohlcv(closes)

    assert detect_cup_and_handle(df, cup_window=10, handle_window=5, tolerance=0.15)

    hit = detect_cup_and_handle(
        df,
        cup_window=10,
        handle_window=5,
        tolerance=0.15,
        return_hit=True,
    )

    assert isinstance(hit, CupHandleHit)
    assert hit.cup_depth_pct > 0.1
