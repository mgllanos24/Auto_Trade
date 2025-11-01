import pandas as pd

from cup_handle_scanner import CupHandleHit, detect_cup_and_handle


def _make_close_series(values):
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"close": values}, index=index)


def test_detect_cup_and_handle_returns_hit_with_metrics():
    cup_left = [100, 98, 96, 94, 92, 90, 88, 86, 84, 82]
    cup_right = [82, 84, 86, 88, 90, 92, 94, 96, 98, 100]
    cup = cup_left + cup_right
    handle = [99.0, 98.8, 98.6, 98.4, 98.2]
    closes = cup + handle

    df = _make_close_series(closes)

    hit = detect_cup_and_handle(df, cup_window=len(cup), handle_window=len(handle))

    assert isinstance(hit, CupHandleHit)
    assert hit.resistance > hit.cup_depth
    assert 0 < hit.cup_depth_pct < 1
    assert hit.handle_length == len(handle)
    assert 0 <= hit.handle_pullback_pct < 0.05
    assert hit.handle_slope < 0
    assert hit.left_peak_idx == 0
    assert hit.cup_low_idx == len(cup_left) - 1
    assert hit.right_peak_idx == len(cup) - 1
    assert hit.handle_start_idx == len(cup)
    assert hit.handle_end_idx == len(cup) + len(handle) - 1
    assert hit.handle_low_idx == len(cup) + len(handle) - 1


def test_detect_cup_and_handle_rejects_shallow_cup():
    shallow_cup = [100, 99.5, 99, 98.5, 98, 98.5, 99, 99.5]
    handle = [99.4, 99.2, 99.1]
    closes = shallow_cup + handle
    df = _make_close_series(closes)

    hit = detect_cup_and_handle(df, cup_window=len(shallow_cup), handle_window=len(handle))

    assert hit is None
