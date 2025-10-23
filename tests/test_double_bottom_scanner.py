import pandas as pd

from double_bottom_scanner import DoubleBottomHit, scan_double_bottoms


def _make_ohlcv(lows, volumes):
    closes = [val + 3 for val in lows]
    highs = [val + 4 for val in lows]
    opens = [val + 2 for val in lows]
    index = pd.date_range("2024-01-01", periods=len(lows), freq="D")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=index,
    )


def test_scan_double_bottoms_detects_classic_pattern():
    lows = [
        100,
        98,
        96,
        94,
        90,
        92,
        94,
        96,
        97,
        98,
        96,
        94,
        92,
        90.5,
        90,
        92,
        95,
        99,
        104,
        110,
        115,
        120,
        125,
        130,
    ]
    volumes = [
        1_500_000,
        1_450_000,
        1_400_000,
        1_350_000,
        1_300_000,
        1_200_000,
        1_150_000,
        1_100_000,
        1_050_000,
        1_000_000,
        950_000,
        900_000,
        850_000,
        800_000,
        780_000,
        800_000,
        850_000,
        900_000,
        1_000_000,
        1_200_000,
        1_400_000,
        1_600_000,
        1_800_000,
        2_000_000,
    ]
    df = _make_ohlcv(lows, volumes)

    hits = scan_double_bottoms(
        df,
        window=len(df),
        tolerance=0.05,
        min_bounce=0.05,
        require_breakout=True,
    )

    assert hits, "Expected to detect at least one double bottom pattern"
    best_hit = max(hits, key=lambda hit: hit.bounce_pct)
    assert isinstance(best_hit, DoubleBottomHit)
    assert best_hit.breakout is True
    assert best_hit.bounce_pct >= 0.05
    assert best_hit.support < best_hit.neckline


def test_scan_double_bottoms_volume_contraction_requirement():
    lows = [
        100,
        98,
        96,
        94,
        90,
        92,
        94,
        96,
        97,
        98,
        96,
        94,
        92,
        90.5,
        90,
        92,
        95,
        99,
        104,
        110,
        115,
        120,
        125,
        130,
    ]
    steady_volume = [1_000_000] * len(lows)
    df = _make_ohlcv(lows, steady_volume)

    hits = scan_double_bottoms(
        df,
        window=len(df),
        tolerance=0.05,
        min_bounce=0.05,
        require_breakout=True,
        require_volume_contraction=True,
    )

    assert hits == []


def test_scan_double_bottoms_filters_out_stale_patterns():
    pattern_section = [
        100,
        98,
        96,
        94,
        90,
        92,
        94,
        96,
        97,
        98,
        96,
        94,
        92,
        90.5,
        90,
        92,
        95,
        99,
    ]
    trailing_section = [
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
    ]
    lows = pattern_section + trailing_section
    volumes = [1_000_000] * len(lows)
    df = _make_ohlcv(lows, volumes)

    stale_hits = scan_double_bottoms(
        df,
        window=len(df),
        tolerance=0.05,
        min_bounce=0.05,
        require_breakout=False,
    )

    assert stale_hits == []

    permissive_hits = scan_double_bottoms(
        df,
        window=len(df),
        tolerance=0.05,
        min_bounce=0.05,
        require_breakout=False,
        max_pattern_age=None,
    )

    assert permissive_hits, "Disabling the age filter should surface the pattern"
