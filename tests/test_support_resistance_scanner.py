from tests.test_pattern_scanner import _install_stub_modules

_install_stub_modules()

import pandas as pd

from pattern_scanner import (
    SupportResistancePattern,
    _collect_pattern_candidates,
    detect_support_resistance_volume,
)


def _build_frame_with_level_volume_spikes() -> pd.DataFrame:
    closes = [106 + (i % 5) * 2 for i in range(120)]
    lows = [c - 2 for c in closes]
    highs = [c + 2 for c in closes]
    volume = [1_000_000] * len(closes)

    support_touch_indices = [20, 45, 70, 95]
    resistance_touch_indices = [30, 60, 90, 110]

    for idx in support_touch_indices:
        lows[idx] = 99.5
        closes[idx] = 100.3
        highs[idx] = 102.0
        volume[idx] = 4_000_000

    for idx in resistance_touch_indices:
        highs[idx] = 121.5
        closes[idx] = 120.8
        lows[idx] = 118.5
        volume[idx] = 3_800_000

    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        }
    )


def test_detect_support_resistance_volume_identifies_levels_with_heavy_volume():
    df = _build_frame_with_level_volume_spikes()

    pattern = detect_support_resistance_volume(df)

    assert isinstance(pattern, SupportResistancePattern)
    assert pattern.support_touch_count >= 3
    assert pattern.resistance_touch_count >= 3
    assert pattern.support_heavy_volume_ratio > 1.3 or pattern.resistance_heavy_volume_ratio > 1.3


def test_collect_candidates_includes_support_resistance_pattern():
    df = _build_frame_with_level_volume_spikes()

    candidates = _collect_pattern_candidates(df, ["Support and Resistance"])

    assert candidates
    assert candidates[0].name == "Support and Resistance"
