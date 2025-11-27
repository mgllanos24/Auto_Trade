"""Deprecated wrapper for building the OHLCV master CSV.

All download logic now lives in :mod:`build_ohlcv_last2y`.  This wrapper keeps
existing entry points working while ensuring the single source of data
construction is consolidated in one place.
"""

from build_ohlcv_last2y import main


if __name__ == "__main__":
    main()
