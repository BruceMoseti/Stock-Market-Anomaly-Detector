"""Preprocessing and feature tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import add_features
from src.preprocessing import clean_prices


def _ohlcv(n: int = 10, ticker: str = "AAPL") -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(100, 109, n)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def test_missing_prices_handled():
    df = _ohlcv(12)
    df.loc[3, "close"] = np.nan
    df.loc[4, "open"] = np.nan
    cleaned = clean_prices(df, max_fill_days=5)
    assert cleaned["close"].isna().sum() == 0
    assert len(cleaned) == 12
    assert cleaned.iloc[3]["close"] == cleaned.iloc[2]["close"]


def test_negative_prices_dropped_if_unfillable():
    df = _ohlcv(8)
    df.loc[0, "close"] = -5
    cleaned = clean_prices(df, max_fill_days=0)
    assert (cleaned["close"] > 0).all()
    assert len(cleaned) == 7


def test_returns_and_volume_ratio():
    df = _ohlcv(30)
    featured = add_features(df, volatility_window=5, volume_window=5)
    expected = df["close"].pct_change().iloc[1]
    assert abs(featured["return"].iloc[1] - expected) < 1e-12
    assert featured["volume_ratio"].iloc[10] > 0
    assert featured["volatility"].iloc[10] >= 0
