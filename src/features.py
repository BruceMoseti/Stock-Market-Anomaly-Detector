"""Market features used by the anomaly detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(
    df: pd.DataFrame,
    volatility_window: int = 20,
    volume_window: int = 20,
) -> pd.DataFrame:
    """Add return, log-return, rolling volatility, volume ratio, and overnight gap."""
    out = df.sort_values(["ticker", "date"]).copy()
    grouped = out.groupby("ticker", sort=False)

    prev_close = grouped["close"].shift(1)
    out["return"] = grouped["close"].pct_change()
    out["log_return"] = np.log(out["close"] / prev_close)
    out["gap"] = (out["open"] - prev_close) / prev_close
    out["volatility"] = grouped["return"].transform(
        lambda s: s.rolling(volatility_window, min_periods=volatility_window).std()
    )
    avg_volume = grouped["volume"].transform(
        lambda s: s.rolling(volume_window, min_periods=volume_window).mean()
    )
    out["volume_ratio"] = out["volume"] / avg_volume.replace(0, np.nan)
    return out
