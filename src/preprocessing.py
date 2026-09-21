"""Clean and validate OHLCV panels."""

from __future__ import annotations

import pandas as pd

REQUIRED = ["date", "ticker", "open", "high", "low", "close", "volume"]


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_prices(df: pd.DataFrame, max_fill_days: int = 5) -> pd.DataFrame:
    """Drop invalid rows, fill short gaps within each ticker, keep a clean panel."""
    validate_schema(df)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    numeric = ["open", "high", "low", "close", "volume"]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out.loc[out["close"] <= 0, "close"] = pd.NA
    out.loc[out["volume"] < 0, "volume"] = pd.NA

    filled = []
    for _, group in out.groupby("ticker", sort=False):
        g = group.copy()
        price_cols = ["open", "high", "low", "close"]
        if max_fill_days > 0:
            g[price_cols] = g[price_cols].ffill(limit=max_fill_days)
            g["volume"] = g["volume"].ffill(limit=max_fill_days)
        g = g.dropna(subset=["close"])
        g["volume"] = g["volume"].fillna(0.0)
        for col in ["open", "high", "low"]:
            g[col] = g[col].fillna(g["close"])
        filled.append(g)

    if not filled:
        return out.iloc[0:0]
    out = pd.concat(filled, ignore_index=True)
    out = out.drop_duplicates(subset=["ticker", "date"], keep="last")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)
