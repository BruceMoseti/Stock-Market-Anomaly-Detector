"""Download and cache historical OHLCV data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def default_tickers() -> list[str]:
    return [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "AMZN", "TSLA", "AMD", "INTC", "AVGO",
        "JPM", "BAC", "GS", "MS", "V",
        "MA", "WMT", "COST", "HD", "NKE",
        "PG", "KO", "PEP", "JNJ", "UNH",
        "PFE", "MRK", "ABBV", "LLY", "XOM",
        "CVX", "COP", "BA", "CAT", "GE",
        "HON", "DIS", "NFLX", "CMCSA", "T",
        "VZ", "ORCL", "CRM", "ADBE", "CSCO",
        "QCOM", "TXN", "AMAT", "MU", "IBM",
        "SPY", "QQQ", "IWM", "DIA", "GLD",
    ]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLUMNS)


def _normalize_downloaded(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return _empty_frame()

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = [str(x).lower() for x in raw.columns.get_level_values(0).unique()]
        price_names = {"open", "high", "low", "close", "adj close", "volume"}
        ticker_is_level0 = set(level0).isdisjoint(price_names)
        for ticker in tickers:
            try:
                part = raw[ticker] if ticker_is_level0 else raw.xs(ticker, axis=1, level=1)
            except (KeyError, ValueError):
                continue
            frames.append(_single_ticker_frame(part, ticker))
    else:
        ticker = tickers[0]
        frames.append(_single_ticker_frame(raw, ticker))

    if not frames:
        return _empty_frame()
    out = pd.concat(frames, ignore_index=True)
    return out[OHLCV_COLUMNS]


def _single_ticker_frame(part: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = part.copy()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    close_col = "close" if "close" in df.columns else "adj_close"
    if close_col not in df.columns:
        return _empty_frame()
    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date", close_col: "close"})
    for col in ["open", "high", "low", "volume"]:
        if col not in df.columns:
            df[col] = np.nan if col != "volume" else 0.0
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[OHLCV_COLUMNS]


def load_cached(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return _empty_frame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df[OHLCV_COLUMNS]


def save_prices(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance and optionally cache it."""
    if use_cache and cache_path:
        cached = load_cached(cache_path)
        if not cached.empty:
            return cached

    import yfinance as yf

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        threads=True,
        progress=False,
        group_by="ticker",
    )
    df = _normalize_downloaded(raw, tickers)
    if df.empty:
        raise RuntimeError(
            "Yahoo Finance returned no rows. Check tickers/dates or use --synthetic."
        )
    if cache_path:
        save_prices(df, cache_path)
    return df


def generate_synthetic_prices(
    tickers: list[str],
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic OHLCV panel for offline runs and tests."""
    dates = pd.bdate_range(start=start, end=end)
    rng = np.random.default_rng(seed)
    frames = []
    for i, ticker in enumerate(tickers):
        n = len(dates)
        drift = 0.00035
        vol = 0.012 + 0.004 * (i % 6)
        returns = rng.normal(drift, vol, size=n)
        close = 50.0 * (1.0 + 0.4 * (i % 5)) * np.cumprod(1.0 + returns)
        prev_close = np.r_[close[0], close[:-1]]
        gap = rng.normal(0.0, 0.003, size=n)
        open_ = prev_close * (1.0 + gap)
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, size=n)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, size=n)))
        volume = rng.lognormal(mean=16.2 - 0.05 * (i % 7), sigma=0.35, size=n)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
